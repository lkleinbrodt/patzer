"""
eval/tournament.py

Play Patzer (our trained model) against Stockfish at various fixed depths.
Results are persisted to eval/results.json so experiments accumulate over time.

Usage:
    # Run a tournament (pulls checkpoint from R2 if needed):
    python eval/tournament.py \\
        --checkpoint checkpoints/patzer_v0/ckpt.pt \\
        --depths 1 3 5 \\
        --games 20 \\
        --stockfish /opt/homebrew/bin/stockfish \\
        --conditioning match_color

    # Try different result-conditioning tokens (use a real .pt path; do not use "..."
    # as the checkpoint — in Fish that expands to ../.. and passes exists() but is not a file):
    python eval/tournament.py --checkpoint checkpoints/patzer_v0/ckpt.pt \\
        --depths 1 --games 10 --conditioning none

    # Print accumulated results:
    python eval/tournament.py --show
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.engine import CONDITIONING_OPTIONS, Patzer, StockfishPlayer

RESULTS_FILE = Path(__file__).parent / "results.json"


def play_game(white, black, max_moves: int = 300) -> str:
    """Play one game; returns '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    move_history: list[str] = []

    while not board.is_game_over(claim_draw=True) and len(move_history) < max_moves * 2:
        player = white if board.turn == chess.WHITE else black

        try:
            uci = player.get_move(board, move_history)
            move = chess.Move.from_uci(uci)
        except Exception as e:
            print(f"  [warn] {player.name} error: {e} — random move")
            move = random.choice(list(board.legal_moves))

        if move not in board.legal_moves:
            print(f"  [warn] {player.name} played illegal {move.uci()} — random move")
            move = random.choice(list(board.legal_moves))

        board.push(move)
        move_history.append(move.uci())

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "1/2-1/2"
    if outcome.winner is chess.WHITE:
        return "1-0"
    if outcome.winner is chess.BLACK:
        return "0-1"
    return "1/2-1/2"


def run_tournament(
    patzer: "Patzer",
    checkpoint_path: str,
    stockfish_binary: str,
    depths: list[int],
    n_games: int,
) -> list[dict]:
    """Play n_games per depth (alternating colors). Returns list of result records."""
    records = []

    for depth in depths:
        sf = StockfishPlayer(stockfish_binary, depth)
        w = l = d = 0

        for game_idx in range(n_games):
            patzer_is_white = game_idx % 2 == 0
            white = patzer if patzer_is_white else sf
            black = sf if patzer_is_white else patzer

            result = play_game(white, black)
            color = "W" if patzer_is_white else "B"
            print(f"  depth={depth} [{game_idx+1}/{n_games}] Patzer={color} → {result}")

            if patzer_is_white:
                if result == "1-0":
                    w += 1
                elif result == "0-1":
                    l += 1
                else:
                    d += 1
            else:
                if result == "0-1":
                    w += 1
                elif result == "1-0":
                    l += 1
                else:
                    d += 1

        sf.close()
        records.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_path,
            "iter_num": patzer.iter_num,
            "stockfish_depth": depth,
            "games": n_games,
            "temperature": patzer.temperature,
            "top_k": patzer.top_k,
            "conditioning": patzer.conditioning,
            "W": w,
            "L": l,
            "D": d,
        })

    return records


def save_results(records: list[dict]):
    existing = load_results()
    existing.extend(records)
    RESULTS_FILE.write_text(json.dumps(existing, indent=2))
    print(f"\n[saved] {len(records)} record(s) → {RESULTS_FILE}")


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    return json.loads(RESULTS_FILE.read_text())


def aggregate_results(records: list[dict]) -> list[dict]:
    """
    Combine records with identical (checkpoint, depth, conditioning, temperature, top_k)
    into single rows by summing W/L/D. This lets you accumulate games across multiple runs.
    """
    from collections import defaultdict
    agg: dict[tuple, dict] = defaultdict(lambda: {"W": 0, "L": 0, "D": 0})
    meta: dict[tuple, dict] = {}

    for r in records:
        key = (
            r["checkpoint"],
            r["stockfish_depth"],
            r.get("conditioning", ""),
            r.get("temperature", 1.0),
            r.get("top_k"),
        )
        agg[key]["W"] += r["W"]
        agg[key]["L"] += r["L"]
        agg[key]["D"] += r["D"]
        meta[key] = {"iter_num": r.get("iter_num", 0), "checkpoint": r["checkpoint"]}

    rows = []
    for key, counts in agg.items():
        ckpt, depth, cond, temp, top_k = key
        rows.append({
            "checkpoint": ckpt,
            "iter_num": meta[key]["iter_num"],
            "stockfish_depth": depth,
            "conditioning": cond,
            "temperature": temp,
            "top_k": top_k,
            **counts,
        })

    rows.sort(key=lambda r: (r["checkpoint"], r["stockfish_depth"], r["conditioning"]))
    return rows


def show_results():
    records = load_results()
    if not records:
        print("No results yet.")
        return

    rows = aggregate_results(records)
    print(f"\n{'Checkpoint':<40} {'Iter':>6} {'Depth':>5} {'Cond':<12} {'T':>4} {'N':>5} {'W':>4} {'L':>4} {'D':>4} {'Score':>7}")
    print("-" * 107)
    for r in rows:
        ckpt = Path(r["checkpoint"]).name
        total = r["W"] + r["L"] + r["D"]
        score = (r["W"] + 0.5 * r["D"]) / total * 100 if total else 0
        print(
            f"{ckpt:<40} {r.get('iter_num', '?'):>6} {r['stockfish_depth']:>5} "
            f"{r['conditioning']:<12} {r['temperature']:>4.1f} "
            f"{total:>5} {r['W']:>4} {r['L']:>4} {r['D']:>4} {score:>6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Patzer vs Stockfish tournament")
    parser.add_argument("--checkpoint", help="Path to checkpoint (local or R2 key)")
    parser.add_argument("--pull-r2", action="store_true", help="Download checkpoint from R2 first")
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--games", type=int, default=20, help="Games per depth (alternates colors)")
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="torch device: cpu | mps | cuda")
    parser.add_argument(
        "--conditioning",
        default="match_color",
        choices=CONDITIONING_OPTIONS,
        help=(
            "Result token prepended to game sequence: "
            "match_color (win-condition per side), white_win, black_win, draw, none"
        ),
    )
    parser.add_argument("--show", action="store_true", help="Print accumulated results and exit")
    args = parser.parse_args()

    if args.show:
        show_results()
        return

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --show")

    ckpt_path = Path(args.checkpoint).expanduser()
    try:
        ckpt_resolved = ckpt_path.resolve()
    except OSError:
        ckpt_resolved = ckpt_path

    if args.pull_r2:
        from patzer.r2 import pull_file
        print(f"[r2] pulling {args.checkpoint} ...")
        if not pull_file(args.checkpoint, ckpt_path):
            print("R2 pull failed — check credentials in .env", file=sys.stderr)
            sys.exit(1)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        print("Use --pull-r2 to download from R2.", file=sys.stderr)
        sys.exit(1)

    if not ckpt_path.is_file():
        print(
            f"Checkpoint must be a file, not a directory: {ckpt_path} → {ckpt_resolved}",
            file=sys.stderr,
        )
        if args.checkpoint.strip() in ("...", "..") or ckpt_path.name == "..":
            print(
                'Hint: Fish expands "..." to ../.. — pass the real .pt path.',
                file=sys.stderr,
            )
        sys.exit(1)

    patzer = Patzer(
        ckpt_path,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioning=args.conditioning,
    )

    records = run_tournament(patzer, args.checkpoint, args.stockfish, args.depths, args.games)
    save_results(records)
    show_results()

    if patzer.illegal_move_count:
        print(f"\n[note] fell back to random move {patzer.illegal_move_count} time(s)")


if __name__ == "__main__":
    main()
