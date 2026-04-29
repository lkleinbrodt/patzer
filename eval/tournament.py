"""
eval/tournament.py

Play Patzer (our trained model) against Stockfish at ELO-limited settings
(recommended) or fixed depths (legacy).
Results are persisted to eval/results.json so experiments accumulate over time.

Usage:
    # Run a smart ELO tournament (recommended):
    python eval/tournament.py \
        --checkpoint checkpoints/patzer_v0/ckpt.pt \
        --smart-elo \
        --games 12 \
        --stockfish /opt/homebrew/bin/stockfish

    # Run a fixed ELO ladder:
    python eval/tournament.py \\
        --checkpoint checkpoints/patzer_v0/ckpt.pt \\
        --stockfish-elo 1000 1200 1400 1600 1800 \\
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


def _play_match(patzer, sf, n_games, label) -> tuple[int, int, int]:
    """Play n_games against sf (alternating colors). Returns (W, L, D) from patzer's perspective."""
    w = l = d = 0
    for game_idx in range(n_games):
        patzer_is_white = game_idx % 2 == 0
        white = patzer if patzer_is_white else sf
        black = sf if patzer_is_white else patzer

        result = play_game(white, black)
        color = "W" if patzer_is_white else "B"
        print(f"  {label} [{game_idx+1}/{n_games}] Patzer={color} → {result}")

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
    return w, l, d


def run_tournament(
    patzer: "Patzer",
    checkpoint_path: str,
    stockfish_binary: str,
    depths: list[int],
    n_games: int,
    elo_limits: list[int] | None = None,
) -> list[dict]:
    """Play n_games per depth/elo (alternating colors). Returns list of result records."""
    records = []

    for depth in depths:
        sf = StockfishPlayer(stockfish_binary, depth=depth)
        w, l, d = _play_match(patzer, sf, n_games, f"depth={depth}")
        sf.close()
        records.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_path,
            "iter_num": patzer.iter_num,
            "stockfish_depth": depth,
            "stockfish_elo": None,
            "games": n_games,
            "temperature": patzer.temperature,
            "top_k": patzer.top_k,
            "conditioning": patzer.conditioning,
            "W": w, "L": l, "D": d,
        })

    for elo in (elo_limits or []):
        sf = StockfishPlayer(stockfish_binary, elo_limit=elo)
        w, l, d = _play_match(patzer, sf, n_games, f"elo={elo}")
        sf.close()
        records.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_path,
            "iter_num": patzer.iter_num,
            "stockfish_depth": None,
            "stockfish_elo": elo,
            "games": n_games,
            "temperature": patzer.temperature,
            "top_k": patzer.top_k,
            "conditioning": patzer.conditioning,
            "W": w, "L": l, "D": d,
        })

    return records


def run_smart_elo_tournament(
    patzer: "Patzer",
    checkpoint_path: str,
    stockfish_binary: str,
    n_games: int,
    anchor_elos: list[int],
    refine_step: int = 100,
    max_refine_rounds: int = 2,
) -> list[dict]:
    """
    Smart workflow:
      1) run anchor ELOs to bracket where score crosses 50%
      2) add midpoint-ish probes around the crossing for sharper estimate
    """
    records: list[dict] = []
    tested: set[int] = set()

    def _run_elo(elo: int) -> tuple[float, dict]:
        sf = StockfishPlayer(stockfish_binary, elo_limit=elo)
        w, l, d = _play_match(patzer, sf, n_games, f"elo={elo}")
        sf.close()
        rec = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": checkpoint_path,
            "iter_num": patzer.iter_num,
            "stockfish_depth": None,
            "stockfish_elo": elo,
            "games": n_games,
            "temperature": patzer.temperature,
            "top_k": patzer.top_k,
            "conditioning": patzer.conditioning,
            "W": w, "L": l, "D": d,
        }
        score = (w + 0.5 * d) / n_games if n_games else 0.0
        return score, rec

    for elo in sorted(anchor_elos):
        if elo in tested:
            continue
        score, rec = _run_elo(elo)
        records.append(rec)
        tested.add(elo)
        print(f"  [smart] elo={elo} score={score:.3f}")

    for _ in range(max_refine_rounds):
        points = sorted(
            ((r["stockfish_elo"], (r["W"] + 0.5 * r["D"]) / r["games"]) for r in records),
            key=lambda x: x[0],
        )
        lower = [p for p in points if p[1] >= 0.5]
        upper = [p for p in points if p[1] < 0.5]
        if not lower or not upper:
            break
        # Use the nearest straddling pair around 50%, not far extremes.
        lo_elo = max(lower, key=lambda x: x[0])[0]  # highest elo with score >= 50%
        hi_elo = min(upper, key=lambda x: x[0])[0]  # lowest elo with score < 50%
        if hi_elo <= lo_elo:
            break
        if hi_elo - lo_elo <= refine_step:
            break
        probe = ((hi_elo + lo_elo) // 2 // refine_step) * refine_step
        if probe in tested:
            # If midpoint already tested due to rounding, try neighboring buckets.
            alt_low = probe - refine_step
            alt_high = probe + refine_step
            if lo_elo < alt_low < hi_elo and alt_low not in tested:
                probe = alt_low
            elif lo_elo < alt_high < hi_elo and alt_high not in tested:
                probe = alt_high
            else:
                break
        score, rec = _run_elo(probe)
        records.append(rec)
        tested.add(probe)
        print(f"  [smart] probe elo={probe} score={score:.3f}")

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
            r.get("stockfish_depth"),
            r.get("stockfish_elo"),
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
        ckpt, depth, elo, cond, temp, top_k = key
        rows.append({
            "checkpoint": ckpt,
            "iter_num": meta[key]["iter_num"],
            "stockfish_depth": depth,
            "stockfish_elo": elo,
            "conditioning": cond,
            "temperature": temp,
            "top_k": top_k,
            **counts,
        })

    rows.sort(key=lambda r: (r["checkpoint"], r["stockfish_depth"], r["conditioning"]))
    return rows



def estimate_model_elo(records: list[dict]) -> list[dict]:
    """Estimate model ELO from elo-limited Stockfish results.

    For each (checkpoint, conditioning, temperature, top_k) group, compute score vs
    each configured Stockfish ELO and interpolate the model ELO at 50% expected score.
    """
    grouped: dict[tuple, list[tuple[int, float, int]]] = {}

    for r in aggregate_results(records):
        elo = r.get("stockfish_elo")
        if elo is None:
            continue
        total = r["W"] + r["L"] + r["D"]
        if total == 0:
            continue
        score = (r["W"] + 0.5 * r["D"]) / total
        key = (
            r["checkpoint"],
            r.get("conditioning", ""),
            r.get("temperature", 1.0),
            r.get("top_k"),
        )
        grouped.setdefault(key, []).append((elo, score, total))

    estimates = []
    for key, points in grouped.items():
        points.sort(key=lambda x: x[0])
        below = [p for p in points if p[1] >= 0.5]
        above = [p for p in points if p[1] < 0.5]

        est = None
        note = ""
        if below and above:
            # Interpolate from the nearest straddling pair around score=50%.
            lo = max(below, key=lambda x: x[0])  # highest elo with score >= 50%
            hi = min(above, key=lambda x: x[0])  # lowest elo with score < 50%
            if lo[0] != hi[0] and lo[1] != hi[1]:
                t = (0.5 - hi[1]) / (lo[1] - hi[1])
                est = hi[0] + t * (lo[0] - hi[0])
                note = "interpolated"
        elif below:
            strongest = max(points, key=lambda x: x[0])
            est = float(strongest[0])
            note = "lower_bound"
        elif above:
            weakest = min(points, key=lambda x: x[0])
            est = float(weakest[0])
            note = "upper_bound"

        estimates.append({
            "checkpoint": key[0],
            "conditioning": key[1],
            "temperature": key[2],
            "top_k": key[3],
            "elo_estimate": est,
            "n_points": len(points),
            "total_games": sum(p[2] for p in points),
            "method": note,
        })

    estimates.sort(key=lambda r: (r["checkpoint"], r["conditioning"]))
    return estimates


def show_elo_estimates():
    records = load_results()
    estimates = estimate_model_elo(records)
    if not estimates:
        print("No ELO-limited results yet. Run with --stockfish-elo first.")
        return

    print(f"\n{'Checkpoint':<40} {'Cond':<12} {'T':>4} {'Games':>6} {'Pts':>4} {'EstElo':>8} {'Method':<14}")
    print("-" * 98)
    for r in estimates:
        ckpt = Path(r["checkpoint"]).name
        elo = "n/a" if r["elo_estimate"] is None else f"{r['elo_estimate']:.0f}"
        print(
            f"{ckpt:<40} {r['conditioning']:<12} {r['temperature']:>4.1f} "
            f"{r['total_games']:>6} {r['n_points']:>4} {elo:>8} {r['method']:<14}"
        )

def show_results():
    records = load_results()
    if not records:
        print("No results yet.")
        return

    rows = aggregate_results(records)
    print(f"\n{'Checkpoint':<40} {'Iter':>6} {'Opponent':<12} {'Cond':<12} {'T':>4} {'N':>5} {'W':>4} {'L':>4} {'D':>4} {'Score':>7}")
    print("-" * 113)
    for r in rows:
        ckpt = Path(r["checkpoint"]).name
        total = r["W"] + r["L"] + r["D"]
        score = (r["W"] + 0.5 * r["D"]) / total * 100 if total else 0
        if r.get("stockfish_elo") is not None:
            opponent = f"elo{r['stockfish_elo']}"
        else:
            opponent = f"d{r['stockfish_depth']}"
        print(
            f"{ckpt:<40} {r.get('iter_num', '?'):>6} {opponent:<12} "
            f"{r['conditioning']:<12} {r['temperature']:>4.1f} "
            f"{total:>5} {r['W']:>4} {r['L']:>4} {r['D']:>4} {score:>6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Patzer vs Stockfish tournament")
    parser.add_argument("--checkpoint", help="Path to checkpoint (local or R2 key)")
    parser.add_argument("--pull-r2", action="store_true", help="Download checkpoint from R2 first")
    parser.add_argument("--depths", nargs="+", type=int, default=[],
                        help="Legacy depth-based opponents (prefer --stockfish-elo or --smart-elo)")
    parser.add_argument("--stockfish-elo", nargs="+", type=int, default=None,
                        help="ELO-limited Stockfish targets (e.g. --stockfish-elo 1200 1500)")
    parser.add_argument("--smart-elo", action="store_true",
                        help="Run adaptive ELO ladder, then estimate model ELO")
    parser.add_argument("--anchor-elos", nargs="+", type=int,
                        default=[900, 1100, 1300, 1500, 1700, 1900],
                        help="Anchor ELOs used by --smart-elo")
    parser.add_argument("--refine-step", type=int, default=100,
                        help="Probe step size for --smart-elo refinement")
    parser.add_argument("--max-refine-rounds", type=int, default=2,
                        help="Extra probe rounds for --smart-elo")
    parser.add_argument("--games", type=int, default=20, help="Games per depth (alternates colors)")
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--temperature", type=float, default=0.1)
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
    parser.add_argument("--estimate-elo", action="store_true", help="Estimate Patzer ELO from elo-limited runs")
    args = parser.parse_args()

    if args.show:
        show_results()
        return

    if args.estimate_elo:
        show_elo_estimates()
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

    if not args.smart_elo and not args.depths and not args.stockfish_elo:
        parser.error("Provide --smart-elo, --stockfish-elo, or --depths")

    patzer = Patzer(
        ckpt_path,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioning=args.conditioning,
    )

    if args.smart_elo:
        records = run_smart_elo_tournament(
            patzer,
            args.checkpoint,
            args.stockfish,
            args.games,
            anchor_elos=args.anchor_elos,
            refine_step=args.refine_step,
            max_refine_rounds=args.max_refine_rounds,
        )
    else:
        records = run_tournament(
            patzer,
            args.checkpoint,
            args.stockfish,
            args.depths,
            args.games,
            elo_limits=args.stockfish_elo,
        )
    save_results(records)
    show_results()
    if args.smart_elo or args.stockfish_elo:
        show_elo_estimates()

    if patzer.illegal_move_count:
        print(f"\n[note] fell back to random move {patzer.illegal_move_count} time(s)")


if __name__ == "__main__":
    main()
