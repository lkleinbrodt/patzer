"""
eval/evaluate.py — single entry point for all Patzer evaluation.

All results are stored in eval/results.db (SQLite, gitignored). One row per
game; Elo estimates and leaderboards are computed on-the-fly from those records.

Prerequisites
-------------
- Stockfish installed: brew install stockfish  (default path: /opt/homebrew/bin/stockfish)
- A trained checkpoint. R2 is the source of truth; checkpoints are pulled automatically
  if not present locally. You can also pull manually:
    python patzer/r2.py pull checkpoints/patzer_v2/weights_best.pt   # single file
    python patzer/r2.py pull checkpoints/patzer_v2                   # full dir (skips existing)

Checkpoint naming convention:
  weights_best.pt          — best val-loss weights; use this for eval (recommended)
  weights_iter_050000.pt   — snapshot at a specific training step
  ckpt.pt                  — latest full checkpoint (includes optimizer state, for resume only)

Quick start
-----------
Estimate Elo against Stockfish (adaptive — plays until confident or max_games):

    python eval/evaluate.py stockfish checkpoints/patzer_v2/weights_best.pt \\
        --games 50 --device mps

  Output (one line per game):
    elo=1320 [1/50] Patzer=W → 1-0 (win) | W-L-D=1-0-0 score=100.0% | estElo=  1850±320
    elo=1850 [2/50] Patzer=B → 1/2-1/2 (draw) | W-L-D=1-0-1 score= 75.0% | estElo=  1740±180
    ...
    [result] patzer_v2@45000: estimated Elo = 1283 ± 42  (50 games)

  Stops early if posterior sigma drops below --stop-sigma (default 50 Elo points).

Compare two checkpoints head-to-head:

    python eval/evaluate.py head2head \\
        checkpoints/patzer_v2/weights_best.pt \\
        checkpoints/patzer_v1/weights_best.pt \\
        --games 20 --device mps

Round-robin across multiple checkpoints (all pairs):

    python eval/evaluate.py head2head \\
        checkpoints/patzer_v2/ckpt_010000.pt \\
        checkpoints/patzer_v2/ckpt_070000.pt \\
        checkpoints/patzer_v2/ckpt_150000.pt \\
        --round-robin --games 10 --device mps

Show unified Elo leaderboard (computed from all stored games via Bradley-Terry):

    python eval/evaluate.py leaderboard
    python eval/evaluate.py leaderboard --min-games 1   # include lightly-tested models

  Stockfish players are anchored at their configured Elo; Patzer models are fitted.
  Output:
    Rank  Player                          Elo     ±  Games  W-L-D
    ---------------------------------------------------------------
    1     patzer_v2@150000               1380    31     50  28-17-5
    2     patzer_v2@45000                1283    42     50  22-21-7
    3     patzer_v1@40000                1050    67     20   6-13-1
    4     stockfish:1600                 1600     —    ...

Show game-by-game history:

    python eval/evaluate.py history patzer_v2   # substring match on player name

Plot Elo vs training step (requires matplotlib, requires prior stockfish runs):

    python eval/evaluate.py progress patzer_v2

Common flags
------------
  --device        cpu | mps | cuda  (default: cpu)
  --temperature   move sampling temperature; 0.0 = greedy (default)
  --conditioning  match_color | white_win | black_win | draw | none (default: match_color)
  --games         number of games to play
  --stockfish     path to stockfish binary (default: /opt/homebrew/bin/stockfish)
  --db            path to results database (default: eval/results.db)
"""

import argparse
import math
import random
import sys
import time
from itertools import combinations
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.db import DB_PATH, insert_game, player_name, query_games, stockfish_name
from eval.elo import compute_ratings
from eval.engine import CONDITIONING_OPTIONS, Patzer, StockfishPlayer
from patzer.r2 import pull_file

DEFAULT_STOCKFISH = "/opt/homebrew/bin/stockfish"


def _ensure_checkpoint(path: Path) -> None:
    """Pull from R2 if the checkpoint doesn't exist locally. Exits on failure."""
    if path.exists():
        return
    r2_key = str(path)
    print(f"[r2] {path} not found locally — attempting pull from R2...")
    ok = pull_file(r2_key, path, skip_existing=False)
    if not ok:
        sys.exit(
            f"Checkpoint not found locally and R2 pull failed: {path}\n"
            f"Pull manually: python patzer/r2.py pull {r2_key}"
        )
    if not path.exists():
        sys.exit(
            f"R2 pull succeeded but file still missing at {path}.\n"
            f"Check that the R2 key exists: {r2_key}"
        )

OPENING_LINES_UCI = [
    ["e2e4", "e7e5", "g1f3", "b8c6"],  # Italian-ish
    ["d2d4", "d7d5", "c2c4", "e7e6"],  # QGD
    ["e2e4", "c7c5", "g1f3", "d7d6"],  # Sicilian
    ["d2d4", "g8f6", "c2c4", "g7g6"],  # King's Indian
    ["c2c4", "e7e5", "b1c3", "g8f6"],  # English
    ["e2e4", "e7e6", "d2d4", "d7d5"],  # French
    ["e2e4", "c7c6", "d2d4", "d7d5"],  # Caro-Kann
    ["d2d4", "d7d5", "g1f3", "g8f6"],  # London-ish
]


# ---------------------------------------------------------------------------
# Game engine helpers
# ---------------------------------------------------------------------------

def play_game(
    white,
    black,
    opening: list[str] | None = None,
    max_moves: int = 300,
) -> str:
    """Play one game; returns '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    move_history: list[str] = []

    if opening:
        for uci in opening:
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                board.push(move)
                move_history.append(uci)

    while not board.is_game_over(claim_draw=True) and len(move_history) < max_moves * 2:
        player = white if board.turn == chess.WHITE else black
        try:
            uci = player.get_move(board, move_history)
            move = chess.Move.from_uci(uci)
        except Exception as e:
            print(f"  [warn] {player.name} error: {e} — random move", file=sys.stderr)
            move = random.choice(list(board.legal_moves))

        if move not in board.legal_moves:
            print(f"  [warn] {player.name} played illegal {move.uci()} — random move", file=sys.stderr)
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


def _score_from_result(result: str, player_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if player_is_white:
        return 1.0 if result == "1-0" else 0.0
    return 1.0 if result == "0-1" else 0.0


# ---------------------------------------------------------------------------
# Bayesian Elo estimation (ported from tournament.py)
# ---------------------------------------------------------------------------

def _elo_expected_score(model_elo: float, opp_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp_elo - model_elo) / 400.0))


def _stockfish_elo_bounds(binary: str) -> tuple[int, int]:
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        opt = engine.options.get("UCI_Elo")
        sf_min = int(opt.min) if opt and opt.min is not None else 0
        sf_max = int(opt.max) if opt and opt.max is not None else 4000
        return sf_min, sf_max
    finally:
        engine.quit()


def _posterior_grid(
    grid_min: int, grid_max: int, step: int, prior_mu: float, prior_sigma: float
) -> tuple[list[int], list[float]]:
    xs = list(range(grid_min, grid_max + 1, step))
    if prior_sigma <= 0:
        p = [0.0] * len(xs)
        mid = min(range(len(xs)), key=lambda i: abs(xs[i] - prior_mu))
        p[mid] = 1.0
        return xs, p
    inv2 = 1.0 / (2.0 * prior_sigma ** 2)
    logp = [-(x - prior_mu) ** 2 * inv2 for x in xs]
    m = max(logp)
    raw = [math.exp(v - m) for v in logp]
    s = sum(raw)
    return xs, [v / s for v in raw]


def _posterior_update(
    xs: list[int], p: list[float], sf_elo: int, score: float
) -> list[float]:
    eps = 1e-9
    logp = []
    for x, prior in zip(xs, p):
        e = _elo_expected_score(float(x), float(sf_elo))
        e = min(max(e, eps), 1.0 - eps)
        ll = score * math.log(e) + (1.0 - score) * math.log(1.0 - e)
        logp.append(math.log(prior + eps) + ll)
    m = max(logp)
    raw = [math.exp(v - m) for v in logp]
    s = sum(raw)
    return [v / s for v in raw]


def _posterior_mean_sigma(xs: list[int], p: list[float]) -> tuple[float, float]:
    mu = sum(x * w for x, w in zip(xs, p))
    var = sum((x - mu) ** 2 * w for x, w in zip(xs, p))
    return mu, math.sqrt(max(0.0, var))


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_stockfish(args: argparse.Namespace) -> None:
    ckpt = Path(args.checkpoint)
    _ensure_checkpoint(ckpt)

    patzer = Patzer(
        ckpt, device=args.device, temperature=args.temperature,
        top_k=args.top_k, conditioning=args.conditioning,
    )
    pname = player_name(ckpt, patzer.iter_num)
    rel_ckpt = str(ckpt)

    sf_min, sf_max = _stockfish_elo_bounds(args.stockfish)
    grid_min = max(0, sf_min - 600)
    grid_max = sf_max + 600
    xs, p = _posterior_grid(grid_min, grid_max, args.elo_step, args.prior_elo, args.prior_sigma)

    total_w = total_l = total_d = 0
    games_played = 0
    sf: StockfishPlayer | None = None

    def _round(x: float) -> int:
        return int(round(x / args.elo_step) * args.elo_step)

    print(f"[stockfish] evaluating {pname} | max_games={args.games} stop_sigma={args.stop_sigma}")

    try:
        while games_played < args.games:
            mu, sigma = _posterior_mean_sigma(xs, p)
            if sigma <= args.stop_sigma and games_played > 0:
                print(f"[stop] sigma={sigma:.1f} ≤ {args.stop_sigma} after {games_played} games")
                break

            target_elo = int(sf_min) if games_played == 0 else _round(mu)
            target_elo = max(sf_min, min(sf_max, target_elo))
            sf_name = stockfish_name(target_elo)

            if sf is None:
                sf = StockfishPlayer(args.stockfish, elo_limit=target_elo)
            else:
                sf.set_elo_limit(target_elo)

            patzer_is_white = games_played % 2 == 0
            white = patzer if patzer_is_white else sf
            black = sf if patzer_is_white else patzer

            result = play_game(white, black)
            score = _score_from_result(result, patzer_is_white)

            if score == 1.0:
                total_w += 1
                tag = "win"
            elif score == 0.0:
                total_l += 1
                tag = "loss"
            else:
                total_d += 1
                tag = "draw"

            games_played += 1
            p = _posterior_update(xs, p, target_elo, score)
            mu2, sigma2 = _posterior_mean_sigma(xs, p)
            running = (total_w + 0.5 * total_d) / games_played
            color = "W" if patzer_is_white else "B"
            print(
                f"  elo={target_elo} [{games_played}/{args.games}] Patzer={color} → {result} ({tag})"
                f" | W-L-D={total_w}-{total_l}-{total_d} score={running*100:5.1f}%"
                f" | estElo={mu2:6.0f}±{sigma2:4.0f}"
            )

            insert_game(
                white=pname if patzer_is_white else sf_name,
                black=sf_name if patzer_is_white else pname,
                result=result,
                white_checkpoint=rel_ckpt if patzer_is_white else None,
                black_checkpoint=None if patzer_is_white else rel_ckpt,
                white_iter=patzer.iter_num if patzer_is_white else None,
                black_iter=None if patzer_is_white else patzer.iter_num,
                temperature=args.temperature,
                top_k=args.top_k,
                conditioning=args.conditioning,
            )
    finally:
        if sf is not None:
            sf.close()

    mu, sigma = _posterior_mean_sigma(xs, p)
    print(f"\n[result] {pname}: estimated Elo = {mu:.0f} ± {sigma:.0f}  ({games_played} games)")


def cmd_head2head(args: argparse.Namespace) -> None:
    checkpoints = [Path(c) for c in args.checkpoints]
    for c in checkpoints:
        _ensure_checkpoint(c)

    if len(checkpoints) > 2 and not args.round_robin:
        sys.exit("Pass --round-robin to compare more than 2 checkpoints.")

    pairs = list(combinations(range(len(checkpoints)), 2)) if args.round_robin else [(0, 1)]

    rng = random.Random(args.seed)

    models: list[Patzer] = []
    for c in checkpoints:
        models.append(Patzer(c, device=args.device, temperature=args.temperature,
                             top_k=args.top_k, conditioning=args.conditioning))
    names = [player_name(c, m.iter_num) for c, m in zip(checkpoints, models)]

    for i, j in pairs:
        a, b = models[i], models[j]
        na, nb = names[i], names[j]
        ra, rb = str(checkpoints[i]), str(checkpoints[j])
        print(f"\n[head2head] {na} vs {nb}  ({args.games} games)")
        w = l = d = 0
        for game_idx in range(args.games):
            opening_line = rng.choice(OPENING_LINES_UCI)
            a_is_white = game_idx % 2 == 0
            white, black = (a, b) if a_is_white else (b, a)
            wname, bname = (na, nb) if a_is_white else (nb, na)
            wrel, brel = (ra, rb) if a_is_white else (rb, ra)
            witer = (a if a_is_white else b).iter_num
            biter = (b if a_is_white else a).iter_num

            result = play_game(white, black, opening=opening_line)
            score_a = _score_from_result(result, a_is_white)
            if score_a == 1.0:
                w += 1; tag = f"{na} wins"
            elif score_a == 0.0:
                l += 1; tag = f"{nb} wins"
            else:
                d += 1; tag = "draw"

            opening_str = " ".join(opening_line)
            print(f"  [{game_idx+1}/{args.games}] {wname}(W) vs {bname}(B) → {result} ({tag})")
            insert_game(
                white=wname, black=bname, result=result,
                white_checkpoint=wrel, black_checkpoint=brel,
                white_iter=witer, black_iter=biter,
                opening=opening_str,
                temperature=args.temperature, top_k=args.top_k,
                conditioning=args.conditioning,
            )

        played = w + l + d
        score = (w + 0.5 * d) / played if played else 0.0
        print(f"  → {na}: W-L-D={w}-{l}-{d}  score={score*100:.1f}%")


def cmd_leaderboard(args: argparse.Namespace) -> None:
    games = query_games()
    if not games:
        print("No games in database.")
        return

    ratings = compute_ratings(games)
    ratings = [r for r in ratings if r.games >= args.min_games]

    if not ratings:
        print(f"No players with ≥{args.min_games} games.")
        return

    print(f"\n{'Rank':<5} {'Player':<28} {'Elo':>6} {'±':>5} {'Games':>6} {'W-L-D'}")
    print("-" * 65)
    for rank, r in enumerate(ratings, 1):
        se = f"{r.stderr:.0f}" if not math.isnan(r.stderr) else "—"
        wld = f"{r.wins}-{r.losses}-{r.draws}"
        print(f"{rank:<5} {r.name:<28} {r.elo:>6.0f} {se:>5} {r.games:>6} {wld}")


def cmd_history(args: argparse.Namespace) -> None:
    games = query_games(player=args.player)
    if not games:
        print(f"No games found matching '{args.player}'.")
        return

    print(f"\n{'Date':<26} {'White':<25} {'Black':<25} {'Result':<8} Opening")
    print("-" * 100)
    for g in games:
        date = g["timestamp"][:19].replace("T", " ")
        opening = g["opening"] or ""
        if len(opening) > 25:
            opening = opening[:22] + "..."
        print(f"{date:<26} {g['white']:<25} {g['black']:<25} {g['result']:<8} {opening}")


def cmd_progress(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib required: pip install matplotlib")

    version = args.version
    games = query_games(player=version)
    if not games:
        print(f"No games found for '{version}'.")
        return

    # Only look at games where one side is a patzer model and the other is stockfish
    sf_games: dict[int, list[dict]] = {}  # iter_num -> list of games
    for g in games:
        w, b = g["white"], g["black"]
        if version in w and b.startswith("stockfish:"):
            it = g.get("white_iter")
        elif version in b and w.startswith("stockfish:"):
            it = g.get("black_iter")
        else:
            continue
        if it is None:
            continue
        sf_games.setdefault(it, []).append(g)

    if not sf_games:
        print(f"No Stockfish games found for '{version}'. Run 'stockfish' subcommand first.")
        return

    iters = sorted(sf_games)
    elos = []
    for it in iters:
        sub_games = sf_games[it]
        ratings = compute_ratings(sub_games)
        patzer_ratings = [r for r in ratings if version in r.name and not r.name.startswith("stockfish:")]
        if patzer_ratings:
            elos.append(patzer_ratings[0].elo)
        else:
            elos.append(float("nan"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, elos, marker="o", linewidth=2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Estimated Elo")
    ax.set_title(f"Elo progression — {version}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = Path(args.output) if args.output else Path(__file__).parent / f"progress_{version}.png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evaluate.py", description="Patzer evaluation CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- stockfish ---
    p_sf = sub.add_parser("stockfish", help="Adaptive Bayesian Elo estimation vs Stockfish")
    p_sf.add_argument("checkpoint", help="Path to .pt checkpoint")
    p_sf.add_argument("--games", type=int, default=50, help="Max games to play (default 50)")
    p_sf.add_argument("--stop-sigma", type=float, default=50.0, dest="stop_sigma",
                       help="Stop when posterior sigma ≤ this (default 50)")
    p_sf.add_argument("--prior-elo", type=float, default=1500.0, dest="prior_elo")
    p_sf.add_argument("--prior-sigma", type=float, default=400.0, dest="prior_sigma")
    p_sf.add_argument("--elo-step", type=int, default=10, dest="elo_step")
    p_sf.add_argument("--stockfish", default=DEFAULT_STOCKFISH, help="Stockfish binary path")
    p_sf.add_argument("--device", default="cpu")
    p_sf.add_argument("--temperature", type=float, default=0.0)
    p_sf.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_sf.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_sf.add_argument("--db", default=str(DB_PATH), help="Database path")

    # --- head2head ---
    p_h2h = sub.add_parser("head2head", help="Model vs model")
    p_h2h.add_argument("checkpoints", nargs="+", help="Two or more checkpoint paths")
    p_h2h.add_argument("--games", type=int, default=20, help="Games per pair (default 20)")
    p_h2h.add_argument("--round-robin", action="store_true", dest="round_robin",
                        help="All-pairs if >2 checkpoints")
    p_h2h.add_argument("--device", default="cpu")
    p_h2h.add_argument("--temperature", type=float, default=0.0)
    p_h2h.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_h2h.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_h2h.add_argument("--seed", type=int, default=42)
    p_h2h.add_argument("--db", default=str(DB_PATH), help="Database path")

    # --- leaderboard ---
    p_lb = sub.add_parser("leaderboard", help="Unified Elo leaderboard")
    p_lb.add_argument("--min-games", type=int, default=5, dest="min_games")
    p_lb.add_argument("--db", default=str(DB_PATH), help="Database path")

    # --- history ---
    p_hist = sub.add_parser("history", help="Per-game log for a model")
    p_hist.add_argument("player", help="Player name substring (e.g. 'patzer_v2')")
    p_hist.add_argument("--db", default=str(DB_PATH), help="Database path")

    # --- progress ---
    p_prog = sub.add_parser("progress", help="Plot Elo vs training step")
    p_prog.add_argument("version", help="Model version (e.g. 'patzer_v2')")
    p_prog.add_argument("--output", default=None, help="Output image path")
    p_prog.add_argument("--db", default=str(DB_PATH), help="Database path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Allow --db override to propagate to db module
    if hasattr(args, "db") and args.db != str(DB_PATH):
        import eval.db as db_mod
        db_mod.DB_PATH = Path(args.db)

    dispatch = {
        "stockfish": cmd_stockfish,
        "head2head": cmd_head2head,
        "leaderboard": cmd_leaderboard,
        "history": cmd_history,
        "progress": cmd_progress,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
