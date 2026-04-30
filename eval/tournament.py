"""
eval/tournament.py

Play Patzer (our trained model) against Stockfish at ELO-limited settings using
an adaptive "cold start" Elo estimation loop.
Results are persisted to eval/results.json so experiments accumulate over time.

Usage:
    # Run an adaptive ELO tournament (default checkpoint auto-picked):
    python eval/tournament.py --stockfish /opt/homebrew/bin/stockfish

    # Or pick a specific checkpoint (prefer weights_best.pt for play/eval):
    python eval/tournament.py --checkpoint checkpoints/patzer_v2/weights_best.pt

    # Print accumulated results:
    python eval/tournament.py --show
"""

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.engine import CONDITIONING_OPTIONS, Patzer, StockfishPlayer

RESULTS_FILE = Path(__file__).parent / "results.json"


def _checkpoint_label(checkpoint: str) -> str:
    """
    Human-friendly model id for display.
    - checkpoints/patzer_v1/weights_best.pt -> patzer_v1
    - checkpoints/patzer_v1/ckpt.pt -> patzer_v1
    - checkpoints/patzer_v2/ckpt_150000.pt -> patzer_v2
    - otherwise fall back to path stem/name.
    """
    p = Path(checkpoint)
    if p.suffix == ".pt" and p.parent.name and p.parent.name.startswith("patzer_v"):
        # Prefer model-version directory names over ckpt filenames.
        # This keeps the "Model" column stable while "Iter" carries the step.
        if p.name.startswith("ckpt"):
            return p.parent.name
    if p.stem:
        return p.stem
    return p.name or str(checkpoint)


def _latest_local_checkpoint_file(checkpoints_dir: Path) -> Path | None:
    """
    Pick the latest local checkpoint assuming directories like checkpoints/patzer_vX/.
    Prefers weights_best.pt (best weights for eval/play); falls back to ckpt_best.pt then ckpt.pt.
    """
    best_v: int | None = None
    best_dir: Path | None = None

    for p in checkpoints_dir.glob("patzer_v*"):
        if not p.is_dir():
            continue
        try:
            v = int(p.name.removeprefix("patzer_v"))
        except ValueError:
            continue
        if best_v is None or v > best_v:
            best_v, best_dir = v, p

    if best_dir is None:
        return None

    for name in ("weights_best.pt", "ckpt_best.pt", "ckpt.pt"):
        path = best_dir / name
        if path.is_file():
            return path
    return None


def play_game(white, black, max_moves: int = 300, timings: dict | None = None) -> str:
    """Play one game; returns '1-0', '0-1', or '1/2-1/2'."""
    board = chess.Board()
    move_history: list[str] = []

    while not board.is_game_over(claim_draw=True) and len(move_history) < max_moves * 2:
        player = white if board.turn == chess.WHITE else black

        try:
            t0 = time.perf_counter()
            uci = player.get_move(board, move_history)
            dt = time.perf_counter() - t0
            if timings is not None:
                key = getattr(player, "name", type(player).__name__)
                slot = timings.setdefault(key, {"secs": 0.0, "plies": 0})
                slot["secs"] += float(dt)
                slot["plies"] += 1
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
        if result == "1/2-1/2":
            outcome_tag = "Patzer draw"
        elif (result == "1-0" and patzer_is_white) or (result == "0-1" and not patzer_is_white):
            outcome_tag = "Patzer win"
        else:
            outcome_tag = "Patzer loss"

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

        played = game_idx + 1
        score = (w + 0.5 * d) / played if played else 0.0
        print(
            f"  {label} [{played}/{n_games}] Patzer={color} → {result} ({outcome_tag})"
            f" | running W-L-D={w}-{l}-{d} score={score*100:5.1f}%"
        )
    return w, l, d


def _stockfish_elo_bounds(stockfish_binary: str) -> tuple[int | None, int | None]:
    """Read Stockfish's UCI_Elo min/max (if exposed by this build)."""
    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_binary)
    try:
        opt = engine.options.get("UCI_Elo")
        if opt is None:
            return None, None
        return opt.min, opt.max
    finally:
        engine.quit()


def _elo_expected_score(model_elo: float, stockfish_elo: float) -> float:
    """Standard Elo expected score (win=1, draw=0.5, loss=0)."""
    return 1.0 / (1.0 + 10 ** ((stockfish_elo - model_elo) / 400.0))


def _result_to_score_perspective_of_patzer(result: str, patzer_is_white: bool) -> float:
    """Map '1-0'/'0-1'/'1/2-1/2' to Patzer score in [0, 0.5, 1]."""
    if result == "1/2-1/2":
        return 0.5
    if patzer_is_white:
        return 1.0 if result == "1-0" else 0.0
    # Patzer is black
    return 1.0 if result == "0-1" else 0.0


def _posterior_grid(
    grid_min: int,
    grid_max: int,
    step: int,
    prior_mu: float,
    prior_sigma: float,
) -> tuple[list[int], list[float]]:
    xs = list(range(grid_min, grid_max + 1, step))
    if prior_sigma <= 0:
        p = [0.0] * len(xs)
        mid = min(range(len(xs)), key=lambda i: abs(xs[i] - prior_mu))
        p[mid] = 1.0
        return xs, p
    inv2 = 1.0 / (2.0 * prior_sigma * prior_sigma)
    logp = [-(x - prior_mu) * (x - prior_mu) * inv2 for x in xs]
    m = max(logp)
    p = [math.exp(v - m) for v in logp]
    s = sum(p)
    return xs, [v / s for v in p]


def _posterior_update(xs: list[int], p: list[float], stockfish_elo: int, score: float) -> list[float]:
    """
    Update posterior over model Elo after observing a single game score in {0, 0.5, 1}.
    Uses a Bernoulli-style likelihood with fractional score (draw -> sqrt(E(1-E))).
    """
    eps = 1e-9
    logp = []
    for x, prior in zip(xs, p):
        e = _elo_expected_score(float(x), float(stockfish_elo))
        e = min(max(e, eps), 1.0 - eps)
        # likelihood ∝ e^score * (1-e)^(1-score)
        ll = score * math.log(e) + (1.0 - score) * math.log(1.0 - e)
        logp.append(math.log(prior + eps) + ll)

    m = max(logp)
    p2 = [math.exp(v - m) for v in logp]
    s = sum(p2)
    return [v / s for v in p2]


def _posterior_mean_sigma(xs: list[int], p: list[float]) -> tuple[float, float]:
    mu = sum(x * w for x, w in zip(xs, p))
    var = sum(((x - mu) ** 2) * w for x, w in zip(xs, p))
    return mu, math.sqrt(max(0.0, var))


def run_adaptive_elo_tournament(
    patzer: "Patzer",
    checkpoint_path: str,
    stockfish_binary: str,
    max_games: int,
    batch_size: int,
    prior_elo: float,
    prior_sigma: float,
    stop_sigma: float,
    elo_step: int = 10,
) -> list[dict]:
    """
    Sequential cold-start Elo estimation.

    Maintain a posterior over Patzer Elo and pick the next Stockfish Elo to be informative
    (near the current posterior mean), updating after each game.
    """
    sf_min, sf_max = _stockfish_elo_bounds(stockfish_binary)
    if sf_min is None:
        sf_min = 0
    if sf_max is None:
        sf_max = 4000

    grid_min = max(0, int(sf_min) - 600)
    grid_max = int(sf_max) + 600
    xs, p = _posterior_grid(grid_min, grid_max, elo_step, prior_elo, prior_sigma)

    # Accumulate per-elo outcomes for persistence.
    per_elo: dict[int, dict[str, int]] = {}

    total_w = total_l = total_d = 0
    games_played = 0
    timings: dict | None = {} if getattr(patzer, "_collect_timings", False) else None
    sf: StockfishPlayer | None = None

    def _round_to_step(x: float, step: int) -> int:
        return int(round(x / step) * step)

    try:
        while games_played < max_games:
            mu, sigma = _posterior_mean_sigma(xs, p)
            if sigma <= stop_sigma and games_played > 0:
                print(f"[stop] posterior sigma={sigma:.1f} <= {stop_sigma:.1f} after {games_played} game(s)")
                break

            # Cold start: begin at the weakest opponent available (models tend to be weak early).
            if games_played == 0:
                target_elo = int(sf_min)
            else:
                target_elo = _round_to_step(mu, 10)
            target_elo = max(int(sf_min), min(int(sf_max), int(target_elo)))

            # Create Stockfish once, then reconfigure Elo between batches.
            if sf is None:
                sf = StockfishPlayer(
                    stockfish_binary,
                    elo_limit=target_elo,
                    move_time=getattr(patzer, "_stockfish_move_time", 0.1),
                )
            else:
                sf.set_elo_limit(target_elo)

            # Play a small batch at this opponent Elo, alternating colors.
            for _ in range(batch_size):
                if games_played >= max_games:
                    break

                patzer_is_white = games_played % 2 == 0
                white = patzer if patzer_is_white else sf
                black = sf if patzer_is_white else patzer

                result = play_game(white, black, timings=timings)
                score = _result_to_score_perspective_of_patzer(result, patzer_is_white)

                if score == 0.5:
                    outcome_tag = "Patzer draw"
                    total_d += 1
                elif score == 1.0:
                    outcome_tag = "Patzer win"
                    total_w += 1
                else:
                    outcome_tag = "Patzer loss"
                    total_l += 1

                games_played += 1
                per_elo.setdefault(target_elo, {"W": 0, "L": 0, "D": 0, "games": 0})
                per_elo[target_elo]["games"] += 1
                if score == 1.0:
                    per_elo[target_elo]["W"] += 1
                elif score == 0.0:
                    per_elo[target_elo]["L"] += 1
                else:
                    per_elo[target_elo]["D"] += 1

                p = _posterior_update(xs, p, target_elo, score)
                mu2, sigma2 = _posterior_mean_sigma(xs, p)
                running_score = (total_w + 0.5 * total_d) / games_played
                color = "W" if patzer_is_white else "B"
                print(
                    f"  elo={target_elo} [{games_played}/{max_games}] Patzer={color} → {result} ({outcome_tag})"
                    f" | total W-L-D={total_w}-{total_l}-{total_d} score={running_score*100:5.1f}%"
                    f" | estElo={mu2:6.0f}±{sigma2:4.0f}"
                )
    finally:
        if sf is not None:
            sf.close()

    if timings:
        print("\n[timing] move generation time (sum over plies):")
        rows = []
        for name, t in timings.items():
            plies = int(t.get("plies", 0))
            secs = float(t.get("secs", 0.0))
            ms = (secs / plies * 1000.0) if plies else float("nan")
            rows.append((secs, plies, name, ms))
        rows.sort(reverse=True)
        for secs, plies, name, ms in rows:
            print(f"  {name:<20} {secs:7.2f}s  {plies:5d} plies  ({ms:6.1f} ms/ply)")

    records: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    for elo, c in sorted(per_elo.items(), key=lambda kv: kv[0]):
        records.append(
            {
                "timestamp": now,
                "checkpoint": checkpoint_path,
                "iter_num": patzer.iter_num,
                "stockfish_depth": None,
                "stockfish_elo": int(elo),
                "games": int(c["games"]),
                "temperature": patzer.temperature,
                "top_k": patzer.top_k,
                "conditioning": patzer.conditioning,
                "W": int(c["W"]),
                "L": int(c["L"]),
                "D": int(c["D"]),
            }
        )
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
    Combine records with identical (checkpoint, iter_num, depth, elo, conditioning, temperature, top_k)
    into single rows by summing W/L/D. iter_num is part of the key so reusing ckpt.pt at a new training
    step does not merge with older runs.
    """
    from collections import defaultdict
    agg: dict[tuple, dict] = defaultdict(lambda: {"W": 0, "L": 0, "D": 0})
    meta: dict[tuple, dict] = {}

    for r in records:
        key = (
            r["checkpoint"],
            r.get("iter_num"),
            r.get("stockfish_depth"),
            r.get("stockfish_elo"),
            r.get("conditioning", ""),
            r.get("temperature", 0.0),
            r.get("top_k"),
        )
        agg[key]["W"] += r["W"]
        agg[key]["L"] += r["L"]
        agg[key]["D"] += r["D"]
        meta[key] = {"iter_num": r.get("iter_num"), "checkpoint": r["checkpoint"]}

    rows = []
    for key, counts in agg.items():
        ckpt, iter_num, depth, elo, cond, temp, top_k = key
        rows.append({
            "checkpoint": ckpt,
            "iter_num": iter_num,
            "stockfish_depth": depth,
            "stockfish_elo": elo,
            "conditioning": cond,
            "temperature": temp,
            "top_k": top_k,
            **counts,
        })

    rows.sort(
        key=lambda r: (r["checkpoint"], r.get("iter_num") or 0, r["stockfish_depth"], r["conditioning"])
    )
    return rows



def estimate_model_elo(records: list[dict]) -> list[dict]:
    """
    Estimate model Elo from elo-limited Stockfish results using a 1D likelihood grid.

    This matches the adaptive tournament's underlying model (Elo expected score curve),
    and produces sensible estimates even from a single point (with high uncertainty),
    unlike the old "50% crossing" interpolation.
    """
    # Group aggregated elo results by model/config.
    grouped: dict[tuple, list[dict]] = {}
    for r in aggregate_results(records):
        elo = r.get("stockfish_elo")
        if elo is None:
            continue
        total = int(r["W"]) + int(r["L"]) + int(r["D"])
        if total <= 0:
            continue
        key = (
            r["checkpoint"],
            r.get("conditioning", ""),
            r.get("temperature", 0.0),
            r.get("top_k"),
            r.get("iter_num"),
        )
        grouped.setdefault(key, []).append(
            {
                "elo": int(elo),
                "W": int(r["W"]),
                "L": int(r["L"]),
                "D": int(r["D"]),
                "games": total,
            }
        )

    estimates: list[dict] = []
    eps = 1e-9

    for key, pts in grouped.items():
        pts.sort(key=lambda x: x["elo"])
        min_elo = pts[0]["elo"]
        max_elo = pts[-1]["elo"]

        # Wide grid around observed opponents.
        grid_min = max(0, min_elo - 800)
        grid_max = max_elo + 800
        step = 5
        xs = list(range(grid_min, grid_max + 1, step))

        # Uniform prior over the grid.
        logp = [0.0 for _ in xs]

        for i, x in enumerate(xs):
            ll = 0.0
            for pnt in pts:
                elo = float(pnt["elo"])
                # Treat draws as half-win/half-loss for likelihood.
                s = float(pnt["W"]) + 0.5 * float(pnt["D"])
                f = float(pnt["L"]) + 0.5 * float(pnt["D"])
                e = _elo_expected_score(float(x), elo)
                e = min(max(e, eps), 1.0 - eps)
                ll += s * math.log(e) + f * math.log(1.0 - e)
            logp[i] = ll

        m = max(logp)
        p = [math.exp(v - m) for v in logp]
        z = sum(p)
        p = [v / z for v in p]

        mu, sigma = _posterior_mean_sigma(xs, p)
        estimates.append(
            {
                "checkpoint": key[0],
                "conditioning": key[1],
                "temperature": key[2],
                "top_k": key[3],
                "iter_num": key[4],
                "elo_estimate": mu,
                "elo_sigma": sigma,
                "n_points": len(pts),
                "total_games": sum(pnt["games"] for pnt in pts),
                "method": "grid_posterior",
            }
        )

    estimates.sort(key=lambda r: (r["checkpoint"], r.get("iter_num") or 0, r["conditioning"]))
    return estimates


def show_elo_estimates():
    records = load_results()
    estimates = estimate_model_elo(records)
    if not estimates:
        print("No ELO-limited results yet. Run the tournament first.")
        return

    print(
        f"\n{'Model':<18} {'Iter':>6} {'Cond':<12} {'T':>4} {'Games':>6} {'Pts':>4} "
        f"{'EstElo':>8} {'±':>1} {'Sig':>4} {'Method':<14}"
    )
    print("-" * 98)
    for r in estimates:
        ckpt = _checkpoint_label(r["checkpoint"])
        elo = "n/a" if r["elo_estimate"] is None else f"{r['elo_estimate']:.0f}"
        sig = "n/a" if r.get("elo_sigma") is None else f"{r['elo_sigma']:.0f}"
        it = r.get("iter_num")
        iter_num = "?" if it is None else str(it)
        print(
            f"{ckpt:<18} {iter_num:>6} {r['conditioning']:<12} {r['temperature']:>4.1f} "
            f"{r['total_games']:>6} {r['n_points']:>4} {elo:>8} ± {sig:>4} {r['method']:<14}"
        )

def show_results():
    records = load_results()
    if not records:
        print("No results yet.")
        return

    # Model-centric summary (aligned with adaptive Elo evaluation).
    from collections import defaultdict

    # Keyed by (checkpoint, conditioning, temperature, top_k, iter_num)
    totals: dict[tuple, dict] = defaultdict(lambda: {"W": 0, "L": 0, "D": 0, "iter_num": "?"})
    for r in records:
        key = (
            r.get("checkpoint"),
            r.get("conditioning", ""),
            r.get("temperature", 0.0),
            r.get("top_k"),
            r.get("iter_num"),
        )
        totals[key]["W"] += int(r.get("W", 0))
        totals[key]["L"] += int(r.get("L", 0))
        totals[key]["D"] += int(r.get("D", 0))
        if totals[key]["iter_num"] == "?" and r.get("iter_num") is not None:
            totals[key]["iter_num"] = r.get("iter_num")

    # Attach Elo estimates where available.
    est_map: dict[tuple, dict] = {}
    for e in estimate_model_elo(records):
        k = (e["checkpoint"], e["conditioning"], e["temperature"], e["top_k"], e.get("iter_num"))
        est_map[k] = e

    rows = []
    for key, c in totals.items():
        ckpt, cond, temp, top_k, _iter_part = key
        total = c["W"] + c["L"] + c["D"]
        score = (c["W"] + 0.5 * c["D"]) / total * 100 if total else 0.0
        e = est_map.get(key, {})
        rows.append(
            {
                "model": _checkpoint_label(str(ckpt)),
                "checkpoint": ckpt,
                "iter_num": c["iter_num"],
                "conditioning": cond,
                "temperature": temp,
                "top_k": top_k,
                "games": total,
                "W": c["W"],
                "L": c["L"],
                "D": c["D"],
                "score": score,
                "elo_estimate": e.get("elo_estimate"),
                "elo_sigma": e.get("elo_sigma"),
                "elo_method": e.get("method", ""),
            }
        )

    rows.sort(
        key=lambda r: (
            float("-inf") if r["elo_estimate"] is None else float(r["elo_estimate"]),
            r["model"],
        ),
        reverse=True,
    )

    print(
        f"\n{'Model':<18} {'Iter':>6} {'Cond':<12} {'T':>4} {'Games':>6} "
        f"{'W':>4} {'L':>4} {'D':>4} {'Score':>7} {'EstElo':>8} {'Sig':>4} {'Method':<12}"
    )
    print("-" * 96)
    for r in rows:
        elo = "n/a" if r["elo_estimate"] is None else f"{r['elo_estimate']:.0f}"
        sig = "n/a"
        if r.get("elo_sigma") is not None:
            sig = f"{r['elo_sigma']:.0f}"
        print(
            f"{r['model']:<18} {r['iter_num']:>6} {r['conditioning']:<12} {r['temperature']:>4.1f} "
            f"{r['games']:>6} {r['W']:>4} {r['L']:>4} {r['D']:>4} {r['score']:>6.1f}% "
            f"{elo:>8} {sig:>4} {r['elo_method']:<12}"
        )


def main():
    parser = argparse.ArgumentParser(description="Patzer vs Stockfish tournament")
    parser.add_argument(
        "--checkpoint",
        help=(
            "Path to checkpoint (local file or R2 key). "
            "If omitted, uses latest local checkpoints/patzer_vX/weights_best.pt (or ckpt_best.pt/ckpt.pt)"
        ),
    )
    parser.add_argument("--pull-r2", action="store_true", help="Download checkpoint from R2 first")
    parser.add_argument("--games", type=int, default=50, help="Max games to play (adaptive Elo)")
    parser.add_argument("--batch-size", type=int, default=2, help="Games per opponent Elo before repicking")
    parser.add_argument("--prior-elo", type=float, default=1600, help="Prior mean for Patzer Elo")
    parser.add_argument("--prior-sigma", type=float, default=400, help="Prior stddev for Patzer Elo")
    parser.add_argument("--stop-sigma", type=float, default=35, help="Stop once posterior stddev <= this")
    parser.add_argument("--grid-step", type=int, default=10, help="Elo grid step size for posterior")
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--device", default="cpu", help="torch device: cpu | mps | cuda")
    parser.add_argument("--timing", action="store_true", help="Print per-move timing breakdown")
    parser.add_argument(
        "--sf-move-time",
        type=float,
        default=0.05,
        help="Seconds per Stockfish move when using ELO-limited mode (lower = faster)",
    )
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

    checkpoints_dir = Path(__file__).parent.parent / "checkpoints"
    if not args.checkpoint:
        ckpt_auto = _latest_local_checkpoint_file(checkpoints_dir)
        if ckpt_auto is None:
            parser.error(
                "--checkpoint not provided and no local checkpoints found at "
                f"{checkpoints_dir}/patzer_vX/weights_best.pt (or ckpt_best.pt/ckpt.pt)"
            )
        args.checkpoint = str(ckpt_auto)
        print(f"[default] using local checkpoint {args.checkpoint} (@eval/tournament.py)")

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

    if args.games <= 0:
        parser.error("--games must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.grid_step <= 0:
        parser.error("--grid-step must be > 0")
    if args.stop_sigma < 0:
        parser.error("--stop-sigma must be >= 0")

    patzer = Patzer(
        ckpt_path,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioning=args.conditioning,
    )
    # Lightweight timing collection toggle. Stored on instance to avoid changing class API.
    patzer._collect_timings = bool(args.timing)
    patzer._stockfish_move_time = float(args.sf_move_time)

    records = run_adaptive_elo_tournament(
        patzer,
        args.checkpoint,
        args.stockfish,
        max_games=args.games,
        batch_size=args.batch_size,
        prior_elo=args.prior_elo,
        prior_sigma=args.prior_sigma,
        stop_sigma=args.stop_sigma,
        elo_step=args.grid_step,
    )
    save_results(records)
    show_results()
    show_elo_estimates()

    if patzer.illegal_move_count:
        print(f"\n[note] fell back to random move {patzer.illegal_move_count} time(s)")


if __name__ == "__main__":
    main()
