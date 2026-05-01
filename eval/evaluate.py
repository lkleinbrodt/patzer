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
Estimate Elo against Stockfish (adaptive — plays until confident or max_games).
You can specify an exact checkpoint or just a prefix — if a prefix is given, the
available weights_*.pt files are listed from R2 and you pick one interactively:

    # Exact path:
    python eval/evaluate.py stockfish checkpoints/patzer_v2/weights_best.pt --games 50 --device mps

    # Shorthand (iter in thousands, i.e. iter/1000):
    python eval/evaluate.py stockfish patzer_v3@180 --games 50 --device mps
    python eval/evaluate.py stockfish --checkpoint patzer_v3@best --games 50 --device mps

    # Prefix (interactive selection from R2):
    python eval/evaluate.py stockfish checkpoints/patzer_v2 --games 50 --device mps

  Output (one line per game):
    elo=1320 [1/50] Patzer=W → 1-0 (win) | W-L-D=1-0-0 score=100.0% | estElo=  1850±320
    elo=1850 [2/50] Patzer=B → 1/2-1/2 (draw) | W-L-D=1-0-1 score= 75.0% | estElo=  1740±180
    ...
    [result] patzer_v2@45: estimated Elo = 1283 ± 42  (50 games)

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
    1     patzer_v2@150                  1380    31     50  28-17-5
    2     patzer_v2@45                   1283    42     50  22-21-7
    3     patzer_v1@40                   1050    67     20   6-13-1
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
import re
import sys
import time
from itertools import combinations
from pathlib import Path
from collections import Counter, defaultdict

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.db import DB_PATH, insert_game, iter_display_k, player_name, query_games, stockfish_name
from eval.elo import compute_ratings
from eval.engine import CONDITIONING_OPTIONS, Patzer, StockfishPlayer
import patzer.r2 as r2

DEFAULT_STOCKFISH = "/opt/homebrew/bin/stockfish"


def _resolve_checkpoint(arg: str) -> Path:
    """
    Resolve a checkpoint argument into a concrete path.

    Supported forms:
      - Exact path: checkpoints/patzer_v3/weights_best.pt
      - R2 prefix:  checkpoints/patzer_v3            (interactive pick from R2)
      - Shorthand:  patzer_v3@180                    (→ weights_iter_180000.pt)
      - Shorthand:  patzer_v3@best                   (→ weights_best.pt)
      - Shorthand:  patzer_v3                        (→ checkpoints/patzer_v3)
    """
    raw = arg.strip()
    if not raw:
        raise ValueError("Empty checkpoint argument")

    # Shorthand: patzer_vN@K or patzer_vN@best (K is in 'k' steps, i.e. iter/1000).
    m = re.match(r"^(patzer_v\d+)@(\d+|best)$", raw)
    if m:
        ver, tag = m.group(1), m.group(2)
        if tag == "best":
            return Path(f"checkpoints/{ver}/weights_best.pt")
        return Path(f"checkpoints/{ver}/weights_iter_{int(tag) * 1000:06d}.pt")

    # Shorthand: patzer_vN means the checkpoints prefix for that model.
    if re.match(r"^patzer_v\d+$", raw):
        raw = f"checkpoints/{raw}"

    if arg.endswith(".pt"):
        return Path(arg)

    print(f"\n[r2] listing checkpoints under '{arg}'...")
    keys = r2.list_weights(arg)
    if not keys:
        client, _ = r2._client()
        if client is None:
            sys.exit(
                f"'{arg}' looks like a prefix but R2 is not configured.\n"
                "Specify an exact checkpoint path (e.g. checkpoints/patzer_v2/weights_best.pt) "
                "or set R2 env vars in .env."
            )
        sys.exit(f"No weights_*.pt files found in R2 under: {arg}")

    print(f"Found {len(keys)} checkpoint(s):")
    for i, key in enumerate(keys):
        print(f"  [{i}] {key}")

    while True:
        try:
            raw = input(f"\nSelect checkpoint (0–{len(keys)-1}): ").strip()
            idx = int(raw)
            if 0 <= idx < len(keys):
                return Path(keys[idx])
        except (ValueError, EOFError):
            pass
        print(f"  Enter a number between 0 and {len(keys)-1}.")


def _sync_checkpoint(path: Path) -> None:
    """Ensure the local checkpoint is present and up-to-date with R2."""
    r2_key = str(path)

    if path.exists():
        if r2.is_fresh(r2_key, path):
            print(f"[checkpoint] {path.name} is up to date")
            return
        print(f"[checkpoint] {path.name} is stale — pulling from R2...")
    else:
        print(f"[checkpoint] {path} not found locally — pulling from R2...")

    ok = r2.pull_file(r2_key, path)
    if not ok:
        if path.exists():
            print(f"[checkpoint] R2 pull failed (credentials?), using existing local copy")
            return
        sys.exit(
            f"Checkpoint not found locally and R2 pull failed: {path}\n"
            f"Pull manually: python patzer/r2.py pull {r2_key}"
        )


_WEIGHTS_ITER_RE = re.compile(r"^weights_iter_(\d+)\.pt$")


def _pick_evenly_spaced_iters(
    available: list[tuple[int, str]],
    n_pick: int,
) -> list[str]:
    """
    Given sorted (iter_num, key) pairs, pick n_pick snapshots spaced evenly in
    iteration space (not list index space), always including endpoints when
    n_pick >= 2. Returns selected keys.
    """
    if n_pick <= 0 or not available:
        return []
    if n_pick >= len(available):
        return [k for _, k in available]

    it_min = available[0][0]
    it_max = available[-1][0]
    if n_pick == 1 or it_min == it_max:
        return [available[-1][1]]

    # Targets evenly spaced in iter space, then pick closest available (deduped).
    remaining = available[:]  # list[(iter, key)], sorted
    selected: list[str] = []
    for i in range(n_pick):
        tgt = it_min + (it_max - it_min) * i / (n_pick - 1)
        # Find closest remaining iter to tgt.
        best_j = min(range(len(remaining)), key=lambda j: abs(remaining[j][0] - tgt))
        _, key = remaining.pop(best_j)
        selected.append(key)

    # Keep output sorted by iteration for readability.
    picked_set = set(selected)
    return [k for _, k in available if k in picked_set]


def _select_checkpoints_from_prefix(
    prefix: str,
    *,
    n_iters: int,
    include_best: bool,
) -> list[Path]:
    """
    Select checkpoints under an R2 prefix:
      - optionally include weights_best.pt
      - include n_iters evenly spaced weights_iter_*.pt snapshots
    """
    keys = r2.list_weights(prefix)
    if not keys:
        sys.exit(f"No weights_*.pt files found in R2 under: {prefix}")

    best_key = None
    iters: list[tuple[int, str]] = []
    for k in keys:
        name = Path(k).name
        if name == "weights_best.pt":
            best_key = k
            continue
        m = _WEIGHTS_ITER_RE.match(name)
        if m:
            iters.append((int(m.group(1)), k))

    iters.sort(key=lambda t: t[0])
    iter_keys = _pick_evenly_spaced_iters(iters, n_iters)

    selected: list[Path] = []
    if include_best and best_key is not None:
        selected.append(Path(best_key))
    for k in iter_keys:
        selected.append(Path(k))

    if not selected:
        sys.exit(f"No selectable checkpoints found under: {prefix}")

    # If weights_best.pt is byte-identical to a weights_iter_*.pt snapshot, don't treat them as
    # separate players and don't download twice. Prefer weights_best.pt (canonical).
    try:
        if best_key is not None:
            best_path = str(Path(best_key))
            best_etag = r2.get_etag(best_key)
            if best_etag is not None:
                selected = [
                    p for p in selected
                    if not (p.name.startswith("weights_iter_") and r2.get_etag(str(p)) == best_etag)
                ]
    except Exception:
        pass

    # Deduplicate paths (e.g. if best is also among iter keys somehow).
    out: list[Path] = []
    seen = set()
    for p in selected:
        s = str(p)
        if s not in seen:
            out.append(p)
            seen.add(s)
    return out

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
    ckpt = _resolve_checkpoint(args.checkpoint)
    _sync_checkpoint(ckpt)

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
    checkpoints = [_resolve_checkpoint(c) for c in args.checkpoints]
    for c in checkpoints:
        _sync_checkpoint(c)

    if len(checkpoints) > 2 and not args.round_robin:
        sys.exit("Pass --round-robin to compare more than 2 checkpoints.")

    pairs = list(combinations(range(len(checkpoints)), 2)) if args.round_robin else [(0, 1)]

    rng = random.Random(args.seed)

    models: list[Patzer] = []
    for c in checkpoints:
        models.append(Patzer(c, device=args.device, temperature=args.temperature,
                             top_k=args.top_k, conditioning=args.conditioning))
    names = [player_name(c, m.iter_num) for c, m in zip(checkpoints, models)]

    # If two checkpoints resolve to the same displayed player name (e.g. weights_best coincides
    # with a weights_iter snapshot at the same iter), collapse them so we don't record self-play
    # rows with identical player labels.
    if len(set(names)) != len(names):
        kept: dict[str, int] = {}
        dropped: list[tuple[str, str]] = []  # (name, checkpoint)
        for i, (n, c) in enumerate(zip(names, checkpoints)):
            if n not in kept:
                kept[n] = i
                continue
            # Prefer weights_best.pt as canonical when duplicate labels occur.
            prev_i = kept[n]
            prev_name = checkpoints[prev_i].name
            cur_name = c.name
            prev_is_best = prev_name == "weights_best.pt"
            cur_is_best = cur_name == "weights_best.pt"
            if cur_is_best and not prev_is_best:
                dropped.append((n, str(checkpoints[prev_i])))
                kept[n] = i
            else:
                dropped.append((n, str(c)))
        if dropped:
            print("[head2head] warning: collapsing duplicate player labels:", file=sys.stderr)
            for n, c in dropped[:8]:
                print(f"  - dropped {n} from {c}", file=sys.stderr)
            if len(dropped) > 8:
                print(f"  - (and {len(dropped) - 8} more)", file=sys.stderr)
        keep_idx = sorted(kept.values())
        checkpoints = [checkpoints[i] for i in keep_idx]
        models = [models[i] for i in keep_idx]
        names = [names[i] for i in keep_idx]

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


def cmd_rr_prefix(args: argparse.Namespace) -> None:
    """
    Common workflow: given an R2 prefix, pull `weights_best.pt` plus N evenly-spaced
    `weights_iter_*.pt` snapshots, then play a full round-robin head2head.
    """
    selected = _select_checkpoints_from_prefix(
        args.prefix,
        n_iters=args.iters,
        include_best=not args.no_best,
    )
    for c in selected:
        _sync_checkpoint(c)

    h2h_args = argparse.Namespace(
        checkpoints=[str(p) for p in selected],
        games=args.games,
        round_robin=True,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioning=args.conditioning,
        seed=args.seed,
        db=args.db,
    )
    cmd_head2head(h2h_args)


def cmd_leaderboard(args: argparse.Namespace) -> None:
    games = query_games()
    if not games:
        print("No games in database.")
        return

    # Full leaderboard (Stockfish anchored, Patzer fitted). We hide Stockfish rows in display.
    ratings_all = compute_ratings(games)
    ratings_all = [r for r in ratings_all if r.games >= args.min_games]

    # H2H-only leaderboard: only include Patzer-vs-Patzer games, no Stockfish anchors.
    h2h_games = [
        g for g in games
        if isinstance(g.get("white"), str)
        and isinstance(g.get("black"), str)
        and g["white"].startswith("patzer_v")
        and g["black"].startswith("patzer_v")
    ]
    ratings_h2h = compute_ratings(h2h_games) if h2h_games else []
    ratings_h2h = [r for r in ratings_h2h if r.games >= args.min_games]
    h2h_rank_by_name = {r.name: i + 1 for i, r in enumerate(ratings_h2h)}

    # Display: Patzer only.
    display = [r for r in ratings_all if r.name.startswith("patzer_v")]
    if not display:
        print(f"No Patzer players with ≥{args.min_games} games.")
        return

    print(
        f"\n{'Rank':<5} {'H2H_Rank':<9} {'Player':<28} {'Elo':>6} {'±':>5} {'Games':>6} {'W-L-D'}"
    )
    print("-" * 75)
    for rank, r in enumerate(display, 1):
        se = f"{r.stderr:.0f}" if not math.isnan(r.stderr) else "—"
        wld = f"{r.wins}-{r.losses}-{r.draws}"
        h2h_rank = h2h_rank_by_name.get(r.name)
        h2h_disp = f"{h2h_rank}" if h2h_rank is not None else "—"
        print(f"{rank:<5} {h2h_disp:<9} {r.name:<28} {r.elo:>6.0f} {se:>5} {r.games:>6} {wld}")


def _checkpoint_map_from_games(games: list[dict]) -> dict[str, str]:
    """
    Map player label -> most common checkpoint path seen in DB for that player.
    Only considers Patzer players where the corresponding checkpoint field is non-null.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for g in games:
        w, b = g.get("white"), g.get("black")
        wc, bc = g.get("white_checkpoint"), g.get("black_checkpoint")
        if isinstance(w, str) and isinstance(wc, str) and w.startswith("patzer_v"):
            counts[w][wc] += 1
        if isinstance(b, str) and isinstance(bc, str) and b.startswith("patzer_v"):
            counts[b][bc] += 1

    out: dict[str, str] = {}
    for player, c in counts.items():
        if not c:
            continue
        ckpt, _ = c.most_common(1)[0]
        out[player] = ckpt
    return out


def cmd_rr_leaderboard(args: argparse.Namespace) -> None:
    """
    Convenience: take all Patzer players currently on the leaderboard (min-games),
    resolve their checkpoint paths from the DB, then run a head2head round-robin.
    """
    games = query_games()
    if not games:
        print("No games in database.")
        return

    ratings_all = compute_ratings(games)
    ratings_all = [r for r in ratings_all if r.games >= args.min_games]
    if not ratings_all:
        print(f"No players with ≥{args.min_games} games.")
        return

    ckpt_map = _checkpoint_map_from_games(games)

    # Display and selection should be Patzer-only (Stockfish games still contribute to Elo,
    # but we don't show Stockfish rows here).
    patzers = [r for r in ratings_all if r.name.startswith("patzer_v")]
    if not patzers:
        print(f"No Patzer players with ≥{args.min_games} games.")
        return

    top = patzers[: min(20, len(patzers))]
    print("\nTop 20 Patzer leaderboard candidates:")
    print(f"{'Rank':<5} {'Player':<28} {'Elo':>6} {'±':>5} {'Games':>6} {'W-L-D'}")
    print("-" * 65)
    for rank, r in enumerate(top, 1):
        se = f"{r.stderr:.0f}" if not math.isnan(r.stderr) else "—"
        wld = f"{r.wins}-{r.losses}-{r.draws}"
        print(f"{rank:<5} {r.name:<28} {r.elo:>6.0f} {se:>5} {r.games:>6} {wld}")

    if args.no_prompt:
        players = [r.name for r in top]
    else:
        print("\nSelect models (by Rank) to run a round-robin head2head now.")
        print("Examples: '1 2 3', '1-5', '1,3,7-10'. Press Enter for default (top 10).")

        def _parse_rank_list(raw: str, n_max: int) -> list[int]:
            raw = raw.strip()
            if not raw:
                return []
            raw = raw.replace(",", " ")
            toks = [t for t in raw.split() if t]
            out: list[int] = []
            for t in toks:
                if "-" in t:
                    a, b = t.split("-", 1)
                    try:
                        lo = int(a); hi = int(b)
                    except ValueError:
                        continue
                    if lo > hi:
                        lo, hi = hi, lo
                    for x in range(lo, hi + 1):
                        if 1 <= x <= n_max:
                            out.append(x)
                else:
                    try:
                        x = int(t)
                    except ValueError:
                        continue
                    if 1 <= x <= n_max:
                        out.append(x)
            # Deduplicate while preserving order
            seen = set()
            dedup: list[int] = []
            for x in out:
                if x not in seen:
                    dedup.append(x)
                    seen.add(x)
            return dedup

        try:
            raw = input("Ranks to include (default: 1-10): ")
        except EOFError:
            return
        ranks = _parse_rank_list(raw, len(top))
        if not ranks:
            ranks = list(range(1, min(10, len(top)) + 1))

        players = [top[i - 1].name for i in ranks]
        if len(players) < 2:
            print("[rr-leaderboard] need at least 2 Patzer models selected; skipping.")
            return

    if args.limit is not None:
        players = players[: max(0, int(args.limit))]

    checkpoints: list[Path] = []
    missing: list[str] = []
    for p in players:
        ck = ckpt_map.get(p)
        if not ck:
            missing.append(p)
            continue
        checkpoints.append(Path(ck))

    # Deduplicate checkpoints while preserving order (in case of label collisions).
    seen = set()
    uniq: list[Path] = []
    for c in checkpoints:
        s = str(c)
        if s not in seen:
            uniq.append(c)
            seen.add(s)
    checkpoints = uniq

    # If multiple leaderboard labels point to byte-identical checkpoints (e.g. weights_best == weights_iter),
    # keep only one to avoid treating the same model as multiple players. Prefer weights_best.pt.
    try:
        keys = [str(c) for c in checkpoints]
        etags = {k: r2.get_etag(k) for k in keys}
        # Group by ETag
        groups: dict[str, list[Path]] = defaultdict(list)
        no_etag: list[Path] = []
        for c in checkpoints:
            et = etags.get(str(c))
            if et is None:
                no_etag.append(c)
            else:
                groups[et].append(c)

        keep: list[Path] = []
        for et, paths in groups.items():
            # Prefer weights_best.pt when present
            best = next((p for p in paths if p.name == "weights_best.pt"), None)
            keep.append(best if best is not None else paths[0])
        keep.extend(no_etag)

        # Preserve original order
        keep_set = {str(p) for p in keep}
        checkpoints = [c for c in checkpoints if str(c) in keep_set]
    except Exception:
        pass

    if missing:
        print(f"[rr-leaderboard] warning: no checkpoint path for {len(missing)} player(s): {', '.join(missing[:8])}")
        if len(missing) > 8:
            print(f"[rr-leaderboard] (and {len(missing) - 8} more)")

    if len(checkpoints) < 2:
        print("[rr-leaderboard] need at least 2 checkpoints to run.")
        return

    for c in checkpoints:
        _sync_checkpoint(c)

    h2h_args = argparse.Namespace(
        checkpoints=[str(p) for p in checkpoints],
        games=args.games,
        round_robin=True,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioning=args.conditioning,
        seed=args.seed,
        db=args.db,
    )
    cmd_head2head(h2h_args)


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
    xs = [iter_display_k(it) for it in iters]
    ax.plot(xs, elos, marker="o", linewidth=2)
    ax.set_xlabel("Training step (iter / 1000)")
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
    p_sf.add_argument("checkpoint", nargs="?", help="Checkpoint path / prefix / shorthand (e.g. patzer_v3@180)")
    p_sf.add_argument(
        "--checkpoint",
        dest="checkpoint_flag",
        default=None,
        help="Alias for the positional checkpoint (e.g. --checkpoint patzer_v3@180)",
    )
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

    # --- rr-prefix ---
    p_rr = sub.add_parser(
        "rr-prefix",
        help="Round-robin: weights_best + N evenly-spaced weights_iter_* under an R2 prefix",
    )
    p_rr.add_argument("prefix", help="R2 prefix, e.g. checkpoints/patzer_v2")
    p_rr.add_argument("--iters", type=int, default=6, help="How many weights_iter_* snapshots to include (default 6)")
    p_rr.add_argument("--no-best", action="store_true", help="Exclude weights_best.pt (default: include)")
    p_rr.add_argument("--games", type=int, default=12, help="Games per pair (default 12)")
    p_rr.add_argument("--device", default="cpu")
    p_rr.add_argument("--temperature", type=float, default=0.0)
    p_rr.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_rr.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_rr.add_argument("--seed", type=int, default=42)
    p_rr.add_argument("--db", default=str(DB_PATH), help="Database path")

    # --- rr-leaderboard ---
    p_rrlb = sub.add_parser(
        "rr-leaderboard",
        help="Round-robin across all Patzer models currently on the leaderboard",
    )
    p_rrlb.add_argument("--min-games", type=int, default=5, dest="min_games",
                        help="Only include players with at least this many games (default 5)")
    p_rrlb.add_argument("--limit", type=int, default=None,
                        help="Optional cap: only take top N players by current Elo")
    p_rrlb.add_argument("--games", type=int, default=12, help="Games per pair (default 12)")
    p_rrlb.add_argument("--device", default="cpu")
    p_rrlb.add_argument("--temperature", type=float, default=0.0)
    p_rrlb.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_rrlb.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_rrlb.add_argument("--seed", type=int, default=42)
    p_rrlb.add_argument("--db", default=str(DB_PATH), help="Database path")
    p_rrlb.add_argument("--no-prompt", action="store_true", help="Use the whole displayed top 20 (Patzer only) without prompting")

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

    # Support --checkpoint as an alias for the positional for stockfish.
    if getattr(args, "cmd", None) == "stockfish":
        if getattr(args, "checkpoint", None) is None and getattr(args, "checkpoint_flag", None):
            args.checkpoint = args.checkpoint_flag
        if getattr(args, "checkpoint", None) is None:
            parser.error("stockfish: the following arguments are required: checkpoint")

    # Allow --db override to propagate to db module
    if hasattr(args, "db") and args.db != str(DB_PATH):
        import eval.db as db_mod
        db_mod.DB_PATH = Path(args.db)

    dispatch = {
        "stockfish": cmd_stockfish,
        "head2head": cmd_head2head,
        "rr-prefix": cmd_rr_prefix,
        "rr-leaderboard": cmd_rr_leaderboard,
        "leaderboard": cmd_leaderboard,
        "history": cmd_history,
        "progress": cmd_progress,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
