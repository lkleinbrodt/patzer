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

    # Several checkpoints — plays each vs Stockfish sequentially (same flags each run):
    python eval/evaluate.py stockfish patzer_v3@best patzer_v3@180 --games 50 --device mps

    # Shorthand (iter in thousands, i.e. iter/1000):
    python eval/evaluate.py stockfish patzer_v3@180 --games 50 --device mps
    python eval/evaluate.py stockfish --checkpoint patzer_v3@best --games 50 --device mps

    # Prefix (interactive selection from R2):
    python eval/evaluate.py stockfish checkpoints/patzer_v2 --games 50 --device mps

  With `--seed`, each checkpoint gets a deterministic but distinct RNG stream (`seed + i·1000003`).
  Stops early if posterior sigma drops below --stop-sigma (default 50 Elo points).

Compare two checkpoints head-to-head:

    python eval/evaluate.py head2head \\
        checkpoints/patzer_v2/weights_best.pt \\
        checkpoints/patzer_v1/weights_best.pt \\
        --games 20 --device mps

Gauntlet: one challenger plays each selected leaderboard opponent in turn (opponents do not play each other):

    python eval/evaluate.py gauntlet patzer_v4@best --games 50 --device mps

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
  --temperature   move sampling temperature; 0.0 = greedy, 0.01 = default
  --conditioning  match_color | white_win | black_win | draw | none (default: match_color)
  --games         number of games to play
  --stockfish     path to stockfish binary (default: /opt/homebrew/bin/stockfish)
  --db            path to results database (default: eval/results.db)
  --visualize     (stockfish, head2head, gauntlet, rr-*) split window: scrollable log + metrics
                  on the left (mirrors eval output), live board on the right; one refresh per
                  half-move and after each log line; closing the window aborts
"""

import argparse
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.db import DB_PATH, insert_game, iter_display_k, player_name, query_games, stockfish_name
from eval.elo import compute_ratings
from eval.engine import CONDITIONING_OPTIONS, Patzer, StockfishPlayer
import patzer.r2 as r2

DEFAULT_STOCKFISH = "/opt/homebrew/bin/stockfish"

# ── Terminal display helpers ─────────────────────────────────────────────────

_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(s: str, *codes: str) -> str:
    return f"\033[{';'.join(codes)}m{s}\033[0m" if _COLOR else s


def _bold(s: str) -> str:   return _c(s, "1")
def _dim(s: str) -> str:    return _c(s, "2")
def _green(s: str) -> str:  return _c(s, "32")
def _red(s: str) -> str:    return _c(s, "31")
def _yellow(s: str) -> str: return _c(s, "33")
def _cyan(s: str) -> str:   return _c(s, "36")


def _dur(sec: float) -> str:
    """Human-readable duration, e.g. 4s, 2m05s, 1h03m."""
    if sec < 60:
        return f"{sec:.0f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _eta(sec: float) -> str:
    """ETA string with appropriate granularity."""
    if sec < 90:
        return f"~{int(sec)}s"
    m = int(sec) // 60
    if m < 90:
        return f"~{m}m"
    h, m = divmod(m, 60)
    return f"~{h}h{m:02d}m"


def _result_colored(result: str, score_a: float) -> str:
    """Color-coded result string from player-A's perspective."""
    short = {"1-0": "1-0", "0-1": "0-1", "1/2-1/2": "½-½"}.get(result, result)
    if score_a == 1.0:
        return _green(short)
    if score_a == 0.0:
        return _red(short)
    return _yellow(short)


def _score_colored(pct: float) -> str:
    s = f"{pct:5.1f}%"
    if pct >= 60:
        return _green(s)
    if pct <= 40:
        return _red(s)
    return _yellow(s)


def _plain_col(s: str, width: int, align: str = "<") -> str:
    """Truncate/pad plain text to exactly `width` characters for monospace tables.

    Format widths must be applied *before* wrapping with ANSI codes, or columns drift.
    """
    if len(s) > width:
        s = s[: max(1, width)]
    if align == ">":
        return s.rjust(width)
    if align == "^":
        return s.center(width)
    return s.ljust(width)


def _parse_rank_list(raw: str, n_max: int) -> list[int]:
    """Parse rank selections like '1 2 3', '1-5', '1,3,7-10'. Empty input → []."""
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
                lo = int(a)
                hi = int(b)
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
    seen: set[int] = set()
    dedup: list[int] = []
    for x in out:
        if x not in seen:
            dedup.append(x)
            seen.add(x)
    return dedup


def _print_patzer_candidate_table(title_inner: str, top: list) -> None:
    """Print a numbered Patzer-only candidate table (PlayerRating rows)."""
    w = max(56, len(title_inner))
    print()
    print(_cyan(_bold("  ╔" + "═" * w + "╗")))
    print(_cyan(_bold(f"  ║{title_inner:<{w}}║")))
    print(_cyan(_bold("  ╚" + "═" * w + "╝")))
    print()

    nt = len(top)
    rk_w = max(2, len(str(nt)))
    games_w = max(5, *(len(str(r.games)) for r in top))
    wld_w = max(5, *(len(f"{r.wins}-{r.losses}-{r.draws}") for r in top))
    plain_hdr = (
        f"  {_plain_col('#', rk_w, '>')}  {_plain_col('Player', 28)}  "
        f"{_plain_col('Elo', 5, '>')}  {_plain_col('±', 4, '>')}  "
        f"{_plain_col('Games', games_w, '>')}  {_plain_col('W-L-D', wld_w)}"
    )
    rule_w = max(0, len(plain_hdr) - 2)
    print(_dim("  " + "─" * rule_w))
    print(_dim(plain_hdr))
    print(_dim("  " + "─" * rule_w))
    for rank, r in enumerate(top, 1):
        se = f"{r.stderr:.0f}" if not math.isnan(r.stderr) else "—"
        wld = _plain_col(f"{r.wins}-{r.losses}-{r.draws}", wld_w)
        elo_str = _plain_col(f"{r.elo:5.0f}", 5, ">")
        rk_s = _plain_col(str(rank), rk_w, ">")
        name_s = _plain_col(r.name, 28)
        se_s = _plain_col(se, 4, ">")
        g_s = _plain_col(str(r.games), games_w, ">")

        if rank <= 3:
            row = (
                f"  {rk_s}  {_bold(name_s)}  {_bold(elo_str)}  "
                f"{_dim(se_s)}  {g_s}  {_dim(wld)}"
            )
        else:
            row = f"  {rk_s}  {name_s}  {elo_str}  {_dim(se_s)}  {g_s}  {_dim(wld)}"
        print(row)
    print(_dim("  " + "─" * rule_w))


def _print_rr_banner(
    n_players: int, n_pairs: int, total_games: int, viz=None,
) -> None:
    inner = f"  Round Robin  ·  {n_players} players  ·  {n_pairs} matches  ·  {total_games:,} games  "
    w = max(58, len(inner))
    print()
    print(_cyan(_bold("  ╔" + "═" * w + "╗")))
    print(_cyan(_bold(f"  ║{inner:<{w}}║")))
    print(_cyan(_bold("  ╚" + "═" * w + "╝")))
    if viz is not None:
        viz.log(
            f"Round Robin · {n_players} players · {n_pairs} matches · {total_games:,} games",
            "accent",
        )


def _print_match_header(
    pair_idx: int,
    n_pairs: int,
    na: str,
    nb: str,
    games_per_pair: int,
    games_done: int,
    total_games: int,
    t_start: float,
    viz=None,
) -> None:
    elapsed = time.time() - t_start
    if games_done > 0:
        rate = games_done / elapsed
        remaining = total_games - games_done
        eta_str = f"  {_eta(remaining / rate)}"
        eta_plain = f" · ETA {_eta(remaining / rate)}"
    else:
        eta_str = ""
        eta_plain = ""

    match_label = _bold(_cyan(f"Match {pair_idx + 1}/{n_pairs}"))
    right = _dim(f"{games_done}/{total_games} games · {_dur(elapsed)} elapsed{eta_str}")
    # Build a divider that fills ~78 chars
    label_plain = f"Match {pair_idx + 1}/{n_pairs}"
    fill = max(4, 76 - len(label_plain) - len(f"{games_done}/{total_games} games · {_dur(elapsed)} elapsed"))
    divider = _dim("─" * fill)

    print()
    print(f"  {match_label}  {divider}  {right}")
    print(f"  {_bold(na)}  {_dim('vs')}  {_bold(nb)}  " + _dim(f"({games_per_pair} games)"))
    if viz is not None:
        viz.log(
            f"Match {pair_idx + 1}/{n_pairs} · {games_done}/{total_games} games · "
            f"{_dur(elapsed)} elapsed{eta_plain}",
            "accent",
        )
        viz.log(f"{na}  vs  {nb}  ({games_per_pair} games)", "normal")


def _print_game_line(
    game_idx: int,
    games_per_pair: int,
    na: str,
    nb: str,
    result: str,
    termination: str,
    wa: int,
    la: int,
    da: int,
    score_a: float,
    a_is_white: bool,
    game_secs: float,
    viz=None,
) -> None:
    """Print one game line. na/nb are always model-A / model-B (stable order).
    Colors (W/B) reflect actual board assignment for this game."""
    played = wa + la + da
    pct = (wa + 0.5 * da) / played * 100 if played else 0.0

    ca = _dim("W") if a_is_white else _dim("B")   # model A's color this game
    cb = _dim("B") if a_is_white else _dim("W")   # model B's color this game
    result_str = _result_colored(result, score_a)
    term_str = _dim(_plain_col(termination or "", 20))
    wld_str = _dim(_plain_col(f"{wa}-{la}-{da}", 11))
    score_str = _score_colored(pct)
    time_str = _dim(f"{game_secs:.1f}s")

    na_c = _bold(_plain_col(na, 24))
    nb_c = _bold(_plain_col(nb, 24))
    print(
        f"  {game_idx + 1:>2}/{games_per_pair}"
        f"  {ca} {na_c}  {cb} {nb_c}"
        f"  {result_str}  {term_str}"
        f"  {wld_str}  {score_str}  {time_str}"
    )
    if viz is not None:
        played_v = wa + la + da
        pct_v = (wa + 0.5 * da) / played_v * 100 if played_v else 0.0
        ca_p = "W" if a_is_white else "B"
        cb_p = "B" if a_is_white else "W"
        viz.log(
            f"{game_idx + 1}/{games_per_pair}  {ca_p} {na[:26]:<26}  {cb_p} {nb[:26]:<26}  "
            f"{result}  {(termination or '')[:18]}  "
            f"{wa}-{la}-{da}  {pct_v:5.1f}%  {game_secs:.1f}s",
            "normal",
        )


def _print_match_footer(
    na: str, nb: str, wa: int, la: int, da: int, match_secs: float, viz=None,
) -> None:
    played = wa + la + da
    pct = (wa + 0.5 * da) / played * 100 if played else 0.0
    avg = match_secs / played if played else 0.0

    if wa > la:
        verdict = _green(f"{na} wins the match")
    elif la > wa:
        verdict = _red(f"{nb} wins the match")
    else:
        verdict = _yellow("match drawn")

    print(_dim("  " + "─" * 76))
    print(
        f"  {_bold(na)}  W-L-D {wa}-{la}-{da}  "
        f"{_score_colored(pct)}  ·  {verdict}  "
        + _dim(f"·  avg {avg:.1f}s/game")
    )
    if viz is not None:
        viz.log("—" * 42, "dim")
        vplain = (
            f"{na} wins the match" if wa > la else f"{nb} wins the match" if la > wa else "match drawn"
        )
        tone = "good" if wa != la else "warn"
        viz.log(
            f"{na}  W-L-D {wa}-{la}-{da}  ·  {pct:.1f}%  ·  {vplain}  ·  avg {avg:.1f}s/game",
            tone,
        )


def _print_standings(
    pair_idx: int,
    n_pairs: int,
    t_start: float,
    db_path: Path,
    n_show: int = 15,
    viz=None,
) -> None:
    """Print current Elo standings computed from all games stored so far."""
    games = query_games(db_path=db_path)
    if not games:
        return
    ratings = [r for r in compute_ratings(games) if r.name.startswith("patzer_v")]
    if not ratings:
        return

    elapsed = time.time() - t_start
    print()
    print(
        f"  {_dim('Standings after')}"
        f"  {_bold(f'{pair_idx}/{n_pairs}')} {_dim('matches')}  "
        f"{_dim('·')}  {_dim(_dur(elapsed) + ' elapsed')}"
    )
    shown = ratings[:n_show]
    rk_w = max(2, len(str(len(shown))))
    gw = max(5, *(len(str(r.games)) for r in shown))
    wld_w = max(5, *(len(f"{r.wins}-{r.losses}-{r.draws}") for r in shown))
    plain_hdr = (
        f"  {_plain_col('#', rk_w, '>')}  {_plain_col('Player', 28)}  "
        f"{_plain_col('Elo', 5, '>')}  {_plain_col('±', 4, '>')}  "
        f"{_plain_col('Games', gw, '>')}  {_plain_col('W-L-D', wld_w)}"
    )
    rule_w = max(0, len(plain_hdr) - 2)
    print(_dim("  " + "─" * rule_w))
    print(_dim(plain_hdr))
    print(_dim("  " + "─" * rule_w))
    for rank, r in enumerate(shown, 1):
        se = f"{r.stderr:.0f}" if r.stderr and not math.isnan(r.stderr) else "—"
        wld = _plain_col(f"{r.wins}-{r.losses}-{r.draws}", wld_w)
        elo_str = _plain_col(f"{r.elo:5.0f}", 5, ">")
        rk_s = _plain_col(str(rank), rk_w, ">")
        name_s = _plain_col(r.name, 28)
        se_s = _plain_col(se, 4, ">")
        g_s = _plain_col(str(r.games), gw, ">")
        if rank == 1:
            row = (
                f"  {rk_s}  {_bold(_green(name_s))}  {_bold(_green(elo_str))}  "
                f"{_dim(se_s)}  {g_s}  {_dim(wld)}"
            )
        elif rank <= 3:
            row = (
                f"  {rk_s}  {_bold(name_s)}  {_bold(elo_str)}  "
                f"{_dim(se_s)}  {g_s}  {_dim(wld)}"
            )
        else:
            row = f"  {rk_s}  {name_s}  {elo_str}  {_dim(se_s)}  {g_s}  {_dim(wld)}"
        print(row)
    print()
    if viz is not None:
        viz.log(f"Standings after {pair_idx}/{n_pairs} matches · {_dur(time.time() - t_start)} elapsed", "accent")
        hdr = f"{'#':>3}  {'Player':<28}  {'Elo':>5}  {'±':>4}  {'Games':>5}  W-L-D"
        viz.log(hdr, "dim")
        for rank, r in enumerate(shown, 1):
            se = f"{r.stderr:.0f}" if r.stderr and not math.isnan(r.stderr) else "—"
            wldp = f"{r.wins}-{r.losses}-{r.draws}"
            line = f"{rank:>3}  {r.name:<28}  {r.elo:5.0f}  {se:>4}  {r.games:5}  {wldp}"
            tnn = "good" if rank == 1 else "accent" if rank <= 3 else "normal"
            viz.log(line, tnn)


def _print_sf_banner(
    pname: str,
    max_games: int,
    stop_sigma: float,
    prior_elo: float,
    prior_sigma: float,
    elo_step: int,
    *,
    run_idx: int | None = None,
    n_runs: int | None = None,
    viz=None,
) -> None:
    prefix = ""
    if n_runs is not None and n_runs > 1 and run_idx is not None:
        prefix = f"Run {run_idx}/{n_runs} · "
    inner = (
        f"  {prefix}Stockfish eval  ·  {pname}  ·  ≤{max_games} games  ·  "
        f"stop σ≤{stop_sigma:g}  ·  prior {prior_elo:.0f}±{prior_sigma:.0f} (step {elo_step})  "
    )
    w = max(58, len(inner))
    print()
    print(_cyan(_bold("  ╔" + "═" * w + "╗")))
    print(_cyan(_bold(f"  ║{inner:<{w}}║")))
    print(_cyan(_bold("  ╚" + "═" * w + "╝")))
    print(
        _dim(
            "  Adaptive UCI_Elo: first game uses minimum rated strength; "
            "later games target the posterior mean (clamped to engine limits)."
        )
    )
    print()
    if viz is not None:
        pfx = ""
        if n_runs is not None and n_runs > 1 and run_idx is not None:
            pfx = f"Run {run_idx}/{n_runs} · "
        viz.log(
            f"{pfx}Stockfish eval · {pname} · ≤{max_games} games · "
            f"stop σ≤{stop_sigma:g} · prior {prior_elo:.0f}±{prior_sigma:.0f} (step {elo_step})",
            "accent",
        )
        viz.log(
            "Adaptive UCI_Elo: first game uses minimum rated strength; "
            "later games target the posterior mean.",
            "dim",
        )


def _print_sf_game_line(
    game_idx: int,
    max_games: int,
    target_elo: int,
    pname: str,
    *,
    result: str,
    termination: str,
    score_patzer: float,
    total_w: int,
    total_l: int,
    total_d: int,
    mu_est: float,
    sigma_est: float,
    patzer_is_white: bool,
    game_secs: float,
    eta_sec: float | None,
    viz=None,
) -> None:
    """One finished game vs Stockfish (compact row, ANSI-safe widths)."""
    idx_w = max(5, len(str(max_games)) * 2 + 1)
    idx_s = _plain_col(f"{game_idx}/{max_games}", idx_w, ">")
    sf_tag = _plain_col(f"SF {target_elo}", 7)
    ca = _dim("W") if patzer_is_white else _dim("B")
    pn = _plain_col(pname, 24)
    opp = _plain_col(stockfish_name(target_elo), 18)
    res = _result_colored(result, score_patzer)
    term = _dim(_plain_col(termination or "", 16))
    wld = _dim(_plain_col(f"{total_w}-{total_l}-{total_d}", 7))
    played = total_w + total_l + total_d
    pct = (total_w + 0.5 * total_d) / played * 100 if played else 0.0
    pct_s = _score_colored(pct)
    est_plain = _plain_col(f"{mu_est:.0f}±{sigma_est:.0f}", 12)
    est_cell = _bold(est_plain)
    time_str = _dim(f"{game_secs:.1f}s")
    eta_part = _dim(f"  ETA {_eta(eta_sec)}") if eta_sec is not None else ""

    print(
        f"  {idx_s}  {sf_tag}  {ca} {pn}  {_dim('v')} {opp}"
        f"  {res}  {term}  {wld}  {pct_s}  {_dim('est')} {est_cell}  {time_str}{eta_part}"
    )
    if viz is not None:
        short = {"1-0": "1-0", "0-1": "0-1", "1/2-1/2": "½-½"}.get(result, result)
        eta_p = f"  ETA {_eta(eta_sec)}" if eta_sec is not None else ""
        ca_p = "W" if patzer_is_white else "B"
        viz.log(
            f"{game_idx}/{max_games}  SF {target_elo}  {ca_p} {pname[:22]:<22}  v  {stockfish_name(target_elo)[:16]:<16}  "
            f"{short}  {(termination or '')[:14]:<14}  {total_w}-{total_l}-{total_d}  "
            f"{pct:5.1f}%  est {mu_est:.0f}±{sigma_est:.0f}  {game_secs:.1f}s{eta_p}",
            "normal",
        )


def _print_sf_stop_confidence(
    sigma: float,
    stop_sigma: float,
    n_games: int,
    elapsed: float,
    viz=None,
) -> None:
    print(_dim("  " + "─" * 72))
    print(
        f"  {_bold(_green('Stopping'))}  {_dim('·')}  posterior σ = {_bold(f'{sigma:.1f}')} "
        f"{_dim('≤')} {_bold(f'{stop_sigma:g}')}  "
        f"{_dim(f'·  {n_games} games · {_dur(elapsed)}')}"
    )
    if viz is not None:
        viz.log(
            f"Stopping · posterior σ = {sigma:.1f} ≤ {stop_sigma:g} · "
            f"{n_games} games · {_dur(elapsed)}",
            "good",
        )


def _print_sf_summary(
    pname: str,
    mu: float,
    sigma: float,
    n_games: int,
    elapsed: float,
    viz=None,
) -> None:
    print(_dim("  " + "─" * 72))
    print(
        f"  {_bold('Result')}  {_bold(pname)}  {_dim('→')}  "
        f"estimated Elo {_bold(f'{mu:.0f}')} ± {_bold(f'{sigma:.0f}')}  "
        f"{_dim(f'({n_games} games · {_dur(elapsed)})')}"
    )
    print()
    if viz is not None:
        viz.log(
            f"Result  {pname}  →  estimated Elo {mu:.0f} ± {sigma:.0f}  "
            f"({n_games} games · {_dur(elapsed)})",
            "accent",
        )


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

OPENINGS_JSON_PATH = Path(__file__).parent / "openings.json"

_opening_book_cache: list[list[str]] | None = None


def _load_opening_book() -> list[list[str]]:
    """
    Load opening sequences from openings.json — a pre-filtered list of balanced
    UCI move sequences (built by eval/build_opening_book.py from Stockfish's
    2moves_v2.pgn, filtered to |eval| ≤ 75 cp).

    Each entry is a list of 4 UCI move strings, e.g. ["e2e4","e7e5","g1f3","b8c6"].
    These are both applied to the board AND pre-filled into move_history so the
    model receives its proper opening context.

    Falls back to a tiny hardcoded set if the file is missing.
    """
    global _opening_book_cache
    if _opening_book_cache is not None:
        return _opening_book_cache

    if OPENINGS_JSON_PATH.exists():
        import json
        _opening_book_cache = json.loads(OPENINGS_JSON_PATH.read_text())
        return _opening_book_cache

    print(
        f"[warn] {OPENINGS_JSON_PATH} not found — using minimal hardcoded fallback.\n"
        "       Run:  python eval/build_opening_book.py",
        file=sys.stderr,
    )
    _opening_book_cache = [
        ["e2e4", "e7e5", "g1f3", "b8c6"],
        ["d2d4", "d7d5", "c2c4", "e7e6"],
        ["e2e4", "c7c5", "g1f3", "d7d6"],
        ["d2d4", "g8f6", "c2c4", "g7g6"],
        ["c2c4", "e7e5", "b1c3", "g8f6"],
        ["e2e4", "e7e6", "d2d4", "d7d5"],
        ["e2e4", "c7c6", "d2d4", "d7d5"],
        ["d2d4", "d7d5", "g1f3", "g8f6"],
    ]
    return _opening_book_cache


# ---------------------------------------------------------------------------
# Game engine helpers
# ---------------------------------------------------------------------------

def play_game(
    white,
    black,
    opening_moves: list[str] | None = None,
    max_moves: int = 300,
    on_ply: Callable[[chess.Board, chess.Move], None] | None = None,
    return_move_history: bool = False,
) -> tuple[str, str] | tuple[str, str, list[str]]:
    """
    Play one game from the starting position (optionally with opening_moves
    pre-applied). Returns (result, termination), or (result, termination, move_history)
    when return_move_history is True (UCI move strings in order played).

    result: '1-0' | '0-1' | '1/2-1/2'
    termination: 'checkmate' | 'stalemate' | 'fifty_moves' | 'threefold_repetition' |
                 'insufficient_material' | 'seventyfive_moves' | 'fivefold_repetition' |
                 'move_limit'

    opening_moves are both applied to the board AND pre-filled into move_history
    so Patzer models receive their proper opening context (the sequence they were
    trained on) rather than being asked to move from an unfamiliar position with
    zero context tokens.
    """
    board = chess.Board()
    move_history: list[str] = []

    if opening_moves:
        for uci in opening_moves:
            m = chess.Move.from_uci(uci)
            if m in board.legal_moves:
                board.push(m)
                move_history.append(uci)
                if on_ply is not None:
                    on_ply(board, m)
            else:
                # Shouldn't happen with a valid book — bail and start from wherever we got
                print(f"  [warn] opening move {uci} illegal in position — truncating", file=sys.stderr)
                break

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
        if on_ply is not None:
            on_ply(board, move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result, termination = "1/2-1/2", "move_limit"
    else:
        termination = outcome.termination.name.lower()
        if outcome.winner is chess.WHITE:
            result = "1-0"
        elif outcome.winner is chess.BLACK:
            result = "0-1"
        else:
            result = "1/2-1/2"

    if return_move_history:
        return result, termination, move_history
    return result, termination


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

def _stockfish_eval_one(
    args: argparse.Namespace,
    ckpt: Path,
    rng: random.Random,
    *,
    run_idx: int,
    n_runs: int,
) -> None:
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
    opening_book = _load_opening_book()
    db_path = Path(args.db)
    t_start = time.time()
    viz = None
    if getattr(args, "visualize", False):
        from eval.eval_viz import EvalBoardViewer

        viz = EvalBoardViewer()

    def _round(x: float) -> int:
        return int(round(x / args.elo_step) * args.elo_step)

    _print_sf_banner(
        pname,
        args.games,
        args.stop_sigma,
        args.prior_elo,
        args.prior_sigma,
        args.elo_step,
        run_idx=run_idx,
        n_runs=n_runs,
        viz=viz,
    )

    try:
        while games_played < args.games:
            mu, sigma = _posterior_mean_sigma(xs, p)
            if sigma <= args.stop_sigma and games_played > 0:
                _print_sf_stop_confidence(
                    sigma,
                    args.stop_sigma,
                    games_played,
                    time.time() - t_start,
                    viz=viz,
                )
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

            opening_moves = rng.choice(opening_book)
            if viz is not None:
                viz.set_players(white.name, black.name)
            t_game = time.time()
            result, termination = play_game(
                white,
                black,
                opening_moves=opening_moves,
                on_ply=viz.ply if viz is not None else None,
            )
            game_secs = time.time() - t_game
            score = _score_from_result(result, patzer_is_white)

            if score == 1.0:
                total_w += 1
            elif score == 0.0:
                total_l += 1
            else:
                total_d += 1

            games_played += 1
            p = _posterior_update(xs, p, target_elo, score)
            mu2, sigma2 = _posterior_mean_sigma(xs, p)

            eta_sec = None
            if games_played >= 3:
                elapsed = time.time() - t_start
                avg_t = elapsed / games_played
                rem = max(0, args.games - games_played)
                eta_sec = rem * avg_t

            _print_sf_game_line(
                games_played,
                args.games,
                target_elo,
                pname,
                result=result,
                termination=termination,
                score_patzer=score,
                total_w=total_w,
                total_l=total_l,
                total_d=total_d,
                mu_est=mu2,
                sigma_est=sigma2,
                patzer_is_white=patzer_is_white,
                game_secs=game_secs,
                eta_sec=eta_sec,
                viz=viz,
            )

            insert_game(
                white=pname if patzer_is_white else sf_name,
                black=sf_name if patzer_is_white else pname,
                result=result,
                termination=termination,
                white_checkpoint=rel_ckpt if patzer_is_white else None,
                black_checkpoint=None if patzer_is_white else rel_ckpt,
                white_iter=patzer.iter_num if patzer_is_white else None,
                black_iter=None if patzer_is_white else patzer.iter_num,
                opening=" ".join(opening_moves),
                temperature=args.temperature,
                top_k=args.top_k,
                conditioning=args.conditioning,
                db_path=db_path,
            )
    finally:
        if sf is not None:
            sf.close()

    mu, sigma = _posterior_mean_sigma(xs, p)
    _print_sf_summary(pname, mu, sigma, games_played, time.time() - t_start, viz=viz)
    if viz is not None:
        viz.close()


def cmd_stockfish(args: argparse.Namespace) -> None:
    ckpts = [_resolve_checkpoint(s) for s in args.checkpoints]
    for c in ckpts:
        _sync_checkpoint(c)

    n = len(ckpts)
    for i, ckpt in enumerate(ckpts):
        if i > 0:
            print()
            print(_dim("  " + "─" * 72))
        if args.seed is None:
            rng = random.Random()
        else:
            rng = random.Random(args.seed + i * 1_000_003)
        _stockfish_eval_one(args, ckpt, rng, run_idx=i + 1, n_runs=n)


def cmd_head2head(args: argparse.Namespace) -> None:
    checkpoints = [_resolve_checkpoint(c) for c in args.checkpoints]
    for c in checkpoints:
        _sync_checkpoint(c)

    if len(checkpoints) > 2 and not args.round_robin:
        sys.exit("Pass --round-robin to compare more than 2 checkpoints.")

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

    # Build pairs AFTER dedup so indices are valid against the final checkpoints/models lists.
    pairs = list(combinations(range(len(checkpoints)), 2)) if args.round_robin else [(0, 1)]

    opening_book = _load_opening_book()
    db_path = Path(args.db)
    n_pairs = len(pairs)
    total_games = n_pairs * args.games
    games_done = 0
    t_start = time.time()
    viz = None
    if getattr(args, "visualize", False):
        from eval.eval_viz import EvalBoardViewer

        viz = EvalBoardViewer()

    try:
        if n_pairs > 1:
            _print_rr_banner(len(names), n_pairs, total_games, viz=viz)

        for pair_idx, (i, j) in enumerate(pairs):
            a, b = models[i], models[j]
            na, nb = names[i], names[j]
            ra, rb = str(checkpoints[i]), str(checkpoints[j])

            _print_match_header(
                pair_idx, n_pairs, na, nb, args.games, games_done, total_games, t_start,
                viz=viz,
            )

            w = l = d = 0
            t_match_start = time.time()

            for game_idx in range(args.games):
                opening_moves = rng.choice(opening_book)
                a_is_white = game_idx % 2 == 0
                white, black = (a, b) if a_is_white else (b, a)
                wname, bname = (na, nb) if a_is_white else (nb, na)
                wrel, brel = (ra, rb) if a_is_white else (rb, ra)
                witer = (a if a_is_white else b).iter_num
                biter = (b if a_is_white else a).iter_num

                if viz is not None:
                    viz.set_players(wname, bname)
                t_game = time.time()
                result, termination = play_game(
                    white,
                    black,
                    opening_moves=opening_moves,
                    on_ply=viz.ply if viz is not None else None,
                )
                game_secs = time.time() - t_game

                score_a = _score_from_result(result, a_is_white)
                if score_a == 1.0:
                    w += 1
                elif score_a == 0.0:
                    l += 1
                else:
                    d += 1

                games_done += 1
                _print_game_line(
                    game_idx, args.games, na, nb, result, termination,
                    w, l, d, score_a, a_is_white, game_secs,
                    viz=viz,
                )

                insert_game(
                    white=wname, black=bname, result=result,
                    termination=termination,
                    white_checkpoint=wrel, black_checkpoint=brel,
                    white_iter=witer, black_iter=biter,
                    opening=" ".join(opening_moves),
                    temperature=args.temperature, top_k=args.top_k,
                    conditioning=args.conditioning,
                    db_path=db_path,
                )

            _print_match_footer(na, nb, w, l, d, time.time() - t_match_start, viz=viz)
            if n_pairs > 1:
                _print_standings(pair_idx + 1, n_pairs, t_start, db_path, viz=viz)

        # Final summary for single head2head (no standings shown mid-run)
        if n_pairs == 1:
            elapsed = time.time() - t_start
            print(_dim(f"\n  Done in {_dur(elapsed)}"))
    finally:
        if viz is not None:
            viz.close()


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
        visualize=getattr(args, "visualize", False),
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

    total_games_db = len(games)
    h2h_note = f"  ·  {len(h2h_games)} head-to-head" if h2h_games else ""
    header_inner = f"  Elo Leaderboard  ·  {total_games_db} games in DB{h2h_note}  "
    w = max(58, len(header_inner))
    print()
    print(_cyan(_bold("  ╔" + "═" * w + "╗")))
    print(_cyan(_bold(f"  ║{header_inner:<{w}}║")))
    print(_cyan(_bold("  ╚" + "═" * w + "╝")))
    print()

    col_note = _dim("  (Elo anchored on Stockfish; H2H = head-to-head-only rank)")
    print(col_note)
    n = len(display)
    rk_w = max(3, len(str(n)) + 1)  # "10." and header "Rk"
    h2h_w = max(3, len(str(n)))
    games_w = max(5, *(len(str(r.games)) for r in display))  # header "Games"
    wld_w = max(5, *(len(f"{r.wins}-{r.losses}-{r.draws}") for r in display))
    plain_hdr = (
        f"  {_plain_col('Rk', rk_w, '>')}  {_plain_col('H2H', h2h_w, '>')}  "
        f"{_plain_col('Player', 28)}  {_plain_col('Elo', 5, '>')}  {_plain_col('±', 4, '>')}  "
        f"{_plain_col('Games', games_w, '>')}  {_plain_col('W-L-D', wld_w)}"
    )
    rule_w = max(0, len(plain_hdr) - 2)
    print(_dim("  " + "─" * rule_w))
    print(_dim(plain_hdr))
    print(_dim("  " + "─" * rule_w))

    for rank, r in enumerate(display, 1):
        se = f"{r.stderr:.0f}" if not math.isnan(r.stderr) else "—"
        wld = _plain_col(f"{r.wins}-{r.losses}-{r.draws}", wld_w)
        h2h_rank = h2h_rank_by_name.get(r.name)
        h2h_val = str(h2h_rank) if h2h_rank is not None else "—"
        elo_str = _plain_col(f"{r.elo:5.0f}", 5, ">")
        name_s = _plain_col(r.name, 28)
        rk_s = _plain_col(f"{rank}.", rk_w, ">")
        h2h_s = _plain_col(h2h_val, h2h_w, ">")
        se_s = _plain_col(se, 4, ">")
        g_s = _plain_col(str(r.games), games_w, ">")

        if rank == 1:
            row = (
                f"  {rk_s}  {_dim(h2h_s)}  {_bold(_green(name_s))}  "
                f"{_bold(_green(elo_str))}  {_dim(se_s)}  {g_s}  {_dim(wld)}"
            )
        elif rank <= 3:
            row = (
                f"  {rk_s}  {_dim(h2h_s)}  {_bold(name_s)}  "
                f"{_bold(elo_str)}  {_dim(se_s)}  {g_s}  {_dim(wld)}"
            )
        else:
            row = f"  {rk_s}  {_dim(h2h_s)}  {name_s}  {elo_str}  {_dim(se_s)}  {g_s}  {_dim(wld)}"
        print(row)

    print(_dim("  " + "─" * rule_w))
    print()


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

    title_inner = f"  Leaderboard Candidates  ·  top {len(top)} of {len(patzers)} Patzer players  "
    _print_patzer_candidate_table(title_inner, top)

    if args.no_prompt:
        players = [r.name for r in top]
    else:
        print(f"\n  {_bold('Select models by rank')} to run a round-robin tournament.")
        print(_dim("  Examples: '1 2 3', '1-5', '1,3,7-10' — Enter = default top 10"))

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
        print(_yellow(f"  ⚠  no checkpoint path for {len(missing)} player(s): {', '.join(missing[:8])}"))
        if len(missing) > 8:
            print(_dim(f"     (and {len(missing) - 8} more)"))

    if len(checkpoints) < 2:
        print(_red("  ✗  need at least 2 resolvable checkpoints to run."))
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
        visualize=getattr(args, "visualize", False),
    )
    cmd_head2head(h2h_args)


def cmd_gauntlet(args: argparse.Namespace) -> None:
    """
    One challenger plays each selected leaderboard opponent in sequence; opponents never play each other.
    """
    challenger_path = _resolve_checkpoint(args.challenger)
    _sync_checkpoint(challenger_path)

    ch = Patzer(
        challenger_path,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioning=args.conditioning,
    )
    challenger_name = player_name(challenger_path, ch.iter_num)
    del ch

    games = query_games()
    if not games:
        print("No games in database.")
        return

    ratings_all = compute_ratings(games)
    ratings_all = [r for r in ratings_all if r.games >= args.min_games]
    patzers = [r for r in ratings_all if r.name.startswith("patzer_v")]
    if not patzers:
        print(f"No Patzer players with ≥{args.min_games} games.")
        return

    others = [r for r in patzers if r.name != challenger_name][:20]
    if not others:
        print("[gauntlet] No opponents available (leaderboard has only the challenger).")
        return

    ckpt_map = _checkpoint_map_from_games(games)

    print(f"\n  {_bold('Challenger:')}  {_bold(challenger_name)}  {_dim(str(challenger_path))}")
    title_inner = f"  Gauntlet opponents  ·  top {len(others)} Patzers (excl. challenger)  "
    _print_patzer_candidate_table(title_inner, others)

    if args.no_prompt:
        ranks = list(range(1, min(10, len(others)) + 1))
    else:
        print(f"\n  {_bold('Select opponent ranks')} for the gauntlet.")
        print(_dim("  Examples: '1 2 3', '1-5', '1,3,7-10' — Enter = ranks 1–10"))
        try:
            raw = input("Opponent ranks (default: 1-10): ")
        except EOFError:
            return
        ranks = _parse_rank_list(raw, len(others))
        if not ranks:
            ranks = list(range(1, min(10, len(others)) + 1))

    opponent_names = [others[i - 1].name for i in ranks]
    if args.limit is not None:
        opponent_names = opponent_names[: max(0, int(args.limit))]

    if not opponent_names:
        print("[gauntlet] No opponents selected; exiting.")
        return

    n_match = len(opponent_names)
    total_games = n_match * args.games
    inner = f"  Gauntlet  ·  {challenger_name}  vs  {n_match} opponent(s)  ·  {total_games} games  "
    w = max(58, len(inner))
    print()
    print(_cyan(_bold("  ╔" + "═" * w + "╗")))
    print(_cyan(_bold(f"  ║{inner:<{w}}║")))
    print(_cyan(_bold("  ╚" + "═" * w + "╝")))

    for idx, opp_name in enumerate(opponent_names):
        opp_ckpt_s = ckpt_map.get(opp_name)
        if not opp_ckpt_s:
            print(_yellow(f"\n  ⚠  no checkpoint in DB for {opp_name}, skipping."))
            continue
        opp_path = Path(opp_ckpt_s)
        if opp_path.resolve() == challenger_path.resolve():
            print(_dim(f"\n  (skip {opp_name}: same file as challenger)"))
            continue
        _sync_checkpoint(opp_path)

        seed = None if args.seed is None else args.seed + idx * 1_000_003
        h2h_args = argparse.Namespace(
            checkpoints=[str(challenger_path), str(opp_path)],
            games=args.games,
            round_robin=False,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            conditioning=args.conditioning,
            seed=seed,
            db=args.db,
            visualize=getattr(args, "visualize", False),
        )
        cmd_head2head(h2h_args)


def cmd_history(args: argparse.Namespace) -> None:
    games = query_games(player=args.player)
    if not games:
        print(f"No games found matching '{args.player}'.")
        return

    print(f"\n{'Date':<26} {'White':<22} {'Black':<22} {'Result':<8} {'End':<22} Opening")
    print("-" * 115)
    for g in games:
        date = g["timestamp"][:19].replace("T", " ")
        opening = g["opening"] or ""
        if len(opening) > 28:
            opening = opening[:25] + "..."
        termination = g.get("termination") or ""
        print(
            f"{date:<26} {g['white']:<22} {g['black']:<22} {g['result']:<8} {termination:<22} {opening}"
        )


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

    # Only look at games where one side is a patzer model and the other is stockfish.
    # Use startswith to avoid patzer_v2 matching patzer_v20, patzer_v22, etc.
    sf_games: dict[int, list[dict]] = {}  # iter_num -> list of games
    for g in games:
        w, b = g["white"], g["black"]
        if w.startswith(version) and b.startswith("stockfish:"):
            it = g.get("white_iter")
        elif b.startswith(version) and w.startswith("stockfish:"):
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
        patzer_ratings = [r for r in ratings if r.name.startswith(version) and not r.name.startswith("stockfish:")]
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
    p_sf.add_argument(
        "checkpoint_args",
        nargs="*",
        help="One or more checkpoints (path / prefix / shorthand). Runs eval on each in order.",
    )
    p_sf.add_argument(
        "--checkpoint",
        dest="checkpoint_flag",
        default=None,
        help="Extra checkpoint (same forms as positional); combine with positional args",
    )
    p_sf.add_argument("--games", type=int, default=50, help="Max games to play (default 50)")
    p_sf.add_argument("--stop-sigma", type=float, default=50.0, dest="stop_sigma",
                       help="Stop when posterior sigma ≤ this (default 50)")
    p_sf.add_argument("--prior-elo", type=float, default=1500.0, dest="prior_elo")
    p_sf.add_argument("--prior-sigma", type=float, default=400.0, dest="prior_sigma")
    p_sf.add_argument("--elo-step", type=int, default=10, dest="elo_step")
    p_sf.add_argument("--stockfish", default=DEFAULT_STOCKFISH, help="Stockfish binary path")
    p_sf.add_argument("--device", default="cpu")
    p_sf.add_argument("--temperature", type=float, default=0.01)
    p_sf.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_sf.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_sf.add_argument("--seed", type=int, default=None,
                       help="RNG seed for opening selection (default: random)")
    p_sf.add_argument("--db", default=str(DB_PATH), help="Database path")
    p_sf.add_argument(
        "--visualize",
        action="store_true",
        help="Open a pygame board window; one refresh per half-move (close to abort)",
    )

    # --- head2head ---
    p_h2h = sub.add_parser("head2head", help="Model vs model")
    p_h2h.add_argument("checkpoints", nargs="+", help="Two or more checkpoint paths")
    p_h2h.add_argument("--games", type=int, default=20, help="Games per pair (default 20)")
    p_h2h.add_argument("--round-robin", action="store_true", dest="round_robin",
                        help="All-pairs if >2 checkpoints")
    p_h2h.add_argument("--device", default="cpu")
    p_h2h.add_argument("--temperature", type=float, default=0.01)
    p_h2h.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_h2h.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_h2h.add_argument("--seed", type=int, default=None,
                        help="RNG seed for opening selection (default: random)")
    p_h2h.add_argument("--db", default=str(DB_PATH), help="Database path")
    p_h2h.add_argument(
        "--visualize",
        action="store_true",
        help="Open a pygame board window; one refresh per half-move (close to abort)",
    )

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
    p_rr.add_argument("--temperature", type=float, default=0.01)
    p_rr.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_rr.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_rr.add_argument("--seed", type=int, default=None,
                       help="RNG seed for opening selection (default: random)")
    p_rr.add_argument("--db", default=str(DB_PATH), help="Database path")
    p_rr.add_argument(
        "--visualize",
        action="store_true",
        help="Open a pygame board window; one refresh per half-move (close to abort)",
    )

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
    p_rrlb.add_argument("--temperature", type=float, default=0.01)
    p_rrlb.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_rrlb.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_rrlb.add_argument("--seed", type=int, default=None,
                         help="RNG seed for opening selection (default: random)")
    p_rrlb.add_argument("--db", default=str(DB_PATH), help="Database path")
    p_rrlb.add_argument("--no-prompt", action="store_true", help="Use the whole displayed top 20 (Patzer only) without prompting")
    p_rrlb.add_argument(
        "--visualize",
        action="store_true",
        help="Open a pygame board window; one refresh per half-move (close to abort)",
    )

    # --- gauntlet ---
    p_gau = sub.add_parser(
        "gauntlet",
        help="Challenger vs each selected leaderboard opponent in turn (no games among opponents)",
    )
    p_gau.add_argument(
        "challenger",
        help="Challenger checkpoint (path, patzer_vN@K, patzer_vN@best, …)",
    )
    p_gau.add_argument(
        "--games",
        type=int,
        default=50,
        help="Games per challenger–opponent pair (default: 50)",
    )
    p_gau.add_argument(
        "--min-games",
        type=int,
        default=5,
        dest="min_games",
        help="Only list Patzer players with at least this many games in the DB (default: 5)",
    )
    p_gau.add_argument(
        "--limit",
        type=int,
        default=None,
        help="After rank selection, keep only the first N opponents (optional)",
    )
    p_gau.add_argument("--device", default="cpu")
    p_gau.add_argument("--temperature", type=float, default=0.01)
    p_gau.add_argument("--top-k", type=int, default=None, dest="top_k")
    p_gau.add_argument("--conditioning", default="match_color", choices=CONDITIONING_OPTIONS)
    p_gau.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for openings; each opponent match adds an offset (default: random)",
    )
    p_gau.add_argument("--db", default=str(DB_PATH), help="Database path")
    p_gau.add_argument(
        "--no-prompt",
        action="store_true",
        help="Use default opponent ranks 1–10 (no interactive selection)",
    )
    p_gau.add_argument(
        "--visualize",
        action="store_true",
        help="Open a pygame board window; one refresh per half-move (close to abort)",
    )

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

    # Merge positional checkpoints + optional --checkpoint into args.checkpoints.
    if getattr(args, "cmd", None) == "stockfish":
        merged: list[str] = []
        merged.extend(getattr(args, "checkpoint_args", None) or [])
        if getattr(args, "checkpoint_flag", None):
            merged.append(args.checkpoint_flag)
        if not merged:
            parser.error(
                "stockfish: pass at least one checkpoint "
                "(positional and/or --checkpoint)"
            )
        args.checkpoints = merged

    # Allow --db override to propagate to db module
    if hasattr(args, "db") and args.db != str(DB_PATH):
        import eval.db as db_mod
        db_mod.DB_PATH = Path(args.db)

    dispatch = {
        "stockfish": cmd_stockfish,
        "head2head": cmd_head2head,
        "rr-prefix": cmd_rr_prefix,
        "rr-leaderboard": cmd_rr_leaderboard,
        "gauntlet": cmd_gauntlet,
        "leaderboard": cmd_leaderboard,
        "history": cmd_history,
        "progress": cmd_progress,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
