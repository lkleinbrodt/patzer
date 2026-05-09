#!/usr/bin/env python3
"""
Analyze opening diversity + basic outcome/length stats across ELO cutoffs for
Patzer's line-based Lichess dump format:

  <RESULT> <WHITE_ELO> <BLACK_ELO> <UCI_MOVE_1> <UCI_MOVE_2> ...

Example line:
  1-0 1858 1857 e2e4 e7e5 g1f3 ...

This script is designed to stream through multi-million-game `.txt.gz` files
without loading them into memory.

Outputs (per cutoff):
  - game counts and % of total
  - result distribution (W/L/D)
  - average game length (plies)
  - Elo summaries (mean min-elo, mean avg-elo)
  - opening signature diversity over first N plies:
      * top-K opening signatures
      * normalized entropy over the top-M signatures (proxy for diversity)

Notes:
  - "cutoff" uses the prepare-style rule: min(WhiteElo, BlackElo) >= cutoff.
  - "opening signature" is the first `--opening-plies` UCI moves joined by spaces.
    It’s an ECO-free proxy that still detects repertoire collapse.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path


def _open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _iter_cutoffs(lo: int, hi: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("--elo-step must be > 0")
    if hi < lo:
        raise ValueError("--elo-high must be >= --elo-low")
    return list(range(lo, hi + 1, step))


def _safe_float(x: float, default: float = 0.0) -> float:
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return x


def _entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return ent


@dataclass
class CutoffStats:
    cutoff: int
    n_games: int = 0
    n_white_win: int = 0
    n_black_win: int = 0
    n_draw: int = 0
    sum_plies: int = 0
    sum_min_elo: int = 0
    sum_avg_elo: int = 0
    openings: Counter[str] | None = None

    def __post_init__(self):
        if self.openings is None:
            self.openings = Counter()

    def add_game(self, result: str, we: int, be: int, moves: list[str], opening_plies: int):
        self.n_games += 1
        if result == "1-0":
            self.n_white_win += 1
        elif result == "0-1":
            self.n_black_win += 1
        else:
            self.n_draw += 1

        plies = len(moves)
        self.sum_plies += plies

        mn = we if we < be else be
        av = (we + be) // 2
        self.sum_min_elo += mn
        self.sum_avg_elo += av

        if opening_plies > 0:
            sig_moves = moves[:opening_plies]
            sig = " ".join(sig_moves)
            self.openings[sig] += 1

    def finalize(self, total_seen: int, top_k: int, entropy_top_m: int) -> dict:
        n = self.n_games
        if n <= 0:
            return {
                "cutoff": self.cutoff,
                "n_games": 0,
                "pct_of_total": 0.0,
            }

        pct = n / total_seen if total_seen > 0 else 0.0
        avg_plies = self.sum_plies / n
        avg_min_elo = self.sum_min_elo / n
        avg_avg_elo = self.sum_avg_elo / n

        top = self.openings.most_common(top_k) if self.openings else []
        top_for_entropy = self.openings.most_common(entropy_top_m) if self.openings else []
        ent_counts = [c for _, c in top_for_entropy]
        ent = _entropy_from_counts(ent_counts)
        max_ent = math.log(len(ent_counts)) if len(ent_counts) > 1 else 0.0
        ent_norm = (ent / max_ent) if max_ent > 0 else 0.0

        return {
            "cutoff": self.cutoff,
            "n_games": n,
            "pct_of_total": _safe_float(pct),
            "results": {
                "white_win": self.n_white_win,
                "black_win": self.n_black_win,
                "draw": self.n_draw,
                "white_win_pct": _safe_float(self.n_white_win / n),
                "black_win_pct": _safe_float(self.n_black_win / n),
                "draw_pct": _safe_float(self.n_draw / n),
            },
            "avg_plies": _safe_float(avg_plies),
            "avg_min_elo": _safe_float(avg_min_elo),
            "avg_avg_elo": _safe_float(avg_avg_elo),
            "opening": {
                "unique_signatures": len(self.openings) if self.openings else 0,
                "top_k": [{"sig": s, "count": c} for s, c in top],
                "entropy_top_m": entropy_top_m,
                "entropy_nats": _safe_float(ent),
                "entropy_norm_0_1": _safe_float(ent_norm),
            },
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--input",
        type=str,
        default="data/lichess_games/games_2021-08.txt.gz",
        help="Path to games_*.txt or .txt.gz",
    )
    p.add_argument("--elo-low", type=int, default=1800)
    p.add_argument("--elo-high", type=int, default=2500)
    p.add_argument("--elo-step", type=int, default=50)
    p.add_argument(
        "--opening-plies",
        type=int,
        default=8,
        help="How many plies to include in the opening signature (UCI moves).",
    )
    p.add_argument("--top-k", type=int, default=20, help="Top openings to list per cutoff.")
    p.add_argument(
        "--entropy-top-m",
        type=int,
        default=2000,
        help="Compute opening entropy over the top-M signatures (proxy diversity).",
    )
    p.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Stop after reading N games (0 = no limit).",
    )
    p.add_argument(
        "--sample-rate",
        type=float,
        default=1.0,
        help="Randomly keep each game with probability p (use <1.0 for fast approximate stats).",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output JSON path (if empty, print summary only).",
    )
    return p.parse_args()


def _parse_line(line: str):
    parts = line.split()
    if len(parts) < 4:
        return None
    result = parts[0]
    try:
        we = int(parts[1])
        be = int(parts[2])
    except ValueError:
        return None
    moves = parts[3:]
    return result, we, be, moves


def main() -> int:
    args = parse_args()
    path = Path(args.input)
    if not path.is_file():
        print(f"error: input not found: {path}", file=sys.stderr)
        return 2

    cutoffs = _iter_cutoffs(args.elo_low, args.elo_high, args.elo_step)
    stats_by_cutoff = {c: CutoffStats(cutoff=c) for c in cutoffs}

    rng = random.Random(args.seed)
    sample_rate = float(args.sample_rate)
    if not (0.0 < sample_rate <= 1.0):
        print("error: --sample-rate must be in (0, 1].", file=sys.stderr)
        return 2

    total_seen = 0
    total_parsed = 0
    total_kept = 0
    bad_lines = 0

    with _open_text(path) as f:
        for line in f:
            total_parsed += 1
            if args.max_games and total_parsed > args.max_games:
                break

            if sample_rate < 1.0 and rng.random() > sample_rate:
                continue

            parsed = _parse_line(line)
            if parsed is None:
                bad_lines += 1
                continue
            result, we, be, moves = parsed
            total_kept += 1

            mn = we if we < be else be
            total_seen += 1

            # Update all cutoffs <= mn. Cutoff list is sorted ascending.
            for c in cutoffs:
                if mn < c:
                    break
                stats_by_cutoff[c].add_game(
                    result=result,
                    we=we,
                    be=be,
                    moves=moves,
                    opening_plies=args.opening_plies,
                )

            if total_kept % 500000 == 0:
                print(
                    f"[progress] kept={total_kept:,} parsed={total_parsed:,} bad={bad_lines:,}",
                    file=sys.stderr,
                )

    results = {
        "input": str(path),
        "sample_rate": sample_rate,
        "max_games": args.max_games,
        "opening_plies": args.opening_plies,
        "total_parsed_lines": total_parsed,
        "total_kept_games": total_kept,
        "bad_lines": bad_lines,
        "cutoffs": [stats_by_cutoff[c].finalize(total_seen=total_seen, top_k=args.top_k, entropy_top_m=args.entropy_top_m) for c in cutoffs],
    }

    # Print a compact console summary.
    print(f"input: {results['input']}")
    print(f"kept_games: {total_kept:,} (sample_rate={sample_rate})  bad_lines: {bad_lines:,}")
    print("")
    print("cutoff   games      pct   draw%   avg_plies   avg_min_elo   open_entropy_norm")
    for row in results["cutoffs"]:
        if row.get("n_games", 0) <= 0:
            print(f"{row['cutoff']:>6}  {0:>8}  {0.0:>6.3f}  {0.0:>6.3f}  {0.0:>9.2f}  {0.0:>11.1f}  {0.0:>16.3f}")
            continue
        draw_pct = row["results"]["draw_pct"]
        print(
            f"{row['cutoff']:>6}  {row['n_games']:>8,}  {row['pct_of_total']:>6.3f}  {draw_pct:>6.3f}  "
            f"{row['avg_plies']:>9.2f}  {row['avg_min_elo']:>11.1f}  {row['opening']['entropy_norm_0_1']:>16.3f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

