#!/usr/bin/env python3
"""
Count lines (= games stored, one game per line) in `games_*.txt` dumps.

Quick mode (default): estimate from file sizes and a sampled average bytes/line
(same idea as pipeline/prepare.py).

Exact mode: scan every newline (full disk read).

Examples (from repo root):
  python pipeline/count_games_txt.py
  python pipeline/count_games_txt.py --estimate
  python pipeline/count_games_txt.py --exact
  python pipeline/count_games_txt.py --input data/lichess_games/games_2014-*.txt

  # Games at each ELO floor (both players >= cutoff); full file read
  python pipeline/count_games_txt.py --elo-distribution
  python pipeline/count_games_txt.py --elo-distribution --elo-low 1600 --elo-high 2800 --elo-step 50
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input",
        type=str,
        nargs="+",
        default=["data/lichess_games/games_*.txt"],
        help="Glob(s); default: %(default)s",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--estimate",
        dest="estimate",
        action="store_true",
        help="Approximate counts from total bytes ÷ sampled bytes/line (default).",
    )
    mode.add_argument(
        "--exact",
        dest="exact",
        action="store_true",
        help="Count every newline via a full-file read.",
    )
    p.set_defaults(estimate=False, exact=False)
    p.add_argument(
        "--elo-distribution",
        action="store_true",
        help=(
            "Full scan: for each cutoff C, count games where both players have ELO >= C "
            "(same rule as prepare.py --min-elo). Prints a table; "
            "default cutoffs: elo-low..elo-high inclusive every elo-step."
        ),
    )
    p.add_argument(
        "--elo-low",
        type=int,
        default=1800,
        metavar="N",
        help="Lowest cutoff in --elo-distribution (default: %(default)s)",
    )
    p.add_argument(
        "--elo-high",
        type=int,
        default=2500,
        metavar="N",
        help="Highest cutoff in --elo-distribution (default: %(default)s)",
    )
    p.add_argument(
        "--elo-step",
        type=int,
        default=100,
        metavar="N",
        help="Spacing between cutoffs (default: %(default)s)",
    )
    return p.parse_args()


def resolve_files(patterns: list[str]) -> list[Path]:
    seen: dict[str, Path] = {}
    for pattern in patterns:
        for raw in sorted(glob.glob(pattern, recursive=False)):
            p = Path(raw)
            if p.is_file():
                seen[str(p.resolve())] = p
            else:
                print(f"warning: skipped non-file match {raw!r}", file=sys.stderr)
    return list(seen.values())


def _sample_avg_bytes_per_line(path: Path, max_bytes: int = 393_216) -> float:
    if not path.is_file():
        return 120.0
    try:
        with open(path, "rb") as f:
            blob = f.read(max_bytes)
    except OSError as e:
        print(f"{path}: read error ({e})", file=sys.stderr)
        return 120.0
    if not blob:
        return 120.0
    n_nl = blob.count(b"\n")
    if n_nl < 8:
        return max(72.0, min(384.0, len(blob)))
    avg = len(blob) / max(1, n_nl)
    return max(48.0, min(512.0, float(avg)))


def estimate_totals(paths: list[Path]) -> tuple[list[tuple[str, int, float]], float, float]:
    """
    Per-(name, estimated_lines, size_mb), summed estimated lines, sampled avg bytes/line.
    """
    if not paths:
        return [], 0.0, 120.0
    rows: list[tuple[str, int, float]] = []
    total_bytes = 0
    avg_bpl = _sample_avg_bytes_per_line(paths[0])
    for p in paths:
        try:
            sz = p.stat().st_size
        except OSError as e:
            print(f"{p}: stat error ({e})", file=sys.stderr)
            continue
        total_bytes += sz
        mb = sz / (1024 * 1024)
        est = max(1, int(sz / avg_bpl)) if sz > 0 else 0
        rows.append((p.name, est, mb))
    est_lines = max(1, int(total_bytes / avg_bpl)) if total_bytes > 0 else 0
    return rows, float(est_lines), avg_bpl


def count_newlines(path: Path, chunk: int = 8 * 1024 * 1024) -> int:
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return n
            n += b.count(b"\n")


def _elo_cutoffs(elo_low: int, elo_high: int, elo_step: int) -> list[int]:
    if elo_step <= 0:
        raise ValueError("elo_step must be positive")
    if elo_low > elo_high:
        raise ValueError("elo_low must be <= elo_high")
    return list(range(elo_low, elo_high + 1, elo_step))


def scan_elo_distribution(
    paths: list[Path],
    elo_low: int,
    elo_high: int,
    elo_step: int,
) -> None:
    """
    For each cutoff C, count games where min(white_elo, black_elo) >= C
    (equivalent to both >= C).
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pipeline.prepare import parse_game_elos

    cutoffs = _elo_cutoffs(elo_low, elo_high, elo_step)
    if not cutoffs:
        print("No cutoffs in range; check --elo-low, --elo-high, --elo-step", file=sys.stderr)
        sys.exit(2)

    counts = {c: 0 for c in cutoffs}
    total_lines = 0
    parseable = 0
    t0 = time.time()

    for p in sorted(paths, key=lambda q: q.name):
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                elos = parse_game_elos(line)
                if elos is None:
                    continue
                parseable += 1
                m = min(elos[0], elos[1])
                for c in cutoffs:
                    if m >= c:
                        counts[c] += 1

    dt = time.time() - t0

    print(
        "ELO cutoffs: games where both White and Black are >= cutoff "
        "(same rule as pipeline/prepare.py --min-elo)."
    )
    print(f"Cutoffs: {cutoffs[0]} .. {cutoffs[-1]} every {elo_step} "
          f"({len(cutoffs)} levels)")
    print()
    w = max(len(str(c)) for c in cutoffs)
    print(f"  {'cutoff':>{w}s}  {'n_games':>14}  {'% of lines':>12}  {'% parseable':>14}")
    for c in cutoffs:
        n = counts[c]
        pct_lines = 100.0 * n / max(total_lines, 1)
        pct_ok = 100.0 * n / max(parseable, 1)
        print(f"  {c:{w}d}  {n:>14,}  {pct_lines:>11.2f}%  {pct_ok:>13.2f}%")
    print()
    print(f"Total lines in files:  {total_lines:,}")
    print(f"Parseable game lines:  {parseable:,}")
    print(f"Read time: {dt:.1f}s ({total_lines / max(dt, 1e-6) / 1e6:.2f}M lines/s)")


def main() -> None:
    args = parse_args()
    paths = sorted(resolve_files(list(args.input)), key=lambda q: q.name)

    if not paths:
        print("No input files matched. Check --input globs.", file=sys.stderr)
        sys.exit(1)

    if args.elo_distribution:
        if args.estimate or args.exact:
            print(
                "Note: --elo-distribution performs a full parse scan "
                "(--estimate / --exact ignored).",
                file=sys.stderr,
            )
        try:
            scan_elo_distribution(paths, args.elo_low, args.elo_high, args.elo_step)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        return

    use_estimate = args.estimate or not args.exact

    if use_estimate:
        rows, est_total, avg_bpl = estimate_totals(paths)
        print(f"Estimated games (≈lines): {int(est_total):,}")
        print(f"Sampled avg line length:  {avg_bpl:.1f} bytes")
        print(f"Files: {len(rows)}")
        print()
        w = max(len(name) for name, _, __ in rows) if rows else 10
        for name, est, mb in rows:
            print(f"  {name:{w}s}  ~{est:>12,}  ({mb:.1f} MiB)")
        print()
        print("Note: lines include malformed rows; prepare.py skips those when tokenizing.")
        return

    t0 = time.time()
    total = 0
    w = max(len(p.name) for p in paths)
    print("Exact newline counts:")
    print()
    for p in sorted(paths, key=lambda q: q.name):
        n = count_newlines(p)
        total += n
        try:
            mb = p.stat().st_size / (1024 * 1024)
        except OSError:
            mb = 0.0
        print(f"  {p.name:{w}s}  {n:>12,}  ({mb:.1f} MiB)")
    dt = time.time() - t0
    print()
    print(f"Total lines (= games rows): {total:,}")
    print(f"Read time: {dt:.1f}s ({total/max(dt, 1e-6)/1e6:.2f}M lines/s)")


if __name__ == "__main__":
    main()
