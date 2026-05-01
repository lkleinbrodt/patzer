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


def main() -> None:
    args = parse_args()
    use_estimate = args.estimate or not args.exact
    paths = sorted(resolve_files(list(args.input)), key=lambda q: q.name)

    if not paths:
        print("No input files matched. Check --input globs.", file=sys.stderr)
        sys.exit(1)

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
