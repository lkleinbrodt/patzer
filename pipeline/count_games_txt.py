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

  # How many games have both players >= 2000? (reads every line; ignores --estimate/--exact)
  python pipeline/count_games_txt.py --min-elo 2000
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
        "--min-elo",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Full scan: count lines where both ratings are >= N (matches prepare.py / "
            "filter_games.py). Reads every line; cannot be combined with byte-based "
            "--estimate."
        ),
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


def scan_min_elo(paths: list[Path], min_elo: int) -> None:
    """Stream all files; count games passing the same ELO rule as prepare.py --min-elo."""
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pipeline.prepare import parse_game_line

    total_lines = 0
    parseable = 0
    pass_elo = 0
    rows: list[tuple[str, int, int, int, float]] = []
    t0 = time.time()

    for p in sorted(paths, key=lambda q: q.name):
        fl_total = 0
        fl_parse = 0
        fl_pass = 0
        try:
            mb = p.stat().st_size / (1024 * 1024)
        except OSError:
            mb = 0.0
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                total_lines += 1
                fl_total += 1
                if parse_game_line(line, min_elo=None) is None:
                    continue
                parseable += 1
                fl_parse += 1
                if parse_game_line(line, min_elo=min_elo) is not None:
                    pass_elo += 1
                    fl_pass += 1
        rows.append((p.name, fl_total, fl_parse, fl_pass, mb))

    dt = time.time() - t0
    rejected = parseable - pass_elo

    print(f"Min ELO: both players >= {min_elo} (same rule as pipeline/prepare.py --min-elo)")
    print()
    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"  {'file':{w}s}  {'lines':>12}  {'parseable':>12}  {'pass ELO':>12}  MiB")
    for name, lt, pr, ps, mb in rows:
        print(f"  {name:{w}s}  {lt:>12,}  {pr:>12,}  {ps:>12,}  {mb:>5.1f}")
    print()
    print(f"Total lines in files:     {total_lines:,}")
    print(f"Parseable game lines:     {parseable:,}  (valid result + integer ELOs + moves)")
    print(f"Games passing min-elo:    {pass_elo:,}  ({100.0 * pass_elo / max(total_lines, 1):.2f}% of all lines)")
    print(f"Parseable but below ELO:  {rejected:,}  ({100.0 * rejected / max(parseable, 1):.2f}% of parseable)")
    print(f"Read time: {dt:.1f}s ({total_lines / max(dt, 1e-6) / 1e6:.2f}M lines/s)")


def main() -> None:
    args = parse_args()
    paths = sorted(resolve_files(list(args.input)), key=lambda q: q.name)

    if not paths:
        print("No input files matched. Check --input globs.", file=sys.stderr)
        sys.exit(1)

    if args.min_elo is not None:
        if args.estimate or args.exact:
            print(
                "Note: with --min-elo, line counts use a full parse scan "
                "(--estimate / --exact ignored).",
                file=sys.stderr,
            )
        scan_min_elo(paths, args.min_elo)
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
