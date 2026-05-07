#!/usr/bin/env python3
"""
Count lines (= games stored, one game per line) in `games_*.txt` dumps.

Quick mode (default): for each `games_YYYY-MM.txt` (or `.txt.gz`), if a sibling
`games_YYYY-MM.stats.json` exists (written by the scrape/parse pipeline), use its
`kept_games` field as the game count (exact for that month). Otherwise fall back
to estimating from file size and a sampled average bytes/line (same idea as
pipeline/prepare.py).

Exact mode: use ``kept_games`` from sibling ``*.stats.json`` when present; otherwise
scan every newline (full disk read).

``--elo-distribution`` tabulates games by average ELO cutoff ((White+Black)//2). It reads
``elo_histogram`` from each ``games_*.stats.json`` when available (fast); otherwise it
parses the text dumps. This is not the same rule as ``prepare.py --min-elo``.

``--min-elo-distribution`` tabulates games by *min player rating* cutoff
(``min(WhiteElo, BlackElo)``). This matches the ``prepare.py --min-elo`` rule and will
use ``min_elo_histogram`` from ``games_*.stats.json`` when available; otherwise it reads
each games file line-by-line.

Examples (from repo root):
  python pipeline/count_games_txt.py
  python pipeline/count_games_txt.py --estimate
  python pipeline/count_games_txt.py --exact
  python pipeline/count_games_txt.py --input data/lichess_games/games_2014-*.txt

  # Average ELO at each cutoff (see --elo-distribution); prefers *.stats.json
  python pipeline/count_games_txt.py --elo-distribution
  python pipeline/count_games_txt.py --elo-distribution --elo-low 1600 --elo-high 2800 --elo-step 50

  # Prepare-style: both players >= cutoff (min player ELO at each cutoff)
  python pipeline/count_games_txt.py --min-elo-distribution
  python pipeline/count_games_txt.py --min-elo-distribution --elo-low 1600 --elo-high 2600 --elo-step 100
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
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
        help="Count newlines (full read) unless games_*.stats.json provides kept_games.",
    )
    p.set_defaults(estimate=False, exact=False)
    p.add_argument(
        "--elo-distribution",
        action="store_true",
        help=(
            "For each cutoff C, count games whose average rating (White+Black)//2 is >= C. "
            "Uses elo_histogram in games_*.stats.json when present (fast); otherwise reads "
            "each games file. Not the same as prepare.py --min-elo (both players >= C)."
        ),
    )
    p.add_argument(
        "--min-elo-distribution",
        action="store_true",
        help=(
            "For each cutoff C, count games where both players are >= C "
            "(i.e. min(WhiteElo, BlackElo) >= C). Uses min_elo_histogram in "
            "games_*.stats.json when present (fast); otherwise reads each games file."
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


def stats_json_path_for_games_file(games_path: Path) -> Path:
    """
    Sibling `games_YYYY-MM.stats.json` for `games_YYYY-MM.txt` or `.txt.gz`
    (same naming as pipeline/parse_pgn.py).
    """
    name = games_path.name
    if name.endswith(".txt.gz"):
        base = name[:-7]
    elif name.endswith(".txt"):
        base = name[:-4]
    else:
        base = games_path.stem
    return games_path.parent / f"{base}.stats.json"


def read_kept_games_from_stats(stats_path: Path) -> int | None:
    """Return kept_games from stats JSON, or None if missing/invalid."""
    if not stats_path.is_file():
        return None
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    kg = data.get("kept_games")
    if isinstance(kg, int) and kg >= 0:
        return kg
    return None


def _load_stats_json(stats_path: Path) -> dict | None:
    if not stats_path.is_file():
        return None
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def load_elo_histogram_from_stats(stats_path: Path) -> dict[int, int] | None:
    """
    Parse ``elo_histogram`` from stats (100-point buckets of *average* ELO).
    Returns None if the file is missing or has no usable histogram.
    """
    data = _load_stats_json(stats_path)
    if not data or "elo_histogram" not in data:
        return None
    raw = data["elo_histogram"]
    if not isinstance(raw, dict):
        return None
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            bk = int(k)
            bv = int(v)
        except (TypeError, ValueError):
            continue
        if bv < 0:
            continue
        out[bk] = out.get(bk, 0) + bv
    return out


def load_histogram_from_stats(stats_path: Path, key: str) -> dict[int, int] | None:
    """
    Load a histogram dict from stats JSON.

    Expected shape: { "<bucket_lo_int>": <count_int>, ... } where bucket_lo is a multiple of 100.
    """
    data = _load_stats_json(stats_path)
    if not data or key not in data:
        return None
    raw = data[key]
    if not isinstance(raw, dict):
        return None
    out: dict[int, int] = {}
    for k, v in raw.items():
        try:
            bk = int(k)
            bv = int(v)
        except (TypeError, ValueError):
            continue
        if bv < 0:
            continue
        out[bk] = out.get(bk, 0) + bv
    return out


def _frac_avg_in_bucket_ge_cutoff(bucket_lo: int, cutoff: int) -> float:
    """
    Fraction of **integer** averages in [bucket_lo, bucket_lo + 99] that are >= cutoff.

    Used when applying a cutoff to 100-wide histogram bins (uniform-within-bin assumption
    for partially overlapped bins).
    """
    hi = bucket_lo + 99
    if cutoff > hi:
        return 0.0
    if cutoff <= bucket_lo:
        return 1.0
    return (hi - cutoff + 1) / 100.0


def _counts_from_elo_histogram(
    hist: dict[int, int],
    cutoffs: list[int],
) -> dict[int, float]:
    acc = {c: 0.0 for c in cutoffs}
    for bucket_lo, n in hist.items():
        for c in cutoffs:
            acc[c] += n * _frac_avg_in_bucket_ge_cutoff(bucket_lo, c)
    return acc


def open_text_games(path: Path):
    """Text read for ``games_*.txt`` or ``games_*.txt.gz``."""
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


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


def estimate_totals(paths: list[Path]) -> tuple[list[tuple[str, int, float, bool]], float, float]:
    """
    Per-(name, line_count, size_mb, from_stats), summed line count, sampled avg bytes/line.

    When ``games_*.stats.json`` exists beside a games file, ``kept_games`` is used
    (exact for that dump); ``from_stats`` is True. Otherwise rows use byte sampling
    and ``from_stats`` is False.
    """
    if not paths:
        return [], 0.0, 120.0
    rows: list[tuple[str, int, float, bool]] = []
    total_lines = 0
    avg_bpl = 120.0
    sampled_for_estimate = False
    for p in paths:
        try:
            sz = p.stat().st_size
        except OSError as e:
            print(f"{p}: stat error ({e})", file=sys.stderr)
            continue
        mb = sz / (1024 * 1024)
        sp = stats_json_path_for_games_file(p)
        kg = read_kept_games_from_stats(sp)
        if kg is not None:
            rows.append((p.name, kg, mb, True))
            total_lines += kg
        else:
            if not sampled_for_estimate:
                avg_bpl = _sample_avg_bytes_per_line(p)
                sampled_for_estimate = True
            est = max(1, int(sz / avg_bpl)) if sz > 0 else 0
            rows.append((p.name, est, mb, False))
            total_lines += est
    return rows, float(total_lines), avg_bpl


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


def run_elo_distribution(
    paths: list[Path],
    elo_low: int,
    elo_high: int,
    elo_step: int,
) -> None:
    """
    For each cutoff C, count games where (white + black) // 2 >= C.

    When ``games_*.stats.json`` includes ``elo_histogram``, buckets are merged and
    cutoffs applied using the same average-ELO rule (with a uniform-within-bucket
    approximation for cutoffs that fall inside a bin). Months without a usable
    histogram are read from the games text file line by line.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pipeline.prepare import parse_game_elos

    cutoffs = _elo_cutoffs(elo_low, elo_high, elo_step)
    if not cutoffs:
        print("No cutoffs in range; check --elo-low, --elo-high, --elo-step", file=sys.stderr)
        sys.exit(2)

    merged_hist: dict[int, int] = {}
    scan_paths: list[Path] = []
    stats_kept_sum = 0
    n_from_stats_files = 0

    for p in sorted(paths, key=lambda q: q.name):
        sp = stats_json_path_for_games_file(p)
        data = _load_stats_json(sp)
        hist = load_elo_histogram_from_stats(sp)
        if hist is None:
            scan_paths.append(p)
            continue
        kg = data.get("kept_games") if data else None
        if not hist:
            if isinstance(kg, int) and kg == 0:
                pass
            else:
                scan_paths.append(p)
                continue
        for b, n in hist.items():
            merged_hist[b] = merged_hist.get(b, 0) + n
        if isinstance(kg, int) and kg >= 0:
            stats_kept_sum += kg
        else:
            stats_kept_sum += sum(hist.values())
        n_from_stats_files += 1

    t0 = time.time()
    count_f = _counts_from_elo_histogram(merged_hist, cutoffs)
    total_lines = stats_kept_sum
    parseable = stats_kept_sum

    for p in scan_paths:
        with open_text_games(p) as f:
            for line in f:
                total_lines += 1
                elos = parse_game_elos(line)
                if elos is None:
                    continue
                parseable += 1
                avg_elo = (elos[0] + elos[1]) // 2
                for c in cutoffs:
                    if avg_elo >= c:
                        count_f[c] += 1.0

    dt = time.time() - t0

    print(
        "ELO cutoffs: average rating per game = (White ELO + Black ELO) // 2 "
        "(integer division; same bucket definition as pipeline/parse_pgn.py elo_histogram)."
    )
    print(
        "  From games_*.stats.json: 100-point average-ELO bins. If a cutoff falls "
        "inside a bin, the contribution is scaled assuming average ELO is uniformly "
        "distributed across that bin’s 100 integer values."
    )
    print(
        "  This is NOT prepare.py --min-elo (both players must be >= N). "
        "For that rule, use --min-elo-distribution."
    )
    if n_from_stats_files and scan_paths:
        print(
            f"  Mixed inputs: {n_from_stats_files} month(s) from stats.json, "
            f"{len(scan_paths)} file(s) read from disk."
        )
    elif n_from_stats_files and not scan_paths:
        print(
            f"  All {n_from_stats_files} input(s) used elo_histogram from stats.json only."
        )
    elif scan_paths and not n_from_stats_files:
        print(
            f"  No usable stats.json next to inputs; all {len(scan_paths)} file(s) read from disk."
        )

    print(f"Cutoffs: {cutoffs[0]} .. {cutoffs[-1]} every {elo_step} "
          f"({len(cutoffs)} levels)")
    print()
    w = max(len(str(c)) for c in cutoffs)
    print(f"  {'cutoff':>{w}s}  {'n_games':>14}  {'% of lines':>12}  {'% parseable':>14}")
    for c in cutoffs:
        n = int(round(count_f[c]))
        pct_lines = 100.0 * n / max(total_lines, 1)
        pct_ok = 100.0 * n / max(parseable, 1)
        print(f"  {c:{w}d}  {n:>14,}  {pct_lines:>11.2f}%  {pct_ok:>13.2f}%")
    print()
    print(f"Total lines in files:  {total_lines:,}")
    print(f"Parseable game lines:  {parseable:,}")
    if scan_paths:
        print(
            f"Read time: {dt:.1f}s ({total_lines / max(dt, 1e-6) / 1e6:.2f}M lines/s; "
            f"includes {len(scan_paths)} full file read(s))"
        )
    else:
        print(f"Read time: {dt:.3f}s (stats.json only)")


def run_min_elo_distribution(
    paths: list[Path],
    elo_low: int,
    elo_high: int,
    elo_step: int,
) -> None:
    """
    For each cutoff C, count games where min(white_elo, black_elo) >= C.

    When ``games_*.stats.json`` includes ``min_elo_histogram``, buckets are merged and
    cutoffs applied using a uniform-within-bucket approximation for cutoffs inside a bin.
    Months without a usable histogram are read from the games text file line by line.
    """
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from pipeline.prepare import parse_game_elos

    cutoffs = _elo_cutoffs(elo_low, elo_high, elo_step)
    if not cutoffs:
        print("No cutoffs in range; check --elo-low, --elo-high, --elo-step", file=sys.stderr)
        sys.exit(2)

    merged_hist: dict[int, int] = {}
    scan_paths: list[Path] = []
    stats_kept_sum = 0
    n_from_stats_files = 0

    for p in sorted(paths, key=lambda q: q.name):
        sp = stats_json_path_for_games_file(p)
        data = _load_stats_json(sp)
        hist = load_histogram_from_stats(sp, "min_elo_histogram")
        if hist is None:
            scan_paths.append(p)
            continue
        kg = data.get("kept_games") if data else None
        if not hist:
            if isinstance(kg, int) and kg == 0:
                pass
            else:
                scan_paths.append(p)
                continue
        for b, n in hist.items():
            merged_hist[b] = merged_hist.get(b, 0) + n
        if isinstance(kg, int) and kg >= 0:
            stats_kept_sum += kg
        else:
            stats_kept_sum += sum(hist.values())
        n_from_stats_files += 1

    t0 = time.time()
    count_f = _counts_from_elo_histogram(merged_hist, cutoffs)
    total_lines = stats_kept_sum
    parseable = stats_kept_sum

    for p in scan_paths:
        with open_text_games(p) as f:
            for line in f:
                total_lines += 1
                elos = parse_game_elos(line)
                if elos is None:
                    continue
                parseable += 1
                min_elo = min(elos[0], elos[1])
                for c in cutoffs:
                    if min_elo >= c:
                        count_f[c] += 1.0

    dt = time.time() - t0

    print(
        "ELO cutoffs: per-game min player rating = min(White ELO, Black ELO). "
        "This matches prepare.py --min-elo."
    )
    print(
        "  From games_*.stats.json: 100-point min-ELO bins. If a cutoff falls "
        "inside a bin, the contribution is scaled assuming min ELO is uniformly "
        "distributed across that bin’s 100 integer values."
    )
    if n_from_stats_files and scan_paths:
        print(
            f"  Mixed inputs: {n_from_stats_files} month(s) from stats.json, "
            f"{len(scan_paths)} file(s) read from disk."
        )
    elif n_from_stats_files and not scan_paths:
        print(f"  All {n_from_stats_files} input(s) used min_elo_histogram from stats.json only.")
    elif scan_paths and not n_from_stats_files:
        print(f"  No usable stats.json next to inputs; all {len(scan_paths)} file(s) read from disk.")

    print(f"Cutoffs: {cutoffs[0]} .. {cutoffs[-1]} every {elo_step} "
          f"({len(cutoffs)} levels)")
    print()
    w = max(len(str(c)) for c in cutoffs)
    print(f"  {'cutoff':>{w}s}  {'n_games':>14}  {'% of lines':>12}  {'% parseable':>14}")
    for c in cutoffs:
        n = int(round(count_f[c]))
        pct_lines = 100.0 * n / max(total_lines, 1)
        pct_ok = 100.0 * n / max(parseable, 1)
        print(f"  {c:{w}d}  {n:>14,}  {pct_lines:>11.2f}%  {pct_ok:>13.2f}%")
    print()
    print(f"Total lines in files:  {total_lines:,}")
    print(f"Parseable game lines:  {parseable:,}")
    if scan_paths:
        print(
            f"Read time: {dt:.1f}s ({total_lines / max(dt, 1e-6) / 1e6:.2f}M lines/s; "
            f"includes {len(scan_paths)} full file read(s))"
        )
    else:
        print(f"Read time: {dt:.3f}s (stats.json only)")


def main() -> None:
    args = parse_args()
    paths = sorted(resolve_files(list(args.input)), key=lambda q: q.name)

    if not paths:
        print("No input files matched. Check --input globs.", file=sys.stderr)
        sys.exit(1)

    if args.elo_distribution or args.min_elo_distribution:
        if args.estimate or args.exact:
            print(
                "Note: --elo-distribution / --min-elo-distribution ignore --estimate / --exact "
                "(uses stats elo_histogram when present, else reads games files).",
                file=sys.stderr,
            )
        try:
            if args.min_elo_distribution:
                run_min_elo_distribution(paths, args.elo_low, args.elo_high, args.elo_step)
            else:
                run_elo_distribution(paths, args.elo_low, args.elo_high, args.elo_step)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(2)
        return

    use_estimate = args.estimate or not args.exact

    if use_estimate:
        rows, est_total, avg_bpl = estimate_totals(paths)
        n_stats = sum(1 for *__, fs in rows if fs)
        print(f"Estimated games (lines ≈ kept games): {int(est_total):,}")
        if n_stats == len(rows) and rows:
            print("Sampled avg line length:  — (unused; all counts from *.stats.json)")
        else:
            print(f"Sampled avg line length:  {avg_bpl:.1f} bytes (files without stats only)")
        print(f"Files: {len(rows)} ({n_stats} from stats.json, {len(rows) - n_stats} byte estimate)")
        print()
        wn = max(len(name) for name, _, __, __ in rows) if rows else 10
        for name, est, mb, from_stats in rows:
            tag = "=" if from_stats else "≈"
            src = "stats" if from_stats else "est "
            print(f"  {name:{wn}s}  {tag}{est:>12,}  ({mb:.1f} MiB)  [{src}]")
        print()
        print("Note: lines include malformed rows; prepare.py skips those when tokenizing.")
        return

    t0 = time.time()
    total = 0
    w = max(len(p.name) for p in paths)
    print("Exact newline counts:")
    print()
    for p in sorted(paths, key=lambda q: q.name):
        sp = stats_json_path_for_games_file(p)
        kg = read_kept_games_from_stats(sp)
        if kg is not None:
            n = kg
            note = " (from stats.json kept_games)"
        else:
            n = count_newlines(p)
            note = ""
        total += n
        try:
            mb = p.stat().st_size / (1024 * 1024)
        except OSError:
            mb = 0.0
        print(f"  {p.name:{w}s}  {n:>12,}  ({mb:.1f} MiB){note}")
    dt = time.time() - t0
    print()
    print(f"Total lines (= games rows): {total:,}")
    print(f"Read time: {dt:.1f}s ({total/max(dt, 1e-6)/1e6:.2f}M lines/s)")


if __name__ == "__main__":
    main()
