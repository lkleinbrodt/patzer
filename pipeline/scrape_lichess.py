#!/usr/bin/env python3
"""
Lichess data pipeline: download → verify → filter → parse → clean up.

Downloads monthly PGN dumps from Lichess, runs them through the ELO filter
and UCI parser, and keeps only the small filtered output. The large .zst files
are deleted after processing to save disk space.

Usage:
    # Process all available months
    python scrape_lichess.py --output-dir ./data

    # Process specific months
    python scrape_lichess.py --output-dir ./data --months 2024-01 2024-02 2024-03

    # Skip months before a cutoff (e.g. you already have older games on another machine)
    python scrape_lichess.py --output-dir ./data --min-month 2020-04

    # Filter options (passed through to filter_games.py and parse_pgn.py)
    python scrape_lichess.py --output-dir ./data --min-elo 1800 --min-moves 10 --max-moves 200

    # Dry run: show what would be downloaded without doing anything
    python scrape_lichess.py --output-dir ./data --dry-run
    
    nohup python scrape_lichess.py --output-dir ./data --min-elo 1800 --max-months 12 &

Resumable: already-processed months are skipped. If a download is interrupted,
it resumes from where it left off. Re-run the script anytime to pick up where
you left off.

Requirements:
    pip install chess requests
    wget must be available on PATH (for resumable downloads)
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

import requests


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _r2_key_for_path(repo_root: Path, path: Path) -> str:
    """S3 key mirroring repo-relative paths (must not use absolute paths as keys)."""
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def push_outputs_to_r2(output_dir: Path, month: str) -> bool:
    """
    Upload games_*.txt, stats, progress, and scrape.log for resumable sync to R2.
    Requires R2 env vars and patzer/r2.py. Returns False if nothing was pushed or r2 unavailable.
    """
    repo_root = _repo_root()
    patzer_dir = repo_root / "patzer"
    if not patzer_dir.is_dir():
        log.error("patzer/ not found next to pipeline/; cannot push to R2")
        return False
    pd = str(patzer_dir)
    if pd not in sys.path:
        sys.path.insert(0, pd)
    import r2  # noqa: E402

    ok = True
    games_candidates = [output_dir / f"games_{month}.txt.gz", output_dir / f"games_{month}.txt"]
    games_path = next((p for p in games_candidates if p.is_file()), None)
    if games_path is None:
        log.error(f"Cannot push missing file (tried .txt.gz and .txt): games_{month}")
        ok = False
    else:
        key = _r2_key_for_path(repo_root, games_path)
        if not r2.push_file(games_path, key):
            log.error(f"R2 push failed for {key} (check R2_* env vars)")
            ok = False

    stats_p = output_dir / f"games_{month}.stats.json"
    if not stats_p.is_file():
        log.error(f"Cannot push missing file: {stats_p}")
        ok = False
    else:
        key = _r2_key_for_path(repo_root, stats_p)
        if not r2.push_file(stats_p, key):
            log.error(f"R2 push failed for {key} (check R2_* env vars)")
            ok = False

    prog = output_dir / "progress.json"
    if prog.is_file():
        if not r2.push_file(prog, _r2_key_for_path(repo_root, prog)):
            ok = False

    # Flush log file so R2 gets complete tail
    for h in log.handlers:
        if isinstance(h, logging.FileHandler):
            h.flush()
    lf = output_dir / "scrape.log"
    if lf.is_file():
        if not r2.push_file(lf, _r2_key_for_path(repo_root, lf)):
            ok = False

    return ok

# ── constants ────────────────────────────────────────────────────────────────

_MONTH_YYYY_MM = re.compile(r"^\d{4}-\d{2}$")

LIST_URL = "https://database.lichess.org/standard/list.txt"
SHA256_URL = "https://database.lichess.org/standard/sha256sums.txt"
BASE_URL = "https://database.lichess.org/standard/"

# Seconds to wait between downloads — be polite to Lichess
POLITE_DELAY = 5

# ── logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def fetch_text(url):
    """Fetch a small text file from a URL."""
    log.info(f"Fetching {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def parse_file_list(list_text):
    files = [line.strip() for line in list_text.splitlines() if line.strip()]
    # Extract just the filename from full URLs if necessary
    files = [Path(f).name if f.startswith('http') else f for f in files]
    files.sort()
    return files


def parse_checksums(sha256_text):
    """
    Parse sha256sums.txt — lines like:
        abc123...  lichess_db_standard_rated_2024-01.pgn.zst
    Returns dict of {filename: expected_sha256}.
    """
    checksums = {}
    for line in sha256_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            checksum, filename = parts
            checksums[filename] = checksum.lower()
    return checksums


def extract_month(filename):
    """Extract YYYY-MM from a Lichess filename."""
    match = re.search(r"(\d{4}-\d{2})", filename)
    return match.group(1) if match else None


def verify_sha256(filepath, expected_hash):
    """Verify SHA256 of a file. Returns True if match."""
    log.info(f"Verifying SHA256 for {filepath.name} ...")
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest().lower()
    if actual != expected_hash.lower():
        log.error(f"Checksum mismatch for {filepath.name}")
        log.error(f"  Expected: {expected_hash}")
        log.error(f"  Got:      {actual}")
        return False
    log.info("Checksum OK")
    return True


def download_file(url, dest_path):
    """
    Download a file using wget with resume support (-c flag).
    wget will resume partial downloads automatically.
    Returns True on success.
    """
    log.info(f"Downloading {url}")
    log.info(f"  → {dest_path}")

    cmd = [
        "wget",
        "--continue",           # Resume partial downloads
        "--progress=bar:force", # Show progress bar
        "--tries=5",            # Retry up to 5 times
        "--waitretry=30",       # Wait 30s between retries
        "--timeout=60",         # 60s connection timeout
        "--output-document", str(dest_path),
        url,
    ]

    result = subprocess.run(cmd)

    if result.returncode != 0:
        log.error(f"wget failed with exit code {result.returncode}")
        return False

    log.info(f"Download complete: {dest_path.name} "
             f"({dest_path.stat().st_size / 1e9:.1f} GB)")
    return True


def run_pipeline(zst_path, output_path, stats_path, args):
    """
    Stream zst → filter → parse → output file.
    Runs entirely in-memory pipeline without writing an intermediate PGN.
    Returns True on success.
    """
    log.info(f"Running pipeline: {zst_path.name} → {output_path.name}")

    # Build the pipeline:
    # zstdcat file.pgn.zst | filter_games.py | parse_pgn.py --output output.txt

    filter_script = Path(__file__).parent / "filter_games.py"
    parse_script = Path(__file__).parent / "parse_pgn.py"

    if not filter_script.exists():
        log.error(f"filter_games.py not found at {filter_script}")
        return False
    if not parse_script.exists():
        log.error(f"parse_pgn.py not found at {parse_script}")
        return False

    filter_cmd = [
        sys.executable, str(filter_script),
        "--min-elo", str(args.min_elo),
        "--log-every", "200000",
    ]
    if args.time_controls:
        filter_cmd += ["--time-controls"] + [str(t) for t in args.time_controls]

    parse_cmd = [
        sys.executable, str(parse_script),
        "--output", str(output_path),
        "--stats-output", str(stats_path),
        "--min-moves", str(args.min_moves),
        "--max-moves", str(args.max_moves),
        "--log-every", "50000",
    ]
    if args.validate:
        parse_cmd.append("--validate")
    if args.workers:
        parse_cmd += ["--workers", str(args.workers)]

    log.info(f"  filter: {' '.join(filter_cmd)}")
    log.info(f"  parse:  {' '.join(parse_cmd)}")

    try:
        zstdcat = subprocess.Popen(
            ["zstdcat", str(zst_path)],
            stdout=subprocess.PIPE,
        )
        filter_proc = subprocess.Popen(
            filter_cmd,
            stdin=zstdcat.stdout,
            stdout=subprocess.PIPE,
        )
        zstdcat.stdout.close()  # Allow zstdcat to receive SIGPIPE if filter exits

        parse_proc = subprocess.Popen(
            parse_cmd,
            stdin=filter_proc.stdout,
        )
        filter_proc.stdout.close()

        parse_proc.wait()
        filter_proc.wait()
        zstdcat.wait()

        if parse_proc.returncode != 0:
            log.error(f"parse_pgn.py exited with code {parse_proc.returncode}")
            return False
        if filter_proc.returncode != 0:
            log.error(f"filter_games.py exited with code {filter_proc.returncode}")
            return False

    except Exception as e:
        log.error(f"Pipeline error: {e}")
        return False

    log.info(f"Pipeline complete → {output_path.name} "
             f"({output_path.stat().st_size / 1e6:.1f} MB)")
    return True


def load_progress(progress_file):
    """Load set of already-completed months from a progress file."""
    if progress_file.exists():
        with open(progress_file) as f:
            return set(json.load(f))
    return set()


def save_progress(progress_file, completed_months):
    """Save set of completed months to a progress file."""
    with open(progress_file, "w") as f:
        json.dump(sorted(completed_months), f, indent=2)


# ── main ──────────────────────────────────────────────────────────────────────

def _parse_min_month(s: str) -> str:
    s = s.strip()
    if not _MONTH_YYYY_MM.match(s):
        raise argparse.ArgumentTypeError("min-month must be YYYY-MM")
    return s


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and process Lichess game database dumps."
    )
    #default output is ./data/lichess_games/
    parser.add_argument("--output-dir", type=str, default="data/lichess_games",
                        help="Directory to store processed output files")
    parser.add_argument("--months", type=str, nargs="+", default=None,
                        help="Specific months to process e.g. 2024-01 2024-02. "
                             "If not set, processes all available months.")
    parser.add_argument(
        "--min-month",
        type=_parse_min_month,
        default=None,
        metavar="YYYY-MM",
        help="Only process dump months on or after this (inclusive). "
             "Use when older months are already available locally.",
    )
    parser.add_argument("--max-months", type=int, default=None,
                        help="Process at most this many months (oldest first). "
                             "Useful for a first run e.g. --max-months 12.")
    parser.add_argument("--min-elo", type=int, default=1800,
                        help="Minimum ELO for both players (default: 1800)")
    parser.add_argument("--time-controls", type=int, nargs="+", default=None,
                        help="Whitelist of base time controls in seconds. "
                             "Default: exclude bullet (<180s), keep everything else.")
    parser.add_argument("--min-moves", type=int, default=10,
                        help="Minimum moves per game (default: 10)")
    parser.add_argument("--max-moves", type=int, default=200,
                        help="Maximum moves per game (default: 200)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers for parse_pgn.py SAN→UCI conversion. "
                             "Default: cpu_count-1 (min 1). Set 1 to disable.")
    parser.add_argument("--validate", action="store_true", default=False,
                        help="Enable full python-chess PGN validation (slower). "
                             "Default: fast mode (skips PGN tree building).")
    parser.add_argument("--keep-zst", action="store_true", default=False,
                        help="Keep the .zst files after processing (default: delete them)")
    parser.add_argument(
        "--no-compress",
        action="store_true",
        default=False,
        help="Write monthly games output as plain .txt instead of .txt.gz (default: compress).",
    )
    parser.add_argument("--skip-verify", action="store_true", default=False,
                        help="Skip SHA256 verification (faster but riskier)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Show what would be processed without doing anything")
    parser.add_argument("--delay", type=int, default=POLITE_DELAY,
                        help=f"Seconds to wait between downloads (default: {POLITE_DELAY})")
    parser.add_argument(
        "--push-r2",
        action="store_true",
        help="After each completed month, upload games/statistics/progress/log to R2 "
             "(same keys as local paths under the repo root). Requires R2_* env vars.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Add file logging
    log_file = output_dir / "scrape.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    log.addHandler(file_handler)
    log.info(f"Logging to {log_file}")

    progress_file = output_dir / "progress.json"
    completed = load_progress(progress_file)
    if completed:
        log.info(f"Already completed months: {sorted(completed)}")

    # Fetch file list and checksums
    try:
        list_text = fetch_text(LIST_URL)
        sha256_text = fetch_text(SHA256_URL)
    except Exception as e:
        log.error(f"Failed to fetch file list: {e}")
        sys.exit(1)

    all_files = parse_file_list(list_text)
    checksums = parse_checksums(sha256_text)

    log.info(f"Found {len(all_files)} months in Lichess database")

    # Filter to requested months
    if args.months:
        requested = set(args.months)
        all_files = [f for f in all_files if extract_month(f) in requested]
        log.info(f"Filtered to {len(all_files)} requested months")

    if args.min_month:
        n_before = len(all_files)
        all_files = [
            f for f in all_files
            if (m := extract_month(f)) is not None and m >= args.min_month
        ]
        log.info(
            f"Filtered to months >= {args.min_month}: {len(all_files)} files "
            f"(from {n_before} after prior filters)"
        )

    # Cap at max_months (applied before skipping completed, so it's a
    # limit on the total universe not just what's left to do)
    if args.max_months:
        all_files = all_files[:args.max_months]
        log.info(f"Capped to first {args.max_months} months (oldest first)")

    # Skip already completed
    todo = [f for f in all_files if extract_month(f) not in completed]
    log.info(f"Months to process: {len(todo)} "
             f"(skipping {len(all_files) - len(todo)} already done)\n")

    if args.dry_run:
        log.info("DRY RUN — would process:")
        for f in todo:
            filename = Path(f).name
            size_note = f"  checksum: {'known' if filename in checksums else 'MISSING'}"
            log.info(f"  {filename}{size_note}")
        return

    if not todo:
        log.info("Nothing to do.")
        return

    success_count = 0
    fail_count = 0

    for i, filename in enumerate(todo):
        month = extract_month(filename)
        url = BASE_URL + filename
        zst_path = output_dir / filename
        output_path = output_dir / (f"games_{month}.txt" if args.no_compress else f"games_{month}.txt.gz")
        stats_path = output_dir / f"games_{month}.stats.json"

        log.info(f"\n{'='*60}")
        log.info(f"[{i+1}/{len(todo)}] Processing {month}")
        log.info(f"{'='*60}")

        # ── Step 1: Download ──────────────────────────────────────────
        if not zst_path.exists():
            success = download_file(url, zst_path)
            if not success:
                log.error(f"Download failed for {filename}, skipping.")
                fail_count += 1
                continue
        else:
            log.info(f"Already downloaded: {zst_path.name} — skipping download")

        # ── Step 2: Verify checksum ───────────────────────────────────
        if not args.skip_verify:
            expected = checksums.get(filename)
            if expected:
                if not verify_sha256(zst_path, expected):
                    log.error(
                        f"Checksum failed for {filename}. "
                        "Deleting local file and retrying download once."
                    )
                    zst_path.unlink(missing_ok=True)

                    # One automatic recovery attempt: fresh download + re-verify.
                    if not download_file(url, zst_path):
                        log.error(f"Retry download failed for {filename}, skipping.")
                        fail_count += 1
                        continue
                    if not verify_sha256(zst_path, expected):
                        log.error(
                            f"Checksum still failed after retry for {filename}. "
                            "Deleting file and skipping."
                        )
                        zst_path.unlink(missing_ok=True)
                        fail_count += 1
                        continue
            else:
                log.warning(f"No checksum found for {filename}, skipping verification.")
        else:
            log.info("Skipping checksum verification (--skip-verify)")

        # ── Step 3: Run pipeline ──────────────────────────────────────
        success = run_pipeline(zst_path, output_path, stats_path, args)
        if not success:
            log.error(f"Pipeline failed for {filename}, skipping.")
            fail_count += 1
            continue

        # ── Step 4: Clean up .zst ─────────────────────────────────────
        if not args.keep_zst:
            log.info(f"Deleting {zst_path.name} to free disk space")
            zst_path.unlink()
        else:
            log.info(f"Keeping {zst_path.name} (--keep-zst)")

        # ── Step 5: Mark as complete ──────────────────────────────────
        completed.add(month)
        save_progress(progress_file, completed)
        success_count += 1
        log.info(f"✓ {month} complete → {output_path.name}")

        if args.push_r2:
            if push_outputs_to_r2(output_dir, month):
                log.info(f"Pushed {month} artifacts to R2")
            else:
                log.error("R2 push failed after completing month (see errors above)")

        # ── Polite delay before next download ─────────────────────────
        if i < len(todo) - 1:
            log.info(f"Waiting {args.delay}s before next download ...")
            time.sleep(args.delay)

    log.info(f"\n{'='*60}")
    log.info(f"All done. Success: {success_count}, Failed: {fail_count}")
    log.info(f"Output files in: {output_dir}")
    log.info(f"Progress saved to: {progress_file}")


if __name__ == "__main__":
    main()