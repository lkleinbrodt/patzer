"""
pipeline/prepare.py

One-time preprocessing step: tokenize game text files into binary files
that the training loop can memory-map efficiently.

Default behavior reads data/lichess_games/games_*.txt, tokenizes every game,
splits into train/val, and saves uint16 binaries + metadata.
"""

import argparse
import glob
import hashlib
import itertools
import json
import os
import re
import sys
import time
from collections import deque
from functools import partial
from datetime import UTC, datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from patzer.tokenizer import ChessTokenizer

_WORKER_TOK = None


def _use_ansi() -> bool:
    if not sys.stderr.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    term = os.environ.get("TERM", "")
    return term != "dumb"


def _c(s: str, *codes: str) -> str:
    if not codes or not _use_ansi():
        return s
    return "".join(codes) + s + "\033[0m"


def _fmt_duration(seconds: float) -> str:
    if not (seconds == seconds):  # NaN
        return "—"
    if seconds < 0:
        seconds = 0
    if seconds < 1:
        return "<1s"
    sec = int(round(seconds))
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _fmt_int_per_sec(rate: float) -> str:
    if not (rate > 0) or not (rate == rate):
        return "—"
    if rate >= 1_000_000:
        return f"{rate/1e6:.2f}M/s"
    if rate >= 1_000:
        return f"{rate/1e3:.1f}k/s"
    return f"{rate:.0f}/s"


def _sample_avg_bytes_per_line(path: Path, max_bytes: int = 393_216) -> float:
    """
    Estimate average encoded line length (bytes) by reading a prefix of `path`.
    Falls back if the sample has too few lines.
    """
    if not path.is_file():
        return 120.0
    try:
        with open(path, "rb") as f:
            blob = f.read(max_bytes)
    except OSError:
        return 120.0
    if not blob:
        return 120.0
    n_nl = blob.count(b"\n")
    if n_nl < 8:
        return max(72.0, min(384.0, len(blob)))
    avg = len(blob) / max(1, n_nl)
    return max(48.0, min(512.0, float(avg)))


def estimate_total_lines(paths: list[str]) -> tuple[int, float]:
    """
    Fast line-count estimate using file sizes ÷ sampled average bytes/line.

    Returns (estimated_lines, avg_bytes_per_line_used).
    """
    path_objs = [Path(p) for p in paths]
    sizes = [(p, p.stat().st_size) for p in path_objs if p.is_file()]
    total_bytes = sum(sz for _, sz in sizes)
    if total_bytes <= 0:
        return max(1, len(paths)), 120.0
    avg_bpl = 120.0
    for p, _sz in sizes:
        avg_bpl = _sample_avg_bytes_per_line(p)
        break
    est_lines = max(1, int(total_bytes / avg_bpl))
    return est_lines, avg_bpl


class PipelineProgressReporter:
    """
    Throttled stderr progress: line scan %, throughput, EWMA ETA for games.
    """

    def __init__(
        self,
        label: str,
        *,
        est_total_lines: int,
        exact_total_games: int | None,
        min_interval_s: float = 0.35,
    ):
        self.label = label
        self.est_total_lines = max(1, est_total_lines)
        self.exact_total_games = exact_total_games
        self.min_interval_s = min_interval_s
        self._t0 = time.time()
        self._last_emit = 0.0
        self.lines_seen = 0
        self.chunk_count = 0
        self.games_processed = 0

    def on_chunk_finish(self, chunk_line_count: int, games_kept_or_written_delta: int) -> None:
        self.lines_seen += chunk_line_count
        self.chunk_count += 1
        self.games_processed += games_kept_or_written_delta
        self.maybe_emit(force=False)

    def maybe_emit(self, *, force: bool) -> None:
        now = time.time()
        if not force and (now - self._last_emit) < self.min_interval_s:
            return
        elapsed = max(1e-6, now - self._t0)
        self._last_emit = now

        line_pct = min(100.0, 100.0 * self.lines_seen / self.est_total_lines)
        bar_w = 18
        filled = int(bar_w * line_pct / 100.0)
        bar_inner = _c("█" * filled, "\033[32m") + _c("░" * (bar_w - filled), "\033[2m")

        g_rate = self.games_processed / elapsed
        lg = self.label
        lg = _c(lg, "\033[1m")

        denom_games = self.exact_total_games
        if denom_games is None and self.lines_seen > 0:
            ratio = self.games_processed / self.lines_seen
            denom_games = max(float(self.games_processed), ratio * self.est_total_lines)

        pct_games_txt = ""
        eta_sec = None
        if denom_games is not None and denom_games > 0:
            pg = min(100.0, 100.0 * self.games_processed / denom_games)
            pct_games_txt = f" │ games {pg:4.1f}%"
            rem = float(denom_games) - float(self.games_processed)
            if rem > 0 and g_rate > 0:
                eta_sec = rem / g_rate

        eta_txt = _fmt_duration(eta_sec) if eta_sec is not None else "—"

        cyan = "\033[36m"
        tty = sys.stderr.isatty()
        leader = "\r" if tty else ""
        pad = "    " if tty else ""
        msg = (
            f"{leader}{lg} │{bar_inner}│ {line_pct:5.1f}% lines │ "
            f"{self.games_processed:,} games │ {_fmt_int_per_sec(g_rate)} │ "
            f"ETA {_c(eta_txt, cyan)} │ {_fmt_duration(elapsed)} elapsed{pct_games_txt}"
            f"{pad}"
        )

        sys.stderr.write(msg)
        if (not tty) or force:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def finish(self) -> None:
        self.maybe_emit(force=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, nargs="+",
                        default=["data/lichess_games/games_*.txt"],
                        help="Input file(s) or glob pattern(s). "
                             "e.g. data/lichess_games/games_2013-*.txt")
    parser.add_argument("--output-dir", type=str, default="data/prepared",
                        help="Output directory for binary files (default: data/prepared)")
    parser.add_argument("--vocab", type=str, default="data/vocab.json",
                        help="Vocab file (default: data/vocab.json). "
                             "Created if it doesn't exist.")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction of games to use for validation (default: 0.05)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Cap number of kept/tokenized games (useful for debugging)")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Context window size, just recorded in meta.json (default: 256)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1),
                        help="Tokenizer worker processes (default: cpu_count-1)")
    parser.add_argument("--chunk-lines", type=int, default=50_000,
                        help="Lines per work chunk (default: 50000)")
    parser.add_argument("--flush-tokens", type=int, default=2_000_000,
                        help="Flush to disk every N buffered tokens (default: 2000000)")
    parser.add_argument("--split-mode", type=str, choices=["boundary", "hash"], default="hash",
                        help=("Split strategy: 'boundary' keeps old behavior and does two passes; "
                              "'hash' is one-pass approximate split by deterministic line hash "
                              "(val ratio is approximate) "
                              "(default: hash)."))
    parser.add_argument("--seed", type=int, default=1337,
                        help="Seed used by hash split mode (default: 1337)")
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=None,
        metavar="N",
        help="Max chunk tasks queued to workers (default: 2 × workers; lower on low-RAM machines)",
    )
    return parser.parse_args()


def parse_game_line(line: str) -> tuple[str, list[str]] | None:
    """
    Parse one line from games.txt.

    Format: {result} {white_elo} {black_elo} {uci_move} {uci_move} ...
    e.g.:   1-0 2100 2000 e2e4 e7e5 g1f3 b8c6

    Returns (result, moves) or None if line is malformed.
    """
    parts = line.strip().split()
    if len(parts) < 4:
        return None

    result = parts[0]
    # parts[1] = white_elo, parts[2] = black_elo — skipped for v1
    moves = parts[3:]

    if result not in ("1-0", "0-1", "1/2-1/2"):
        return None

    return result, moves


def _init_worker(vocab_path: str):
    global _WORKER_TOK
    _WORKER_TOK = ChessTokenizer.load(Path(vocab_path))


def _encode_line(line: str) -> list[int] | None:
    parsed = parse_game_line(line)
    if parsed is None:
        return None
    result, moves = parsed
    try:
        return _WORKER_TOK.encode_game(moves, result=result)
    except ValueError:
        return None


def _count_chunk_parse_only(lines: list[str]) -> tuple[int, int, int]:
    kept = 0
    skipped = 0
    for line in lines:
        if parse_game_line(line) is None:
            skipped += 1
        else:
            kept += 1
    n_lines = len(lines)
    return kept, skipped, n_lines


def _tokenize_chunk(lines: list[str]) -> tuple[list[list[int]], int, int]:
    games: list[list[int]] = []
    skipped = 0
    for line in lines:
        token_ids = _encode_line(line)
        if token_ids is None:
            skipped += 1
            continue
        games.append(token_ids)
    n_lines = len(lines)
    return games, skipped, n_lines


def _stable_split_value(line: str, seed: int) -> float:
    # Deterministic pseudo-random in [0, 1), stable across runs/machines.
    digest = hashlib.md5(f"{seed}:{line}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / float(2**128)


def _tokenize_chunk_for_hash_split(
    lines: list[str], seed: int
) -> tuple[list[tuple[float, list[int]]], int, int]:
    """
    Like _tokenize_chunk, but also compute the hash-split score per line in-process.

    Returning (split_val, token_ids) instead of (line, token_ids) avoids pickling
    every full game line back to the parent, which is very large on big runs.
    """
    games: list[tuple[float, list[int]]] = []
    skipped = 0
    for line in lines:
        token_ids = _encode_line(line)
        if token_ids is None:
            skipped += 1
            continue
        games.append((_stable_split_value(line, seed), token_ids))
    n_lines = len(lines)
    return games, skipped, n_lines


def _bounded_map(executor, fn, iterable, *, max_in_flight: int):
    """
    Memory-bounded alternative to executor.map.

    executor.map eagerly drains the input iterator and submits all futures at
    once, causing unbounded memory growth on large datasets. This helper keeps
    at most `max_in_flight` futures pending at any time by reading the input
    lazily. Result order is preserved.
    """
    pending: deque = deque()
    it = iter(iterable)
    for item in itertools.islice(it, max_in_flight):
        pending.append(executor.submit(fn, item))
    for item in it:
        yield pending.popleft().result()
        pending.append(executor.submit(fn, item))
    while pending:
        yield pending.popleft().result()


def iter_line_chunks(
    input_files: list[str],
    chunk_lines: int,
    *,
    label: str = "",
):
    chunk: list[str] = []
    n_files = len(input_files)
    for idx, input_path in enumerate(input_files):
        p = Path(input_path)
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
        except OSError:
            size_mb = 0.0
        lbl = label or "read"
        hdr = (
            _c("[prepare]", "\033[2m")
            + " "
            + _c(lbl, "\033[1m")
            + f"  file {idx + 1}/{n_files}: {p.name}  "
            + _c(f"({size_mb:.1f} MiB)", "\033[2m")
        )
        sys.stderr.write(hdr + "\n")
        sys.stderr.flush()
        with open(input_path, "r") as f:
            for line in f:
                chunk.append(line)
                if len(chunk) >= chunk_lines:
                    yield chunk
                    chunk = []
    if chunk:
        yield chunk


class BufferedBinWriter:
    def __init__(self, out_path: Path, flush_tokens: int):
        self.out_path = out_path
        self.flush_tokens = flush_tokens
        self._buffers: list[np.ndarray] = []
        self._buffered_tokens = 0
        self.tokens_written = 0
        self.games_written = 0
        self._fh = open(out_path, "wb")

    def append(self, token_ids: list[int]):
        arr = np.asarray(token_ids, dtype=np.uint16)
        self._buffers.append(arr)
        self._buffered_tokens += arr.size
        self.games_written += 1
        if self._buffered_tokens >= self.flush_tokens:
            self.flush()

    def flush(self):
        if not self._buffers:
            return
        joined = np.concatenate(self._buffers)
        joined.tofile(self._fh)
        self.tokens_written += int(joined.size)
        self._buffers.clear()
        self._buffered_tokens = 0

    def close(self):
        self.flush()
        self._fh.close()


def main():
    args = parse_args()
    if args.split_mode == "boundary" and args.max_games is not None:
        print("Error: --max-games is supported only with --split-mode hash.")
        print("Use hash mode for capped-size smoke tests, or run boundary mode without --max-games.")
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or create tokenizer
    vocab_path = Path(args.vocab)
    if vocab_path.exists():
        print(f"Loading vocab from {vocab_path}")
        tok = ChessTokenizer.load(vocab_path)
    else:
        print(f"Building vocab and saving to {vocab_path}")
        tok = ChessTokenizer()
        tok.save(vocab_path)

    # resolve all input files from globs
    input_files = []
    for pattern in args.input:
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"Warning: no files matched {pattern}")
        input_files.extend(matched)

    if not input_files:
        print("No input files found. Exiting.")
        sys.exit(1)

    print(f"Vocab size: {tok.vocab_size}")
    print(f"Input files: {len(input_files)}")
    for f in input_files:
        print(f"  {f}")
    print(f"Output:     {output_dir}")
    print(f"Val split:  {args.val_fraction:.1%}")
    print(f"Workers:    {args.workers}")
    print(f"Chunk:      {args.chunk_lines:,} lines")
    max_in_flight = (
        max(1, args.max_in_flight)
        if args.max_in_flight is not None
        else max(1, args.workers * 2)
    )
    if args.workers > 1:
        print(f"Max flight: {max_in_flight} pending chunks (bounded executor queue)")
    print(f"Split mode: {args.split_mode}\n")

    est_lines, avg_bpl = estimate_total_lines(input_files)
    if _use_ansi():
        sys.stderr.write(
            "\033[2m[prepare]\033[0m"
            f" ~{est_lines:,} lines estimated · sampled avg {avg_bpl:.0f} B/line"
            "\n"
        )
    else:
        sys.stderr.write(
            f"[prepare] ~{est_lines:,} lines estimated · sampled avg {avg_bpl:.0f} B/line\n"
        )
    sys.stderr.flush()

    train_writer = BufferedBinWriter(output_dir / "train.bin", args.flush_tokens)
    val_writer = BufferedBinWriter(output_dir / "val.bin", args.flush_tokens)

    total_kept = 0
    total_skipped = 0
    split_boundary = None
    start = time.time()

    if args.workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(str(vocab_path),),
        )
    else:
        executor = None
        _init_worker(str(vocab_path))

    try:
        if args.split_mode == "boundary":
            print("Pass 1/2: counting valid games for boundary split…")
            parse_skipped = 0
            prog1 = PipelineProgressReporter(
                "Pass 1/2 · count valid games",
                est_total_lines=est_lines,
                exact_total_games=None,
            )
            chunks = iter_line_chunks(input_files, args.chunk_lines, label="Pass 1/2 · read")
            if executor:
                count_results = _bounded_map(
                    executor,
                    _count_chunk_parse_only,
                    chunks,
                    max_in_flight=max_in_flight,
                )
            else:
                count_results = map(_count_chunk_parse_only, chunks)
            for kept, skipped, n_lines in count_results:
                total_kept += kept
                parse_skipped += skipped
                prog1.on_chunk_finish(n_lines, kept)
            prog1.finish()
            print(
                f"  Parse-only pass: {parse_skipped:,} malformed lines "
                "(not included in final skipped metric)."
            )

            n_val = max(1, int(total_kept * args.val_fraction))
            n_train = total_kept - n_val
            split_boundary = n_train
            print(f"Boundary split: train={n_train:,}, val={n_val:,}")

            print("\nPass 2/2: tokenizing + streaming write…")
            seen_games = 0
            prog2 = PipelineProgressReporter(
                "Pass 2/2 · tokenize & write",
                est_total_lines=est_lines,
                exact_total_games=total_kept,
            )
            chunks = iter_line_chunks(input_files, args.chunk_lines, label="Pass 2/2 · read")
            if executor:
                tok_results = _bounded_map(
                    executor,
                    _tokenize_chunk,
                    chunks,
                    max_in_flight=max_in_flight,
                )
            else:
                tok_results = map(_tokenize_chunk, chunks)
            for games, encode_skipped, n_lines in tok_results:
                total_skipped += encode_skipped
                written_this_chunk = 0
                for token_ids in games:
                    if args.max_games and seen_games >= args.max_games:
                        break
                    if seen_games < split_boundary:
                        train_writer.append(token_ids)
                    else:
                        val_writer.append(token_ids)
                    seen_games += 1
                    written_this_chunk += 1
                prog2.on_chunk_finish(n_lines, written_this_chunk)
                if args.max_games and seen_games >= args.max_games:
                    break
            prog2.finish()
        else:
            # Hash split is one-pass and deterministic by (seed, line content).
            print("One-pass hash split: tokenizing + streaming write…")
            exact_cap = args.max_games if args.max_games is not None else None
            prog = PipelineProgressReporter(
                "Hash split · tokenize & write",
                est_total_lines=est_lines,
                exact_total_games=exact_cap,
            )
            chunks = iter_line_chunks(input_files, args.chunk_lines, label="Hash split · read")
            hash_fn = partial(_tokenize_chunk_for_hash_split, seed=args.seed)
            if executor:
                tok_results = _bounded_map(
                    executor,
                    hash_fn,
                    chunks,
                    max_in_flight=max_in_flight,
                )
            else:
                tok_results = map(hash_fn, chunks)

            for games, skipped, n_lines in tok_results:
                total_skipped += skipped
                kept_this_chunk = 0
                for split_val, token_ids in games:
                    if args.max_games and total_kept >= args.max_games:
                        break
                    total_kept += 1
                    kept_this_chunk += 1
                    if split_val < args.val_fraction:
                        val_writer.append(token_ids)
                    else:
                        train_writer.append(token_ids)
                prog.on_chunk_finish(n_lines, kept_this_chunk)
                if args.max_games and total_kept >= args.max_games:
                    break
            prog.finish()
    finally:
        if executor:
            executor.shutdown(wait=True)
        train_writer.close()
        val_writer.close()

    n_train = train_writer.games_written
    n_val = val_writer.games_written
    total_kept = n_train + n_val
    total_tokens = train_writer.tokens_written + val_writer.tokens_written
    train_tokens = train_writer.tokens_written
    val_tokens = val_writer.tokens_written

    wall_s = time.time() - start
    print(f"\nDone ({_fmt_duration(wall_s)}). {total_kept:,} games tokenized, {total_skipped:,} skipped.")
    print(f"Train games: {n_train:,}")
    print(f"Val games:   {n_val:,}")
    print(f"train.bin:   {train_tokens:,} tokens ({(output_dir / 'train.bin').stat().st_size / 1e6:.1f} MB)")
    print(f"val.bin:     {val_tokens:,} tokens ({(output_dir / 'val.bin').stat().st_size / 1e6:.1f} MB)")

    meta = {
        "prepared_at": datetime.now(UTC).isoformat(),
        "source_files": [Path(f).name for f in input_files],
        "months": sorted([
            m for f in input_files
            for m in [re.search(r"(\d{4}-\d{2})", Path(f).name)]
            if m for m in [m.group(1)]
        ]),
        "min_elo_filter": "see source filter_games.py args",
        "vocab_size": tok.vocab_size,
        "block_size": args.block_size,
        "split_mode": args.split_mode,
        "seed": args.seed if args.split_mode == "hash" else None,
        "workers": args.workers,
        "max_in_flight": max_in_flight,
        "chunk_lines": args.chunk_lines,
        "flush_tokens": args.flush_tokens,
        "total_games": total_kept,
        "train_games": n_train,
        "val_games": n_val,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "avg_tokens_per_game": round(total_tokens / total_kept, 1) if total_kept else 0.0,
        "skipped_games": total_skipped,
        "split_boundary_index": split_boundary,
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nMetadata saved to {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
