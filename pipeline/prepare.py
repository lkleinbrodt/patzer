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
import json
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from patzer.tokenizer import ChessTokenizer

_WORKER_TOK = None


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


def _count_chunk_parse_only(lines: list[str]) -> tuple[int, int]:
    kept = 0
    skipped = 0
    for line in lines:
        if parse_game_line(line) is None:
            skipped += 1
        else:
            kept += 1
    return kept, skipped


def _tokenize_chunk(lines: list[str]) -> tuple[list[list[int]], int]:
    games: list[list[int]] = []
    skipped = 0
    for line in lines:
        token_ids = _encode_line(line)
        if token_ids is None:
            skipped += 1
            continue
        games.append(token_ids)
    return games, skipped


def _tokenize_chunk_with_lines(lines: list[str]) -> tuple[list[tuple[str, list[int]]], int]:
    games: list[tuple[str, list[int]]] = []
    skipped = 0
    for line in lines:
        token_ids = _encode_line(line)
        if token_ids is None:
            skipped += 1
            continue
        games.append((line, token_ids))
    return games, skipped


def _stable_split_value(line: str, seed: int) -> float:
    # Deterministic pseudo-random in [0, 1), stable across runs/machines.
    digest = hashlib.md5(f"{seed}:{line}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / float(2**128)


def iter_line_chunks(input_files: list[str], chunk_lines: int, phase: str | None = None):
    chunk: list[str] = []
    prefix = f"[{phase}] " if phase else ""
    for input_path in input_files:
        print(f"{prefix}Scanning {Path(input_path).name}...")
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
    print(f"Split mode: {args.split_mode}\n")

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
            print("Pass 1/2: counting valid games for boundary split...")
            parse_skipped = 0
            chunks = iter_line_chunks(input_files, args.chunk_lines, phase="pass1")
            if executor:
                count_results = executor.map(_count_chunk_parse_only, chunks, chunksize=4)
            else:
                count_results = map(_count_chunk_parse_only, chunks)
            for kept, skipped in count_results:
                total_kept += kept
                parse_skipped += skipped
                if args.max_games and total_kept >= args.max_games:
                    total_kept = args.max_games
                    break
                if total_kept and total_kept % 500_000 == 0:
                    print(f"  Counted {total_kept:,} kept games...")
            print(f"  Parse-only pass counted {parse_skipped:,} malformed lines (not included in final skipped metric).")

            n_val = max(1, int(total_kept * args.val_fraction))
            n_train = total_kept - n_val
            split_boundary = n_train
            print(f"Boundary split: train={n_train:,}, val={n_val:,}")

            print("\nPass 2/2: tokenizing + streaming write...")
            seen_games = 0
            chunks = iter_line_chunks(input_files, args.chunk_lines, phase="pass2")
            if executor:
                tok_results = executor.map(_tokenize_chunk, chunks, chunksize=4)
            else:
                tok_results = map(_tokenize_chunk, chunks)
            for games, encode_skipped in tok_results:
                total_skipped += encode_skipped
                for token_ids in games:
                    if args.max_games and seen_games >= args.max_games:
                        break
                    if seen_games < split_boundary:
                        train_writer.append(token_ids)
                    else:
                        val_writer.append(token_ids)
                    seen_games += 1
                    if seen_games % 200_000 == 0:
                        elapsed = time.time() - start
                        print(f"  Written {seen_games:,} games ({elapsed:.1f}s)")
                if args.max_games and seen_games >= args.max_games:
                    break
        else:
            # Hash split is one-pass and deterministic by (seed, line content).
            print("One-pass hash split: tokenizing + streaming write...")
            chunks = iter_line_chunks(input_files, args.chunk_lines, phase="pass1")
            if executor:
                tok_results = executor.map(_tokenize_chunk_with_lines, chunks, chunksize=4)
            else:
                tok_results = map(_tokenize_chunk_with_lines, chunks)

            for games, skipped in tok_results:
                total_skipped += skipped
                for line, token_ids in games:
                    if args.max_games and total_kept >= args.max_games:
                        break
                    total_kept += 1
                    if _stable_split_value(line, args.seed) < args.val_fraction:
                        val_writer.append(token_ids)
                    else:
                        train_writer.append(token_ids)
                    if total_kept % 200_000 == 0:
                        elapsed = time.time() - start
                        print(f"  Written {total_kept:,} games ({elapsed:.1f}s)")
                if args.max_games and total_kept >= args.max_games:
                    break
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

    print(f"\nDone. {total_kept:,} games tokenized, {total_skipped:,} skipped.")
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
