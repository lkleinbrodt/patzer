#!/usr/bin/env python3
"""
eval/build_opening_book.py

One-time script to build eval/openings.json — a curated set of balanced chess
opening sequences for use by the eval harness.

Source: Stockfish's 2moves_v2.pgn (12 092 real-game 2-move openings).
Filter: keep only positions where Stockfish evaluates the resulting position
within +/- cp_limit centipawns at the given analysis depth.  This ensures both
sides start each evaluation game from a roughly equal position.

The output openings.json is a JSON array of UCI move lists, e.g.:
    [["e2e4","e7e5","g1f3","b8c6"], ...]

Commit the generated file.  The eval harness reads it at runtime without
needing Stockfish for opening selection.

Usage
-----
    # Use local PGN (fastest if you already have the file):
    python eval/build_opening_book.py --pgn data/2moves_v2.pgn

    # Download automatically:
    python eval/build_opening_book.py

    # Tune filtering:
    python eval/build_opening_book.py --pgn data/2moves_v2.pgn --cp-limit 50 --depth 10
    python eval/build_opening_book.py --pgn data/2moves_v2.pgn --no-filter
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import chess
import chess.pgn

PGN_URL = "https://raw.githubusercontent.com/official-stockfish/books/master/2moves_v2.pgn.zip"
OUTPUT_PATH = Path(__file__).parent / "openings.json"


def _get_pgn_text(pgn_path: str | None) -> str:
    if pgn_path:
        p = Path(pgn_path)
        print(f"Reading {p} ({p.stat().st_size:,} bytes)...", file=sys.stderr)
        return p.read_text()

    print(f"Downloading {PGN_URL} ...", file=sys.stderr)
    with urllib.request.urlopen(PGN_URL, timeout=60) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.endswith(".pgn"))
        text = z.read(name).decode("utf-8")
    print(f"Downloaded {len(text):,} bytes.", file=sys.stderr)
    return text


def _parse_pgn(pgn_text: str) -> list[list[str]]:
    """Return list of UCI move sequences (one per game) from a PGN string."""
    games: list[list[str]] = []
    f = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(f)
        if game is None:
            break
        moves: list[str] = []
        node = game
        while node.variations:
            node = node.variations[0]
            moves.append(node.move.uci())
        if moves:
            games.append(moves)
    print(f"Parsed {len(games):,} game sequences from PGN.", file=sys.stderr)
    return games


def _filter_balanced(
    sequences: list[list[str]],
    stockfish_path: str,
    cp_limit: int,
    depth: int,
) -> list[list[str]]:
    """Keep only positions where |Stockfish eval| <= cp_limit at the given depth."""
    import chess.engine

    n = len(sequences)
    print(
        f"Filtering {n:,} positions — keeping |eval| ≤ {cp_limit} cp at depth {depth}",
        file=sys.stderr,
    )
    print(f"Using Stockfish: {stockfish_path}", file=sys.stderr)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)

    kept: list[list[str]] = []
    skipped_invalid = 0
    skipped_imbalanced = 0
    skipped_mate = 0
    t0 = time.time()
    log_every = max(1, n // 40)   # ~40 progress lines total

    for i, moves in enumerate(sequences):
        if i % log_every == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 and i > 0 else 0
            eta_s = int((n - i) / rate) if rate > 0 else 0
            print(
                f"  [{i:>6}/{n}]  kept={len(kept):>5}  invalid={skipped_invalid}"
                f"  imbalanced={skipped_imbalanced}  mate={skipped_mate}"
                f"  {elapsed:5.0f}s elapsed  ETA ~{eta_s}s",
                file=sys.stderr,
            )

        # Replay moves onto board
        board = chess.Board()
        valid = True
        for uci in moves:
            m = chess.Move.from_uci(uci)
            if m not in board.legal_moves:
                valid = False
                break
            board.push(m)
        if not valid or board.is_game_over():
            skipped_invalid += 1
            continue

        # Quick Stockfish analysis
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        score = info["score"].white()

        if score.is_mate():
            skipped_mate += 1
            continue

        cp = score.score()
        if cp is None or abs(cp) > cp_limit:
            skipped_imbalanced += 1
            continue

        kept.append(moves)

    engine.quit()
    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s  ({elapsed / n * 1000:.1f} ms/pos).",
        file=sys.stderr,
    )
    print(
        f"Kept {len(kept):,} / {n:,}  "
        f"(invalid={skipped_invalid}, imbalanced={skipped_imbalanced}, mate-in-N={skipped_mate}).",
        file=sys.stderr,
    )
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eval/openings.json — balanced UCI opening sequences."
    )
    parser.add_argument(
        "--pgn",
        default=None,
        metavar="PATH",
        help="Path to local 2moves_v2.pgn (skips download if provided)",
    )
    parser.add_argument(
        "--cp-limit",
        type=int,
        default=75,
        dest="cp_limit",
        help="Max centipawn imbalance to keep a position (default: 75)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=8,
        help="Stockfish analysis depth per position (default: 8)",
    )
    parser.add_argument(
        "--stockfish",
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish binary",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=f"Output JSON path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        dest="no_filter",
        help="Skip Stockfish balance filtering (faster, less balanced)",
    )
    args = parser.parse_args()

    pgn_text = _get_pgn_text(args.pgn)
    sequences = _parse_pgn(pgn_text)

    if args.no_filter:
        print("Skipping balance filter (--no-filter).", file=sys.stderr)
        result = sequences
    else:
        result = _filter_balanced(sequences, args.stockfish, args.cp_limit, args.depth)

    output = Path(args.output)
    output.write_text(json.dumps(result, separators=(",", ":")))
    print(f"Wrote {len(result):,} openings to {output}.", file=sys.stderr)


if __name__ == "__main__":
    main()
