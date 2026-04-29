"""
TODO: accept multiple games files (when we scrape we will have one file for each month)

pipeline/prepare.py

One-time preprocessing step: tokenize games.txt into binary files that
the training loop can memory-map efficiently.

Reads data/games.txt, tokenizes every game, concatenates into one flat
token sequence, then splits into train/val and saves as uint16 binary files.

Usage:
    python pipeline/prepare.py
    python pipeline/prepare.py --input data/games.txt --output-dir data/prepared
    python pipeline/prepare.py --val-fraction 0.05

Output:
    data/prepared/train.bin   - training tokens as uint16
    data/prepared/val.bin     - validation tokens as uint16
    data/prepared/meta.json   - vocab size, token counts, split info
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from patzer.tokenizer import ChessTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/games.txt",
                        help="Input games file (default: data/games.txt)")
    parser.add_argument("--output-dir", type=str, default="data/prepared",
                        help="Output directory for binary files (default: data/prepared)")
    parser.add_argument("--vocab", type=str, default="data/vocab.json",
                        help="Vocab file (default: data/vocab.json). "
                             "Created if it doesn't exist.")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction of games to use for validation (default: 0.05)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Cap number of games to process (useful for debugging)")
    parser.add_argument("--block-size", type=int, default=256,
                        help="Context window size, just recorded in meta.json (default: 256)")
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


def main():
    args = parse_args()

    input_path = Path(args.input)
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

    print(f"Vocab size: {tok.vocab_size}")
    print(f"Input:      {input_path}")
    print(f"Output:     {output_dir}")
    print(f"Val split:  {args.val_fraction:.1%}\n")

    # ── Pass 1: tokenize all games ────────────────────────────────────────────
    # We collect all token sequences in a list first so we can do a clean
    # train/val split by game (not by token) before concatenating.

    all_games: list[list[int]] = []
    skipped = 0
    start = time.time()

    with open(input_path, "r") as f:
        for i, line in enumerate(f):
            if args.max_games and i >= args.max_games:
                break

            parsed = parse_game_line(line)
            if parsed is None:
                skipped += 1
                continue

            result, moves = parsed

            try:
                token_ids = tok.encode_game(moves, result=result)
            except ValueError:
                # Unknown move token — shouldn't happen but skip gracefully
                skipped += 1
                continue

            all_games.append(token_ids)

            if (i + 1) % 100_000 == 0:
                elapsed = time.time() - start
                print(f"  {i+1:,} lines processed, {len(all_games):,} games kept "
                      f"({elapsed:.1f}s)")

    print(f"\nDone. {len(all_games):,} games tokenized, {skipped:,} skipped.")

    # ── Train / val split ─────────────────────────────────────────────────────
    # Split by game index so no game straddles the boundary.

    n_val = max(1, int(len(all_games) * args.val_fraction))
    n_train = len(all_games) - n_val

    train_games = all_games[:n_train]
    val_games = all_games[n_train:]

    print(f"Train games: {n_train:,}")
    print(f"Val games:   {n_val:,}")

    # ── Concatenate and save ──────────────────────────────────────────────────

    for split_name, games in [("train", train_games), ("val", val_games)]:
        # Flatten list of lists into one array
        tokens = np.concatenate([np.array(g, dtype=np.uint16) for g in games])

        out_path = output_dir / f"{split_name}.bin"
        tokens.tofile(out_path)

        print(f"\n{split_name}.bin:")
        print(f"  {len(tokens):,} tokens")
        print(f"  {out_path.stat().st_size / 1e6:.1f} MB")
        print(f"  Saved to {out_path}")

    # ── Save metadata ─────────────────────────────────────────────────────────

    total_tokens = sum(len(g) for g in all_games)
    train_tokens = sum(len(g) for g in train_games)
    val_tokens = sum(len(g) for g in val_games)

    meta = {
        "vocab_size": tok.vocab_size,
        "block_size": args.block_size,
        "total_games": len(all_games),
        "train_games": n_train,
        "val_games": n_val,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "avg_tokens_per_game": round(total_tokens / len(all_games), 1),
    }

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nMetadata saved to {meta_path}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
