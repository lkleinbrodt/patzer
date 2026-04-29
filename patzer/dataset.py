"""
chess/dataset.py

PyTorch Dataset that memory-maps the binary token files produced by
pipeline/prepare.py.

The dataset treats the entire token sequence as one flat array and returns
(input, target) pairs of length block_size. Input and target are identical
sequences offset by one position — the standard language model setup.

Usage:
    from chess.dataset import ChessDataset

    train_ds = ChessDataset("data/prepared/train.bin", block_size=256)
    val_ds   = ChessDataset("data/prepared/val.bin",   block_size=256)

    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    x, y = next(iter(loader))
    # x.shape == y.shape == (64, 256)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """
    Memory-mapped dataset over a flat binary token file.

    Each item is a (input, target) pair of token ID tensors, both of length
    block_size. Target is input shifted left by one position — at each
    position i, the model is trained to predict token i+1 given tokens 0..i.

    Memory mapping means the OS pages in only the parts of the file that are
    actually accessed. The full file never needs to fit in RAM.
    """

    def __init__(self, bin_path: str | Path, block_size: int = 256):
        """
        Args:
            bin_path:   Path to a .bin file produced by pipeline/prepare.py
            block_size: Number of tokens per training example.
                        Must match the block_size used when training nanoGPT.
        """
        self.bin_path = Path(bin_path)
        self.block_size = block_size

        if not self.bin_path.exists():
            raise FileNotFoundError(
                f"Token file not found: {self.bin_path}\n"
                f"Run pipeline/prepare.py first."
            )

        # Memory-map the file as a read-only uint16 array.
        # np.memmap doesn't load anything into RAM — the OS pages data in
        # on demand as individual indices are accessed.
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode="r")

        # Number of complete (input, target) pairs we can extract.
        # Each pair needs block_size + 1 tokens (input of block_size,
        # target is shifted by 1 so needs one extra token at the end).
        self.n_samples = len(self.data) - block_size

        if self.n_samples <= 0:
            raise ValueError(
                f"File too small for block_size={block_size}. "
                f"File has {len(self.data)} tokens, need at least {block_size + 1}."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (input, target) tensors of shape (block_size,).

        input:  tokens[idx : idx + block_size]
        target: tokens[idx + 1 : idx + block_size + 1]

        Casting from uint16 to int64 because PyTorch embedding layers
        and cross-entropy loss expect int64 (torch.long).
        """
        chunk = self.data[idx : idx + self.block_size + 1]

        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))

        return x, y

    def __repr__(self) -> str:
        return (
            f"ChessDataset("
            f"path={self.bin_path}, "
            f"tokens={len(self.data):,}, "
            f"block_size={self.block_size}, "
            f"samples={self.n_samples:,}"
            f")"
        )


# ── Convenience loader ────────────────────────────────────────────────────────

def load_datasets(
    prepared_dir: str | Path = "data/prepared",
    block_size: int = 256,
) -> tuple[ChessDataset, ChessDataset]:
    """
    Load train and val datasets from a prepared directory.

    Args:
        prepared_dir: Directory containing train.bin, val.bin, meta.json
        block_size:   Token sequence length

    Returns:
        (train_dataset, val_dataset)
    """
    prepared_dir = Path(prepared_dir)
    train_ds = ChessDataset(prepared_dir / "train.bin", block_size=block_size)
    val_ds   = ChessDataset(prepared_dir / "val.bin",   block_size=block_size)
    return train_ds, val_ds


def load_meta(prepared_dir: str | Path = "data/prepared") -> dict:
    """Load metadata saved by pipeline/prepare.py."""
    meta_path = Path(prepared_dir) / "meta.json"
    with open(meta_path) as f:
        return json.load(f)


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    bin_path = sys.argv[1] if len(sys.argv) > 1 else "data/prepared/train.bin"
    block_size = 256

    print(f"Loading dataset from {bin_path}")
    ds = ChessDataset(bin_path, block_size=block_size)
    print(ds)

    # Check a single item
    x, y = ds[0]
    print(f"\nSample 0:")
    print(f"  x shape: {x.shape}, dtype: {x.dtype}")
    print(f"  y shape: {y.shape}, dtype: {y.dtype}")
    print(f"  x[:8]: {x[:8].tolist()}")
    print(f"  y[:8]: {y[:8].tolist()}")
    assert (x[1:] == y[:-1]).all(), "Target should be input shifted by 1"
    print("  ✓ x/y offset relationship correct")

    # Check DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    xb, yb = next(iter(loader))
    print(f"\nBatch from DataLoader:")
    print(f"  xb shape: {xb.shape}")
    print(f"  yb shape: {yb.shape}")
    assert xb.shape == (32, block_size)
    assert yb.shape == (32, block_size)
    print("  ✓ Batch shapes correct")

    print("\nAll checks passed.")
