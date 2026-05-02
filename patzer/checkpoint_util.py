"""Shared checkpoint I/O for train, sample, and tooling."""

from __future__ import annotations

import torch


def load_checkpoint(path, map_location=None):
    """
    Load a training/play checkpoint (.pt).

    Prefer weights_only=True (PyTorch 2.6+ security default); fall back if the
    file or torch version cannot load it (legacy pickles, extra types).
    """
    from model import GPTConfig
    from torch.serialization import add_safe_globals

    add_safe_globals([GPTConfig])
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)
