"""
Homemade engine for lichess-bot: loads Patzer from checkpoints.

Import only from a running lichess-bot process (so `lib.engine_wrapper` exists),
after the Patzer repo root is on sys.path and PATZER_ROOT is set (see bot/templates/homemade_shim.py).
"""

from __future__ import annotations

import logging
import os
import random
import time
import sys
from pathlib import Path

import chess
from chess.engine import PlayResult

from lib.engine_wrapper import MinimalEngine

logger = logging.getLogger(__name__)


def _patzer_repo_root() -> Path:
    root = os.environ.get("PATZER_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[1]


def _resolve_checkpoint(raw: str | None) -> Path:
    if not raw:
        raise ValueError(
            "Set engine.homemade_options.patzer_checkpoint in your lichess-bot config "
            "(path to weights_best.pt or ckpt.pt, relative to PATZER_ROOT or absolute)."
        )
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = _patzer_repo_root() / p
    return p.resolve()


def _pick_device(preference: str | None) -> str:
    pref = (preference or "auto").strip().lower()
    if pref not in ("", "auto"):
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _opt_str(options: dict, key: str) -> str | None:
    v = options.get(key)
    if v is None:
        return None
    return str(v)


def _opt_float(options: dict, key: str) -> float | None:
    v = options.get(key)
    if v is None:
        return None
    return float(v)


class PatzerEngine(MinimalEngine):
    """Patzer GPT — checkpoint and device from engine.homemade_options."""

    def __init__(self, commands, options, stderr, draw_or_resign, game, debug, **kwargs):
        super().__init__(commands, options, stderr, draw_or_resign, game, debug, **kwargs)
        repo = _patzer_repo_root()
        rp = str(repo)
        if rp not in sys.path:
            sys.path.insert(0, rp)

        from eval.engine import Patzer

        ckpt_path = _resolve_checkpoint(_opt_str(options, "patzer_checkpoint"))
        device = _pick_device(_opt_str(options, "device"))
        temperature = float(_opt_str(options, "temperature") or "0")
        conditioning = _opt_str(options, "conditioning") or "match_color"
        # Default to a tiny delay to prevent bursty move submissions triggering 429s
        # (especially in bullet with high concurrency and near-instant engines).
        self.min_think_ms = int(_opt_float(options, "min_think_ms") or 50)
        self.think_jitter_ms = int(_opt_float(options, "think_jitter_ms") or 50)

        logger.info("PatzerEngine loading checkpoint=%s device=%s", ckpt_path, device)
        self.patzer = Patzer(
            checkpoint_path=ckpt_path,
            device=device,
            temperature=temperature,
            top_k=None,
            conditioning=conditioning,
        )

    def search(self, board: chess.Board, *args) -> PlayResult:
        if self.min_think_ms > 0:
            jitter = random.randint(0, max(0, self.think_jitter_ms))
            time.sleep((self.min_think_ms + jitter) / 1000.0)
        move_history = [m.uci() for m in board.move_stack]
        uci = self.patzer.get_move(board, move_history)
        return PlayResult(chess.Move.from_uci(uci), None)
