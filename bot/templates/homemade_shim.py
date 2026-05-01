"""
Thin shim for lichess-bot: copy or symlink this file to lichess-bot/homemade.py

The real PatzerEngine lives in the Patzer repo under bot/lichess_homemade.py.
PATZER_ROOT must point at the Patzer repository root (the directory that contains
eval/, patzer/, checkpoints/, bot/).

deploy_bot.py syncs this file to your lichess-bot checkout automatically.
"""
from __future__ import annotations

import os
import sys

_patzer = os.environ.get("PATZER_ROOT", os.path.expanduser("~/Projects/patzer"))
_patzer = os.path.abspath(_patzer)
if _patzer not in sys.path:
    sys.path.append(_patzer)

from bot.lichess_homemade import PatzerEngine

__all__ = ["PatzerEngine"]
