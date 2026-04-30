"""
eval/uci_engine.py

Minimal UCI wrapper around `eval.engine.Patzer`, so you can use Patzer from any
GUI that supports UCI engines (Cute Chess, Banksia, Arena, etc.).

Example:
  python eval/uci_engine.py --checkpoint checkpoints/patzer_v1/ckpt_best.pt --device mps
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import chess
except ModuleNotFoundError as e:  # pragma: no cover
    if e.name != "chess":
        raise
    raise SystemExit(
        "Missing dependency: python-chess (import name 'chess').\n\n"
        "Install deps in your project venv, e.g.:\n"
        "  uv pip install -r requirements.txt\n"
    ) from e

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.engine import CONDITIONING_OPTIONS, Patzer


def _send(line: str) -> None:
    sys.stdout.write(line.rstrip("\n") + "\n")
    sys.stdout.flush()


def _parse_position(cmd: str) -> tuple[chess.Board, list[str]]:
    """
    Parse a UCI "position ..." command.

    Supports:
      - position startpos [moves ...]
      - position fen <fen...> [moves ...]

    Returns (board, move_history_uci).
    """
    parts = cmd.strip().split()
    if len(parts) < 2 or parts[0] != "position":
        raise ValueError("not a position command")

    idx = 1
    board: chess.Board

    if parts[idx] == "startpos":
        board = chess.Board()
        idx += 1
    elif parts[idx] == "fen":
        idx += 1
        # FEN is 6 fields. Some GUIs may append "moves" after.
        fen_fields = parts[idx : idx + 6]
        if len(fen_fields) < 6:
            raise ValueError("incomplete fen in position command")
        fen = " ".join(fen_fields)
        board = chess.Board(fen)
        idx += 6
    else:
        raise ValueError("position must specify startpos or fen")

    move_history: list[str] = []
    if idx < len(parts) and parts[idx] == "moves":
        for uci in parts[idx + 1 :]:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                # Still apply it (many engines do) to keep protocol moving, but
                # it means downstream behaviour is undefined.
                raise ValueError(f"illegal move in position: {uci}")
            board.push(move)
            move_history.append(uci)

    return board, move_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Patzer UCI engine wrapper")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--device", default="cpu", help="torch device: cpu | mps | cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--conditioning",
        default="match_color",
        choices=CONDITIONING_OPTIONS,
        help="Result token strategy used by Patzer",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).expanduser()
    if not ckpt.exists() or not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found (must be a file): {ckpt}")

    # UCI state
    board = chess.Board()
    move_history: list[str] = []
    patzer: Patzer | None = None

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        cmd = line.strip()
        if not cmd:
            continue

        if cmd == "uci":
            _send("id name Patzer")
            _send("id author patzer")
            _send("uciok")
            continue

        if cmd == "isready":
            if patzer is None:
                patzer = Patzer(
                    ckpt,
                    device=args.device,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    conditioning=args.conditioning,
                )
            _send("readyok")
            continue

        if cmd == "ucinewgame":
            board = chess.Board()
            move_history = []
            continue

        if cmd.startswith("position "):
            try:
                board, move_history = _parse_position(cmd)
            except Exception:
                # Don't crash the GUI; reset position if malformed.
                board = chess.Board()
                move_history = []
            continue

        if cmd.startswith("go"):
            # minimal support: ignore time controls and just move immediately
            if patzer is None:
                patzer = Patzer(
                    ckpt,
                    device=args.device,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    conditioning=args.conditioning,
                )

            if board.is_game_over(claim_draw=True):
                _send("bestmove 0000")
                continue

            uci = patzer.get_move(board, move_history)
            _send(f"bestmove {uci}")
            continue

        if cmd in {"quit", "stop"}:
            # We don't do async search, so stop==quit is fine.
            if cmd == "quit":
                break
            continue

        # Commonly sent but optional UCI commands we safely ignore:
        # - setoption name ...
        # - ponderhit
        # - debug on/off


if __name__ == "__main__":
    main()

