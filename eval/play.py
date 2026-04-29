"""
eval/play.py

Play a simple human-vs-Patzer game in the terminal.

Examples:
  python eval/play.py --checkpoint checkpoints/patzer_v0/ckpt.pt --human-color white --device mps
  python eval/play.py --checkpoint checkpoints/patzer_v0/ckpt.pt --human-color black --temperature 0.8 --top-k 50
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import chess
    import chess.pgn
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


def _print_board(board: chess.Board) -> None:
    # python-chess has a clean text board; unicode is nicer if terminal supports it.
    try:
        print(board.unicode(invert_color=True))
    except Exception:
        print(board)


def _format_legal_hint(board: chess.Board, limit: int = 20) -> str:
    moves = list(board.legal_moves)
    if not moves:
        return "(no legal moves)"
    sample = " ".join(m.uci() for m in moves[:limit])
    if len(moves) > limit:
        sample += f" … (+{len(moves) - limit} more)"
    return sample


def _read_human_move(board: chess.Board) -> chess.Move:
    while True:
        raw = input("Your move (UCI like e2e4 or SAN like Nf3; 'help'): ").strip()
        if not raw:
            continue
        cmd = raw.lower()
        if cmd in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        if cmd in {"help", "h", "?"}:
            print(
                "Commands:\n"
                "  - enter a move in UCI (e2e4, e7e8q) or SAN (Nf3, O-O)\n"
                "  - 'moves' to show a sample of legal UCI moves\n"
                "  - 'fen' to print current FEN\n"
                "  - 'quit' to exit\n"
            )
            continue
        if cmd == "fen":
            print(board.fen())
            continue
        if cmd == "moves":
            print(_format_legal_hint(board))
            continue

        # Prefer SAN parsing (lets the user type O-O, exd5, etc.)
        try:
            move = board.parse_san(raw)
            return move
        except ValueError:
            pass

        # Fall back to UCI
        try:
            move = chess.Move.from_uci(raw)
        except ValueError:
            print("Couldn't parse that move. Try SAN (e.g. Nf3) or UCI (e.g. g1f3).")
            continue

        if move not in board.legal_moves:
            print(f"Illegal move for this position: {move.uci()}")
            print(f"Legal move sample: {_format_legal_hint(board)}")
            continue
        return move


@dataclass
class _Human:
    name: str = "Human"

    def get_move(self, board: chess.Board, move_history: list[str]) -> str:
        return _read_human_move(board).uci()


def play_game(
    *,
    checkpoint: Path,
    human_color: str,
    device: str,
    temperature: float,
    top_k: int | None,
    conditioning: str,
    fen: str | None,
    max_plies: int,
) -> None:
    board = chess.Board(fen) if fen else chess.Board()
    move_history: list[str] = []

    patzer = Patzer(
        checkpoint,
        device=device,
        temperature=temperature,
        top_k=top_k,
        conditioning=conditioning,
    )
    human = _Human()
    human_is_white = human_color.lower().startswith("w")

    print()
    print(f"White: {'Human' if human_is_white else 'Patzer'}")
    print(f"Black: {'Patzer' if human_is_white else 'Human'}")
    if fen:
        print(f"Start position (FEN): {fen}")
    print()

    try:
        while not board.is_game_over(claim_draw=True) and len(move_history) < max_plies:
            _print_board(board)
            print()
            side = "White" if board.turn == chess.WHITE else "Black"
            print(f"{side} to move. Fullmove={board.fullmove_number}")

            human_to_play = (board.turn == chess.WHITE) == human_is_white
            if human_to_play:
                move = _read_human_move(board)
                print(f"  {human.name}: {board.san(move)} ({move.uci()})")
            else:
                uci = patzer.get_move(board, move_history)
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    # Shouldn't happen due to legal masking, but keep it safe.
                    move = next(iter(board.legal_moves))
                print(f"  {patzer.name}: {board.san(move)} ({move.uci()})")

            board.push(move)
            move_history.append(move.uci())
            print()

    except KeyboardInterrupt:
        print("\nExiting game.")
        return

    _print_board(board)
    print()

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        print("Result: 1/2-1/2 (move limit)")
    else:
        print(f"Result: {outcome.result()}  ({outcome.termination.name})")

    # Print a small PGN for copy/paste.
    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "Human" if human_is_white else "Patzer"
    game.headers["Black"] = "Patzer" if human_is_white else "Human"
    game.headers["Result"] = outcome.result() if outcome else "1/2-1/2"
    print("\nPGN:\n")
    print(game, end="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against Patzer (terminal game)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--human-color", default="white", choices=["white", "black"])
    parser.add_argument("--device", default="cpu", help="torch device: cpu | mps | cuda")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--conditioning",
        default="match_color",
        choices=CONDITIONING_OPTIONS,
        help="Result token strategy (same as eval/tournament.py)",
    )
    parser.add_argument("--fen", default=None, help="Optional starting FEN")
    parser.add_argument("--max-plies", type=int, default=600, help="Hard move limit (plies)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint).expanduser()
    if not ckpt.exists() or not ckpt.is_file():
        print(f"Checkpoint not found (must be a file): {ckpt}", file=sys.stderr)
        sys.exit(2)

    play_game(
        checkpoint=ckpt,
        human_color=args.human_color,
        device=args.device,
        temperature=args.temperature,
        top_k=args.top_k,
        conditioning=args.conditioning,
        fen=args.fen,
        max_plies=args.max_plies,
    )


if __name__ == "__main__":
    main()

