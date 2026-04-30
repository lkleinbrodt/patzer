"""
eval/play_gui.py — Play a game against Patzer in a pygame window.

Usage:
    python eval/play_gui.py --checkpoint checkpoints/patzer_v1/ckpt_best.pt
    python eval/play_gui.py --checkpoint ... --color black --device cpu
"""

import argparse
import queue
import sys
import threading
from pathlib import Path

import chess
import pygame

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.engine import Patzer

# ── Layout ────────────────────────────────────────────────────────────────────
SQ = 80
MARGIN = 28       # left/bottom space for coordinate labels
STATUS_H = 44
W = MARGIN + SQ * 8
H = SQ * 8 + MARGIN + STATUS_H

# ── Colors ────────────────────────────────────────────────────────────────────
LIGHT        = (240, 217, 181)
DARK         = (181, 136,  99)
SEL_OVERLAY  = ( 20,  85,  30, 170)
DOT_COLOR    = ( 20,  85,  30, 150)
LAST_OVERLAY = (205, 210, 106, 140)
BG           = ( 30,  30,  30)
STATUS_FG    = (220, 220, 220)

UNICODE_PIECES = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}
LETTER_PIECES = {
    (chess.KING,   chess.WHITE): "K", (chess.QUEEN,  chess.WHITE): "Q",
    (chess.ROOK,   chess.WHITE): "R", (chess.BISHOP, chess.WHITE): "B",
    (chess.KNIGHT, chess.WHITE): "N", (chess.PAWN,   chess.WHITE): "P",
    (chess.KING,   chess.BLACK): "k", (chess.QUEEN,  chess.BLACK): "q",
    (chess.ROOK,   chess.BLACK): "r", (chess.BISHOP, chess.BLACK): "b",
    (chess.KNIGHT, chess.BLACK): "n", (chess.PAWN,   chess.BLACK): "p",
}


def _load_piece_font(size: int) -> tuple[pygame.font.Font, dict]:
    for name in ("Apple Symbols", "Arial Unicode MS", "DejaVu Sans", "Segoe UI Symbol"):
        f = pygame.font.SysFont(name, size)
        w = f.render("♔", True, (0, 0, 0)).get_width()
        if w > size // 3:
            return f, UNICODE_PIECES
    return pygame.font.SysFont(None, size), LETTER_PIECES


def _sq_to_px(sq: int, flipped: bool) -> tuple[int, int]:
    col = chess.square_file(sq)
    row = chess.square_rank(sq)
    if flipped:
        col, row = 7 - col, 7 - row
    return MARGIN + col * SQ, (7 - row) * SQ


def _px_to_sq(px: int, py: int, flipped: bool) -> int | None:
    col = (px - MARGIN) // SQ
    row = 7 - (py // SQ)
    if not (0 <= col <= 7 and 0 <= row <= 7):
        return None
    if flipped:
        col, row = 7 - col, 7 - row
    return chess.square(col, row)


def _draw(
    screen: pygame.Surface,
    board: chess.Board,
    piece_font: pygame.font.Font,
    piece_map: dict,
    label_font: pygame.font.Font,
    selected: int | None,
    legal_targets: set[int],
    last_move: chess.Move | None,
    flipped: bool,
) -> None:
    for sq in chess.SQUARES:
        x, y = _sq_to_px(sq, flipped)
        f, r = chess.square_file(sq), chess.square_rank(sq)
        pygame.draw.rect(screen, LIGHT if (f + r) % 2 == 0 else DARK, (x, y, SQ, SQ))

    overlay = pygame.Surface((SQ, SQ), pygame.SRCALPHA)

    if last_move:
        overlay.fill(LAST_OVERLAY)
        for sq in (last_move.from_square, last_move.to_square):
            screen.blit(overlay, _sq_to_px(sq, flipped))

    if selected is not None:
        overlay.fill(SEL_OVERLAY)
        screen.blit(overlay, _sq_to_px(selected, flipped))

    for tgt in legal_targets:
        overlay.fill((0, 0, 0, 0))
        cx, cy = SQ // 2, SQ // 2
        if board.piece_at(tgt):
            pygame.draw.circle(overlay, DOT_COLOR, (cx, cy), SQ // 2 - 3, 6)
        else:
            pygame.draw.circle(overlay, DOT_COLOR, (cx, cy), SQ // 9)
        screen.blit(overlay, _sq_to_px(tgt, flipped))

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        sym = piece_map[(piece.piece_type, piece.color)]
        x, y = _sq_to_px(sq, flipped)
        cx = x + SQ // 2
        cy = y + SQ // 2
        if piece.color == chess.WHITE:
            # thin shadow so white pieces are visible on light squares
            shadow = piece_font.render(sym, True, (80, 60, 40))
            screen.blit(shadow, (cx - shadow.get_width() // 2 + 1, cy - shadow.get_height() // 2 + 1))
            text = piece_font.render(sym, True, (255, 255, 255))
        else:
            text = piece_font.render(sym, True, (15, 15, 15))
        screen.blit(text, (cx - text.get_width() // 2, cy - text.get_height() // 2))

    files = "abcdefgh"
    ranks = "12345678"
    for i in range(8):
        fl = label_font.render(files[i if not flipped else 7 - i], True, STATUS_FG)
        screen.blit(fl, (MARGIN + i * SQ + SQ // 2 - fl.get_width() // 2, SQ * 8 + 5))
        rl = label_font.render(ranks[7 - i if not flipped else i], True, STATUS_FG)
        screen.blit(rl, (4, i * SQ + SQ // 2 - rl.get_height() // 2))


def _game_over_msg(board: chess.Board, human: chess.Color) -> str | None:
    if board.is_checkmate():
        return "Checkmate — you win!" if (not board.turn) == human else "Checkmate — Patzer wins."
    if board.is_stalemate():
        return "Stalemate — draw."
    if board.is_insufficient_material():
        return "Draw (insufficient material)."
    if board.is_seventyfive_moves():
        return "Draw (75-move rule)."
    if board.is_fivefold_repetition():
        return "Draw (fivefold repetition)."
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--color", choices=["white", "black"], default="white")
    ap.add_argument("--temperature", type=float, default=0.1)
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    human = chess.WHITE if args.color == "white" else chess.BLACK
    flipped = human == chess.BLACK

    print(f"Loading Patzer from {args.checkpoint} on {args.device}…", file=sys.stderr)
    patzer = Patzer(args.checkpoint, device=args.device, temperature=args.temperature)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Patzer")
    clock = pygame.time.Clock()

    piece_font, piece_map = _load_piece_font(int(SQ * 0.78))
    label_font = pygame.font.SysFont(None, 20)
    status_font = pygame.font.SysFont(None, 30)

    board = chess.Board()
    move_history: list[str] = []
    selected: int | None = None
    legal_targets: set[int] = set()
    last_move: chess.Move | None = None
    ai_queue: queue.Queue[str] = queue.Queue()
    ai_thinking = False

    def start_ai() -> None:
        nonlocal ai_thinking
        ai_thinking = True
        b_copy = board.copy()
        h_copy = list(move_history)
        threading.Thread(
            target=lambda: ai_queue.put(patzer.get_move(b_copy, h_copy)),
            daemon=True,
        ).start()

    status = "Your turn"
    if human == chess.BLACK:
        start_ai()
        status = "Patzer thinking…"

    running = True
    while running:
        # ── AI result ─────────────────────────────────────────────────────────
        if ai_thinking and not ai_queue.empty():
            uci = ai_queue.get()
            move = chess.Move.from_uci(uci)
            if move in board.legal_moves:
                board.push(move)
                move_history.append(uci)
                last_move = move
            ai_thinking = False
            status = _game_over_msg(board, human) or "Your turn"

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if ai_thinking or _game_over_msg(board, human) or board.turn != human:
                    continue

                sq = _px_to_sq(event.pos[0], event.pos[1], flipped)
                if sq is None:
                    selected, legal_targets = None, set()
                    continue

                if selected is not None and sq in legal_targets:
                    uci = chess.square_name(selected) + chess.square_name(sq)
                    # auto-promote to queen
                    move = chess.Move.from_uci(uci + "q") if chess.Move.from_uci(uci + "q") in board.legal_moves else chess.Move.from_uci(uci)
                    if move in board.legal_moves:
                        board.push(move)
                        move_history.append(move.uci())
                        last_move = move
                        selected, legal_targets = None, set()
                        status = _game_over_msg(board, human) or "Patzer thinking…"
                        if not _game_over_msg(board, human):
                            start_ai()
                elif board.piece_at(sq) and board.piece_at(sq).color == human:
                    selected = sq
                    legal_targets = {m.to_square for m in board.legal_moves if m.from_square == sq}
                else:
                    selected, legal_targets = None, set()

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)
        _draw(screen, board, piece_font, piece_map, label_font,
              selected, legal_targets, last_move, flipped)

        s = status_font.render(status, True, STATUS_FG)
        screen.blit(s, (MARGIN, SQ * 8 + MARGIN + (STATUS_H - s.get_height()) // 2))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
