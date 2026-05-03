"""
Split eval view: scrollable log (left) + live board (right). No input loop.

Used by evaluate.py --visualize. Closing the window raises KeyboardInterrupt.
"""

from __future__ import annotations

import chess
from collections import deque

SQ = 52
MARGIN = 20
STATUS_H = 48
LOG_PANEL_W = 460
GUTTER = 12
BOARD_LEFT = LOG_PANEL_W + GUTTER
BOARD_AREA_W = MARGIN + SQ * 8 + 16
TOTAL_W = BOARD_LEFT + BOARD_AREA_W
H = SQ * 8 + MARGIN + STATUS_H

LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
LAST_OVERLAY = (205, 210, 106, 140)
BOARD_BG = (28, 28, 32)
LOG_BG = (14, 16, 20)
DIVIDER = (55, 58, 68)
STATUS_FG = (210, 212, 220)

# Log line tones (terminal-ish)
_TONES = {
    "normal": (205, 210, 220),
    "dim": (118, 124, 136),
    "accent": (110, 190, 255),
    "good": (110, 215, 150),
    "bad": (255, 115, 105),
    "warn": (235, 195, 95),
}

UNICODE_PIECES = {
    (chess.KING, chess.WHITE): "♔",
    (chess.QUEEN, chess.WHITE): "♕",
    (chess.ROOK, chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN, chess.WHITE): "♙",
    (chess.KING, chess.BLACK): "♚",
    (chess.QUEEN, chess.BLACK): "♛",
    (chess.ROOK, chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN, chess.BLACK): "♟",
}
LETTER_PIECES = {
    (chess.KING, chess.WHITE): "K",
    (chess.QUEEN, chess.WHITE): "Q",
    (chess.ROOK, chess.WHITE): "R",
    (chess.BISHOP, chess.WHITE): "B",
    (chess.KNIGHT, chess.WHITE): "N",
    (chess.PAWN, chess.WHITE): "P",
    (chess.KING, chess.BLACK): "k",
    (chess.QUEEN, chess.BLACK): "q",
    (chess.ROOK, chess.BLACK): "r",
    (chess.BISHOP, chess.BLACK): "b",
    (chess.KNIGHT, chess.BLACK): "n",
    (chess.PAWN, chess.BLACK): "p",
}


def _sq_to_px(sq: int, board_left: int) -> tuple[int, int]:
    col = chess.square_file(sq)
    row = chess.square_rank(sq)
    return board_left + MARGIN + col * SQ, (7 - row) * SQ


def _load_piece_font(pg, size: int) -> tuple:
    for name in ("Apple Symbols", "Arial Unicode MS", "DejaVu Sans", "Segoe UI Symbol"):
        f = pg.font.SysFont(name, size)
        w = f.render("♔", True, (0, 0, 0)).get_width()
        if w > size // 3:
            return f, UNICODE_PIECES
    return pg.font.SysFont(None, size), LETTER_PIECES


def _load_log_font(pg):
    for name in ("Menlo", "Monaco", "Consolas", "Courier New", "Courier"):
        try:
            f = pg.font.SysFont(name, 13)
            if f:
                return f
        except Exception:
            pass
    return pg.font.SysFont(None, 13)


def _truncate_to_width(font, text: str, max_px: int) -> str:
    if font.size(text)[0] <= max_px:
        return text
    ell = "…"
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        t = text[:mid] + ell
        if font.size(t)[0] <= max_px:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + ell if lo > 0 else ell


def _wrap_line(font, text: str, max_px: int) -> list[str]:
    """Word-wrap to pixel width; single long tokens are truncated."""
    if font.size(text)[0] <= max_px:
        return [text]
    words = text.split()
    if not words:
        return [_truncate_to_width(font, text, max_px)]
    lines: list[str] = []
    cur = words[0]
    for w in words[1:]:
        trial = cur + " " + w
        if font.size(trial)[0] <= max_px:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    out: list[str] = []
    for ln in lines:
        if font.size(ln)[0] <= max_px:
            out.append(ln)
        else:
            out.append(_truncate_to_width(font, ln, max_px))
    return out


def _draw_board(
    pg,
    screen,
    board: chess.Board,
    piece_font,
    piece_map,
    label_font,
    last_move: chess.Move | None,
    board_left: int,
) -> None:
    for sq in chess.SQUARES:
        x, y = _sq_to_px(sq, board_left)
        f, r = chess.square_file(sq), chess.square_rank(sq)
        pg.draw.rect(screen, LIGHT if (f + r) % 2 == 0 else DARK, (x, y, SQ, SQ))

    overlay = pg.Surface((SQ, SQ), pg.SRCALPHA)
    if last_move:
        overlay.fill(LAST_OVERLAY)
        for sq in (last_move.from_square, last_move.to_square):
            screen.blit(overlay, _sq_to_px(sq, board_left))

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        sym = piece_map[(piece.piece_type, piece.color)]
        x, y = _sq_to_px(sq, board_left)
        cx = x + SQ // 2
        cy = y + SQ // 2
        if piece.color == chess.WHITE:
            shadow = piece_font.render(sym, True, (80, 60, 40))
            screen.blit(shadow, (cx - shadow.get_width() // 2 + 1, cy - shadow.get_height() // 2 + 1))
            text = piece_font.render(sym, True, (255, 255, 255))
        else:
            text = piece_font.render(sym, True, (15, 15, 15))
        screen.blit(text, (cx - text.get_width() // 2, cy - text.get_height() // 2))

    files = "abcdefgh"
    ranks = "12345678"
    for i in range(8):
        fl = label_font.render(files[i], True, STATUS_FG)
        screen.blit(fl, (board_left + MARGIN + i * SQ + SQ // 2 - fl.get_width() // 2, SQ * 8 + 4))
        rl = label_font.render(ranks[7 - i], True, STATUS_FG)
        screen.blit(rl, (board_left + 4, i * SQ + SQ // 2 - rl.get_height() // 2))


class EvalBoardViewer:
    """Log panel + board; redraw on each ply and after each log line."""

    __slots__ = (
        "_pg",
        "_screen",
        "_piece_font",
        "_piece_map",
        "_label_font",
        "_status_font",
        "_log_font",
        "_lines",
        "_white",
        "_black",
        "_last_board",
        "_last_move",
    )

    def __init__(self) -> None:
        try:
            import pygame
        except ImportError as e:
            raise SystemExit(
                "pygame is required for --visualize (uv pip install pygame)"
            ) from e
        self._pg = pygame
        pygame.init()
        self._screen = pygame.display.set_mode((TOTAL_W, H))
        pygame.display.set_caption("Patzer eval")
        self._piece_font, self._piece_map = _load_piece_font(pygame, int(SQ * 0.78))
        self._label_font = pygame.font.SysFont(None, 17)
        self._status_font = pygame.font.SysFont(None, 20)
        self._log_font = _load_log_font(pygame)
        self._lines: deque[tuple[str, str]] = deque(maxlen=700)
        self._white = "White"
        self._black = "Black"
        self._last_board: chess.Board | None = None
        self._last_move: chess.Move | None = None

    def set_players(self, white: str, black: str) -> None:
        self._white = white
        self._black = black

    def log(self, text: str, tone: str = "normal") -> None:
        """Append log lines (use tone: normal|dim|accent|good|bad|warn). Refreshes display."""
        if not text:
            return
        for raw in text.split("\n"):
            self._lines.append((raw.rstrip("\r"), tone))
        self._redraw()

    def clear_log(self) -> None:
        self._lines.clear()
        self._redraw()

    def _pump(self) -> None:
        for event in self._pg.event.get():
            if event.type == self._pg.QUIT:
                raise KeyboardInterrupt

    def _draw_log_panel(self) -> None:
        pad = 8
        max_w = LOG_PANEL_W - 2 * pad
        line_h = self._log_font.get_height() + 2
        avail_h = H - 2 * pad
        max_lines = max(1, avail_h // line_h)

        self._pg.draw.rect(self._screen, LOG_BG, (0, 0, LOG_PANEL_W, H))
        # Header strip
        hdr = self._log_font.render("patzer eval — log", True, _TONES["accent"])
        self._screen.blit(hdr, (pad, pad))

        y = pad + line_h + 2
        entries: list[tuple[str, str]] = []
        for text, tone in list(self._lines)[-200:]:
            for sub in _wrap_line(self._log_font, text, max_w):
                entries.append((sub, tone))
        visible = entries[-max_lines:]
        for text, tone in visible:
            color = _TONES.get(tone, _TONES["normal"])
            surf = self._log_font.render(text, True, color)
            self._screen.blit(surf, (pad, y))
            y += line_h

    def _redraw(self) -> None:
        self._pump()
        self._pg.display.set_caption(f"Patzer eval — {self._white} vs {self._black}")

        self._draw_log_panel()
        self._pg.draw.line(self._screen, DIVIDER, (LOG_PANEL_W, 0), (LOG_PANEL_W, H), 1)

        bx0 = BOARD_LEFT
        self._pg.draw.rect(self._screen, BOARD_BG, (bx0, 0, TOTAL_W - bx0, H))

        if self._last_board is not None:
            _draw_board(
                self._pg,
                self._screen,
                self._last_board,
                self._piece_font,
                self._piece_map,
                self._label_font,
                self._last_move,
                bx0,
            )
            side = "White" if self._last_board.turn == chess.WHITE else "Black"
            wn = self._white if len(self._white) <= 24 else self._white[:21] + "…"
            bn = self._black if len(self._black) <= 24 else self._black[:21] + "…"
            line = f"{wn}  vs  {bn}   ·   {side} to move   ·   ply {len(self._last_board.move_stack)}"
            s = self._status_font.render(line, True, STATUS_FG)
            max_sw = BOARD_AREA_W - 2 * MARGIN
            if s.get_width() > max_sw:
                line = f"{side} to move  ·  ply {len(self._last_board.move_stack)}"
                s = self._status_font.render(line, True, STATUS_FG)
            sx = bx0 + MARGIN
            self._screen.blit(s, (sx, SQ * 8 + MARGIN + (STATUS_H - s.get_height()) // 2))
        else:
            hint = self._status_font.render("Waiting for first move…", True, _TONES["dim"])
            self._screen.blit(hint, (bx0 + MARGIN, H // 2 - hint.get_height() // 2))

        self._pg.display.flip()

    def ply(self, board: chess.Board, last_move: chess.Move) -> None:
        self._last_board = board.copy()
        self._last_move = last_move
        self._redraw()

    def close(self) -> None:
        self._pg.quit()
