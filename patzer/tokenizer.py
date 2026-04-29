"""
chess/tokenizer.py

Move-level tokenizer for Attenchess.

Vocabulary:
  - All legal UCI moves (e.g. e2e4, g1f3, e7e8q)
  - Special tokens: <PAD>, <GAME_START>, <GAME_END>,
                    <WHITE_WIN>, <BLACK_WIN>, <DRAW>

The vocabulary is built from all possible UCI moves on a standard chess board
(not derived from training data) so it is fixed and complete — no unknown
moves at inference time.

Usage:
    from chess.tokenizer import ChessTokenizer

    tok = ChessTokenizer()
    tok.save("data/vocab.json")

    tok = ChessTokenizer.load("data/vocab.json")

    ids = tok.encode_game(moves=["e2e4", "e7e5"], result="1-0")
    text = tok.decode(ids)
    moves = tok.decode_moves(ids)
"""

import json
from pathlib import Path
from typing import Optional

import chess


# ── Special tokens ────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    "<PAD>",
    "<GAME_START>",
    "<GAME_END>",
    "<WHITE_WIN>",
    "<BLACK_WIN>",
    "<DRAW>",
]

RESULT_TO_TOKEN = {
    "1-0":       "<WHITE_WIN>",
    "0-1":       "<BLACK_WIN>",
    "1/2-1/2":   "<DRAW>",
}

TOKEN_TO_RESULT = {v: k for k, v in RESULT_TO_TOKEN.items()}


# ── Vocabulary generation ─────────────────────────────────────────────────────

def _generate_all_uci_moves() -> list[str]:
    """
    Generate every possible UCI move on a standard chess board.

    This is done by enumerating all (from_square, to_square) pairs plus
    all promotion variants. This gives a fixed, complete vocabulary
    independent of training data.

    Includes:
      - Normal moves:     e2e4, g1f3, ...
      - Promotions:       e7e8q, e7e8r, e7e8b, e7e8n (and all files/colors)

    Does NOT exclude illegal moves — we enumerate squares, not positions,
    so some generated moves won't be reachable in real games. That's fine;
    the vocabulary just needs to contain every move that could legally appear.
    Illegal moves won't appear in our training data, so we don't need to worry about them.
    (we could micro-optimize by filtering them here, but i dont think it's worth the savings at this point.)
    """
    moves = set()
    promotion_pieces = ["q", "r", "b", "n"]

    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue

            move = chess.Move(from_sq, to_sq)
            moves.add(move.uci())

            # Add promotions for pawn moves to back rank
            from_rank = chess.square_rank(from_sq)
            to_rank = chess.square_rank(to_sq)
            from_file = chess.square_file(from_sq)
            to_file = chess.square_file(to_sq)

            # White pawn promotion: rank 6 → rank 7, file diff <= 1
            if from_rank == 6 and to_rank == 7 and abs(from_file - to_file) <= 1:
                for piece in promotion_pieces:
                    moves.add(chess.Move(from_sq, to_sq, promotion=chess.piece_symbol_to_int(piece) if hasattr(chess, 'piece_symbol_to_int') else _piece_char_to_int(piece)).uci())

            # Black pawn promotion: rank 1 → rank 0, file diff <= 1
            if from_rank == 1 and to_rank == 0 and abs(from_file - to_file) <= 1:
                for piece in promotion_pieces:
                    moves.add(chess.Move(from_sq, to_sq, promotion=_piece_char_to_int(piece)).uci())

    return sorted(moves)


def _piece_char_to_int(piece: str) -> int:
    """Convert promotion piece character to python-chess piece type int."""
    return {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}[piece]


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class ChessTokenizer:
    """
    Move-level tokenizer for chess games.

    Maps UCI move strings and special tokens to integer IDs and back.
    Vocabulary is fixed at construction time — no fitting on data needed.
    """

    def __init__(self):
        all_moves = _generate_all_uci_moves()

        # Build vocabulary: special tokens first, then moves sorted
        vocab_tokens = SPECIAL_TOKENS + all_moves

        self.token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(vocab_tokens)}
        self.id_to_token: dict[int, str] = {i: tok for i, tok in enumerate(vocab_tokens)}

        # Cache frequently-used IDs
        self.pad_id         = self.token_to_id["<PAD>"]
        self.game_start_id  = self.token_to_id["<GAME_START>"]
        self.game_end_id    = self.token_to_id["<GAME_END>"]
        self.white_win_id   = self.token_to_id["<WHITE_WIN>"]
        self.black_win_id   = self.token_to_id["<BLACK_WIN>"]
        self.draw_id        = self.token_to_id["<DRAW>"]

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def special_token_ids(self) -> set[int]:
        return {self.token_to_id[t] for t in SPECIAL_TOKENS}

    @property
    def move_token_ids(self) -> set[int]:
        return set(self.token_to_id.values()) - self.special_token_ids

    # ── Core encode / decode ──────────────────────────────────────────────────

    def encode(self, token: str) -> int:
        """Encode a single token string to its integer ID."""
        if token not in self.token_to_id:
            raise ValueError(
                f"Unknown token: '{token}'. "
                f"All UCI moves should be in vocabulary. "
                f"Check that input is a valid UCI move string."
            )
        return self.token_to_id[token]

    def decode(self, token_id: int) -> str:
        """Decode a single integer ID to its token string."""
        if token_id not in self.id_to_token:
            raise ValueError(f"Unknown token ID: {token_id}")
        return self.id_to_token[token_id]

    def encode_sequence(self, tokens: list[str]) -> list[int]:
        """Encode a list of token strings to a list of integer IDs."""
        return [self.encode(t) for t in tokens]

    def decode_sequence(self, token_ids: list[int]) -> list[str]:
        """Decode a list of integer IDs to a list of token strings."""
        return [self.decode(i) for i in token_ids]

    # ── Game-level encode / decode ────────────────────────────────────────────

    def encode_game(
        self,
        moves: list[str],
        result: Optional[str] = None,
        include_start: bool = True,
        include_end: bool = True,
    ) -> list[int]:
        """
        Encode a full game to a sequence of token IDs.

        Format: [<GAME_START>] [<RESULT>] move1 move2 ... [<GAME_END>]

        The result token is prepended (after GAME_START) so the model sees
        the outcome before the moves — this is the conditioning signal.
        Including it at the start lets us later do ELO/result conditioning
        by swapping this token.

        Args:
            moves:         List of UCI move strings e.g. ["e2e4", "e7e5", ...]
            result:        Game result string: "1-0", "0-1", or "1/2-1/2"
                           If None, no result token is added.
            include_start: Whether to prepend <GAME_START>
            include_end:   Whether to append <GAME_END>

        Returns:
            List of integer token IDs.
        """
        tokens = []

        if include_start:
            tokens.append(self.game_start_id)

        if result is not None:
            result_token = RESULT_TO_TOKEN.get(result)
            if result_token is None:
                raise ValueError(
                    f"Unknown result: '{result}'. "
                    f"Expected one of: {list(RESULT_TO_TOKEN.keys())}"
                )
            tokens.append(self.token_to_id[result_token])

        for move in moves:
            tokens.append(self.encode(move))

        if include_end:
            tokens.append(self.game_end_id)

        return tokens

    def decode_moves(self, token_ids: list[int]) -> list[str]:
        """
        Decode a sequence of token IDs to just the UCI move strings,
        stripping all special tokens.
        """
        moves = []
        for tid in token_ids:
            token = self.decode(tid)
            if token not in SPECIAL_TOKENS and token not in TOKEN_TO_RESULT.values():
                moves.append(token)
        return moves

    def decode_game(self, token_ids: list[int]) -> dict:
        """
        Decode a sequence of token IDs to a structured game dict.

        Returns:
            {
                "moves":  ["e2e4", "e7e5", ...],
                "result": "1-0" | "0-1" | "1/2-1/2" | None,
            }
        """
        moves = []
        result = None

        for tid in token_ids:
            token = self.decode(tid)
            if token in TOKEN_TO_RESULT:
                result = TOKEN_TO_RESULT[token]
            elif token not in SPECIAL_TOKENS:
                moves.append(token)

        return {"moves": moves, "result": result}

    # ── Padding ───────────────────────────────────────────────────────────────

    def pad(self, token_ids: list[int], length: int) -> list[int]:
        """Pad a sequence to exactly `length` with <PAD> tokens."""
        if len(token_ids) >= length:
            return token_ids[:length]
        return token_ids + [self.pad_id] * (length - len(token_ids))

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save vocabulary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved vocabulary ({self.vocab_size} tokens) to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ChessTokenizer":
        """Load a previously saved vocabulary from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        tok = cls.__new__(cls)
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = {int(i): t for t, i in tok.token_to_id.items()}

        tok.pad_id        = tok.token_to_id["<PAD>"]
        tok.game_start_id = tok.token_to_id["<GAME_START>"]
        tok.game_end_id   = tok.token_to_id["<GAME_END>"]
        tok.white_win_id  = tok.token_to_id["<WHITE_WIN>"]
        tok.black_win_id  = tok.token_to_id["<BLACK_WIN>"]
        tok.draw_id       = tok.token_to_id["<DRAW>"]

        return tok

    def __repr__(self) -> str:
        return (
            f"ChessTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"special_tokens={SPECIAL_TOKENS}"
            f")"
        )


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building tokenizer...")
    tok = ChessTokenizer()
    print(tok)
    print(f"\nVocab size: {tok.vocab_size}")
    print(f"Special token IDs: { {t: tok.token_to_id[t] for t in SPECIAL_TOKENS} }")

    # Encode/decode a game
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    result = "1-0"

    ids = tok.encode_game(moves, result=result)
    print(f"\nEncoded game ({len(ids)} tokens):")
    print(f"  IDs:    {ids}")
    print(f"  Tokens: {tok.decode_sequence(ids)}")

    decoded = tok.decode_game(ids)
    print(f"\nDecoded game:")
    print(f"  Moves:  {decoded['moves']}")
    print(f"  Result: {decoded['result']}")

    assert decoded["moves"] == moves, "Move round-trip failed"
    assert decoded["result"] == result, "Result round-trip failed"
    print("\n✓ Round-trip encode/decode passed")

    # Test promotion tokens exist
    assert "e7e8q" in tok.token_to_id, "Missing promotion token e7e8q"
    assert "a2a1n" in tok.token_to_id, "Missing promotion token a2a1n"
    print("✓ Promotion tokens present")

    # Test padding
    padded = tok.pad(ids, length=256)
    assert len(padded) == 256
    assert padded[-1] == tok.pad_id
    print("✓ Padding works")

    # Save and reload
    tok.save("data/vocab.json")
    tok2 = ChessTokenizer.load("data/vocab.json")
    assert tok2.vocab_size == tok.vocab_size
    assert tok2.encode_game(moves, result=result) == ids
    print("✓ Save/load round-trip passed")

    print(f"\nAll checks passed. Vocab size: {tok.vocab_size}")
