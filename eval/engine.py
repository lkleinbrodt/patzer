"""
eval/engine.py

PatzaPlayer: loads a trained GPT checkpoint and generates legal chess moves
using the model's probability distribution with legal-move masking.
"""

import random
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from patzer.model import GPT, GPTConfig
from patzer.tokenizer import ChessTokenizer

# How to choose the result-conditioning token prepended before moves.
# "match_color" = <WHITE_WIN> when playing white, <BLACK_WIN> when black (sensible default)
# "white_win"   = always <WHITE_WIN>
# "black_win"   = always <BLACK_WIN>
# "draw"        = always <DRAW>
# "none"        = no result token (just <GAME_START>)
CONDITIONING_OPTIONS = ("match_color", "white_win", "black_win", "draw", "none")


class Patzer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        temperature: float = 0.1,
        top_k: int | None = None,
        conditioning: str = "match_color",
    ):
        assert conditioning in CONDITIONING_OPTIONS, (
            f"conditioning must be one of {CONDITIONING_OPTIONS}"
        )
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.conditioning = conditioning
        self.tokenizer = ChessTokenizer()
        self.model, self.iter_num = self._load_model(Path(checkpoint_path))
        self.checkpoint_path = Path(checkpoint_path)
        self.illegal_move_count = 0

        device_type = "cuda" if "cuda" in device else "cpu"
        self._ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        )

    def _load_model(self, path: Path) -> tuple[GPT, int]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = GPTConfig(**checkpoint["model_args"])
        # IMPORTANT: keep stdout clean for UCI protocol users of this class.
        # `GPT.__init__` prints parameter counts; route that to stderr.
        with redirect_stdout(sys.stderr):
            model = GPT(config)
        state = checkpoint["model"]
        state = {
            (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
            for k, v in state.items()
        }
        model.load_state_dict(state)
        model.eval()
        model.to(self.device)
        iter_num = checkpoint.get("iter_num", 0)
        print(
            f"[patzer] loaded {path.name}  (iter {iter_num}, conditioning={self.conditioning})",
            file=sys.stderr,
        )
        return model, iter_num

    @property
    def name(self) -> str:
        return f"Patzer"

    def _result_token_id(self, board: chess.Board) -> int | None:
        tok = self.tokenizer
        if self.conditioning == "none":
            return None
        if self.conditioning == "white_win":
            return tok.white_win_id
        if self.conditioning == "black_win":
            return tok.black_win_id
        if self.conditioning == "draw":
            return tok.draw_id
        # match_color: condition on winning from our side
        return tok.white_win_id if board.turn == chess.WHITE else tok.black_win_id

    def get_move(self, board: chess.Board, move_history: list[str]) -> str:
        """Return a UCI move string for the current board position."""
        legal_moves = [m.uci() for m in board.legal_moves]
        if not legal_moves:
            raise ValueError("No legal moves available")

        result_id = self._result_token_id(board)
        token_ids = [self.tokenizer.game_start_id]
        if result_id is not None:
            token_ids.append(result_id)
        token_ids += [self.tokenizer.encode(m) for m in move_history]

        # Crop to model block size
        block_size = self.model.config.block_size
        if len(token_ids) > block_size:
            token_ids = token_ids[-block_size:]

        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            with self._ctx:
                logits, _ = self.model(x)
        logits = logits[0, -1, :]  # (vocab_size,)

        # Legal move masking — zero out everything not in the legal set
        legal_ids = {self.tokenizer.token_to_id[m] for m in legal_moves}
        mask = torch.full_like(logits, float("-inf"))
        for tid in legal_ids:
            mask[tid] = logits[tid]
        logits = mask

        if self.top_k is not None:
            v, _ = torch.topk(logits, min(self.top_k, len(legal_ids)))
            logits[logits < v[-1]] = float("-inf")

        if self.temperature == 0:
            chosen_id = torch.argmax(logits).item()
        else:
            logits = logits / self.temperature
            probs = F.softmax(logits, dim=-1)

            if torch.isnan(probs).any() or probs.sum() < 1e-6:
                self.illegal_move_count += 1
                return random.choice(legal_moves)

            chosen_id = torch.multinomial(probs, num_samples=1).item()
        return self.tokenizer.decode(chosen_id)


class StockfishPlayer:
    def __init__(self, binary_path: str, depth: int | None = None, elo_limit: int | None = None):
        import chess.engine
        assert depth is not None or elo_limit is not None, "Provide depth or elo_limit"
        self.depth = depth
        self.elo_limit = elo_limit
        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)
        if elo_limit is not None:
            self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo_limit})

    @property
    def name(self) -> str:
        if self.elo_limit is not None:
            return f"Stockfish(elo{self.elo_limit})"
        return f"Stockfish(d{self.depth})"

    def get_move(self, board: chess.Board, move_history: list[str]) -> str:
        import chess.engine
        limit = chess.engine.Limit(depth=self.depth) if self.elo_limit is None else chess.engine.Limit(time=0.1)
        result = self.engine.play(board, limit)
        return result.move.uci()

    def close(self):
        self.engine.quit()
