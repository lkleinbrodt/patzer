"""
eval/engine.py

PatzaPlayer: loads a trained GPT checkpoint and generates legal chess moves
using the model's probability distribution with legal-move masking.
Use checkpoints/.../weights_best.pt for best-val weights (eval defaults); ckpt.pt is latest-for-resume.
"""

import random
import sys
from contextlib import nullcontext, redirect_stdout
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.db import player_name

from patzer.checkpoint_util import load_checkpoint
from patzer.model import GPT, GPTConfig
from patzer.tokenizer import ChessTokenizer

# How to choose the result-conditioning token prepended before moves.
# "match_color" = <WHITE_WIN> when playing white, <BLACK_WIN> when black (sensible default)
# "white_win"   = always <WHITE_WIN>
# "black_win"   = always <BLACK_WIN>
# "draw"        = always <DRAW>
# "none"        = no result token (just <GAME_START>)
CONDITIONING_OPTIONS = ("match_color", "white_win", "black_win", "draw", "none")


def _apply_legal_move_mask(logits: torch.Tensor, legal_ids: set[int]) -> torch.Tensor:
    """Restrict logits to legal token ids; all other positions are -inf (1-D logits)."""
    mask = torch.full_like(logits, float("-inf"))
    if not legal_ids:
        return mask
    idx = torch.tensor(tuple(legal_ids), device=logits.device, dtype=torch.long)
    mask[idx] = logits[idx]
    return mask


class _MoveTokenCache:
    """Incremental encode of move_history UCI strings into tokenizer ids (play is append-only)."""

    __slots__ = ("_tokenizer", "_hist_tuple", "_ids")

    def __init__(self, tokenizer: ChessTokenizer):
        self._tokenizer = tokenizer
        self._hist_tuple: tuple[str, ...] = ()
        self._ids: list[int] = []

    def cropped_move_token_ids(self, move_history: list[str], max_move_tokens: int) -> list[int]:
        """Full encode on first use or after history forks; else extend by one move when possible."""
        h = tuple(move_history)
        if h == self._hist_tuple:
            pass
        elif len(h) == len(self._hist_tuple) + 1 and self._hist_tuple == h[:-1]:
            self._ids.append(self._tokenizer.encode(h[-1]))
            self._hist_tuple = h
        else:
            self._ids = [self._tokenizer.encode(m) for m in move_history]
            self._hist_tuple = h
        if len(self._ids) > max_move_tokens:
            return self._ids[-max_move_tokens:]
        return list(self._ids)


class Patzer:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        temperature: float = 0.01,
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
        self._move_tok_cache = _MoveTokenCache(self.tokenizer)
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
        checkpoint = load_checkpoint(path, map_location=self.device)
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
            f"[patzer] loaded {path.name}  ({player_name(path, iter_num)}, conditioning={self.conditioning})",
            file=sys.stderr,
        )
        return model, iter_num

    @property
    def name(self) -> str:
        return player_name(self.checkpoint_path, self.iter_num)

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
        prefix = [self.tokenizer.game_start_id]
        if result_id is not None:
            prefix.append(result_id)
        # Crop move tail to block_size while keeping GAME_START (+ optional result).
        block_size = self.model.config.block_size
        max_move_tokens = block_size - len(prefix)
        move_tokens = self._move_tok_cache.cropped_move_token_ids(move_history, max_move_tokens)
        token_ids = prefix + move_tokens

        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            with self._ctx:
                logits, _ = self.model(x)
        logits = logits[0, -1, :]  # (vocab_size,)

        legal_ids = {self.tokenizer.token_to_id[m] for m in legal_moves}
        logits = _apply_legal_move_mask(logits, legal_ids)

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
    def __init__(
        self,
        binary_path: str,
        depth: int | None = None,
        elo_limit: int | None = None,
        move_time: float = 0.05,
    ):
        import chess.engine
        assert depth is not None or elo_limit is not None, "Provide depth or elo_limit"
        self.depth = depth
        self.elo_limit = elo_limit
        self.move_time = float(move_time)
        self.engine = chess.engine.SimpleEngine.popen_uci(binary_path)
        if elo_limit is not None:
            # Some Stockfish builds enforce a minimum/maximum UCI_Elo. Clamp so callers
            # (e.g. smart ELO sweeps) don't crash when probing outside the supported range.
            self.set_elo_limit(elo_limit)

    def set_elo_limit(self, elo_limit: int) -> None:
        """
        Update Elo-limited strength without restarting the engine process.
        No-op if this player was created with depth-based limits instead.
        """
        if self.depth is not None:
            return
        try:
            opt = self.engine.options.get("UCI_Elo")
            if opt is not None:
                if opt.min is not None:
                    elo_limit = max(int(opt.min), int(elo_limit))
                if opt.max is not None:
                    elo_limit = min(int(opt.max), int(elo_limit))
        except Exception:
            # If option introspection fails, fall back to requested value.
            pass
        self.elo_limit = int(elo_limit)
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": int(elo_limit)})

    @property
    def name(self) -> str:
        if self.elo_limit is not None:
            return f"Stockfish(elo{self.elo_limit})"
        return f"Stockfish(d{self.depth})"

    def get_move(self, board: chess.Board, move_history: list[str]) -> str:
        import chess.engine
        limit = (
            chess.engine.Limit(depth=self.depth)
            if self.elo_limit is None
            else chess.engine.Limit(time=self.move_time)
        )
        result = self.engine.play(board, limit)
        return result.move.uci()

    def close(self):
        self.engine.quit()
