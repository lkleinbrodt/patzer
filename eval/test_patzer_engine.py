"""
Regression tests for eval/engine.py: legal-move masking and move-token cache.

Run from repository root:
  python -m unittest eval/test_patzer_engine.py -v
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.engine import (  # noqa: E402
    _MoveTokenCache,
    _apply_legal_move_mask,
    Patzer,
)
from patzer.tokenizer import ChessTokenizer  # noqa: E402


def _apply_legal_move_mask_reference(logits: torch.Tensor, legal_ids: set[int]) -> torch.Tensor:
    """Original per-id loop (kept here for regression only)."""
    mask = torch.full_like(logits, float("-inf"))
    for tid in legal_ids:
        mask[tid] = logits[tid]
    return mask


def _ref_cropped_move_tokens(move_history: list[str], tokenizer: ChessTokenizer, max_move_tokens: int) -> list[int]:
    ids = [tokenizer.encode(m) for m in move_history]
    if len(ids) > max_move_tokens:
        return ids[-max_move_tokens:]
    return ids


def _apply_top_k(logits: torch.Tensor, legal_count: int, top_k: int | None) -> torch.Tensor:
    if top_k is None:
        return logits
    k = min(top_k, legal_count)
    v, _ = torch.topk(logits, k)
    logits = logits.clone()
    logits[logits < v[-1]] = float("-inf")
    return logits


class _LogitsStub:
    """Minimal model: returns fixed last-step logits (ignores token ids)."""

    def __init__(self, vocab_size: int, logits1d: torch.Tensor):
        self.config = self
        self.vocab_size = vocab_size
        self.block_size = 4096
        self._logits1d = logits1d

    def __call__(self, idx: torch.Tensor):
        b, t, = idx.shape
        out = torch.empty(b, t, self.vocab_size, device=idx.device, dtype=self._logits1d.dtype)
        out[..., :] = float("-inf")
        out[:, -1, :] = self._logits1d.to(device=idx.device, dtype=out.dtype)
        return out, None


class TestLegalMoveMask(unittest.TestCase):
    def test_mask_matches_reference_random_cpu(self):
        g = torch.Generator()
        for trial in range(30):
            g.manual_seed(1000 + trial)
            vocab = 200 + trial * 17
            logits = torch.randn(vocab, generator=g)
            n_legal = min(vocab, 3 + (trial % 40))
            legal_ids = set(random.sample(range(vocab), n_legal))
            a = _apply_legal_move_mask(logits, legal_ids)
            b = _apply_legal_move_mask_reference(logits, legal_ids)
            self.assertTrue(torch.equal(a, b), msg=f"trial {trial}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_mask_matches_reference_cuda(self):
        logits_cpu = torch.randn(500)
        legal_ids = set(random.sample(range(500), 50))
        logits = logits_cpu.cuda()
        a = _apply_legal_move_mask(logits, legal_ids)
        b = _apply_legal_move_mask_reference(logits, legal_ids)
        self.assertTrue(torch.equal(a, b))

    def test_top_k_pipeline_matches_reference(self):
        g = torch.Generator().manual_seed(42)
        logits = torch.randn(300, generator=g)
        legal_ids = set(random.sample(range(300), 25))
        for top_k in (None, 3, 10, 25, 100):
            a0 = _apply_legal_move_mask(logits, legal_ids)
            b0 = _apply_legal_move_mask_reference(logits, legal_ids)
            a = _apply_top_k(a0.clone(), len(legal_ids), top_k)
            b = _apply_top_k(b0.clone(), len(legal_ids), top_k)
            self.assertTrue(torch.equal(a, b))


class TestMoveTokenCache(unittest.TestCase):
    def test_incremental_matches_full_encode_many_games(self):
        tok = ChessTokenizer()
        block_size = 256
        random.seed(2026)
        for game_i in range(15):
            board = chess.Board()
            hist: list[str] = []
            cache = _MoveTokenCache(tok)
            while not board.is_game_over(claim_draw=True) and len(hist) < 120:
                uci = random.choice([m.uci() for m in board.legal_moves])
                board.push_uci(uci)
                hist.append(uci)
                for prefix_len in (1, 2):
                    max_mv = block_size - prefix_len
                    got = cache.cropped_move_token_ids(hist, max_mv)
                    want = _ref_cropped_move_tokens(hist, tok, max_mv)
                    self.assertEqual(
                        got,
                        want,
                        msg=f"game {game_i} len={len(hist)} prefix_len={prefix_len}",
                    )

    def test_fork_history_rebuilds(self):
        tok = ChessTokenizer()
        cache = _MoveTokenCache(tok)
        cache.cropped_move_token_ids(["e2e4", "e7e5"], 500)
        # Non-extension: different line — must match full encode from scratch
        got = cache.cropped_move_token_ids(["d2d4"], 500)
        want = _ref_cropped_move_tokens(["d2d4"], tok, 500)
        self.assertEqual(got, want)


class TestPatzerGetMoveRegression(unittest.TestCase):
    def _make_patzer_stub(self, logits1d: torch.Tensor, **kwargs):
        def fake_load(patzer_self, path: Path):
            model = _LogitsStub(patzer_self.tokenizer.vocab_size, logits1d)
            return model, 0

        with patch.object(Patzer, "_load_model", fake_load):
            return Patzer("dummy.pt", device="cpu", **kwargs)

    def test_greedy_same_as_reference_pipeline(self):
        tok = ChessTokenizer()
        vocab = tok.vocab_size
        g = torch.Generator().manual_seed(777)
        logits1d = torch.randn(vocab, generator=g)
        board = chess.Board()
        hist: list[str] = []
        for _ in range(25):
            legal_uci = [m.uci() for m in board.legal_moves]
            legal_ids = {tok.token_to_id[u] for u in legal_uci}
            logits = logits1d.clone()
            masked = _apply_legal_move_mask(logits, legal_ids)
            expect_id = torch.argmax(masked).item()
            expect_uci = tok.decode(expect_id)

            p = self._make_patzer_stub(logits1d, temperature=0.0, top_k=None)
            got = p.get_move(board, hist)
            self.assertEqual(got, expect_uci)

            board.push_uci(got)
            hist.append(got)

    def test_temperature_sample_matches_reference(self):
        tok = ChessTokenizer()
        vocab = tok.vocab_size
        torch.manual_seed(99)
        logits1d = torch.randn(vocab)
        temperature = 0.7
        top_k = 8

        board = chess.Board()
        hist: list[str] = []
        p = self._make_patzer_stub(logits1d, temperature=temperature, top_k=top_k)

        for _ in range(12):
            legal_uci = [m.uci() for m in board.legal_moves]
            legal_ids = {tok.token_to_id[u] for u in legal_uci}
            logits = logits1d.clone()
            logits = _apply_legal_move_mask(logits, legal_ids)
            logits = _apply_top_k(logits, len(legal_ids), top_k)
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            torch.manual_seed(12345 + len(hist))
            ref_id = torch.multinomial(probs, 1).item()
            ref_uci = tok.decode(ref_id)

            torch.manual_seed(12345 + len(hist))
            got = p.get_move(board, hist)
            self.assertEqual(got, ref_uci)
            board.push_uci(got)
            hist.append(got)


if __name__ == "__main__":
    unittest.main()
