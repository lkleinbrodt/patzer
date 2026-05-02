"""
Sample legal chess moves from a trained Patzer checkpoint (next-token prediction).

Run from the `patzer/` directory (same as train.py) so configurator + imports work.

Example:
  python sample.py --out_dir=checkpoints/patzer_v3 --num_samples=3 --max_new_tokens=80
  python sample.py --out_dir=checkpoints/patzer_v3 --start='e2e4 e7e5 g1f3' --conditioning=match_color
"""
from __future__ import annotations

import os
import random
from contextlib import nullcontext
from pathlib import Path

import chess
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
from tokenizer import ChessTokenizer

# -----------------------------------------------------------------------------
init_from = "resume"
out_dir = "checkpoints/patzer_v3"
# Space-separated UCI opening moves, or "FILE:moves.txt" (one line of moves).
start = ""
num_samples = 3
max_new_tokens = 80
temperature = 0.8
top_k = 200  # None to disable
seed = 1337
device = "auto"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = True
# match_color | white_win | black_win | draw | none — same semantics as eval/engine.Patzer
conditioning = "match_color"
exec(open("configurator.py").read())
# -----------------------------------------------------------------------------

CONDITIONING_OPTIONS = ("match_color", "white_win", "black_win", "draw", "none")


def _parse_start_moves(start: str) -> list[str]:
    if not start.strip():
        return []
    if start.startswith("FILE:"):
        path = Path(start[5:].strip())
        line = path.read_text(encoding="utf-8").strip().splitlines()[0]
        return line.split()
    return start.split()


def _result_token_id(tok: ChessTokenizer, board: chess.Board) -> int | None:
    if conditioning == "none":
        return None
    if conditioning == "white_win":
        return tok.white_win_id
    if conditioning == "black_win":
        return tok.black_win_id
    if conditioning == "draw":
        return tok.draw_id
    if conditioning == "match_color":
        return tok.white_win_id if board.turn == chess.WHITE else tok.black_win_id
    raise ValueError(f"conditioning must be one of {CONDITIONING_OPTIONS}, got {conditioning!r}")


def _build_token_ids(tok: ChessTokenizer, board: chess.Board, move_history: list[str]) -> list[int]:
    rid = _result_token_id(tok, board)
    ids = [tok.game_start_id]
    if rid is not None:
        ids.append(rid)
    ids.extend(tok.encode(m) for m in move_history)
    return ids


def _crop_sequence(token_ids: list[int], block_size: int, tok: ChessTokenizer) -> list[int]:
    if len(token_ids) <= block_size:
        return token_ids
    if len(token_ids) >= 2 and token_ids[0] == tok.game_start_id:
        tail_budget = block_size - 2
        if tail_budget < 1:
            return token_ids[:block_size]
        tail = token_ids[2:][-tail_budget:]
        return token_ids[:2] + tail
    return token_ids[-block_size:]


def main() -> None:
    if conditioning not in CONDITIONING_OPTIONS:
        raise ValueError(f"conditioning must be one of {CONDITIONING_OPTIONS}, got {conditioning!r}")

    if device == "auto":
        if torch.cuda.is_available():
            device_t = "cuda"
            compile_effective = compile
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_t = "mps"
            compile_effective = False
        else:
            device_t = "cpu"
            compile_effective = False
    else:
        device_t = device
        compile_effective = compile and "cuda" in device_t

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = "cuda" if "cuda" in device_t else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    tok = ChessTokenizer()

    if init_from != "resume":
        raise SystemExit("Patzer sample.py only supports init_from='resume' (checkpoint in out_dir).")

    ckpt_best = os.path.join(out_dir, "weights_best.pt")
    ckpt_fallback = os.path.join(out_dir, "ckpt.pt")
    ckpt_path = ckpt_best if os.path.exists(ckpt_best) else ckpt_fallback
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"No checkpoint at {ckpt_best} or {ckpt_fallback}")

    checkpoint = torch.load(ckpt_path, map_location=device_t, weights_only=False)
    ma = checkpoint["model_args"]
    if ma.get("vocab_size") != tok.vocab_size:
        raise SystemExit(
            f"vocab_size mismatch: checkpoint {ma.get('vocab_size')} vs tokenizer {tok.vocab_size}"
        )

    gptconf = GPTConfig(**ma)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device_t)
    if compile_effective:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    block_size = model.config.block_size
    opening_moves = _parse_start_moves(start)

    for sample_i in range(num_samples):
        board = chess.Board()
        move_history: list[str] = []
        for uci in opening_moves:
            m = chess.Move.from_uci(uci)
            if m not in board.legal_moves:
                raise ValueError(f"Illegal opening move {uci!r} in start=")
            board.push(m)
            move_history.append(uci)

        for _ in range(max_new_tokens):
            if board.is_game_over(claim_draw=True):
                break
            legal = [x.uci() for x in board.legal_moves]
            if not legal:
                break

            token_ids = _build_token_ids(tok, board, move_history)
            token_ids = _crop_sequence(token_ids, block_size, tok)
            x = torch.tensor([token_ids], dtype=torch.long, device=device_t)

            with torch.no_grad():
                with ctx:
                    logits, _ = model(x)
            logits = logits[0, -1, :].clone()

            legal_ids = {tok.token_to_id[m] for m in legal}
            mask = torch.full_like(logits, float("-inf"))
            for tid in legal_ids:
                mask[tid] = logits[tid]
            logits = mask

            if top_k is not None and top_k > 0:
                k = min(top_k, len(legal_ids))
                v, _ = torch.topk(logits, k)
                logits[logits < v[-1]] = float("-inf")

            if temperature == 0.0:
                chosen_id = torch.argmax(logits).item()
                uci = tok.decode(chosen_id)
            else:
                logits_scaled = logits / temperature
                probs = F.softmax(logits_scaled, dim=-1)
                if torch.isnan(probs).any() or probs.sum() < 1e-6:
                    uci = random.choice(legal)
                else:
                    chosen_id = torch.multinomial(probs, num_samples=1).item()
                    uci = tok.decode(chosen_id)

            if uci not in legal:
                uci = random.choice(legal)
            board.push_uci(uci)
            move_history.append(uci)

        print(f"--- sample {sample_i + 1}/{num_samples} ---")
        print(board.fen())
        print(" ".join(move_history))
        if board.is_game_over(claim_draw=True):
            print(board.result())


if __name__ == "__main__":
    main()
