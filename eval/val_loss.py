"""
Compute validation loss for a checkpoint on a prepared val.bin.

This is a lightweight, eval-only utility meant for cross-dataset experiments
(e.g., "v6 best model on v5 val set") without downloading train.bin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch

from patzer.checkpoint_util import load_checkpoint
from patzer.model import GPT, GPTConfig


def _device_auto() -> tuple[str, bool]:
    if torch.cuda.is_available():
        return "cuda", True
    if torch.backends.mps.is_available():
        return "mps", False
    return "cpu", False


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="Path to weights_best.pt or ckpt.pt")
    ap.add_argument(
        "--data_dir",
        default="data/prepared",
        help="Directory containing val.bin and meta.json (default: data/prepared)",
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--dtype", default="float16", choices=["float32", "bfloat16", "float16"])
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(str(ckpt_path))

    data_dir = Path(args.data_dir)
    val_bin = data_dir / "val.bin"
    meta_json = data_dir / "meta.json"
    if not val_bin.exists():
        raise FileNotFoundError(str(val_bin))

    meta = {}
    if meta_json.exists():
        meta = json.loads(meta_json.read_text())

    if args.device == "auto":
        device, allow_compile = _device_auto()
    else:
        device, allow_compile = args.device, args.device == "cuda"

    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    print(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "data_dir": str(data_dir),
                "batch_size": args.batch_size,
                "eval_iters": args.eval_iters,
                "device": device,
                "dtype": args.dtype,
            }
        ),
        flush=True,
    )

    checkpoint = load_checkpoint(str(ckpt_path), map_location=device)
    model_args = checkpoint.get("model_args")
    if not isinstance(model_args, dict):
        raise ValueError(f"checkpoint missing model_args: {ckpt_path}")

    # Ensure vocab_size is set (meta.json usually contains it for prepared datasets).
    if model_args.get("vocab_size") is None:
        if "vocab_size" in meta:
            model_args["vocab_size"] = int(meta["vocab_size"])
        else:
            raise ValueError("vocab_size missing from checkpoint model_args and meta.json")

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError(f"checkpoint missing model state_dict: {ckpt_path}")

    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(
        f"loaded model (block_size={int(model_args['block_size'])}, vocab_size={int(model_args['vocab_size'])})",
        flush=True,
    )

    # torch.compile can be a net loss for this short eval, and doesn't work on MPS.
    _ = allow_compile  # reserved for a future flag

    block_size = int(model_args["block_size"])
    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    if len(data) <= block_size + 1:
        raise ValueError(f"val.bin too small ({len(data)=}) for {block_size=}")

    losses = torch.zeros(args.eval_iters, dtype=torch.float32)
    for i in range(args.eval_iters):
        ix = torch.randint(len(data) - block_size - 1, (args.batch_size,))
        x = torch.stack([torch.from_numpy((data[j : j + block_size]).astype(np.int64)) for j in ix])
        y = torch.stack([torch.from_numpy((data[j + 1 : j + 1 + block_size]).astype(np.int64)) for j in ix])
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        losses[i] = loss.float().item()
        if (i + 1) % max(1, min(10, args.eval_iters // 10)) == 0:
            print(f"progress {i+1}/{args.eval_iters}", flush=True)

    mean = float(losses.mean().item())
    std = float(losses.std(unbiased=False).item())
    iter_num = checkpoint.get("iter_num")
    ckpt_val = checkpoint.get("val_loss")
    print(
        json.dumps(
            {
                "checkpoint": str(ckpt_path),
                "iter_num": int(iter_num) if isinstance(iter_num, int) else iter_num,
                "checkpoint_val_loss": float(ckpt_val) if isinstance(ckpt_val, (int, float)) else ckpt_val,
                "data_dir": str(data_dir),
                "val_bin": str(val_bin),
                "batch_size": args.batch_size,
                "block_size": block_size,
                "eval_iters": args.eval_iters,
                "device": device,
                "dtype": args.dtype,
                "val_loss_mean": mean,
                "val_loss_std": std,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

