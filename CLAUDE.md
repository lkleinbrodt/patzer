# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Patzer is a transformer-based chess engine trained via next-token prediction on Lichess game data. The goal is a series of versioned models (v1, v2, ...) each trained, deployed as a Lichess bot, and evaluated for ELO — iterating on architecture and data quality.

## Commands

### Install dependencies
```bash
pip install -r requirements.txt
# Also requires: torch, numpy, wandb (not in requirements.txt yet)
# System dependency: zstd (for pipeline), wget (for downloads), vastai CLI (for cloud training)
```

### Data pipeline (run in order)
```bash
# 1. Download + filter + parse Lichess monthly dumps (resumable, skips completed months)
python pipeline/scrape_lichess.py --output-dir ./data/lichess_games --min-elo 1800 --max-months 12

# 2. Tokenize games into binary train/val files
cd patzer
python ../pipeline/prepare.py --input ../data/lichess_games/games_*.txt --output-dir ../data/prepared

# Smoke test with capped game count:
python ../pipeline/prepare.py --max-games 10000 --output-dir ../data/prepared_test
```

### Train
```bash
# Local (MPS on Mac, CPU fallback)
cd patzer
python train.py config/train_patzer.py --device=auto

# Single GPU (CUDA)
python train.py config/train_patzer.py

# DDP (multi-GPU)
torchrun --standalone --nproc_per_node=4 train.py config/train_patzer.py

# Resume from checkpoint
python train.py config/train_patzer.py --init_from=resume
```

### Cloud training on Vast.ai
```bash
# List available GPU offers
python launch.py --search-only

# Rent cheapest GPU and start training (prompts for confirmation)
python launch.py --config train_patzer

# Run on an existing instance
python launch.py --instance <ID> --config train_patzer

# Resume training from R2 checkpoint
python launch.py --config train_patzer --resume

# List your running instances
python launch.py --list
```

### Checkpoint sync (Cloudflare R2)
```bash
cd patzer
python r2.py push data/prepared          # upload tokenized data
python r2.py pull checkpoints/patzer_v0  # download checkpoint
```

### Tokenizer sanity check
```bash
cd patzer
python tokenizer.py   # builds vocab, encodes/decodes a sample game, validates round-trip
```

### Dataset sanity check
```bash
cd patzer
python dataset.py data/prepared/train.bin
```

### Sample from a trained model
```bash
cd patzer
python sample.py --out_dir=checkpoints/patzer_v0
```

## Architecture

### End-to-end data flow
```
Lichess .pgn.zst dumps
  → pipeline/scrape_lichess.py   (download + SHA256 verify + filter + parse, per month)
  → data/lichess_games/games_YYYY-MM.txt  (one game per line: result elo elo uci_moves...)
  → pipeline/prepare.py          (tokenize + split → uint16 binary)
  → data/prepared/train.bin + val.bin + meta.json
  → patzer/train.py              (nanoGPT training loop, memmap batches from .bin files)
  → checkpoints/patzer_vN/ckpt.pt (+ ckpt_best.pt on val improvement, R2 sync)
```

### Configuration system (`patzer/configurator.py`)
`train.py` and `sample.py` use a "poor man's configurator": `exec(open('configurator.py').read())` is called after setting default globals. `configurator.py` reads `sys.argv` and either execs a named config file (e.g. `config/train_patzer.py`) or applies `--key=value` overrides directly into `globals()`. This means config files simply reassign module-level variables, and CLI flags override them. **All scripts must be run from within the `patzer/` directory** because `configurator.py` is loaded by relative path.

### Tokenizer (`patzer/tokenizer.py`)
Vocabulary is built deterministically from all possible UCI move strings (not from training data) plus 6 special tokens: `<PAD>`, `<GAME_START>`, `<GAME_END>`, `<WHITE_WIN>`, `<BLACK_WIN>`, `<DRAW>`. Vocab size is ~4214. Game encoding format: `<GAME_START> <RESULT> move1 move2 ... <GAME_END>` — the result token is prepended so the model is conditioned on outcome from position 0.

### Model (`patzer/model.py`)
Standard nanoGPT (Karpathy): `Block = CausalSelfAttention + MLP`, pre-norm with `LayerNorm`, weight-tied token embeddings and LM head. Flash attention used when PyTorch ≥ 2.0. `GPTConfig` holds all architecture hyperparameters. `GPT.configure_optimizers` applies weight decay only to 2D parameters (matmul weights + embeddings), not biases or layernorm.

### Training loop (`patzer/train.py`)
Direct port of nanoGPT. Data loading uses `np.memmap` (recreated each batch to avoid memory leaks). Supports DDP via `torchrun`. With `always_save_checkpoint=True`, `ckpt.pt` is the **latest** eval (optimizer + iter for resume). `ckpt_best.pt` is written only when val improves (weights for play/eval). Stamped `ckpt_{iter:06d}.pt` on R2 preserves history. Optional `early_stop_patience_evals` / `early_stop_min_iters` stop when val plateaus. Eval estimates loss over `eval_iters` batches for both train and val splits.

### R2 storage (`patzer/r2.py`)
Cloudflare R2 (S3-compatible) is used to persist training data and checkpoints. Mirrors local path structure exactly. All functions are silent no-ops when R2 env vars are unset — safe to run locally without credentials. Required env vars: `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `R2_ACCOUNT_ID` (set in `.env`).

### Cloud launch (`launch.py`)
Manages Vast.ai GPU instances via the `vastai` CLI. On new instance creation, builds a bootstrap shell script that: exports R2 env vars, clones the repo, pip-installs, optionally pulls a checkpoint from R2, then runs `train.py` inside a `tmux` session. Training output is tailed at `/workspace/train.log`.

## Key conventions

- **Run training scripts from `patzer/`**, not the repo root — `configurator.py` and `r2.py` are loaded with relative paths.
- **Config files** live in `patzer/config/` and are plain Python that reassigns globals. To create a new model version, copy `train_patzer.py` and increment the version.
- **`device='auto'`** in config files detects cuda → mps → cpu and disables `torch.compile` on non-CUDA devices automatically (see `train.py` lines 83–91).
- **Data files are gitignored** (`data/*`). All data lives locally or in R2; never commit binary data.
- **Checkpoint keys**: `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `evals_without_improvement`, `config`. Resume from `ckpt.pt` (latest). Use `ckpt_best.pt` for eval/play (`eval/tournament.py` auto-pick prefers it). When resuming, only architecture args (`n_layer`, `n_head`, `n_embd`, `block_size`, `bias`, `vocab_size`) are forced to match; other hyperparams can change.
- The `_orig_mod.` prefix stripping in `train.py` and `sample.py` handles state dict keys from `torch.compile`.
