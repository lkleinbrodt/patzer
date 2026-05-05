# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Patzer is a transformer-based chess engine trained via next-token prediction on Lichess game data. The goal is a series of versioned models (v1, v2, ...) each trained, deployed as a Lichess bot, and evaluated for ELO — iterating on architecture and data quality.

See **[PROJECT_LOG.md](PROJECT_LOG.md)** for a running history of development work, interesting findings, and rationale behind architectural decisions.

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
python train.py config/train_patzer_v4.py --device=auto

# Single GPU (CUDA)
python train.py config/train_patzer_v4.py

# DDP (multi-GPU)
torchrun --standalone --nproc_per_node=4 train.py config/train_patzer_v4.py

# Resume from checkpoint
python train.py config/train_patzer_v4.py --init_from=resume
```

Replace `train_patzer_v4.py` with the versioned config you want to train (e.g., `train_patzer_v1.py`, `train_patzer_v2.py`, etc.).

### Cloud training on Vast.ai

```bash
# List available GPU offers
python launch.py train --search-only

# Rent cheapest GPU and start training (prompts for confirmation)
python launch.py train --config train_patzer_v4

# Run on an existing instance
python launch.py train --instance <ID> --config train_patzer_v4

# Resume training from R2 checkpoint
python launch.py train --config train_patzer_v4 --resume

# List your running instances
python launch.py --list
```

The `train` subcommand is optional for backward compatibility: `python launch.py --config train_patzer_v4` still works. Replace `train_patzer_v4` with the versioned config you want to train.

### Lichess scrape on a DigitalOcean droplet (recommended)

Short copy-paste sequence (SSH + tmux): **[pipeline/DROPLET_SCRAPE.md](pipeline/DROPLET_SCRAPE.md)**.

Run `pipeline/scrape_lichess.py` on a small CPU droplet; **`--push-r2`** uploads `data/lichess_games/` (same paths as locally) to R2. Put **`R2_*`** credentials in a repo-root **`.env`** on your laptop, sync code + `.env` to the droplet, install apt/python deps once, then run the scraper (optionally under **`tmux`** so it survives disconnects).

```bash
export PATZER_DROPLET='root@YOUR_DROPLET_IP'

./pipeline/droplet_scrape.sh sync       # rsync tree to ~/patzer (excludes data/, .venv, …)
./pipeline/droplet_scrape.sh push-env   # scp .env → ~/patzer/.env
./pipeline/droplet_scrape.sh setup      # wget, zstd, python3, pip install scrape deps

# Foreground (prints logs):
./pipeline/droplet_scrape.sh run -- --output-dir data/lichess_games --push-r2 \
  --min-month 2020-04 --min-elo 1800

# Or SSH in and use tmux (after setup: use venv Python):
ssh "$PATZER_DROPLET"
tmux new -s scrape
cd ~/patzer && .venv/bin/python pipeline/scrape_lichess.py --output-dir data/lichess_games --push-r2 \
  --min-month 2020-04 --min-elo 1800 2>&1 | tee scrape.log
```

Pull games elsewhere (e.g. workstation): **`cd patzer && python r2.py pull data/lichess_games`**.

### Checkpoint sync (Cloudflare R2)

```bash
cd patzer
python r2.py push data/prepared                                      # upload tokenized data
python r2.py pull checkpoints/patzer_v2/weights_best.pt              # pull single file (skips if exists)
python r2.py pull checkpoints/patzer_v2                              # pull full dir (skips existing)
python r2.py pull checkpoints/patzer_v2/weights_best.pt --force      # re-download even if local copy exists
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
# Chess-aware: ChessTokenizer, legal-move masking, same conditioning modes as eval (`match_color`, etc.).
# Default `out_dir` is relative to cwd (`patzer/`). If checkpoints are at repo root, use e.g. `--out_dir=../checkpoints/patzer_v3`.
python sample.py --out_dir=../checkpoints/patzer_v3 --num_samples=3 --max_new_tokens=80
python sample.py --out_dir=../checkpoints/patzer_v3 --start='e2e4 e7e5 g1f3' --conditioning=match_color
```

### Evaluate

```bash
# Estimate a model's Elo vs Stockfish (adaptive Bayesian, stops when confident)
python eval/evaluate.py stockfish checkpoints/patzer_v2/weights_best.pt --games 50 --device mps

# Same, but several checkpoints in one invocation (sequential runs)
python eval/evaluate.py stockfish patzer_v3@best patzer_v3@180 patzer_v4@40 --games 50 --device mps

# Compare two models head-to-head
python eval/evaluate.py head2head checkpoints/patzer_v2/weights_best.pt checkpoints/patzer_v1/weights_best.pt --games 20 --device mps

# Round-robin across multiple checkpoints
python eval/evaluate.py head2head checkpoints/patzer_v2/weights_iter_010000.pt checkpoints/patzer_v2/weights_iter_050000.pt checkpoints/patzer_v2/weights_best.pt --round-robin --games 10 --device mps

# Gauntlet: challenger vs each selected leaderboard opponent (no games among opponents); interactive rank pick or --no-prompt (default ranks 1–10)
python eval/evaluate.py gauntlet patzer_v4@best --games 50 --device mps

# Show unified Elo leaderboard (computed from all stored games)
python eval/evaluate.py leaderboard

# Show game history for a model
python eval/evaluate.py history patzer_v2

# Plot Elo progression over training steps (requires prior stockfish runs)
python eval/evaluate.py progress patzer_v2

# Engine regression: legal-mask + move-token cache vs reference (stub model; no weights)
python -m unittest eval/test_patzer_engine.py -v
```

All results are stored in `eval/results.db` (SQLite, gitignored). One row per game — no aggregation. Pull checkpoints first with `python patzer/r2.py pull checkpoints/patzer_vN` before evaluating.

### Lichess bot (deploy)

Configs and a thin `homemade.py` shim live under `bot/`. The [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) repo stays elsewhere (e.g. `~/Projects/lichess-bot`); `deploy_bot.py` symlinks `bot/templates/homemade_shim.py` → `lichess-bot/homemade.py` and runs `lichess-bot.py --config …` with `PATZER_ROOT` set.

Use **`python bot/cycle_bots.py`** to rotate one bot at a time on a shared machine (defaults: dwell 3600s, cycles every `patzer_*.yml` under `bot/configs/`).

**One-time setup:**

```bash
# 1. Clone lichess-bot and install its dependencies
git clone https://github.com/lichess-bot-devs/lichess-bot ~/Projects/lichess-bot
cd ~/Projects/lichess-bot && python -m venv .venv && .venv/bin/pip install -r requirements.txt

# 2. Install the shim (run from Patzer repo root)
python bot/deploy_bot.py install-shim
```

**Run or manage bots:**

```bash
# Start a bot
python bot/deploy_bot.py run v2

# Upgrade an account to a bot account (one-time, irreversible)
python bot/deploy_bot.py upgrade v2

# Run multiple bots simultaneously in separate terminals
python bot/deploy_bot.py run v1   # terminal 1
python bot/deploy_bot.py run v2   # terminal 2
```

Per-bot secrets: `LICHESS_BOT_TOKEN` env, `PATZER_VN_TOKEN` in `.env`, or `token:` in `bot/configs/patzer_vN.yml` (gitignored). See `bot/README.md` for token lookup order and config options.

### Analysis dashboard

Web-based dashboard (Flask backend + React frontend) for viewing model evaluations and Lichess bot game history.

**Setup:**

```bash
cd dashboard
pip install -r requirements.txt
```

**Run:**

```bash
# From repo root or dashboard/ dir
python dashboard/run.py
```

Runs on `http://localhost:5050` (port configurable via `DASHBOARD_PORT` env var).

**Pages:**

- **Eval Leaderboard**: Unified leaderboard of all evaluation results across models and checkpoints (data from `eval/results.db`)
- **Lichess Analysis**: Game history and stats for deployed Lichess bots (data from `dashboard/lichess_games.db`, separate from eval results)

**Bot token configuration:**

Dashboard syncs Lichess bot game history via the Lichess API. Add tokens to `.env` at repo root:

```
PATZER_V1_TOKEN=lip_xxx
PATZER_V2_TOKEN=lip_yyy
PATZER_VN_USERNAME=account_name    # optional; v2 defaults to `patzer_v2b`
```

Tokens are looked up as `PATZER_VN_TOKEN` where N is the version number. Bot game data is stored in `dashboard/lichess_games.db` (separate from training eval results in `eval/results.db`).

## Architecture

### End-to-end data flow

```
Lichess .pgn.zst dumps
  → pipeline/scrape_lichess.py   (download + SHA256 verify + filter + parse, per month)
  → data/lichess_games/games_YYYY-MM.txt  (one game per line: result elo elo uci_moves...)
  → pipeline/prepare.py          (tokenize + split → uint16 binary)
  → data/prepared/train.bin + val.bin + meta.json
  → patzer/train.py              (nanoGPT training loop, memmap batches from .bin files)
 → checkpoints/patzer_vN/ckpt.pt (+ weights_best.pt on val improvement, R2 sync)
```

### Configuration system (`patzer/configurator.py`)

`train.py` and `sample.py` use a "poor man's configurator": `exec(open('configurator.py').read())` is called after setting default globals. `configurator.py` reads `sys.argv` and either execs a named config file (e.g. `config/train_patzer.py`) or applies `--key=value` overrides directly into `globals()`. This means config files simply reassign module-level variables, and CLI flags override them. **All scripts must be run from within the `patzer/` directory** because `configurator.py` is loaded by relative path.

### Tokenizer (`patzer/tokenizer.py`)

Vocabulary is built deterministically from all possible UCI move strings (not from training data) plus 6 special tokens: `<PAD>`, `<GAME_START>`, `<GAME_END>`, `<WHITE_WIN>`, `<BLACK_WIN>`, `<DRAW>`. Vocab size is ~4214. Game encoding format: `<GAME_START> <RESULT> move1 move2 ... <GAME_END>` — the result token is prepended so the model is conditioned on outcome from position 0.

### Model (`patzer/model.py`)

Standard nanoGPT (Karpathy): `Block = CausalSelfAttention + MLP`, pre-norm with `LayerNorm`, weight-tied token embeddings and LM head. Flash attention used when PyTorch ≥ 2.0. `GPTConfig` holds all architecture hyperparameters. `GPT.configure_optimizers` applies weight decay only to 2D parameters (matmul weights + embeddings), not biases or layernorm.

### Training loop (`patzer/train.py`)

Direct port of nanoGPT. Data loading uses `np.memmap` (recreated each batch to avoid memory leaks). Supports DDP via `torchrun`. With `always_save_checkpoint=True`, `ckpt.pt` is the **latest** eval (optimizer + iter for resume). `weights_best.pt` is written only when val improves (weights-only for play/eval). Optional `weights_iter_*.pt` snapshots can be created on best improvements. Optional `early_stop_patience_evals` / `early_stop_min_iters` stop when val plateaus. Eval estimates loss over `eval_iters` batches for both train and val splits.

### LR schedules (`lr_schedule` config key)

Two schedules are supported:

- `**cosine`** (default, v1–v3): warmup → cosine decay to `min_lr` over `lr_decay_iters` → flat at `min_lr`. Original nanoGPT behavior.
- `**wsd**` (v4+): Warmup-Stable-Decay. Warmup → constant `learning_rate` → optional linear cooldown to `min_lr`. Decouples training duration from decay schedule.

WSD config knobs: `cooldown_start_iter` (None = no cooldown, constant LR), `cooldown_iters` (length of linear ramp-down), `auto_cooldown` (bool, default False). Two workflows:

- **Manual two-phase**: run phase 1 with `cooldown_start_iter=None` and early stopping; create a second config with `init_from=resume`, `cooldown_start_iter=<phase1_iter_num>`, and `early_stop_patience_evals=0`.
- **Automatic single-job**: set `auto_cooldown=True`. When early stopping would fire, training instead sets `cooldown_start_iter=iter_num`, extends `max_iters` by `cooldown_iters`, resets patience, and decays LR to `min_lr` before stopping. `cooldown_start_iter` is saved in `ckpt.pt` so mid-cooldown restarts pick up the correct decay curve.

### R2 storage (`patzer/r2.py`)

Cloudflare R2 (S3-compatible) is used to persist training data and checkpoints. Mirrors local path structure exactly. All functions are silent no-ops when R2 env vars are unset — safe to run locally without credentials. Required env vars: `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `R2_ACCOUNT_ID` (set in `.env`).

Uploads use **`put_object`** (streamed body) instead of boto3 `upload_file` / S3Transfer so there is no hidden nested thread pool during interpreter shutdown. Large files (`ckpt.pt`, `weights_best.pt` rate-limited) default to **`push_async`** (one background worker) so training does not block for multi-minute uploads; **`flush_r2_uploads()`** runs when the training loop ends and again from atexit so the queue drains before process exit. **`metrics.jsonl`** is uploaded synchronously (small file). If `ThreadPoolExecutor.submit` fails, uploads fall back to the foreground. Set **`r2_async_uploads = False`** in config for fully synchronous large uploads (slow, easiest to reason about).

### Cloud launch (`launch.py`)

Manages **Vast.ai GPU** training via the `vastai` CLI. **`launch.py train`** uses the PyTorch CUDA image: exports R2 env vars, clones the repo, pip-installs, optionally pulls a checkpoint from R2, then runs `train.py` in `tmux`; logs at `/workspace/train.log`. Lichess scraping runs on your own CPU host (e.g. **DigitalOcean** droplet — see above), not via `launch.py`.

## Key conventions

- **Run training scripts from `patzer/`**, not the repo root — `configurator.py` and `r2.py` are loaded with relative paths.
- **Config files** live in `patzer/config/` and are versioned (e.g., `train_patzer_v1.py`, `train_patzer_v2.py`, etc.). They are plain Python that reassigns globals. To create a new model version, copy an existing version config and increment the version number.
- `**device='auto'`** in config files detects cuda → mps → cpu and disables `torch.compile` on non-CUDA devices automatically (see `train.py` lines 83–91).
- **Data files are gitignored** (`data/`*). All data lives locally or in R2; never commit binary data.
- **Checkpoint naming**: `ckpt.pt` = latest full checkpoint (optimizer + weights, for resume). `weights_best.pt` = best val-loss weights only (use for eval/play). `weights_iter_XXXXXX.pt` = step snapshots. When resuming, only architecture args (`n_layer`, `n_head`, `n_embd`, `block_size`, `bias`, `vocab_size`) are forced to match; other hyperparams can change.
- **Checkpoint state dict keys**: `model`, `optimizer`, `model_args`, `iter_num`, `best_val_loss`, `evals_without_improvement`, `config`.
- The `_orig_mod.` prefix stripping in `train.py` and `sample.py` handles state dict keys from `torch.compile`.

