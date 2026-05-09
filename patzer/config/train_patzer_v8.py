# patzer v8 — ~205M params (16L / 16H / 1024d), elite-only data (2250+), WSD + auto_cooldown
#
# Goal: push the behavioral-cloning ceiling upward by filtering to truly elite games,
# while keeping model size moderate (no need for 300M+ if quality does most of the work).
#
# Architecture: 16L / 16H / 1024d  (~205M params)
#   - Same depth as v5/v6 (16L) for a clean comparison
#   - Same width/head_dim style as v7 (1024d, 16 heads → 64 dim/head)
#
# Data: prepared_min_elo_2250 (both players >= 2250). Token counts should come from
# the prepare meta in that dataset directory.
# ~21M games, ~1.8B tokens
#
# Schedule: WSD with auto_cooldown + post-cooldown min_lr tail (implemented in train.py).
#

out_dir = 'checkpoints/patzer_v8'
eval_interval = 1000
# Keep eval sequence coverage comparable across runs:
# 100 × batch 64 = 6,400 seqs/split (same as v7).
eval_iters = 100
log_interval = 100

always_save_checkpoint = True
early_stop_patience_evals = 25
# v5/v6 showed the stable phase can plateau well before 150k; v7 uses 100k.
early_stop_min_iters = 100000
ckpt_save_interval = 10000
weights_snapshot_interval = 10000
ckpt_best_min_delta = 0.001
ckpt_best_cooldown_steps = 2500

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v8'

dataset = 'prepared_min_elo_2250'
gradient_accumulation_steps = 2
batch_size = 64
block_size = 256

vocab_size = 4214

gradient_checkpointing = False

# Model: 16L / 16H / 1024d  (~205M params)
n_layer = 16
n_head = 16
n_embd = 1024
bias = False
dropout = 0.0

# WSD + auto_cooldown
lr_schedule = 'wsd'
learning_rate = 4e-4
max_iters = 350000
min_lr = 5e-6
warmup_iters = 5000
cooldown_start_iter = None
cooldown_iters = 60000
auto_cooldown = True

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True

# GPU peak BF16 FLOPS for MFU reporting (RTX 4090 baseline).
peak_flops = 165.2e12

gradient_checkpoint_freq = 1

