# patzer v4 — 40M param GPT, 3× data, WSD schedule (phase 1)
#
# Same architecture as v3; isolates data scaling as the single variable.
# v3 hit gen gap +0.025 on ~868M tokens — data-bound signal.
# v4 trains on ~2.85B tokens (36M games) to test if more data closes the gap.
#
# Schedule: Warmup-Stable-Decay (WSD). This config is PHASE 1 (warmup + constant LR).
# Phase 1 trains until early stop triggers, then phase 2 runs a short cooldown
# from that checkpoint (see train_patzer_v4_cooldown.py, created after phase 1).
#
# Data (1800+ ELO filter):
#   total_games: 38,162,737
#   train_games: 36,257,665
#   val_games:    1,905,072
#   train_tokens: 2,848,252,943
#   val_tokens:     149,593,437
#   avg_tokens/game: 78.6
#
# Throughput: batch=128 * block=256 = 32,768 tokens/iter → 86,898 iters/epoch
# max_iters=600k is a generous cap; early stop will end phase 1 much sooner.

out_dir = 'checkpoints/patzer_v4'
eval_interval = 1000
eval_iters = 50
log_interval = 100

always_save_checkpoint = True
early_stop_patience_evals = 25   # ~25k iters of no val improvement
early_stop_min_iters = 100000    # don't early-stop before 100k iters
ckpt_save_interval = 20000
weights_snapshot_interval = 20000
# Only push weights_best.pt when val improves by at least this amount.
# Avoids flooding R2 with tiny improvements early in training.
ckpt_best_min_delta = 0.001
# At most one weights_best.pt upload per this many steps.
# With eval_interval=1000, a setting of 5000 means at most ~1 upload / 5 evals.
ckpt_best_cooldown_steps = 5000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v4'

dataset = 'prepared'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

vocab_size = 4214

# Model: 12L / 8H / 512d  (~40M params) — identical to v3
n_layer = 12
n_head = 8
n_embd = 512
bias = False
dropout = 0.0

# WSD schedule: warmup → constant LR → (cooldown in phase 2)
lr_schedule = 'wsd'
learning_rate = 6e-4
max_iters = 600000              # generous cap; early stop ends phase 1
min_lr = 1e-5                   # used during cooldown (phase 2)
warmup_iters = 3000
cooldown_start_iter = None      # disabled — phase 1 runs at constant LR
cooldown_iters = 30000          # set here for documentation; unused until phase 2

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True
