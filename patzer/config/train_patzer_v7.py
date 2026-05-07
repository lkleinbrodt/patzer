# patzer v7 — 48L / 16H / 1024d (~608M params); architecture scale-up from v5/v6 (116M).
#
# Architecture: 48L / 16H / 1024d
#   48 × 12 × 1024² + 4214×1024 ≈ 608M params (~5.2× jump from v5/v6's 116M)
#   head_dim = 64 — clean power-of-2, GPU-efficient
#
# Data: 2100+ ELO (~7.8B train tokens, ~100M games)
#   Same as v6 (which beat v5's 1800+ data in H2H play). Data quality > quantity.
#   Tokens/iter: 128 × 256 = 32,768 → ~238k iters/epoch on 2100+ train split.
#   v7 intentionally over-trains relative to Chinchilla (~13 tok/param vs optimal ~20) —
#   consistent with our entire history of capacity-limited, never-overfit models.
#
# Key changes from v6/v5 (116M → 608M + schedule fixes):
#
#   Architecture:
#     n_layer: 16 → 48, n_embd: 768 → 1024
#
#   LR:
#     learning_rate: 6e-4 → 4e-4  (scale down ~33% for 5× larger model; standard practice)
#     warmup_iters: 5k → 8k  (larger models are more sensitive to early LR spikes)
#
#   Schedule:
#     early_stop_min_iters: 80k → 150k
#       80k was tuned for 116M on ~1.7B tokens (~3.4 epochs before plateau). At this scale,
#       80k iters = only 0.34 epochs — model has barely seen the data once. Expect stable
#       plateau at 150–250k iters; 150k gate avoids premature cooldown trigger.
#     cooldown_iters: 50k → 65k
#       v6's 38k was cut off mid-drop; v7-as-116M fixed to 50k. For 5× larger model
#       with a longer stable phase (~250k), ~25% = 62k → 65k.
#     Post-cooldown min_lr tail: auto_cooldown=True + patience stops only when val plateaus.
#       In v5/v6 training terminated immediately when cooldown ended — v6 was still falling.
#
#   No regularization changes: dropout=0.0 and weight_decay=0.1 unchanged.
#   We have never observed overfit across any run (gen gap ~0 from v4 onward). Adding
#   regularization here would be wrong.
#
# Expected total run: stable phase ~200-250k + cooldown 65k + patience tail ~25k ≈ 290-340k iters.
#
# GPU memory: 608M fp32 weights + Adam states + activations at batch=128 requires ~18-22 GB.
#   Rent a 24 GB GPU (RTX 4090, A5000, A6000) on Vast.ai.
#   16 GB fallback: set batch_size=64, gradient_accumulation_steps=2 (same effective batch).

out_dir = 'checkpoints/patzer_v7'
eval_interval = 1000
eval_iters = 50
log_interval = 100

always_save_checkpoint = True
early_stop_patience_evals = 25
early_stop_min_iters = 150000      # 80k too aggressive at this scale — see notes above
ckpt_save_interval = 20000
weights_snapshot_interval = 20000
ckpt_best_min_delta = 0.001
ckpt_best_cooldown_steps = 5000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v7'

dataset = 'prepared_min_elo_2100'
gradient_accumulation_steps = 1    # set to 2 and halve batch_size if 24 GB GPU not available
batch_size = 128
block_size = 256

vocab_size = 4214

# Model: 48L / 16H / 1024d — ~608M params
n_layer = 48
n_head  = 16
n_embd  = 1024
bias    = False
dropout = 0.0                      # no regularization needed; we have never overfit any run

# WSD + auto_cooldown
lr_schedule = 'wsd'
learning_rate = 4e-4               # scaled down from 6e-4 for 5× larger model
max_iters = 600000
min_lr = 5e-6
warmup_iters = 8000                # up from 5k; larger model benefits from longer warmup
cooldown_start_iter = None
cooldown_iters = 65000             # ~25% of expected ~250k stable phase; v6 was cut off at 38k
auto_cooldown = True               # triggers cooldown on plateau, then runs min_lr tail w/ patience

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True
