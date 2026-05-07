# patzer v7 — 40L / 16H / 1024d (~508M params); architecture scale-up from v5/v6 (116M).
#
# Architecture: 40L / 16H / 1024d
#   40 × 12 × 1024² + 4214×1024 ≈ 508M params (~4.4× jump from v5/v6's 116M)
#   head_dim = 64 — clean power-of-2, GPU-efficient
#
# Data: 2100+ ELO (~6.5B tokens, ~83M games — ~avg 78 tokens/game)
#   Same as v6 (which beat v5's 1800+ data in H2H play). Data quality > quantity.
#   Effective batch unchanged from v5/v6: gradient_accum × batch × block = 1 × 128 × 256 = 32,768
#   tokens/iter (~198k iters/epoch if ~6.5B train tokens in memmap split).
#   gradient_checkpointing freed enough VRAM to run batch_size=128 in a single forward
#   (previously OOM'd without checkpointing), so accum steps dropped from 2 → 1 for
#   the same effective batch with less Python overhead.
#   v7 intentionally over-trains vs Chinchilla (~10 tok/param vs optimal ~20) —
#   consistent with capacity-limited, never-overfit Patzer runs.
#
# Key changes from v6/v5 (116M → ~508M + schedule fixes):
#
#   Architecture:
#     n_layer: 16 → 40, n_embd: 768 → 1024
#
#   LR:
#     learning_rate: 6e-4 → 4e-4  (scale down ~33% for ~3× larger model; standard practice)
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
# GPU memory (CUDA): bf16 autocast forward + fp32 grads/Adam + activation storage for backward.
#   On 24 GB cards, `gradient_checkpointing=True` is the main lever that makes ~500M+ models
#   feasible. Without checkpointing, 40L/1024d OOM'd at batch_size=128. With checkpointing,
#   batch_size=128 fits cleanly — accumulation steps reduced to 1 (less Python overhead,
#   same effective batch of 32,768 tokens/iter).

out_dir = 'checkpoints/patzer_v7'
eval_interval = 1000
eval_iters = 100                  # 100 × batch 64 = 6,400 seqs/split = same coverage as eval_iters=50 @ batch 128 (v5/v6)
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
gradient_accumulation_steps = 1    # single forward per step — gradient_checkpointing makes batch=128 fit on 24 GB
batch_size = 128                   # effective batch = 1 × 128 × 256 = 32,768 tokens/iter (same as original 2×64)
block_size = 256

vocab_size = 4214

# Reduce activation memory by recomputing blocks on backward (slower, much lower VRAM).
gradient_checkpointing = True

# Model: 40L / 16H / 1024d — ~508M params
n_layer = 40
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
