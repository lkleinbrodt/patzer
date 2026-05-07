# patzer v7 — 40L / 16H / 1024d (~508M params); architecture scale-up from v5/v6 (116M).
#
# Architecture: 40L / 16H / 1024d
#   40 × 12 × 1024² + 4214×1024 ≈ 508M params (~4.4× jump from v5/v6's 116M)
#   head_dim = 64 — clean power-of-2, GPU-efficient
#
# Data: 2100+ ELO (~6.5B tokens, ~83M games — ~avg 78 tokens/game)
#   Same as v6 (which beat v5's 1800+ data in H2H play). Data quality > quantity.
#   Effective batch: gradient_accum × batch × block = 1 × 256 × 256 = 65,536 tokens/iter.
#   Max 300k iters → ~19.66B total tokens (same data budget as original 600k iters at batch=128).
#   gradient_checkpointing makes batch=256 fit in 14GB (RTX 4090); on 4060 Ti use batch=128.
#   v7 intentionally over-trains vs Chinchilla (~5 tok/param vs optimal ~20) —
#   consistent with capacity-limited, never-overfit Patzer runs.
#
# Key changes from v6/v5 (116M → ~508M + schedule fixes):
#
#   Architecture:
#     n_layer: 16 → 40, n_embd: 768 → 1024
#
#   LR (original v7 design; now tuned for 4090):
#     v6→v7: 6e-4 → 4e-4  (scale down for larger model; standard practice)
#     v7→4090: 4e-4 → 5.7e-4  (√2 scale for 2× batch size)
#     warmup_iters: 5k → 8k  (larger models benefit from longer warmup)
#
#   Schedule (4090-tuned):
#     max_iters: 600k → 300k  (2× batch → half iterations for same data budget)
#     early_stop_min_iters: 150k → 75k  (25% threshold, scaled proportionally)
#     cooldown_iters: 65k → 32k  (~25% of ~130k expected stable phase, scaled for 300k max)
#     Post-cooldown: auto_cooldown=True + patience controls tail; no hard stop at cooldown end.
#
#   Regularization: dropout=0.0, weight_decay=0.1 (no overfit observed across any run).
#
# Expected total run (4090, 3-5 days): stable ~100-125k + cooldown 32k + tail ~25k ≈ 150-170k iters;
#   faster than 4060 Ti's ~10 days due to 3.75× GPU + 2× batch (assuming 1-1.5 sec/iter on 4090).
#
# GPU memory (CUDA): bf16 autocast forward + fp32 grads/Adam + activation storage for backward.
#   Configured for RTX 4090 24GB with batch=256. Gradient checkpointing is still required
#   (activations: ~10GB MLP pre-GELU + 7.5GB Q/K/V without checkpointing; with checkpointing
#   only 40 block inputs stored: 5.1GB). Peak VRAM usage ~14GB.
#   Learning rate scaled by √2 (5.7e-4) for 2× batch size; max_iters halved to 300k
#   for same data budget as original 600k at batch=128.
#
#   On 4060 Ti (12GB): reduce batch_size to 128, learning_rate to 4e-4, max_iters to 600000.

out_dir = 'checkpoints/patzer_v7'
eval_interval = 1000
eval_iters = 25                   # 25 × batch 256 = 6,400 seqs/split (same eval coverage as v6)
log_interval = 100

always_save_checkpoint = True
early_stop_patience_evals = 25
early_stop_min_iters = 75000       # scaled down proportionally with max_iters (25% threshold)
ckpt_save_interval = 20000
weights_snapshot_interval = 20000
ckpt_best_min_delta = 0.001
ckpt_best_cooldown_steps = 5000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v7'

dataset = 'prepared_min_elo_2100'
gradient_accumulation_steps = 1    # single forward per step (no accumulation needed on 4090)
batch_size = 256                   # 4090 24GB allows 2× batch; effective batch = 1 × 256 × 256 = 65,536 tokens/iter
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
learning_rate = 5.7e-4             # 4e-4 × √2 scaling rule for 2× batch size (4090 with batch=256)
max_iters = 300000                 # halved to match data budget (2× batch → same tokens per iter, so half the iters)
min_lr = 5e-6
warmup_iters = 8000                # unchanged (same number of iters as before; proportionally shorter due to larger batch)
cooldown_start_iter = None
cooldown_iters = 32000             # ~25% of expected ~130k stable phase (scaled down from 65k for max_iters 300k)
auto_cooldown = True               # triggers cooldown on plateau, then runs min_lr tail w/ patience

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True

# GPU peak BF16 FLOPS for accurate MFU reporting.
# This config is tuned for RTX 4090 (24GB). For other GPUs:
#   RTX 4060 Ti (12GB): batch_size=128, learning_rate=4e-4, max_iters=600000, peak_flops=44.12e12
#   A100 SXM (40/80GB): batch_size=512+, learning_rate=8e-4+, peak_flops=312e12
peak_flops = 165.2e12  # RTX 4090 BF16 tensor-core peak (no sparsity)

# Gradient checkpointing is always needed for this model size. freq=1 checkpoints every
# block (minimum VRAM). On a 24GB GPU, bump batch_size to 256 instead of disabling.
gradient_checkpoint_freq = 1
