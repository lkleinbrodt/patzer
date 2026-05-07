# patzer v7 — 24L / 16H / 1024d (~306M params); GPT-2-medium-style scale-up from v5/v6 (116M).
#
# Architecture: 24L / 16H / 1024d
#   ~24 × 12 × 1024² + 4214×1024 ≈ 306M params (~2.6× jump from v5/v6's 116M)
#   head_dim = 64 — clean power-of-2, GPU-efficient
#
# Data: 2100+ ELO (~6B train tokens in budget — same dataset family as v6).
#   Effective batch: gradient_accum × batch × block = 2 × 64 × 256 = 32,768 tokens/iter
#   (matches v4/v5/v6's 1 × 128 × 256).
#   ~400k iters → ~13.1B tokens seen (~2.2 passes over ~6B tokens).
#
# Key changes from the abandoned ~508M v7 (40L):
#   Shallower (24L) so 4090 24GB fits without gradient checkpointing — faster steps.
#   Micro-batch 64 × accum 2 keeps VRAM comfortable and matches prior effective batch.
#
# Schedule:
#   WSD + auto_cooldown; longer cooldown (60k) per v5/v6 retrospective.
#   early_stop_min_iters=100k (was too conservative at 150k on v5/v6).
#
# Regularization: dropout=0.0, weight_decay=0.1 (no overfit observed across any run).
#
# Expected wall time (RTX 4090): ~1–2 days to typical early-stop + cooldown vs multi-day 500M runs.

out_dir = 'checkpoints/patzer_v7'
eval_interval = 1000
eval_iters = 100                  # 100 × batch 64 = 6,400 seqs/split (same eval coverage as v6's 50×128)
log_interval = 100

always_save_checkpoint = True
early_stop_patience_evals = 25
early_stop_min_iters = 100000     # v5/v6 retrospective: 150k gate delayed cooldown too long
ckpt_save_interval = 10000
weights_snapshot_interval = 10000
ckpt_best_min_delta = 0.001
ckpt_best_cooldown_steps = 2500

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v7'

dataset = 'prepared_min_elo_2100'
gradient_accumulation_steps = 2    # micro-batch 64 → effective batch 128 (same tokens/step as v6)
batch_size = 64                    # RTX 4090 24GB, no gradient checkpointing at 24L/1024d
block_size = 256

vocab_size = 4214

gradient_checkpointing = False

# Model: 24L / 16H / 1024d — ~306M params
n_layer = 24
n_head  = 16
n_embd  = 1024
bias    = False
dropout = 0.0

# WSD + auto_cooldown
lr_schedule = 'wsd'
learning_rate = 4e-4               # slightly below v6 stable LR (6e-4) for larger width/depth
max_iters = 400000
min_lr = 5e-6
warmup_iters = 5000
cooldown_start_iter = None
cooldown_iters = 60000             # ~25–30% of expected stable phase
auto_cooldown = True

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True

# GPU peak BF16 FLOPS for accurate MFU reporting.
# Tuned for RTX 4090 (24GB). On 12GB GPUs try batch_size=32, gradient_accumulation_steps=4
# (same effective batch) or batch 48 / accum 2 if memory allows.
peak_flops = 165.2e12  # RTX 4090 BF16 tensor-core peak (no sparsity)

gradient_checkpoint_freq = 1
