# patzer v5 — 16L / 16H / 768d, same data as v4, WSD + auto_cooldown
#
# Param count: `GPT(...)` prints `model.get_num_params()` on init (~116.5M for this
# vocab/block_size with weight tying — wte shares lm_head; slightly higher than a
# naive “GPT-2 medium” thumb rule). Quick check before a long run:
#   cd patzer && python -c 'from model import GPT, GPTConfig; GPT(GPTConfig(block_size=256, vocab_size=4214, n_layer=16, n_head=16, n_embd=768, dropout=0.0, bias=False))'
#
# v4 showed capacity-bound scaling at 40M on 2.85B tokens; v5 scales width/depth.
# 768d is a natural checkpoint (GPT-2 medium territory); 16 heads × 48d/head is fine.
#
# Schedule: WSD with auto_cooldown=True — one job: stable LR until early stop, then
# linear cooldown to min_lr (no separate phase-2 config or restarts).
#
# Memory: ~117M params @ batch=128, block=256, bf16/fp16 ≈ 10–12GB was a rough plan;
# verify on target GPU. If OOM, drop batch_size to 96 (same fallback idea as v4).
#
# Data: identical to v4 (1800+ ELO filter, same prepared train/val).
#   See train_patzer_v4.py header for token counts.
#
# Wall time: expect slower steps than v4 (~n_embd²); plan longer Vast.ai runs.
#
# Throughput: batch=128 * block=256 = 32,768 tokens/iter (same as v4 if batch held).

out_dir = 'checkpoints/patzer_v5'
eval_interval = 1000
eval_iters = 50
log_interval = 100

always_save_checkpoint = True
early_stop_patience_evals = 25
# Higher than v4 (100k): larger model on the same data converges slower — don’t early-stop the stable phase before it has settled.
early_stop_min_iters = 150000
ckpt_save_interval = 20000
weights_snapshot_interval = 20000
ckpt_best_min_delta = 0.001
ckpt_best_cooldown_steps = 5000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v5'

dataset = 'prepared'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

vocab_size = 4214

# Model: 16L / 16H / 768d  (see header — ~116.5M non-wpe params at train init)
n_layer = 16
n_head = 16
n_embd = 768
bias = False
dropout = 0.0

# WSD: warmup → constant LR → auto cooldown when early stop would fire
lr_schedule = 'wsd'
learning_rate = 6e-4
max_iters = 600000
min_lr = 1e-5
# Slightly longer than v3/v4 (3k): larger matrices benefit from a gentler ramp (cheap vs total train length).
warmup_iters = 5000
cooldown_start_iter = None
# Length of the linear LR decay after auto_cooldown triggers — compare to *stable-phase* iters (iter at cooldown),
# not max_iters. E.g. stable ends ~120–150k → 38k is ~25–32% of stable (v4 manual phase used ~20%).
cooldown_iters = 38000
auto_cooldown = True

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True
