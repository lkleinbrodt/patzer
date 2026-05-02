# patzer v4 — Phase 2: WSD cooldown
#
# Resumes from Phase 1 checkpoint (iter 140000, val 1.6548) and applies a
# linear LR cooldown from learning_rate → min_lr over 30k iters.
#
# Phase 1 stopped at iter 145001 (25 evals without improvement).
# ckpt.pt is at iter 140000 (evals_without_improvement=20 in checkpoint).
# Setting cooldown_start_iter=140000 starts the decay on the very first
# resumed iter; LR reaches min_lr at iter 170000, then training ends.
#
# Early stopping is DISABLED — let the fixed cooldown run to completion.

out_dir = 'checkpoints/patzer_v4'
eval_interval = 1000
eval_iters = 50
log_interval = 100

init_from = 'resume'

always_save_checkpoint = True
early_stop_patience_evals = 0        # disabled: let cooldown run to completion
early_stop_min_iters = 0
ckpt_save_interval = 10000           # save ckpt.pt every 10k iters during cooldown
weights_snapshot_interval = 5000     # snapshot every 5k iters during cooldown (val drops steadily)
ckpt_best_min_delta = 0.0001         # lower threshold during cooldown (smaller gains expected)
ckpt_best_cooldown_steps = 2000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v4'         # resumes the same W&B run via wandb_run_id in checkpoint

dataset = 'prepared'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

vocab_size = 4214

# Architecture must match Phase 1 checkpoint exactly
n_layer = 12
n_head = 8
n_embd = 512
bias = False
dropout = 0.0

# WSD cooldown: linear decay from learning_rate → min_lr over cooldown_iters
lr_schedule = 'wsd'
learning_rate = 6e-4
min_lr = 1e-5
warmup_iters = 3000                  # only applies if iter < warmup_iters (won't trigger on resume)
cooldown_start_iter = 140000         # matches checkpoint iter_num — decay starts immediately
cooldown_iters = 30000               # decay to min_lr by iter 170000
max_iters = 171000                   # a small buffer past end of cooldown

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True
