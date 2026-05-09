# patzer v7 — Phase 2: WSD cooldown
#
# Resumes from the 180k checkpoint and jumps directly into the 60k linear LR
# cooldown, then runs a min_lr tail with early stopping.
#
# v7 hit 1.50 val at 75k and only 1.48 by 180k — a 0.02 drop over 105k iters.
# The stable phase plateau wasn't worth continuing; jumping to cooldown now.
#
# early_stop_min_iters=241000 blocks early stopping until cooldown finishes (240k),
# then the post-cooldown patience reset fires and we get 15 clean evals at min_lr.

out_dir = 'checkpoints/patzer_v7'
eval_interval = 1000
eval_iters = 100
log_interval = 100

init_from = 'resume'

always_save_checkpoint = True
early_stop_patience_evals = 15      # shorter tail patience (15k iters at min_lr)
early_stop_min_iters = 241000       # block early stop until cooldown ends (180k+60k+1k)
ckpt_save_interval = 10000
weights_snapshot_interval = 5000    # more frequent snapshots — val drops steadily in cooldown
ckpt_best_min_delta = 0.0001        # smaller threshold for cooldown's consistent small gains
ckpt_best_cooldown_steps = 1000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v7'        # resumes same W&B run via wandb_run_id in checkpoint

dataset = 'prepared_min_elo_2100'
gradient_accumulation_steps = 2
batch_size = 64
block_size = 256

vocab_size = 4214

gradient_checkpointing = False

# Architecture must match Phase 1 checkpoint exactly
n_layer = 24
n_head  = 16
n_embd  = 1024
bias    = False
dropout = 0.0

# WSD: jump straight into cooldown from 180k
lr_schedule = 'wsd'
learning_rate = 4e-4
min_lr = 5e-6
warmup_iters = 5000                 # unused on resume (iter_num >> warmup_iters)
cooldown_start_iter = 180000        # start decay immediately — LR reaches min_lr at 240k
cooldown_iters = 60000
auto_cooldown = True                # required: enables the post-cooldown patience reset at 240k
max_iters = 270000                  # 240k + 30k well past the 15-eval patience tail

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True

peak_flops = 165.2e12
gradient_checkpoint_freq = 1
