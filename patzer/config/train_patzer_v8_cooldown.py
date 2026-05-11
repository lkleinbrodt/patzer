# patzer v8 — cooldown resume
#
# Resume from the stable-phase checkpoint at iter 112000 and immediately begin
# the 60k-iter WSD linear cooldown (learning_rate → min_lr).
#
# What changed vs train_patzer_v8.py:
#   - init_from = 'resume'
#   - cooldown_start_iter = 112000  (forces cooldown from the first step)
#   - max_iters extended to cover cooldown + a patience tail at min_lr
#   - early_stop_window_evals = 0   (window trigger disabled; the plateau
#     history from the stable phase would fire it immediately on the first eval
#     and kill the run before cooldown has a chance to help)
#   - early_stop_patience_evals kept at 25; evals_without_improvement is reset
#     to 0 automatically when cooldown completes (_post_cooldown_phase logic)
#   - auto_cooldown = True so the post-cooldown patience-reset fires when we
#     enter the min_lr tail (cooldown_start_iter is not None guard prevents
#     auto_cooldown from re-triggering a second cooldown)

out_dir = 'checkpoints/patzer_v8'
init_from = 'resume'

eval_interval = 1000
eval_iters = 100
log_interval = 100

always_save_checkpoint = True
ckpt_save_interval = 10000
weights_snapshot_interval = 10000
ckpt_best_min_delta = 0.001
ckpt_best_cooldown_steps = 2500

# Patience: 25 evals × 1000 steps = 25k steps without improvement → stop.
# Reset fires automatically when cooldown completes (entering min_lr tail).
early_stop_patience_evals = 25
early_stop_min_iters = 0  # no gate — we're past the stable phase
# Window trigger disabled: the val-loss history from the plateau would fire
# it on the first eval and terminate before cooldown can do anything useful.
early_stop_window_evals = 0
early_stop_window_min_improvement = 0.002

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v8_cooldown'

dataset = 'prepared_min_elo_2250'
gradient_accumulation_steps = 2
batch_size = 64
block_size = 256

vocab_size = 4214

gradient_checkpointing = False

# Model: 16L / 16H / 1024d  (~205M params) — must match saved checkpoint
n_layer = 16
n_head = 16
n_embd = 1024
bias = False
dropout = 0.0

# WSD cooldown — starts immediately at resume iter
lr_schedule = 'wsd'
learning_rate = 4e-4
min_lr = 5e-6
warmup_iters = 5000

# Cooldown begins at the checkpoint iter; completes 60k steps later (~172k).
cooldown_start_iter = 112000
cooldown_iters = 60000

# Keep auto_cooldown True so the _post_cooldown_phase patience reset fires
# when we enter the min_lr tail. The `cooldown_start_iter is None` guard in
# train.py prevents it from trying to start a second cooldown.
auto_cooldown = True

# Hard cap: cooldown end (172k) + 35 patience evals (35k) = 207k
max_iters = 207000

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True

peak_flops = 165.2e12
gradient_checkpoint_freq = 1
