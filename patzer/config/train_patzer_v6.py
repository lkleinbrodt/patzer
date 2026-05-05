# patzer v6 — same architecture as v5 (16L / 16H / 768d); stronger players only (2100+ ELO)
#
# Param count: same as v5 — `GPT(...)` prints ~116.5M non-wpe params at train init.
#
# Data directory: `data/prepared_min_elo_2100/` (from prepare meta below).
# Isolates data quality (2100+ vs v5’s 1800+) while holding architecture constant.
#
# Prepare meta (hash split, seed 1337, min_elo_prepare 2100):
#   total_games:    22,161,998
#   train_games:    21,053,408
#   val_games:       1,108,590
#   train_tokens:    1,732,426,176
#   val_tokens:         91,216,001
#   avg_tokens/game: ~82.3
#
# Throughput: batch=128 * block=256 = 32,768 tokens/iter → ~52.9k iters/epoch (train split).

out_dir = 'checkpoints/patzer_v6'
eval_interval = 1000
eval_iters = 50
log_interval = 100

always_save_checkpoint = True
early_stop_patience_evals = 25
early_stop_min_iters = 150000
ckpt_save_interval = 20000
weights_snapshot_interval = 20000
ckpt_best_min_delta = 0.001
ckpt_best_cooldown_steps = 5000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v6'

dataset = 'prepared_min_elo_2100'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

vocab_size = 4214

# Model: 16L / 16H / 768d — match v5
n_layer = 16
n_head = 16
n_embd = 768
bias = False
dropout = 0.0

# WSD + auto_cooldown — match v5
lr_schedule = 'wsd'
learning_rate = 6e-4
max_iters = 600000
min_lr = 1e-5
warmup_iters = 5000
cooldown_start_iter = None
cooldown_iters = 38000
auto_cooldown = True

beta1 = 0.9
beta2 = 0.95
weight_decay = 1e-1

device = 'auto'
compile = True
