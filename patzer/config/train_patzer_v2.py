# patzer v2 — v1-sized (~12M) model on the big dataset (1800+)
#
# Goal: isolate the gain from more data while keeping the original small architecture.
#
# Data (1800+ scrape floor):
#   total_games: 11,738,474
#   train_games: 11,152,529
#   val_games:      585,945
#   train_tokens: 867,638,605
#   val_tokens:    45,557,921
#   avg_tokens/game: 77.8
#
# Throughput: (batch=32 * accum=4) * block=256 = 32,768 tokens/iter
# Iters/epoch: 867.6M / 32,768 = 26,478
# 150k iters ≈ 5.67 epochs (~4.92B tokens seen)

out_dir = 'checkpoints/patzer_v2'
eval_interval = 1000
eval_iters = 200
log_interval = 100

always_save_checkpoint = True
ckpt_save_interval = 10000
weights_snapshot_interval = 10000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v2'

dataset = 'prepared'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 256

vocab_size = 4214

# Model: 6L / 6H / 384d  (~12M params)
n_layer = 6
n_head = 6
n_embd = 384
bias = False
dropout = 0.0

# Keep v1 LR for comparability; train longer because we have vastly more data.
learning_rate = 1e-3
max_iters = 250000
lr_decay_iters = 250000
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.99
warmup_iters = 3000     # ~2% of max_iters

device = 'auto'
