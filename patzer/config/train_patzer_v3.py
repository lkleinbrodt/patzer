# patzer v3 — ~40M param GPT on the big dataset (1800+)
#
# Data (1800+ scrape floor):
#   total_games: 11,738,474
#   train_games: 11,152,529
#   val_games:      585,945
#   train_tokens: 867,638,605
#   val_tokens:    45,557,921
#   avg_tokens/game: 77.8
#
# Throughput: (batch=32 * accum=4) * block=256 = 32,768 tokens/iter → 26,478 iters/epoch
# 150,000 iters ≈ 5.67 epochs (~4.92B tokens seen)

out_dir = 'checkpoints/patzer_v3'
eval_interval = 1000
eval_iters = 200
log_interval = 100

always_save_checkpoint = True
# Generous early stop: ~15k steps without val improvement, only after 10k steps.
early_stop_patience_evals = 15
early_stop_min_iters = 10000
ckpt_save_interval = 10000
weights_snapshot_interval = 10000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v3'

dataset = 'prepared'
gradient_accumulation_steps = 4
batch_size = 32
block_size = 256

vocab_size = 4214

# Model: 12L / 8H / 512d  (~40M params)
n_layer = 12
n_head = 8
n_embd = 512
bias = False
dropout = 0.0

learning_rate = 6e-4
max_iters = 300000
lr_decay_iters = 150000
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.95
warmup_iters = 3000     # ~2% of max_iters

device = 'auto'
