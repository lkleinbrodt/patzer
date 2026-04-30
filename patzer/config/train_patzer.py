# patzer v0 — 12M param GPT on Lichess games
#
# Data: 250M train tokens, vocab=4214, block_size=256
# Throughput: batch=128 * block=256 = 32,768 tokens/iter → 7,612 iters/epoch
# 40,000 iters ≈ 5.25 epochs (~1.3B tokens seen, near Chinchilla-optimal for 12M params)

out_dir = 'checkpoints/patzer_v1'
eval_interval = 1000   # eval every ~0.4 epochs
eval_iters = 200
log_interval = 100

always_save_checkpoint = True
# Generous early stop: ~15k steps without val improvement, only after 10k steps.
early_stop_patience_evals = 15
early_stop_min_iters = 10000

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v1'

dataset = 'prepared'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

vocab_size = 4214

n_layer = 6
n_head = 6
n_embd = 384
bias = False
dropout = 0.0

learning_rate = 1e-3
max_iters = 40000
lr_decay_iters = max_iters
min_lr = 1e-4
beta1 = 0.9
beta2 = 0.99
warmup_iters = 1000     # ~2% of max_iters

device = 'auto'        # cuda on Vast, mps on Mac — compile handled automatically
