# patzer v2 — ~40M param GPT on Lichess games
#
# Changes from v1:
#   - Bigger model: 12L / 8H / 512d  (~40M params vs ~12M)
#   - More data: 3.15M train games, 249M train tokens (vs ~250M tokens, same iters/epoch)
#   - Longer training: 100k iters, LR decays to 1e-5 (vs 1e-4 in v1, so keeps learning longer)
#   - block_size stays 256: avg game = 79 tokens, 256 already spans ~3 full games
#   - batch_size back to 128: same 32k tok/iter as v1 but 2x gradient updates per iter vs 64-batch
#
# Throughput: batch=128 * block=256 = 32,768 tokens/iter
# Iters/epoch: 249M / 32,768 = 7,599
# 100k iters ≈ 13.2 epochs (~3.3B tokens seen)

out_dir = 'checkpoints/patzer_v2'
eval_interval = 1000
eval_iters = 200
log_interval = 100

always_save_checkpoint = True

wandb_log = True
wandb_project = 'patzer'
wandb_run_name = 'patzer_v2'

dataset = 'prepared'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 256

vocab_size = 4214

n_layer = 12
n_head = 8
n_embd = 512
bias = False
dropout = 0.0

learning_rate = 6e-4
max_iters = 100000
lr_decay_iters = 100000
min_lr = 1e-5
beta1 = 0.9
beta2 = 0.95
warmup_iters = 2000     # ~2% of max_iters

device = 'auto'
