# v0 patzer model
# good for debugging and playing on macbooks and such

out_dir = 'checkpoints/patzer_v0'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'patzer'
wandb_run_name = 'patzer_v0'

dataset = 'prepared'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

vocab_size = 4214

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.00

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 10
lr_decay_iters = max_iters # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device = 'mps' 
compile = False # do not torch compile the model
