"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import json
import os
import time
from pathlib import Path
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, broadcast

from model import GPTConfig, GPT
import r2

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
# Save the "latest" checkpoint (`ckpt.pt`) at most every N steps (0 = every eval).
# `weights_best.pt` is still written whenever val improves.
ckpt_save_interval = 0
# Best-weights checkpoint save policy (to control bandwidth/egress):
# - `weights_best.pt` is used for play/eval and does NOT need optimizer state.
# - It is always weights-only (no optimizer state).
# Only save `weights_best.pt` if val loss improves by at least this absolute amount.
ckpt_best_min_delta = 0.0
# Optional cooldown: only save best at most once every N steps (0 = no cooldown).
ckpt_best_cooldown_steps = 0
# Create weights snapshots `weights_iter_*.pt` (weights-only) at most every N steps,
# but only when a new best is saved. 0 disables snapshots.
weights_snapshot_interval = 10000
# Early stopping on val loss (0 = disabled). Counts consecutive evals without val improvement.
early_stop_patience_evals = 0
early_stop_min_iters = 0 # require at least this many steps before early stop can trigger
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# Persisted in checkpoints so `--init_from=resume` can continue the same run.
# Empty string means "create a fresh run".
wandb_run_id = ""
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(Path(__file__).parent / 'configurator.py').read())
config = {k: globals()[k] for k in config_keys} # will be useful for logging

if device == 'auto':
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
        compile = False
    else:
        device = 'cpu'
        compile = False
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)

# pull training binaries from R2 if they don't exist locally
if master_process and not os.path.exists(os.path.join(data_dir, 'train.bin')):
    print(f"[r2] {data_dir}/train.bin not found locally, pulling from R2...")
    r2.pull_dir(data_dir, data_dir)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9
evals_without_improvement = 0

meta_path = os.path.join(data_dir, 'meta.json')
meta_vocab_size = None
if os.path.exists(meta_path):   
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    meta_vocab_size = meta['vocab_size']

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"init_from=resume but checkpoint not found at {ckpt_path}. "
            f"Did you forget to download from R2 (or set R2 env vars)?"
        )
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    evals_without_improvement = int(checkpoint.get('evals_without_improvement', 0))
    wandb_run_id = str(checkpoint.get('wandb_run_id', wandb_run_id or ""))
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb_init_kwargs = dict(project=wandb_project, name=wandb_run_name, config=config)
    # If we have a stored run id and we're resuming training, keep logging into that same run.
    # Using explicit `step=iter_num` on wandb.log keeps the x-axis consistent across restarts.
    if init_from == 'resume' and wandb_run_id:
        wandb_init_kwargs.update(id=wandb_run_id, resume="must")
    run = wandb.init(**wandb_init_kwargs)
    # Make `iter` the canonical x-axis in W&B. This avoids confusion if W&B defaults
    # charts to "Step" (log call index) instead of the actual training iteration.
    wandb.define_metric("iter")
    wandb.define_metric("*", step_metric="iter")
    # Capture id for checkpointing (fresh run or missing id in old checkpoints).
    if not wandb_run_id:
        wandb_run_id = run.id

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        val_loss = losses['val'].item()
        train_loss_eval = losses['train'].item()
        print(f"step {iter_num}: train loss {train_loss_eval:.4f}, val loss {val_loss:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": train_loss_eval,
                "val/loss": val_loss,
                "lr": lr,
                "mfu": running_mfu*100,
            }, step=iter_num)
        # append to metrics log and push to R2 so training curve is always visible
        import json as _json, time as _time
        metrics_local = os.path.join(out_dir, 'metrics.jsonl')
        with open(metrics_local, 'a') as _f:
            _f.write(_json.dumps({
                "iter": iter_num,
                "train_loss": train_loss_eval,
                "val_loss": val_loss,
                "lr": lr,
                "mfu": round(running_mfu * 100, 2),
                "ts": _time.time(),
            }) + "\n")
        r2.push_file(metrics_local, f"{out_dir}/metrics.jsonl")
        improved = val_loss < (best_val_loss - ckpt_best_min_delta)
        if early_stop_patience_evals > 0:
            if improved:
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1
        # Save behavior:
        # - `weights_best.pt` on val improvement (weights-only; for eval/play)
        # - optional `weights_iter_*.pt` snapshots created by copying `weights_best.pt` (no extra upload)
        # - `ckpt.pt` ("latest") at most every `ckpt_save_interval` steps (or every eval if interval == 0)
        save_latest = False
        if always_save_checkpoint and iter_num > 0:
            save_latest = (ckpt_save_interval == 0) or (iter_num % ckpt_save_interval == 0)
        if improved or save_latest:
            if improved:
                best_val_loss = val_loss
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                    'evals_without_improvement': evals_without_improvement,
                    'config': config,
                    'wandb_run_id': wandb_run_id,
                }
                if save_latest:
                    print(f"saving checkpoint to {out_dir}")
                    ckpt_local = os.path.join(out_dir, 'ckpt.pt')
                    torch.save(checkpoint, ckpt_local)
                    r2.push_file(ckpt_local)
                if improved:
                    # Best val checkpoint is for play/eval; keep it small (weights-only) to reduce R2 egress.
                    can_save_best = True
                    if ckpt_best_cooldown_steps and (iter_num % eval_interval == 0):
                        # cooldown expressed in steps; if non-zero, enforce at most one best save per window
                        # by checking the last saved iter in the existing file (if present).
                        best_weights_local = os.path.join(out_dir, 'weights_best.pt')
                        if os.path.exists(best_weights_local):
                            try:
                                prev = torch.load(best_weights_local, map_location='cpu')
                                prev_it = int(prev.get('iter_num', -10**18))
                                if (iter_num - prev_it) < ckpt_best_cooldown_steps:
                                    can_save_best = False
                            except Exception:
                                pass

                    if can_save_best:
                        best_weights_local = os.path.join(out_dir, 'weights_best.pt')
                        best_weights = {
                            'model': checkpoint['model'],
                            'model_args': checkpoint['model_args'],
                            'iter_num': checkpoint['iter_num'],
                            'val_loss': checkpoint['val_loss'],
                            'best_val_loss': checkpoint['best_val_loss'],
                            'config': checkpoint['config'],
                        }
                        torch.save(best_weights, best_weights_local)
                        pushed_best = r2.push_file(best_weights_local)

                        # Optionally create a rate-limited weights snapshot on improvement.
                        if weights_snapshot_interval and pushed_best:
                            try:
                                import re as _re
                                from pathlib import Path as _Path

                                outp = _Path(out_dir)
                                last_it = None
                                for p in outp.glob("weights_iter_*.pt"):
                                    m = _re.match(r"weights_iter_(\d{6,})\.pt$", p.name)
                                    if m:
                                        try:
                                            it = int(m.group(1))
                                            last_it = it if (last_it is None or it > last_it) else last_it
                                        except ValueError:
                                            pass
                                if last_it is None or (iter_num - last_it) >= int(weights_snapshot_interval):
                                    snap_key = f"{out_dir}/weights_iter_{iter_num:06d}.pt"
                                    # Server-side copy avoids re-uploading.
                                    r2.copy_object(f"{out_dir}/weights_best.pt", snap_key, overwrite=False)
                            except Exception:
                                pass
    if ddp and iter_num % eval_interval == 0:
        sync = torch.tensor([evals_without_improvement], dtype=torch.int, device=device)
        broadcast(sync, src=0)
        evals_without_improvement = int(sync.item())
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
    if (
        early_stop_patience_evals > 0
        and iter_num >= early_stop_min_iters
        and evals_without_improvement >= early_stop_patience_evals
    ):
        if master_process:
            print(
                f"early stop: val loss did not improve for {early_stop_patience_evals} evals "
                f"(eval_interval={eval_interval}), at iter {iter_num}"
            )
        break

if ddp:
    destroy_process_group()
