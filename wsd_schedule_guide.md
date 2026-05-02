# WSD Schedule — Implementation Guide

Replacing the cosine LR schedule with Warmup-Stable-Decay (WSD) for v4 onward. The point is to decouple "how long to train" from "how to decay LR" — phase 1 trains at constant LR until early stop, then phase 2 runs a short cooldown from that checkpoint.

## 1. Replace `get_lr()`

Find the existing cosine `get_lr()` in `train.py` (or wherever it lives in the nanoGPT codebase). Replace with:

```python
def get_lr(it):
    # Phase 1: linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Phase 3: linear cooldown (only if active)
    if cooldown_start_iter is not None and it >= cooldown_start_iter:
        progress = min((it - cooldown_start_iter) / cooldown_iters, 1.0)
        return learning_rate + progress * (min_lr - learning_rate)
    # Phase 2: stable (constant LR)
    return learning_rate
```

That's the entire schedule. No cosine, no `lr_decay_iters`. Linear ramp on both ends, flat in the middle.

## 2. New config knobs

Add to the config (replacing `lr_decay_iters` which is no longer used):

```python
cooldown_start_iter = None   # None = no cooldown, stay at constant LR
cooldown_iters     = 30000   # length of linear ramp-down when active
```

Keep `warmup_iters`, `learning_rate`, `min_lr` as they are.

## 3. Two-stage workflow

**Phase 1 — warmup + constant LR, rely on early stopping.**

Config for v4 phase 1:

```python
out_dir = 'checkpoints/patzer_v4_phase1'
warmup_iters = 3000
learning_rate = 6e-4
min_lr = 1e-5
cooldown_start_iter = None     # disabled
cooldown_iters = 30000         # unused in phase 1, set anyway

max_iters = 600000             # generous cap; early stop will end it sooner
early_stop_patience_evals = 25 # ~25k iters of no val improvement
early_stop_min_iters = 100000

# everything else (model, data, batch sizes, betas) stays the same as v3
```

Phase 1 ends when early stopping triggers. Note the iter number at which it stops — call it `STOP_ITER`.

**Phase 2 — cooldown from that checkpoint.**

Config for v4 phase 2:

```python
out_dir = 'checkpoints/patzer_v4_final'
init_from = 'resume'
resume_from = 'checkpoints/patzer_v4_phase1'  # or wherever phase 1 wrote

warmup_iters = 0               # already warmed up
learning_rate = 6e-4           # cooldown starts from this
min_lr = 1e-5                  # cooldown ends here
cooldown_start_iter = STOP_ITER   # ← fill in actual phase-1 stop iter
cooldown_iters = 30000

max_iters = STOP_ITER + 30000  # stop right after cooldown completes
early_stop_patience_evals = 999  # disable early stop during cooldown
```

The cooldown checkpoint at `STOP_ITER + 30000` is the final v4 model.

## 4. Notes on resume behavior

- nanoGPT's resume restores `iter_num` from the checkpoint. The schedule function above keys off the absolute `iter_num`, so cooldown starts at the right point as long as `cooldown_start_iter` matches the resume iter.
- The optimizer state (Adam moments) is restored on resume — this is correct, we want to continue with the existing momentum.
- Don't change `batch_size`, `gradient_accumulation_steps`, `block_size`, or model params between phases. Only the schedule changes.

## 5. Sanity checks before kicking off the long phase-1 run

- Run for ~500 iters, log LR every step, confirm LR linearly ramps up to `learning_rate` over `warmup_iters` then stays flat. Should not decay.
- Resume that short run with phase-2 config, confirm LR linearly decays from `learning_rate` to `min_lr` over `cooldown_iters`.
- Then kick off the real phase-1 run.

