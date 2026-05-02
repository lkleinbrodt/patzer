# Patzer Model Log

---

## Unified eval insights (`eval/evaluate.py leaderboard`)

Ratings fit a Bradley–Terry model on every row in `eval/results.db`: Patzer **self-play**, **Patzer vs capped Stockfish**, and cross-checkpoint mixes. Player labels are `patzer_vN@{k}` with `k` = training step in **thousands**. Anchored Stockfish opponents are omitted from the printed table; the scale is an internal ladder, not Lichess Elo. Refresh with `python eval/evaluate.py leaderboard`.

### Peak strength per model family


| Family | Best checkpoint (logged) | Elo ± σ       | Notes                                       |
| ------ | ------------------------ | ------------- | ------------------------------------------- |
| v3     | `patzer_v3@180`          | **1632 ± 12** | Strongest in DB; 1134 games                 |
| v2     | `patzer_v2@288`          | **1475 ± 9**  | Tied within noise with `@274` (1474 ± 10)   |
| v1     | `patzer_v1@40`           | **1203 ± 13** | Only v1 snapshot in DB — no iteration sweep |


Across families, v3’s best sits about **~160 Elo** above v2’s best and **~430** above v1 on this combined ladder.

### Does more training always mean higher ladder Elo?

**v3 — mostly yes (in the logged band).** Checkpoints `@40` → `@80` → `@111` → `@145` → `@180` show **strictly increasing** Elo (1430 → 1508 → 1601 → 1625 → 1632). The early `**@10`** bucket is an exception (1197 ± 16, fewer games, wide σ): it behaves like an under-trained snapshot rather than a step in the same curve. Interpreting “later is better” should start from `@40` onward, where the trend is clean.

**v2 — no.** Elo is **not monotonic** in training step. Ordering by `k`: `@50` is weakest (~~1190); `**@120` (~~1417) beats `@180` (~~1340)** by a large margin; `@220` (~~1414) still trails `@120`. The best two entries are the latest large-step snapshots, `**@274`** and `**@288`** (~1474–1475). So the run had a **mid-training regression** (roughly 120k–220k) versus the ~120k checkpoint, then recovery and a late peak. That is exactly the regime where val-loss–based “best weights” and multi-snapshot evals matter: “last iter” or a single mid-run save would mis-rank strength.

**v1 — undetermined.** A single checkpoint (`@40`) is on the ladder; we cannot test iteration vs Elo within v1 from this DB.

---

## v1 — baseline

**Checkpoint:** `checkpoints/patzer_v1`
**Trained:** April 2026 on Vast.ai

### Architecture


| param      | value |
| ---------- | ----- |
| n_layer    | 6     |
| n_head     | 6     |
| n_embd     | 384   |
| block_size | 256   |
| params     | ~12M  |


### Training


| param            | value                             |
| ---------------- | --------------------------------- |
| data             | ~80M tokens, 1800+ ELO, ~1M games |
| batch_size       | 128 (32 × 4 accum)                |
| max_iters        | 40,000                            |
| lr               | 1e-3 → 1e-4 (cosine)              |
| val loss (final) | ~1.84                             |
| hardware         | Vast.ai (GPU unrecorded)          |
| step time        | unrecorded                        |
| throughput       | unrecorded                        |
| eval time        | unrecorded                        |


**Note:** Val loss was still declining at iter 40k — underfit; LR floor 1e-4. Best val was ~30k, not 40k.

### Eval

**Unified ladder:** `patzer_v1@40` — **1203 ± 13** over 1118 games (see insights above). Only one v1 step appears in `results.db`. Older `**eval/tournament.py`** run (~1254 ± 26 from 190 games, temp **0.1**) is a separate pipeline and is **not** in `results.db`.

### Lessons

- Play/eval default is temp **0** (greedy) in `eval/engine.py`; saved tournament rows here are **0.1**.
- 6 layers is too small to learn tactical patterns; mostly learns opening statistics
- LR min should be 1e-5, not 1e-4 — v1 stopped learning too early
- More training beyond 30k had diminishing returns given the LR floor

---

## v2 — small model, big data (rerun)

**Checkpoint:** `checkpoints/patzer_v2`
**Config:** `patzer/config/train_patzer_v2.py`
**Trained:** (planned) overnight Apr 2026

### Architecture


| param      | value |
| ---------- | ----- |
| n_layer    | 6     |
| n_head     | 6     |
| n_embd     | 384   |
| block_size | 256   |
| params     | ~12M  |


### Training


| param      | value                                                    |
| ---------- | -------------------------------------------------------- |
| data       | ~868M train tokens, 1800+ ELO, ~11.15M games             |
| batch_size | 128 (32 × 4 accum)                                       |
| max_iters  | 150,000 (was still learning; later resumed toward 250k)  |
| lr         | 1e-3 → 1e-5 (cosine)                                     |
| hardware   | RTX 3060                                                 |
| step time  | ~155 ms/step                                             |
| throughput | ~211k tokens/sec                                         |
| eval time  | ~6,280 ms (200 iters × batch 32 × 2 splits = 12.8k seqs) |


### Eval (unified ladder)

See **Unified eval insights** — peak `**patzer_v2@288`** / `**@274`** (~1475). Mid-run `**@180`** is weaker than `**@120`** on the same ladder; not monotonic in step.

---

## v3 — ~40M model, big data

**Checkpoint:** `checkpoints/patzer_v3`
**Config:** `patzer/config/train_patzer_v3.py`
**Trained:** (planned) overnight Apr 2026

### Architecture


| param      | value |
| ---------- | ----- |
| n_layer    | 12    |
| n_head     | 8     |
| n_embd     | 512   |
| block_size | 256   |
| params     | ~40M  |


### Training


| param           | value                                                      |
| --------------- | ---------------------------------------------------------- |
| data            | ~868M train tokens, 1800+ ELO, ~11.15M games               |
| batch_size      | 128 (32 × 4 accum)                                         |
| max_iters       | 150,000                                                    |
| lr              | 6e-4 → 1e-5 (cosine)                                       |
| val loss (best) | **1.514** at iter 145k                                     |
| gen gap         | +0.025 (first overfitting signal in any run)               |
| hardware        | RTX 3060 (phase 1) / RTX 4060 Ti (phase 2)                 |
| step time       | ~470 ms/step (3060) · ~321 ms/step (4060 Ti)               |
| throughput      | ~70k tokens/sec (3060) · ~102k tokens/sec (4060 Ti)        |
| eval time       | ~32,000 ms (3060) · ~34,000 ms (4060 Ti); includes R2 sync |


### Eval (unified ladder)

See **Unified eval insights** — best `**patzer_v3@180`** (**1632 ± 12**); ladder rises monotonically from `@40` through `@180` among logged steps; `**@10`** is an early outlier.

### Lessons

- 40M params is roughly the right size for ~868M tokens. Gen gap at +0.025 means capacity is no longer the binding constraint — data is.
- The cosine tail (LR 1e-4 → 1e-5) did real work: 0.07 nats of improvement from iter 100k → 145k. The earlier "starvation" hypothesis was wrong.
- First genuine convergence plateau: last 10k iters bounced within ~0.007 noise floor.
- `lr_decay_iters` (150k) vs `max_iters` (300k) misalignment meant half the scheduled run was at dead min_lr — wasted compute.
- Speedup checklist identified: drop `gradient_accumulation_steps` to 1, raise physical `batch_size`, enable `torch.compile`, align `lr_decay_iters = max_iters`.

---

## v4 — same architecture, 3× data, WSD schedule

**Checkpoint:** `checkpoints/patzer_v4`
**Config:** `patzer/config/train_patzer_v4.py`
**Trained:** (planned) May 2026

### Goal

Test the data scaling hypothesis. v3 showed a gen gap of +0.025 on ~~868M tokens, indicating the 40M-param model was becoming data-bound. v4 keeps the identical architecture and trains on 3.3× more data (~~2.85B tokens, 36M games) to see if the gap closes and val loss drops. This is the highest-EV experiment in the current plan — directly isolating data quantity as the variable.

Also introduces the **WSD (Warmup-Stable-Decay)** LR schedule, replacing cosine. This decouples "how long to train" from "how to decay LR": phase 1 trains at constant LR until early stop, then phase 2 runs a short linear cooldown from that checkpoint.

**Expected outcome:** val loss in the 1.40–1.45 range, gen gap closing back toward v2-like levels (~0.005–0.010). If the gap stays open at +0.025 with 3× data, the model is hitting a capacity wall and v5 should scale architecture rather than data quality.

### Architecture


| param      | value             |
| ---------- | ----------------- |
| n_layer    | 12                |
| n_head     | 8                 |
| n_embd     | 512               |
| block_size | 256               |
| params     | ~40M (same as v3) |


### Training


| param       | value                                            |
| ----------- | ------------------------------------------------ |
| data        | ~2.85B train tokens, 1800+ ELO, ~36.3M games     |
| batch_size  | 128 (physical, no gradient accumulation)         |
| lr_schedule | WSD (warmup → stable → linear cooldown)          |
| lr          | 6e-4 (constant after warmup)                     |
| cooldown    | 30k iters, linear ramp to 1e-5                   |
| max_iters   | 600k cap (early stop ends phase 1 sooner)        |
| compile     | True                                             |
| hardware    | RTX 4060 Ti (16 GB)                              |
| step time   | ~318 ms/step (observed at iter 74k–75k)          |
| throughput  | ~103k tokens/sec                                 |
| eval time   | ~58,000 ms/eval (compute ~40s + R2 sync ~16–20s) |
| mfu         | ~7–8% (rising; 7.23% at eval, ~8% mid-run)       |


**Throughput:** 128 × 256 = 32,768 tokens/iter → ~86,898 iters/epoch.

**Two-phase workflow:**

1. **Phase 1** (`train_patzer_v4.py`): warmup (3k iters) → constant LR at 6e-4. Early stop triggers when val plateaus (`patience=25`, `min_iters=100k`). Note the stop iter.
2. **Phase 2** (`train_patzer_v4_cooldown.py`, created after phase 1): resume from phase 1 checkpoint, 30k-iter linear cooldown from 6e-4 → 1e-5. This produces the final v4 model.

### Changes from v3

1. **3.3× more training data** (2.85B vs 868M tokens, 36M vs 11M games). Same 1800+ ELO filter.
2. **WSD schedule** replaces cosine. Constant LR during training lets early stop decide when to decay, instead of baking a fixed decay length into the config. The v3 retrospective showed `lr_decay_iters` misalignment wasted compute; WSD eliminates that failure mode entirely.
3. `**gradient_accumulation_steps` = 1, `batch_size` = 128** — same effective batch, fewer Python/optimizer overheads. Expected ~1.3–1.7× faster step time. Falls back to batch 96 if 128 OOMs on 12GB.
4. `**compile = True`** — explicit torch.compile for ~1.2–1.4× on Ampere+.

### Lessons so far (in-progress)

- **Step throughput matched v3 exactly.** v3 on 4060 Ti = 321 ms/step; v4 on 4060 Ti = 318 ms/step. The "speedup checklist" from v3 (drop accum, explicit compile) delivered no actual gain because `device='auto'` on CUDA was already enabling compile in v3, and swapping 4 × batch-32 for 1 × batch-128 is identical compute. The optimization wins were already present.
- **eval_iters=200 with batch_size=128 was a silent 4× blowup.** `estimate_loss()` runs `eval_iters` batches for each of the train and val splits = 400 total forward passes. v2/v3 used batch=32 → 12,800 sequences per eval. v4 inherited the same `eval_iters=200` but with batch=128 → **51,200 sequences** — 4× more work, ~40s of eval compute, up to 58s total including R2 sync. This wasn't caught because (a) eval time isn't printed separately in logs — it's baked into the slow iteration's `time` field, and (b) `batch_size` and `eval_iters` are set independently with no guard. Fixed mid-run by setting `eval_iters=50`, which gives the same 12,800-sequence coverage as v2/v3 and drops eval time to ~10s.
- **R2 uploads were blocking training and uploading too frequently.** Switched to async uploads (`r2.push_async`) so training resumes immediately after `torch.save`. Added `ckpt_best_min_delta=0.001` and `ckpt_best_cooldown_steps=5000` to prevent R2 from being hammered during the early-training phase when val improves every eval. Also fixed a latent bug where snapshot `copy_object` could race against the upload and capture the previous weights; the copy is now chained inside the async task.
- **MFU is low (~7–8%) and expected.** A 40M model on a gaming GPU is memory-bandwidth-bound, not FLOP-bound. MFU rises over the run as the running average warms up; actual steady-state is ~8%.

---

## Hardware & compute reference


| Model | GPU         | Arch        | Params | Effective batch | Step time | Tokens/sec | Eval time (excl. R2) | eval_iters × batch |
| ----- | ----------- | ----------- | ------ | --------------- | --------- | ---------- | -------------------- | ------------------ |
| v1    | unknown     | 6L/6H/384d  | ~12M   | 128 (32×4)      | —         | —          | —                    | 200 × 32           |
| v2    | RTX 3060    | 6L/6H/384d  | ~12M   | 128 (32×4)      | ~155 ms   | ~211k      | ~6 s                 | 200 × 32           |
| v3    | RTX 3060    | 12L/8H/512d | ~40M   | 128 (32×4)      | ~470 ms   | ~70k       | ~32 s                | 200 × 32           |
| v3    | RTX 4060 Ti | 12L/8H/512d | ~40M   | 128 (32×4)      | ~321 ms   | ~102k      | ~34 s                | 200 × 32           |
| v4    | RTX 4060 Ti | 12L/8H/512d | ~40M   | 128 (128×1)     | ~318 ms   | ~103k      | ~40 s (+16s R2)      | 200 × 128          |


**Notes:**

- v3→v4 on the same GPU: identical step time confirms accum-drop and explicit compile were already implicit in v3 on CUDA.
- v4 eval covers 4× more sequences per eval than v2/v3 (51,200 vs 12,800), making each eval ~4× slower with no statistical benefit. Reduce `eval_iters` to 50 in v5.
- MFU of 7–8% is normal for small models on gaming GPUs (memory-bandwidth-bound, not FLOP-bound).
- "Eval time" above is pure forward-pass compute; add R2 sync time (~16–20 s) whenever a new best val is saved.