# Patzer Model Log

---

## Unified eval insights (`eval/evaluate.py leaderboard`)

Ratings fit a Bradley–Terry model on every row in `eval/results.db`: Patzer **self-play**, **Patzer vs capped Stockfish**, and cross-checkpoint mixes. Player labels are `patzer_vN@{k}` with `k` = training step in **thousands**. **Stockfish opponents are anchored** at their matchup Elo (hidden from the printed table); Patzer logits are **fitted** on the same graph, so the printed Elo is a **Stockfish-centered scale** — comparable across runs in one DB snapshot, but **not** the same numeric scale as older “internal ladder only” exports. Not Lichess rating. Refresh with `python eval/evaluate.py leaderboard`. *Local DB snapshot (May 2026): **7268** games total in `results.db`.*

### Peak strength per model family


| Family | Best checkpoint (logged) | Elo ± σ       | Notes                                            |
| ------ | ------------------------ | ------------- | ------------------------------------------------ |
| v4     | `patzer_v4@201`          | **1600 ± 11** | #1 on ladder; 1124 games on this row             |
| v3     | `patzer_v3@180`          | **1579 ± 10** | Was strongest before v4 evals landed; 1484 games |
| v2     | `patzer_v2@288`          | **1402 ± 11** | Within noise of `@274` (1389 ± 14)               |
| v1     | `patzer_v1@40`           | **1196 ± 13** | Only v1 snapshot in DB — no iteration sweep      |


Across families, v4’s best sits about **~200** above v2’s best and **~400** above v1 on this anchored ladder (v3’s best is **~21** below v4’s best).

### Does more training always mean higher ladder Elo?

**v3 — mostly yes (in the logged band).** Checkpoints `@40` → `@80` → `@111` → `@145` → `@180` show **strictly increasing** Elo (1373 → 1476 → 1530 → 1569 → 1579). The early `**@10`** bucket is an exception (1124 ± 21, fewer games, wide σ): it behaves like an under-trained snapshot rather than a step in the same curve. Interpreting “later is better” should start from `@40` onward, where the trend is clean.

**v2 — no.** Elo is **not monotonic** in training step. Ordering by `k`: `@50` is weakest (~~1195); **`@120` (~~1302) beats `@180` (~~1289)** by a clear margin; `@220` (~~1384) still trails `@120`. The best two entries are the latest large-step snapshots, `**@274`** and `**@288**` (~1389–1402). So the run had a **mid-training regression** (roughly 120k–220k) versus the ~120k checkpoint, then recovery and a late peak. That is exactly the regime where val-loss–based “best weights” and multi-snapshot evals matter: “last iter” or a single mid-run save would mis-rank strength.

**v4 — yes among logged steps.** `@40` → `@81` → `@104` → `@201` rise monotonically (1361 → 1403 → 1416 → 1600). Mid-run `@104` still trails **v3 `@111`** (1530) on the same ladder — extra data does not guarantee dominance at equal *optimizer* step vs the older run’s snapshot curve.

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
| val loss (best)  | **1.841** at iter **39k** (`metrics.jsonl` from R2) |
| val loss (final) | **1.843** at iter **40k**                           |
| hardware         | Vast.ai (GPU unrecorded)                              |
| step time        | unrecorded                                            |
| throughput       | unrecorded                                            |
| eval time        | unrecorded                                            |


**Note:** Val loss was still improving into the late 30k band; **best val at iter 39k** (final row 40k is slightly worse). Underfit at LR floor 1e-4.

### Eval

**Unified ladder:** `patzer_v1@40` — **1196 ± 13** over 1162 games (see insights above). Only one v1 step appears in `results.db`. Older `**eval/tournament.py`** run (~1254 ± 26 from 190 games, temp **0.1**) is a separate pipeline and is **not** in `results.db`.

### Lessons

- Play/eval default is temp **0** (greedy) in `eval/engine.py`; saved tournament rows here are **0.1**.
- 6 layers is too small to learn tactical patterns; mostly learns opening statistics
- LR min should be 1e-5, not 1e-4 — v1 stopped learning too early
- More training beyond **39k** (best val) had diminishing returns given the LR floor

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
| val loss (best) | **1.650** at iter **288k** (`metrics.jsonl` from R2) |
| hardware   | RTX 3060                                                 |
| step time  | ~155 ms/step                                             |
| throughput | ~211k tokens/sec                                         |
| eval time  | ~6,280 ms (200 iters × batch 32 × 2 splits = 12.8k seqs) |


### Eval (unified ladder)

See **Unified eval insights** — peak `**patzer_v2@288`** / `**@274`** (~1402 / ~1389). Mid-run `**@180`** is weaker than `**@120`** on the same ladder; not monotonic in step.

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
| val loss (best) | **1.510** at iter **180k** (`metrics.jsonl` from R2; 145k was **1.514**) |
| gen gap         | +0.025 (first overfitting signal in any run)               |
| hardware        | RTX 3060 (phase 1) / RTX 4060 Ti (phase 2)                 |
| step time       | ~470 ms/step (3060) · ~321 ms/step (4060 Ti)               |
| throughput      | ~70k tokens/sec (3060) · ~102k tokens/sec (4060 Ti)        |
| eval time       | ~32,000 ms (3060) · ~34,000 ms (4060 Ti); includes R2 sync |


### Eval (unified ladder)

See **Unified eval insights** — best `**patzer_v3@180`** (**1579 ± 10**); ladder rises monotonically from `@40` through `@180` among logged steps; `**@10`** is an early outlier.

### Lessons

- 40M params is roughly the right size for ~868M tokens. Gen gap at +0.025 means capacity is no longer the binding constraint — data is.
- The cosine tail (LR 1e-4 → 1e-5) did real work: 0.07 nats of improvement from iter 100k → 145k. The earlier "starvation" hypothesis was wrong.
- First genuine convergence plateau: last 10k iters bounced within ~0.007 noise floor.
- `lr_decay_iters` (150k) vs `max_iters` (300k) misalignment meant half the scheduled run was at dead min_lr — wasted compute.
- Speedup checklist identified: drop `gradient_accumulation_steps` to 1, raise physical `batch_size`, enable `torch.compile`, align `lr_decay_iters = max_iters`.

---

## v4 — same architecture, 3× data, WSD schedule

**Checkpoint:** `checkpoints/patzer_v4`
**Config:** `patzer/config/train_patzer_v4.py` (phase 1) + `train_patzer_v4_cooldown.py` (phase 2; see `PROJECT_LOG.md`)
**Trained:** May 2026 (phase 1 early stop ~145k → manual cooldown resume → min_lr tail). **Best val at iter 201k**; **stopped manually ~203k** — no improvement after 201k (last `metrics.jsonl` row shows val ~1.506 vs best ~1.501).

### Goal

Test the data scaling hypothesis. v3 showed a gen gap of +0.025 on ~~868M tokens, indicating the 40M-param model was becoming data-bound. v4 keeps the identical architecture and trains on 3.3× more data (~~2.85B tokens, 36M games) to see if the gap closes and val loss drops. This is the highest-EV experiment in the current plan — directly isolating data quantity as the variable.

Also introduces the **WSD (Warmup-Stable-Decay)** LR schedule, replacing cosine. This decouples "how long to train" from "how to decay LR": phase 1 trains at constant LR until early stop, then phase 2 runs a short linear cooldown from that checkpoint.

**Pre-run expectation:** val loss in the 1.40–1.45 range, gen gap closing back toward v2-like levels (~0.005–0.010).

**What happened:** Val **best ~1.501** at iter **201k** (`metrics.jsonl`). Last logged row **203k** has val **~1.506** (no further improvement after 201k; small uptick vs best). Only **~0.009 nats** better than v3’s best val (**~1.510** at 180k), not the hoped 1.40s band — LM loss still looks **capacity-limited** at 40M. **Gen gap nearly vanished** at the end (e.g. iter 201k: train ~1.500 vs val ~1.501), unlike v3’s +0.025 — consistent with “more diverse tokens, same capacity.”

**Play strength (unified ladder, Stockfish-anchored):** `patzer_v4@201` (**1600 ± 11**, 1124 games) ranks **#1** in `eval/results.db` as of the May 2026 snapshot, **~21 Elo** above `patzer_v3@180` (**1579 ± 10**). So the small LM gain + long schedule **did** translate to measurable strength on the eval graph, even though raw val loss did not blow past v3.

### Architecture


| param      | value             |
| ---------- | ----------------- |
| n_layer    | 12                |
| n_head     | 8                 |
| n_embd     | 512               |
| block_size | 256               |
| params     | ~40M (same as v3) |


### Training


| param            | value                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| data             | ~2.85B train tokens, 1800+ ELO, ~36.3M games                                                                               |
| batch_size       | 128 (physical, no gradient accumulation)                                                                                   |
| lr_schedule      | WSD (warmup → stable → linear cooldown)                                                                                    |
| lr               | 6e-4 (constant after warmup)                                                                                               |
| cooldown         | 30k iters, linear ramp to 1e-5                                                                                             |
| max_iters        | 600k cap; phase 1 early stop ~145k; cooldown + min_lr tail; **stopped ~203k** (manual — flat after best at 201k)           |
| compile          | True                                                                                                                       |
| val loss (best)  | **1.50137** (iter **201k**; `metrics.jsonl` / `weights_best` alignment)                                                    |
| stable phase end | iter ~145k @ val ~1.657 (early stop, patience=25)                                                                          |
| cooldown end     | iter ~170k @ val ~1.507 (linear ramp to min_lr)                                                                            |
| hardware         | RTX 4060 Ti (16 GB)                                                                                                        |
| step time        | ~318 ms/step (observed at iter 74k–75k)                                                                                    |
| throughput       | ~103k tokens/sec                                                                                                           |
| eval_iters       | **50** × batch 128 (= 12,800 seqs/split, same as v2/v3 coverage); early run briefly used 200×128 (~58s/eval) — see Lessons |
| eval time        | ~10s compute/eval after fix + R2 when a new best is saved                                                                  |
| mfu              | ~7–8% (rising; 7.23% at eval, ~8% mid-run)                                                                                 |


**Throughput:** 128 × 256 = 32,768 tokens/iter → ~86,898 iters/epoch.

**Two-phase workflow (completed):**

1. **Phase 1** (`train_patzer_v4.py`): warmup (3k iters) → constant LR at 6e-4. Early stop when val plateaus (`patience=25`, `min_iters=100k`) — landed ~145k.
2. **Phase 2** (`train_patzer_v4_cooldown.py`): resume, 30k-iter linear cooldown 6e-4 → 1e-5, then min_lr continuation; **best val at 201k**, then flat until **manual stop ~203k** (see `PROJECT_LOG` for patience / `max_iters` tweaks).

### Eval (unified ladder)

Full table: `python eval/evaluate.py leaderboard`. v4 rows in the May 2026 DB snapshot:


| Checkpoint      | Elo ± σ   | Games |
| --------------- | --------- | ----- |
| `patzer_v4@201` | 1600 ± 11 | 1124  |
| `patzer_v4@104` | 1416 ± 13 | 840   |
| `patzer_v4@81`  | 1403 ± 14 | 700   |
| `patzer_v4@40`  | 1361 ± 15 | 602   |


`@201` is the **best-val** step and matches `**weights_best.pt*`* for play/eval (training ended ~203k without beating 201k). **H2H-only rank** matches overall rank (#1) for `@201` in that snapshot — not true for every checkpoint in every DB.

### Changes from v3

1. **3.3× more training data** (2.85B vs 868M tokens, 36M vs 11M games). Same 1800+ ELO filter.
2. **WSD schedule** replaces cosine. Constant LR during training lets early stop decide when to decay, instead of baking a fixed decay length into the config. The v3 retrospective showed `lr_decay_iters` misalignment wasted compute; WSD eliminates that failure mode entirely.
3. `**gradient_accumulation_steps` = 1, `batch_size` = 128** — same effective batch, fewer Python/optimizer overheads. Expected ~1.3–1.7× faster step time. Falls back to batch 96 if 128 OOMs on 12GB.
4. `**compile = True`** — explicit torch.compile for ~1.2–1.4× on Ampere+.

### Lessons

- **Step throughput matched v3 exactly.** v3 on 4060 Ti = 321 ms/step; v4 on 4060 Ti = 318 ms/step. The "speedup checklist" from v3 (drop accum, explicit compile) delivered no actual gain because `device='auto'` on CUDA was already enabling compile in v3, and swapping 4 × batch-32 for 1 × batch-128 is identical compute. The optimization wins were already present.
- **eval_iters=200 with batch_size=128 was a silent 4× blowup.** `estimate_loss()` runs `eval_iters` batches for each of the train and val splits = 400 total forward passes. v2/v3 used batch=32 → 12,800 sequences per eval. v4 inherited the same `eval_iters=200` but with batch=128 → **51,200 sequences** — 4× more work, ~40s of eval compute, up to 58s total including R2 sync. This wasn't caught because (a) eval time isn't printed separately in logs — it's baked into the slow iteration's `time` field, and (b) `batch_size` and `eval_iters` are set independently with no guard. Fixed mid-run by setting `eval_iters=50`, which gives the same 12,800-sequence coverage as v2/v3 and drops eval time to ~10s. **Current `train_patzer_v4.py` has `eval_iters=50`.**
- **R2 uploads were blocking training and uploading too frequently.** Switched to async uploads (`r2.push_async`) so training resumes immediately after `torch.save`. Added `ckpt_best_min_delta=0.001` and `ckpt_best_cooldown_steps=5000` to prevent R2 from being hammered during the early-training phase when val improves every eval. Also fixed a latent bug where snapshot `copy_object` could race against the upload and capture the previous weights; the copy is now chained inside the async task.
- **MFU is low (~7–8%) and expected.** A 40M model on a gaming GPU is memory-bandwidth-bound, not FLOP-bound. MFU rises over the run as the running average warms up; actual steady-state is ~8%.

#### WSD schedule retrospective

Per-iter val-loss progress across phases:


| Phase                       | iter range  | Δ val loss           | nats / 1k iters                                |
| --------------------------- | ----------- | -------------------- | ---------------------------------------------- |
| Stable, mid                 | 50k → 100k  | -0.030               | 0.0006                                         |
| Stable, late                | 100k → 145k | -0.013               | 0.0003                                         |
| Cooldown (30k, 6e-4 → 1e-5) | 140k → 170k | -0.149               | 0.0050                                         |
| Post-cooldown @ min_lr      | 170k → 201k | -0.006 (1.507→1.501) | 0.0002 (slow drift, same order as late stable) |


- **The cooldown drop is *not* evidence the stable phase was wasted.** This is the expected WSD pattern (DeepSeek, MiniCPM): the stable phase accumulates representation work that the cooldown realizes as lower loss. The "10× faster per iter" ratio is misleading — without the long stable phase, the cooldown has less to extract. Confirmation: ~~30k post-cooldown iters at min_lr produced only -0.006 nats of further drift (1.507 → 1.501), at a rate (~~0.0002 nats/1k iters) that's the same order as the late stable phase before the cooldown. v4 is approaching the basin floor of this architecture+data combo, with diminishing returns at min_lr.
- **3.3× data → +0.009 nats over v3 (1.510 → 1.501).** Marginal on val loss, but **anchored-ladder Elo** still moved v4’s best **past** v3’s best (~+21 vs `v3@180` in the local DB). v3's gen-gap signal of "data-bound" has flipped: v4 looks **capacity-bound** on loss, not on “can’t improve play.” v5 should still **scale architecture** for LM headroom; data/schedule alone won’t deliver large val drops.
- **Manual two-phase workflow caused two restart blips (140k, 160k).** `auto_cooldown=True` (already wired in `train.py`) eliminates this — when early stop would fire, training instead triggers cooldown in-job. Use this for v5.
- **Cooldown ratio of 30k / 145k ≈ 20% was on the low end of MiniCPM's 10–20% range.** A longer cooldown (e.g. 25–30%, ~40k iters here) might have squeezed out marginally more, but the post-cooldown stagnation suggests the headroom is small (≤0.01 nats). For v5 default to ~25–30%.
- **Don’t lower the constant LR.** 6e-4 is doing the exploration job correctly. A lower stable LR would slow the stable phase and give the cooldown less accumulated work to realize.

---

## Hardware & compute reference


| Model | GPU         | Arch        | Params | Effective batch | Step time | Tokens/sec | Eval time (excl. R2) | eval_iters × batch |
| ----- | ----------- | ----------- | ------ | --------------- | --------- | ---------- | -------------------- | ------------------ |
| v1    | unknown     | 6L/6H/384d  | ~12M   | 128 (32×4)      | —         | —          | —                    | 200 × 32           |
| v2    | RTX 3060    | 6L/6H/384d  | ~12M   | 128 (32×4)      | ~155 ms   | ~211k      | ~6 s                 | 200 × 32           |
| v3    | RTX 3060    | 12L/8H/512d | ~40M   | 128 (32×4)      | ~470 ms   | ~70k       | ~32 s                | 200 × 32           |
| v3    | RTX 4060 Ti | 12L/8H/512d | ~40M   | 128 (32×4)      | ~321 ms   | ~102k      | ~34 s                | 200 × 32           |
| v4    | RTX 4060 Ti | 12L/8H/512d | ~40M   | 128 (128×1)     | ~318 ms   | ~103k      | ~10 s (+R2 on best)  | 50 × 128           |


**Notes:**

- v3→v4 on the same GPU: identical step time confirms accum-drop and explicit compile were already implicit in v3 on CUDA.
- v4 briefly used **200 × 128** evals (4× v2/v3 sequence count, ~40s compute); config is now **50 × 128** to match v2/v3 coverage. Guard `eval_iters × batch_size` in future configs.
- MFU of 7–8% is normal for small models on gaming GPUs (memory-bandwidth-bound, not FLOP-bound).
- "Eval time" above is pure forward-pass compute; add R2 sync time (~16–20 s) whenever a new best val is saved.