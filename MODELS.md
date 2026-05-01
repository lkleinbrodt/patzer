# Patzer Model Log

---

## Unified leaderboard (eval harness)

[Elo estimates](https://en.wikipedia.org/wiki/Elo_rating_system) below come from `**python eval/evaluate.py leaderboard**` (May 2026 snapshot). They fit a Bradley–Terry model on every game row in `**eval/results.db**` (SQLite): **head-to-head Patzer self-play**, **Patzer vs capped Stockfish** evals, and cross-checkpoint mixes. Anchored Stockfish sides are omitted from the printed table; Patzer checkpoints are labelled `patzer_vN@{k}` where `k` is training step in **thousands** (e.g. `@145` ≈ iter 145,000).


| Rank | Player        | Elo  | ±   | Games | W–L–D       |
| ---- | ------------- | ---- | --- | ----- | ----------- |
| 1    | patzer_v3@180 | 1632 | 12  | 1134  | 673–222–239 |
| 2    | patzer_v3@145 | 1625 | 12  | 1180  | 743–212–225 |
| 3    | patzer_v3@111 | 1601 | 12  | 1002  | 574–229–199 |
| 4    | patzer_v3@80  | 1508 | 12  | 1002  | 412–329–261 |
| 5    | patzer_v2@288 | 1475 | 9   | 1745  | 809–498–438 |
| 6    | patzer_v2@274 | 1474 | 10  | 1487  | 665–460–362 |
| 7    | patzer_v3@40  | 1430 | 14  | 775   | 265–293–217 |
| 8    | patzer_v2@120 | 1417 | 10  | 1298  | 464–417–417 |
| 9    | patzer_v2@220 | 1414 | 11  | 1248  | 465–439–344 |
| 10   | patzer_v2@180 | 1340 | 11  | 1248  | 315–548–385 |
| 11   | patzer_v1@40  | 1203 | 13  | 1118  | 96–694–328  |
| 12   | patzer_v3@10  | 1197 | 16  | 780   | 63–523–194  |
| 13   | patzer_v2@50  | 1190 | 12  | 1236  | 95–790–351  |


Within this ladder, rankings are reproducible given the stored games; absolute Elo is an internal yardstick (not Lichess rating). Recompute anytime with `**eval/evaluate.py leaderboard`**.

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
| batch_size       | 128                               |
| max_iters        | 40,000                            |
| lr               | 1e-3 → 1e-4 (cosine)              |
| val loss (final) | ~1.84                             |


**Note:** Val loss was still declining at iter 40k — underfit; LR floor 1e-4. Best val was ~30k, not 40k.

### Eval

**Unified ladder:** `**patzer_v1@40`** — **1203 ± 13** Elo over 1118 games (see leaderboard above). Older one-off `**eval/tournament.py`** run (~1254 ± 26 from 190 games, temp **0.1**) used a different pipeline and is **not** mixed into `results.db`.

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


| param      | value                                                   |
| ---------- | ------------------------------------------------------- |
| data       | ~868M train tokens, 1800+ ELO, ~11.15M games            |
| batch_size | 128                                                     |
| max_iters  | 150,000 (was still learning; later resumed toward 250k) |
| lr         | 1e-3 → 1e-5 (cosine)                                    |


### Eval (unified ladder)

Strongest logged v2 checkpoints: `**patzer_v2@288`** **1475 ± 9** (1745 games), `**patzer_v2@274`** **1474 ± 10** (1487 games). Earlier / weaker checkpoints in DB include `@220`, `@180`, `@120`, `@50` (~1190–1417 Elo). See the leaderboard table above.

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


| param           | value                                        |
| --------------- | -------------------------------------------- |
| data            | ~868M train tokens, 1800+ ELO, ~11.15M games |
| batch_size      | 128 (32 × 4 accum)                           |
| max_iters       | 150,000                                      |
| lr              | 6e-4 → 1e-5 (cosine)                         |
| val loss (best) | **1.514** at iter 145k                       |
| gen gap         | +0.025 (first overfitting signal in any run) |


### Eval (unified ladder)

Best entries in `**eval/results.db`**: `**patzer_v3@180**` — **1632 ± 12** over 1134 games; `**@145`** 1625 ± 12; `**@111**` 1601 ± 12 — clear lift over strongest v2 (~~1475). Weaker checkpoints (`@80`, `@40`, `@10`) sit lower on the same ladder (~~1197–1508); see table above.

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


| param       | value                                        |
| ----------- | -------------------------------------------- |
| data        | ~2.85B train tokens, 1800+ ELO, ~36.3M games |
| batch_size  | 128 (physical, no gradient accumulation)     |
| lr_schedule | WSD (warmup → stable → linear cooldown)      |
| lr          | 6e-4 (constant after warmup)                 |
| cooldown    | 30k iters, linear ramp to 1e-5               |
| max_iters   | 600k cap (early stop ends phase 1 sooner)    |
| compile     | True                                         |


**Throughput:** 128 × 256 = 32,768 tokens/iter → ~86,898 iters/epoch.

**Two-phase workflow:**

1. **Phase 1** (`train_patzer_v4.py`): warmup (3k iters) → constant LR at 6e-4. Early stop triggers when val plateaus (`patience=25`, `min_iters=100k`). Note the stop iter.
2. **Phase 2** (`train_patzer_v4_cooldown.py`, created after phase 1): resume from phase 1 checkpoint, 30k-iter linear cooldown from 6e-4 → 1e-5. This produces the final v4 model.

### Changes from v3

1. **3.3× more training data** (2.85B vs 868M tokens, 36M vs 11M games). Same 1800+ ELO filter.
2. **WSD schedule** replaces cosine. Constant LR during training lets early stop decide when to decay, instead of baking a fixed decay length into the config. The v3 retrospective showed `lr_decay_iters` misalignment wasted compute; WSD eliminates that failure mode entirely.
3. `**gradient_accumulation_steps` = 1, `batch_size` = 128** — same effective batch, fewer Python/optimizer overheads. Expected ~1.3–1.7× faster step time. Falls back to batch 96 if 128 OOMs on 12GB.
4. `**compile = True`** — explicit torch.compile for ~1.2–1.4× on Ampere+.

