# Patzer v2/v3 Training Retrospective

*Notes from analyzing the training logs, deciding what to do next, and being honest about what we got wrong along the way.*

## Context

Two models were trained on the same dataset (~11.7M Lichess games, 1800+ ELO filter, ~868M training tokens):

- **v2**: 12M params (6 layers, 6 heads, 384 embd) — same architecture as v1, scaled-up data
- **v3**: 40M params (12 layers, 8 heads, 512 embd) — bigger model, same data

Both used cosine-decayed AdamW with linear warmup. v2 used peak LR 1e-3 and β₂=0.99; v3 dropped to 6e-4 and β₂=0.95 — standard adjustments for scaling up.

## v2 in retrospect

Final val loss **1.652** (perplexity ~5.22), generalization gap **+0.006**. Practically no overfitting. The model was capacity-bound at 12M params, not data-bound — important to note because it would have justified scaling up regardless of any other consideration.

Two operational quirks worth remembering:

**Schedule misalignment.** The original config had `lr_decay_iters=150000`, then was extended to `max_iters=350000` mid-run. Result: 29% of training ran at `min_lr=1e-5` (effectively dead).

**Accidental warm restart at iter 150k.** When `lr_decay_iters` was changed to 250k mid-run, the cosine schedule recomputed and LR jumped from 1e-5 back to ~3.5e-4. Val loss spiked from 1.67 → 1.79 in 1k iters, then descended again, ultimately reaching 1.65 by iter 270k. **This is the only place across our runs where LR has been "kicked" into a non-cosine pattern, and it produced a measurable gain.** It became important evidence later.

The ELO leaderboard told the same story:

```
v2@120 (pre-restart):  1562 ELO
v2@180 (mid-restart):  1448 ELO  ← took a real hit during the spike
v2@220 (post-restart): 1557 ELO  ← recovered
v2@288 (final):        1621 ELO  ← net +60 over pre-restart
```

The mid-cycle dip is a useful warning for any future SGDR experiment: don't measure ELO at the middle of a cycle, only at cycle ends.

## v3 mid-run: a hypothesis that didn't hold up

Around iter 112k, with v3 sitting at val 1.563 and LR cosine-decayed to ~1e-4, I formed a confident hypothesis: the cosine schedule was about to starve the model. The val curve appeared to track the LR curve closely, which suggested loss was decreasing simply because steps were being taken — not because the model was approaching any real minimum.

The diagnostic I used was **val_loss vs cumulative ∫LR**: integrate the learning rate over iterations, plot val against that instead of against iter. This converts schedule-time into "effective gradient distance traveled" and asks whether progress is linear (LR-limited) or curving toward an asymptote (converging).

At iter 112k, the curve looked nearly linear, and the slope of dval/d(∫LR) was *steeper* in the late stage than the middle (-0.012 vs -0.005 per unit ∫LR). My read: each unit of LR was buying *more* descent in the late phase, and we were about to throw most of that away by letting cosine bottom to 1e-5.

Recommendation at the time: hold LR at ~1e-4 instead of decaying to 1e-5, and use min_lr ≈ 10% of peak (modern best practice from Llama-era runs) for v4+.

## v3 in full: real plateau, widening gen gap

The full curve (through iter 151k) softened that read substantially.


|                               | iter | val       | LR     |
| ----------------------------- | ---- | --------- | ------ |
| iter 112k (where I diagnosed) | 112k | 1.563     | 1.0e-4 |
| iter 145k (best ever)         | 145k | **1.514** | 1.2e-5 |
| iter 150k (cosine bottom)     | 150k | 1.518     | 1.0e-5 |


Two things became clear:

**The cosine tail did real work.** From iter 100k → 145k, val dropped from 1.585 → 1.514 (0.07 nats) while LR fell from 1.6e-4 to 1.2e-5. The integrated-LR slope in this region (∫LR 35→46) was the steepest in the entire run, ~3× more productive per unit LR than the middle phase. The model used those small steps effectively to fine-tune into a real basin. The "starvation" hypothesis was wrong, or at least had been an artifact of partial data.

**Real plateau, finally.** Last 10k iters: val bounced in [1.516, 1.520], all changes within the ~0.007 noise floor. First time across either v2 or v3 we'd hit something that genuinely looks like convergence rather than schedule-induced stalling.

**Generalization gap widened.** Across the runs: +0.005 (v2 final) → +0.017 (v3 iter 112k) → **+0.025 (v3 iter 150k)**. Train was now visibly below val. The model had begun memorizing, even though the dataset still soaked up most of its capacity. This is the first overfitting signal in any run so far, and it's the most important number on the page.

## What we actually learned

**Partial curves lie. Full curves don't.** The val-vs-∫LR diagnostic is genuinely useful, but it has to be applied to a finished run. At iter 112k it pointed to "starving model"; at iter 151k it pointed to "schedule was approximately right." The framework was sound; the regime had simply not arrived yet. Lesson: make schedule decisions *after* schedules finish, not in the middle of them.

**The cosine schedule we ran was approximately correct.** Decaying to 1e-5 (1.7% of peak) wasn't obviously wasteful as full-run data showed. Modern advice to set min_lr ~10% of peak might be marginally better — that's a v4 ablation worth one run, not a strong recommendation.

**40M params is roughly the right size for this dataset.** Gen gap at +0.025 says capacity is no longer the binding constraint. v2 at 12M was capacity-bound (gen gap ~0). v3 at 40M is approaching data-bound. Going to 80M+ without scaling data will accelerate overfitting, not val improvement.

**The expected payoff from LR/schedule tuning is small.** Optimistic estimate for an SGDR ablation against vanilla v3 cosine: ±0.02 nats, maybe ±30 ELO. Compare to expected gain from 10× more data: 0.05–0.10 nats and likely 100+ ELO. Schedule tuning is a side quest, not a main path.

## Decisions for v4 and beyond

**v3 final.** Let it run until early-stop triggers (~iter 160-170k expected). Save as `patzer_v3_final` for the leaderboard. Move on.

**SGDR ablation deferred.** Worth doing later as a principled experiment, but not before v4. The recipe when run: total 150k iters split into two cycles (75k each), peak 6e-4 → 3e-4 across cycles, cosine within each. Train from scratch for clean comparison. Read leaderboard ELO at cycle ends only.

**v4: same architecture, more data, same 1800+ filter.** This is the highest-EV experiment in the plan. Directly tests the gen-gap hypothesis. With ~10× more games coming online, expect val to drop into the 1.40–1.45 range and gen gap to close. If gap stays open at 40M with 10× data, we have a different problem than I'm currently expecting and v5+ plans need rethinking.

**v5: quality filter (2200+) at matched token count.** Tests data quality independent of quantity. Requires enough raw games that 2200+ post-filter still gives ≥30M games for training. Use a fixed eval set across v3/v4/v5 (probably 1800+ holdout) so val loss is comparable across runs — eval set composition can't change between experiments or comparisons break.

**v6: scale architecture, on whichever data won.** Last in the sequence. Worth waiting for v4/v5 results before deciding what to scale and on what.

## Data scaling notes

Project sources will eventually have access to ~7.7B raw Lichess games. Current pipeline has ~100M downloaded; v3 trained on 11M. After 1800+ filter, current pool gives ~25-30M filtered games — already enough for v4.

Run experiments as data accumulates rather than waiting for the full download. Downloads run in parallel with training (different bottlenecks, different machines), and each experiment's result shapes the next one. Waiting delays the learning loop without saving any compute.

Practical hobby-project disk thresholds:


| post-filter games | tokenized | regime                                                      |
| ----------------- | --------- | ----------------------------------------------------------- |
| 30M               | ~5 GB     | comfortable                                                 |
| 100M              | ~15 GB    | comfortable                                                 |
| 500M              | ~75 GB    | needs S3 or similar for cloud transfers                     |
| 1B+               | ~150 GB   | real big-data; transfers to vast.ai become significant time |


For models <100M params, the Chinchilla-ish rule (20-200 tokens/param) caps useful dataset size around 200-400M filtered games. Beyond that, returns diminish before disk does.

## Speedup checklist (apply to v4)

In rough ROI order, given RTX 3060 / vast.ai:

1. **Drop `gradient_accumulation_steps` to 1, raise `batch_size` to ~128.** Same effective batch (32K tokens), one forward/backward instead of four. Probably 1.3–1.7× faster step time. Verify it fits in 12GB at v4's params/block_size before committing — fall back to 96 if 128 OOMs.
2. **Verify `compile=True`.** torch.compile on Ampere is a ~1.2-1.4× free win.
3. **Verify `dtype='bfloat16'`.** nanoGPT's default if Ampere is detected; worth confirming. fp32 leaves massive tensor-core throughput on the table.
4. **Verify Flash Attention is active** (PyTorch SDPA dispatches automatically with recent PyTorch + nanoGPT).
5. **Align `lr_decay_iters` with actual `max_iters`.** No more "30-50% of run at min_lr" — that's wasted compute. If unsure how long to train, set them equal and rely on early stopping.
6. **Stop earlier.** v3 plateaued by iter 145k; the last 5k iters before iter 150k were noise. Trim 10-25% off planned iters once curves stabilize.

Expected combined effect on v4 wall-clock: noticeable. v3 was running at ~300ms/iter on a 3060 — there's room for that to come down to ~150-200ms with the first two changes alone, with no quality impact.

---

*Last updated after analyzing the v3 run through iter 151k. Next entry: v4 results.*