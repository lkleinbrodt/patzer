# Patzer Model Log

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

### Local eval (Elo-limited Stockfish, `eval/tournament.py` → `eval/results.json`)

**~1254 ± 26** Elo from 190 games (`ckpt.pt` @ 40k, `match_color`, temp **0.1**). Roughly low–mid human club level on this ladder; most games are vs weak Stockfish caps so the number is an internal yardstick, not Lichess rating.

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

| param      | value |
| ---------- | ----- |
| data       | ~868M train tokens, 1800+ ELO, ~11.15M games |
| batch_size | 128   |
| max_iters  | 150,000 |
| lr         | 1e-3 → 1e-5 (cosine) |

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

| param      | value |
| ---------- | ----- |
| data       | ~868M train tokens, 1800+ ELO, ~11.15M games |
| batch_size | 128   |
| max_iters  | 150,000 |
| lr         | 6e-4 → 1e-5 (cosine) |



