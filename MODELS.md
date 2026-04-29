# Patzer Model Log

---

## v1 — baseline

**Checkpoint:** `checkpoints/patzer_v1`
**Trained:** April 2026 on Vast.ai

### Architecture
| param | value |
|-------|-------|
| n_layer | 6 |
| n_head | 6 |
| n_embd | 384 |
| block_size | 256 |
| params | ~12M |

### Training
| param | value |
|-------|-------|
| data | ~250M tokens, 1800+ ELO, unknown game count |
| batch_size | 128 |
| max_iters | 40,000 |
| lr | 1e-3 → 1e-4 (cosine) |
| val loss (final) | ~1.84 |

**Note:** Val loss was still declining at iter 40k — model was underfit. LR had already hit its minimum (1e-4), so the last ~5k iters barely moved. Best checkpoint was 30k, not 40k.

### Eval vs Stockfish (temp=1.0 — wrong, do not use as baseline)
| depth | score | W-D-L |
|-------|-------|-------|
| 0 | 15.0% | 0-54-126 |
| 1 | 12.8% | 0-46-134 |
| 3 | 5.3% | 0-19-161 |

Zero wins. Terrible. Root cause: temperature=1.0 was randomly sampling moves from the distribution, throwing away the model's predictions.

### Eval vs Stockfish (temp=0.1 — correct)
| depth | score | W-D-L |
|-------|-------|-------|
| 0 | 22.5% | 2-77-101 |
| 1 | 23.1% | 2-79-99 |
| 3 | 14.4% | 0-52-128 |

Wins appeared. Draw rate jumped. Best checkpoint: **30k** at 26.7% overall (1-30-29 across all depths). Model can hold roughly equal positions but can't convert advantages. Zero wins vs depth 3 — any 3-ply search exploits the model's tactical blindness.

**Estimated ELO:** ~1200–1300 (based on depth-0 Stockfish being ~1500–1700 with NNUE)

### Lessons
- Default eval temperature must be 0.1 or lower — fixed in `eval/engine.py`
- 6 layers is too small to learn tactical patterns; mostly learns opening statistics
- LR min should be 1e-5, not 1e-4 — v1 stopped learning too early
- More training beyond 30k had diminishing returns given the LR floor

---

## v2 — bigger model, longer training

**Checkpoint:** `checkpoints/patzer_v2`
**Config:** `patzer/config/train_patzer_v2.py`
**Status:** training

### Architecture
| param | value |
|-------|-------|
| n_layer | 12 |
| n_head | 8 |
| n_embd | 512 |
| block_size | 256 |
| params | ~40M |

### Training
| param | value |
|-------|-------|
| data | 249M train tokens, 3.15M games, 1800+ ELO |
| batch_size | 128 |
| max_iters | 100,000 |
| lr | 6e-4 → 1e-5 (cosine) |
| val loss (final) | TBD |

### Changes from v1
- **3.3× more parameters** — primary lever; model should learn tactics, not just opening statistics
- **2.5× more training** — 100k vs 40k iters; more epochs over the same data
- **LR decays 10× lower** — 1e-5 vs 1e-4; model keeps learning into late training
- **Same data ELO floor** — kept 1800+ to maximize game count (3.15M games); can raise to 2000+ in v3 to trade quantity for quality
- **Same block_size** — 256 is sufficient; avg game is 79 tokens, so each window already spans ~3 full games

### Eval vs Stockfish
TBD
