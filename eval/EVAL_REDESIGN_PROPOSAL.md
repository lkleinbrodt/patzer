# Eval System Redesign Proposal

## Problem Statement

The evaluation system has grown organically into a Frankenstein of overlapping scripts, redundant result files, and over-engineered abstractions. The core goal is simple: **determine how strong each model checkpoint is**, both in absolute terms (vs Stockfish at known Elo) and relative terms (vs each other). The current system makes this harder than it should be.

---

## Current State

### Files

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `eval/tournament.py` | 762 | Model vs Elo-constrained Stockfish (adaptive Bayesian estimation) | Works, but verbose |
| `eval/model_tournament.py` | 1238 | Model vs model with R2 discovery, SPRT, 2-stage pruning, aggregation buckets | Over-engineered |
| `eval/sweep.py` | 292 | Run tournament across all checkpoints + plot | **Broken** (calls deleted `run_tournament` API) |
| `eval/engine.py` | 205 | `Patzer` and `StockfishPlayer` player abstractions | Clean, keep |
| `eval/uci_engine.py` | 181 | UCI protocol wrapper for GUIs | Clean, keep |
| `eval/play.py` | — | Human play interface | Separate concern |
| `eval/play_gui.py` | — | GUI play interface | Separate concern |
| `eval/inspect_training_run.py` | — | Training run inspection | Separate concern |
| `eval/results.json` | 1317 lines | Stockfish tournament results | Messy accumulation |
| `eval/model_results.json` | 1703 lines | Model-vs-model results | Messy accumulation |
| `eval/old_results.json` | 1433 lines | Legacy results from earlier iterations | Dead weight |

### Models / Checkpoints

Local (`checkpoints/`):

```
checkpoints/
├── patzer_v0/
│   └── ckpt.pt              (148MB, iter 10000)
├── patzer_v1/
│   ├── ckpt_best.pt         (148MB, iter 40000)
│   └── metrics.jsonl
└── patzer_v2/
    ├── ckpt.pt              (481MB — this is the larger v3-architecture model?)
    ├── ckpt_best.pt         (148MB, iter ~45000)
    ├── ckpt_010000.pt       (148MB)
    ├── ckpt_015000.pt       (481MB)
    ├── ckpt_020000.pt       (481MB)
    ├── ckpt_035000.pt       (481MB)
    ├── ckpt_050000.pt       (481MB)
    ├── ckpt_055000.pt       (481MB)
    ├── ckpt_060000.pt       (481MB)
    ├── ckpt_070000.pt       (148MB)
    └── ckpt_150000.pt       (148MB)
```

Training configs (`patzer/config/`):
- `train_patzer_v1.py` — ~12M params
- `train_patzer_v2.py` — ~12M params, bigger 1800+ Elo dataset
- `train_patzer_v3.py` — ~40M params (12 layers, 8 heads, 512 embed)

R2 mirrors the same `checkpoints/patzer_vN/` structure.

### Current Results (sample)

**results.json** (Stockfish tournaments): Records per-Elo-level W/L/D aggregates. Example entry:
```json
{
    "timestamp": "2026-04-29T20:00:09",
    "checkpoint": "/Users/lando/Projects/patzer/checkpoints/patzer_v1/ckpt.pt",
    "iter_num": 40000,
    "stockfish_elo": 1320,
    "games": 6,
    "temperature": 0.1,
    "conditioning": "match_color",
    "W": 5, "L": 1, "D": 0
}
```

**model_results.json** (model-vs-model): SPRT match records. Example entry:
```json
{
    "model_a": "checkpoints/patzer_v2/ckpt_010000.pt",
    "model_b": "checkpoints/patzer_v2/ckpt_best.pt",
    "iter_a": 10000,
    "iter_b": 45000,
    "games": 8,
    "W": 0, "L": 5, "D": 3,
    "score": 0.1875,
    "sprt": { "decision": "inconclusive", ... },
    "type": "stage1_vs_baseline",
    "settings": { "temperature": 0.0, "conditioning": "match_color", ... }
}
```

### Specific Problems

1. **Two result files, two schemas** — no unified view of model strength
2. **Aggregated W/L/D per Elo** — you lose individual game data and can't re-analyze
3. **`model_tournament.py` is 1238 lines** — 60% is R2 discovery, label formatting, and aggregation-bucket logic that belongs elsewhere
4. **`sweep.py` is broken** — references a `run_tournament()` function that was deleted during the adaptive-Elo refactor
5. **Results accumulate forever** — absolute paths baked in (`/Users/lando/...`), no schema versioning, duplicate entries from partial runs
6. **Too many entry points** — unclear which script to run; `tournament.py --show`, `model_tournament.py --analyze`, `sweep.py --plot-only` all show slightly different views
7. **SPRT is overkill** — with ~8-12 games per pair, SPRT almost never reaches a decision ("inconclusive" on most entries in `model_results.json`)

---

## Proposed Design

### Philosophy

- **Store raw games, derive everything else.** Individual game results are the atomic unit. Elo estimates, leaderboards, and win rates are all computed views.
- **One tool, subcommands.** `eval/evaluate.py` is the single entry point.
- **Checkpoint paths are just paths.** No R2 discovery baked into eval. Pull checkpoints separately with `r2.py pull`.
- **SQLite, not JSON.** Queryable, no migrations needed (single table, append-only).

### Architecture

```
eval/
├── evaluate.py          # Single CLI entry point (subcommands)
├── engine.py            # Patzer + StockfishPlayer (unchanged)
├── db.py                # Thin SQLite wrapper (create table, insert, query)
├── elo.py               # Elo estimation: Bradley-Terry MLE from game records
├── results.db           # SQLite database (gitignored)
├── uci_engine.py        # UCI wrapper (unchanged)
├── play.py              # Human play (unchanged)
├── play_gui.py          # GUI play (unchanged)
└── inspect_training_run.py  # (unchanged)
```

### Database Schema

```sql
CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601 UTC
    white TEXT NOT NULL,               -- e.g. "patzer_v2@45000" or "stockfish:1320"
    black TEXT NOT NULL,
    result TEXT NOT NULL,              -- '1-0', '0-1', '1/2-1/2'
    white_checkpoint TEXT,             -- relative path: "checkpoints/patzer_v2/ckpt_best.pt"
    black_checkpoint TEXT,
    white_iter INTEGER,                -- training iteration (NULL for stockfish)
    black_iter INTEGER,
    opening TEXT,                      -- UCI opening string or NULL
    temperature REAL DEFAULT 0.0,
    top_k INTEGER,
    conditioning TEXT DEFAULT 'match_color',
    notes TEXT                         -- freeform JSON or text for metadata
);
```

No migrations needed — this is the only table. If we ever want to add columns, SQLite supports `ALTER TABLE ADD COLUMN` with defaults, which is non-destructive.

**Player naming convention:**
- Model checkpoints: `patzer_v2@45000` (version + iter)
- Stockfish: `stockfish:1320` (engine + Elo limit)

### CLI Interface

```bash
# Estimate a model's Elo against Stockfish (adaptive)
python eval/evaluate.py stockfish checkpoints/patzer_v2/ckpt_best.pt \
    --games 50 --device mps

# Play two models head-to-head
python eval/evaluate.py head2head \
    checkpoints/patzer_v2/ckpt_best.pt \
    checkpoints/patzer_v1/ckpt_best.pt \
    --games 20 --device mps

# Show the leaderboard (computed from all stored games)
python eval/evaluate.py leaderboard

# Show raw game history for a model
python eval/evaluate.py history patzer_v2

# Quick comparison: v2 checkpoints at different training steps
python eval/evaluate.py head2head \
    checkpoints/patzer_v2/ckpt_010000.pt \
    checkpoints/patzer_v2/ckpt_070000.pt \
    checkpoints/patzer_v2/ckpt_150000.pt \
    --round-robin --games 10 --device mps
```

### Elo Computation (`eval/elo.py`)

Simple Bradley-Terry maximum likelihood estimation:
1. Query all games from the DB
2. Each unique player (model or stockfish-at-Elo) gets a rating parameter
3. Stockfish players are **anchored** at their configured Elo (fixed, not fitted)
4. Iterative MLE to convergence (or simple closed-form for paired comparisons)
5. Output: rating ± confidence interval for each model

This is ~50-80 lines of math, no external dependencies beyond numpy (optional — can do pure Python).

### What Gets Deleted

- `eval/results.json` — replaced by `results.db`
- `eval/model_results.json` — replaced by `results.db`
- `eval/old_results.json` — dead weight
- `eval/sweep.py` — broken, functionality subsumed by `evaluate.py stockfish` with multiple checkpoints
- `eval/sweep_plot.png` — artifact of deleted sweep
- `eval/tournament.py` — replaced by `evaluate.py`
- `eval/model_tournament.py` — replaced by `evaluate.py`

### What Stays Unchanged

- `eval/engine.py` — `Patzer` and `StockfishPlayer` classes
- `eval/uci_engine.py` — UCI protocol wrapper
- `eval/play.py`, `eval/play_gui.py` — human interaction
- `eval/inspect_training_run.py` — training analysis

### Key Simplifications

| Before | After |
|--------|-------|
| 1238-line model_tournament.py with R2 discovery, SPRT, 2-stage pruning, aggregation buckets, label formatting | ~100-line head2head subcommand: load models, play N games, record results |
| 762-line tournament.py with show/estimate/aggregate modes | ~150-line stockfish subcommand: adaptive Elo loop (keep the Bayesian math, it's good) |
| 150 lines of R2 checkpoint listing/caching | Removed from eval. Use `python patzer/r2.py pull checkpoints/patzer_v2` separately |
| SPRT early stopping (never conclusive with 8 games) | Just play N games. User picks N based on time budget |
| Two separate JSON files with different schemas | One SQLite table, one schema, individual game records |
| Three different "show results" commands | One `leaderboard` subcommand |
| Absolute paths in results (`/Users/lando/...`) | Relative paths only (`checkpoints/patzer_v2/...`) |

### Openings

Keep the built-in opening book from `model_tournament.py` (8 standard openings as UCI move lists). Use them for all head-to-head games to reduce variance. For Stockfish games, start from the standard position (the adaptive algorithm handles variance through volume).

### Future Extensions (not in v1)

- `evaluate.py progress` — plot Elo over training steps (replaces sweep.py)
- `evaluate.py export` — dump results.db to CSV/JSON for external analysis
- Parallel game execution (multiple Stockfish processes)
- Integration with the Lichess bot for online Elo measurement

---

## Migration Plan

1. Write `eval/db.py` (SQLite helper — ~40 lines)
2. Write `eval/elo.py` (Bradley-Terry estimation — ~80 lines)
3. Write `eval/evaluate.py` (CLI + subcommands — ~400 lines total)
4. Delete old files: `tournament.py`, `model_tournament.py`, `sweep.py`, all JSON results
5. Update `CLAUDE.md` with new eval commands
6. Update `.gitignore` to include `eval/results.db`

Total new code: ~500 lines (down from ~2300 across tournament + model_tournament + sweep).
