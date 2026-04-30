# Transformer Chess Engine — Project Plan

## What We're Building

A series of transformer-based chess engines trained via next-token prediction on large datasets of real chess games. Each engine is a version in an ongoing experiment: train a model, deploy it as a bot on Lichess, measure its ELO, then iterate with improvements to architecture, training data, or both.

The end goal is a living project that demonstrates how different design choices — model size, data quality, input representation, training signal — affect the playing strength of a purely neural, search-free chess engine. Results are tracked on a public leaderboard page on the personal website.

### Why This Is Interesting

Traditional chess engines (Stockfish, etc.) rely on hand-crafted evaluation functions and explicit tree search. A newer line of research asks: can a transformer learn to play strong chess the same way it learns language — by absorbing patterns from large amounts of data, with no explicit search at inference time? This project explores that question at a hobbyist scale, building up from a simple baseline and layering in improvements systematically.

---

## High-Level Architecture

```
Lichess game database
        ↓
  Data pipeline (filter, parse, tokenize)
        ↓
  NanoGPT-based transformer (trained on cloud GPU)
        ↓
  Local eval (vs. Stockfish at various depths)
        ↓
  Lichess bot deployment (inference via Modal)
        ↓
  ELO measurement (bot vs. bot games on Lichess)
        ↓
  Leaderboard page on personal website
```

---

## Tech Stack


| Component                | Tool                                |
| ------------------------ | ----------------------------------- |
| Model architecture       | NanoGPT (Karpathy)                  |
| Training framework       | PyTorch                             |
| Chess logic / validation | python-chess                        |
| Training data            | Lichess open database               |
| Local development        | MacBook Air M2 (MPS backend)        |
| Cloud training           | Vast.ai (RTX 4090)                  |
| Experiment tracking      | Weights & Biases                    |
| Bot inference serving    | Modal (serverless)                  |
| Lichess integration      | lichess-bot + Lichess API           |
| Website                  | Existing personal site (React/Vite) |


---

## Phases

---

### Phase 1 — Data Pipeline

The foundation. A clean, well-validated data pipeline makes every subsequent experiment faster and more trustworthy.

**Goals:**

- Download and filter a meaningful slice of Lichess game data
- Parse PGN files into sequences of UCI moves
- Build a move-level tokenizer with a clean vocabulary
- Create a PyTorch Dataset that can feed the training loop
- Validate the pipeline end-to-end: tokenize a game, decode it, replay it with python-chess, confirm all moves are legal

**Key decisions:**

- ELO filter threshold (start with 1800+ for both players)
- Time control filter (avoid bullet — too noisy; prefer rapid or classical)
- Context window length (target: 256 tokens, covers most full games)
- Special tokens: `<GAME_START>`, `<GAME_END>`, `<WHITE_WIN>`, `<BLACK_WIN>`, `<DRAW>`, `<PAD>`

**Deliverables:**

- `data/` pipeline scripts (download, filter, parse, tokenize)
- Saved vocabulary file (JSON)
- PyTorch Dataset and DataLoader
- Sanity check script with distribution stats (game length, move frequency, ELO distribution)

---

### Phase 2 — Model and Training

**Goals:**

- Adapt NanoGPT for the chess token vocabulary and context length
- Get a working training loop running locally on M2 (tiny data, just to validate)
- Run real training jobs on cloud GPU
- Establish a fast local eval loop against Stockfish before bothering with Lichess deployment

**Key decisions:**

- Start small (10M params) to validate pipeline, then scale to 50M+ for real runs
- Use MPS device locally, CUDA on cloud — keep device config clean and switchable
- Log with W&B from the start: loss curves, tokens/sec, checkpoint metadata

**Local eval approach:**

- Play N games (e.g. 200) between the model and Stockfish at various ELO constraints
- Win/loss/draw rates per constraint give a relative strength signal across model versions
- Just an estimate of ELO here, but sufficient to answer "is v2 better than v1?"

**Deliverables:**

- `model/` — NanoGPT adapted for chess
- `train.py` — training loop with W&B logging and checkpointing
- `eval.py` — local tournament script vs. Stockfish
- Trained v1 checkpoint

---

### Phase 3 — Lichess Deployment

**Goals:**

- Wrap the model in a move-generation interface
- Deploy inference on Modal (serverless, pay-per-use)
- Connect to Lichess via lichess-bot
- Let the bot play enough games to get a stable ELO rating

**Move generation details:**

- At each turn: tokenize game history, run forward pass, get logits over vocabulary
- Mask all illegal moves (use python-chess to enumerate legal moves)
- Softmax + sample from legal moves only (temperature is a tunable parameter)
- Lower temperature = more deterministic; higher = more human-like variance

**Infrastructure:**

- Separate Lichess account flagged as a bot account
- Modal function loads model weights from persistent volume, returns UCI move given game state
- lichess-bot handles the Lichess API communication and calls the Modal endpoint

**Deliverables:**

- `serve/` — Modal inference function
- `bot/` — lichess-bot config and integration
- Bot live on Lichess with a stable rating

---

### Phase 4 — Iterate and Improve

Each experiment produces a new model version with a tracked ELO. The goal is to move the number up with principled changes, not random tweaks.

**Planned experiments in rough order:**

The question your experiments should answer in order is: data quantity → model size → data quality → training signal quality. That's roughly cheapest-to-most-expensive in terms of iteration cost, and it builds on itself cleanly.


| Version | Key change                             | One-liner                                                                                            |
| ------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------- |
|         |                                        |                                                                                                      |
| v1      | 12M params, 1M games                   | Baseline                                                                                             |
| v2      | 40M params, 3M games, longer training. | Current                                                                                              |
| v3      | 40M params, 15-30M games               | Data scaling — does more data dwarf bigger model?We've pulled all the data from lichess that we can |
| v4      | 100M+ params, 15-30M games             | Model scaling on top of data                                                                         |
| v5      | 2200+ ELO filter on same data pipeline | Data quality vs. quantity                                                                            |
| v6      | ChessBench / Stockfish annotations     | Best-move supervision                                                                                |


Each version gets a row on the leaderboard. Architecture and training details are recorded for every run.

**Deliverables:**

- Trained checkpoints for each version
- Updated leaderboard data
- Notes on what worked and what didn't

---

### Phase 5 — Website

A page on the personal website that makes the project legible to visitors.

**Must-haves:**

- Leaderboard table: version, params, training data description, key change, Lichess ELO
- Link to each bot's Lichess profile
- Brief written explanation of the project and approach

**Nice-to-haves:**

- Live Lichess game embed (shows the bot's current or most recent game)
- Live stats pulled from Lichess API (current rating, games played, win rate)
- ONNX in-browser inference so visitors can play against a model directly (stretch goal)

---

## Guiding Principles

- **Pipeline first.** Sloppy data = unreliable experiments. Get the data pipeline right before worrying about model quality.
- **Fast feedback loops.** Local Stockfish eval should give signal within 20 minutes of a training run. Don't deploy to Lichess to answer "is this better?"
- **One variable at a time.** Each model version changes one thing. Otherwise it's impossible to know what moved the needle.
- **Write the core code yourself.** The training loop, tokenizer, and data pipeline should be code you understand line by line. Use AI for boilerplate, not for the interesting parts.
- **Keep costs manageable.** Each training run on Vast.ai should cost $1–5. Inference on Modal is nearly free at hobby bot traffic levels. Don't scale up spend until you have a reason to.

---

## Reference

- [NanoGPT](https://github.com/karpathy/nanogpt) — base architecture
- [Lichess open database](https://database.lichess.org) — training data
- [python-chess](https://python-chess.readthedocs.io) — chess logic, move validation
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) — bot framework
- [DeepMind Searchless Chess (Ruoss et al., NeurIPS 2024)](https://github.com/google-deepmind/searchless_chess) — most relevant prior work; ChessBench dataset available for Phase 4
- [Weights & Biases](https://wandb.ai) — experiment tracking

