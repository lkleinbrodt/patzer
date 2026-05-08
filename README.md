# Patzer

Patzer is a **transformer-based chess engine** trained with next-token prediction on real games from the [Lichess open database](https://database.lichess.org). There is **no explicit search** at play time: the model assigns probability over legal UCI moves from the current position, same idea as language modeling applied to chess.

The project tracks **versioned models** (v1, v2, …): scrape and tokenize data, train on GPU (often via [Vast.ai](https://vast.ai)), evaluate against Stockfish and on a unified internal ladder, then optionally run **Lichess bots** that load checkpoints from this repo.

---

## Documentation map

| Doc | What it’s for |
|-----|----------------|
| [**PLAN.md**](PLAN.md) | Product and research plan: phases, architecture, guiding principles, references. |
| [**PROJECT_LOG.md**](PROJECT_LOG.md) | Running dev log: training changes, pipeline fixes, eval tooling, operational notes. |
| [**MODELS.md**](MODELS.md) | Model family write-ups: architecture, data, val loss, ladder Elo, lessons learned. |
| [**CLAUDE.md**](CLAUDE.md) | Full command reference: data pipeline, training, R2 sync, eval, dashboard, config conventions. |
| [**bot/README.md**](bot/README.md) | Lichess bot setup, tokens, cycling multiple bots, YAML options. |

Bots are started with [`bot/deploy_bot.py`](bot/deploy_bot.py), which installs the homemade-engine shim into an external [`lichess-bot`](https://github.com/lichess-bot-devs/lichess-bot) checkout and launches the right config.

---

## How it fits together

```
Lichess monthly dumps (.pgn.zst)
    → pipeline (filter, parse, tokenize → uint16 .bin)
    → patzer/train.py (nanoGPT-style transformer)
    → checkpoints/patzer_vN/
    → eval/evaluate.py (Stockfish, head-to-head, leaderboard in eval/results.db)
    → optional: Lichess via lichess-bot + bot/deploy_bot.py
```

Training code lives under **`patzer/`** (tokenizer, model, loop, configs in `patzer/config/`). **Run training and sampling from `patzer/`** so `configurator.py` resolves paths correctly.

---

## Quick start

**1. Python environment** (repo root):

```bash
uv venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
uv pip install torch numpy    # if not already satisfied
```

**2. Data → prepared shards** (see **CLAUDE.md** for full scrape/prepare flags):

```bash
python pipeline/scrape_lichess.py --output-dir ./data/lichess_games --min-elo 1800 --max-months 12
cd patzer && python ../pipeline/prepare.py --input ../data/lichess_games/games_*.txt --output-dir ../data/prepared
```

**3. Train** (example config):

```bash
cd patzer
python train.py config/train_patzer_v4.py --device=auto
```

**4. Evaluate** (after pulling checkpoints if they live only in R2):

```bash
python eval/evaluate.py stockfish ../checkpoints/patzer_v2/weights_best.pt --games 50 --device mps
python eval/evaluate.py leaderboard
```

**5. Lichess bot** (one-time: clone `lichess-bot`, then from repo root):

```bash
python bot/deploy_bot.py install-shim
python bot/deploy_bot.py run v2
```

Details: **[bot/README.md](bot/README.md)** and **`python bot/deploy_bot.py --help`**.

---

## Repository layout (sketch)

| Path | Role |
|------|------|
| `patzer/` | Training, tokenizer, model, `sample.py`, R2 helper (`r2.py`), configs |
| `pipeline/` | Scrape, prepare, droplet helpers for long-running scrapes |
| `eval/` | Engine, Stockfish / head-to-head eval, SQLite results |
| `bot/` | `deploy_bot.py`, `cycle_bots.py`, configs, homemade engine glue |
| `dashboard/` | Flask + React UI for eval leaderboard and Lichess game history |

---

## License and credits

Architecture is inspired by [nanoGPT](https://github.com/karpathy/nanogpt). Chess rules and move validation use [python-chess](https://python-chess.readthedocs.io). For background on search-free neural chess, see for example [DeepMind’s Searchless Chess](https://github.com/google-deepmind/searchless_chess) (research direction; this repo is an independent hobby implementation).

---

*For day-to-day commands, environment variables (R2, Lichess tokens), and cloud training workflows, start with [**CLAUDE.md**](CLAUDE.md).*
