I want to write one (or many) blog posts about this project. So we should keep a runningg log here of everything we do to make it easy to remember. Basically any time we do something meaningful or interesting, we shoud write a short note here. (for example, making the model better, expanding training data, fixing a pesky bug, changing our eval system, etc.)

- **2026-04-29:** After rsync from chewy: **45** `data/lichess_games/games_*.txt` months (**2013-01** … **2016-09**), **11,738,474** games. Lines are already **≥1800 both sides** (scrape floor); stricter cutoffs e.g. **≥2000 → 3.23M**, **≥2200 → 593k**.

- **2026-04-29:** `pipeline/transfer_games_from_chewy.sh` rsync filter: only `games_*.txt` (no `*.zst` / `.pgn.zst`, no `progress.json` / `scrape.log` / `*.stats.json`).

- **2026-04-29:** `pipeline/transfer_games_from_chewy.sh` now `cd`s to repo root, creates `data/lichess_games`, and rsync excludes a nested `lichess_games/` directory so transfers cannot recreate `data/lichess_games/lichess_games/`. Removed the redundant nested copy locally.

- **2026-04-29:** `eval/inspect_training_run.py` now streams a proper CSV to stdout (`iter`, `train_loss`, `val_loss`, `lr`) via `run.scan_history()` so we get the full run, not a 500-row `history()` sample.

- **2026-04-29:** Training: generous val-based early stop (`early_stop_patience_evals`, `early_stop_min_iters` in `train.py`; enabled in `config/train_patzer.py` as 15 evals without improvement after 10k steps). R2 keeps `ckpt.pt` as latest eval for resume; added `ckpt_best.pt` (push on val improvement) for play/eval. DDP syncs the patience counter via `broadcast`.

- **2026-04-29:** Eval tooling defaults to `ckpt_best.pt`: `eval/tournament.py` auto-pick and labels, `eval/sweep.py` dedupes R2 listing when both `ckpt.pt` and `ckpt_best.pt` exist; play/uci docstrings updated.

- **2026-04-29:** Move policy defaults: **temperature 0** (greedy argmax over legal moves) for `eval.engine.Patzer`, UCI/play/tournament/sweep CLIs, and tournament JSON fallbacks; avoids accidental full-temperature play. Lichess homemade engine renamed **`PatzerEngine`** (was `PatzrEngine`); `config.yml` `engine.name` updated to match.

- **2026-04-29:** `MODELS.md` local-eval blurbs only (EstElo ± σ from `tournament.py --show`); v2 now 200 games → **1273 ± 26** (tighter σ, mean down vs 100-game snapshot).

- **2026-04-30:** `eval/tournament.py` — results aggregation and `--show` / `--estimate-elo` now key by **`iter_num`** as well as checkpoint path + temp (R2 always writes the same `ckpt.pt` path). Previously different training steps were merged into one row and `--estimate-elo` showed iter from the **first** JSON row matching the path only, so e.g. T=0.0 could display the wrong iter.

- **2026-04-30:** `MODELS.md` v2 local eval updated for **57k / T=0.0 / 100 games** (~**1380 ± 35**, 51% score); noted separate **0.1** rows at 46k vs 56k from `tournament.py --show`.

- **2026-04-30:** Tournament speed profiling + improvements: added `eval/tournament.py --timing` to measure per-ply time split (Patzer vs Stockfish). Found Stockfish (Elo-limited) was the main bottleneck due to hardcoded 0.1s/move. Added `--sf-move-time` and updated `eval/engine.StockfishPlayer` to reuse a single Stockfish process across Elo changes via `set_elo_limit()`.
- **2026-04-30:** Default Stockfish move time for Elo-limited tournament runs set to **0.05s/move** (quality/speed middle ground).
- **2026-04-30:** Added `eval/model_tournament.py`: model-vs-model league tournament. Discovers checkpoints from R2, supports interactive or `--select/--indices` selection, safely caches downloads (ETag-aware for `ckpt_best.pt`), runs a 2-stage pruning tournament (baseline filter + per-pair SPRT early stopping), prints a leaderboard, and persists to `eval/model_results.json`.
- **2026-04-30:** `eval/model_tournament.py` design choice: each run is **fresh** (no resume bookkeeping), but every match record is stamped with `prefix/temperature/top_k/conditioning/openings` so a separate aggregator combines results across runs sharing a settings bucket. Added `--analyze` (no games, just print combined leaderboard) and `--no-aggregate`. Aggregated Elo uses iterative replay of all games (order-independent), so reruns automatically tighten ratings instead of restarting them.
- **2026-04-30:** `eval/model_tournament.py` supports **multiple `--prefix` values** (concatenated R2 catalog + global indices), fair `--select N` split across prefixes, `--cross-best` (one `ckpt_best.pt` per prefix), and `settings.prefix` as a sorted `|` join for aggregation; baseline when multiple bests picks highest `iter_num`.
- **2026-04-30:** `patzer/r2.py` — added server-side `copy_object()` + CLI `python r2.py copy <src> <dst> [--force]`. Used it to duplicate R2 `checkpoints/patzer_v1/ckpt.pt` → `checkpoints/patzer_v1/ckpt_best.pt` so v1 has both keys for model tournament / eval defaults.

- **2026-04-29:** Model versioning reset after bad v2 run: set up **v2** = v1-sized (6L/6H/384d, ~12M) trained on the new **11.7M-game / 913M-token** dataset, and **v3** = 12L/8H/512d (~40M) on the same dataset. `train_patzer.py` now points at v3 by default; configs live in `patzer/config/train_patzer_v2.py` and `patzer/config/train_patzer_v3.py`.

- **2026-04-29:** Vast (RTX 3060 12GB) OOM guard: switched v2/v3 configs to `batch_size=32` with `gradient_accumulation_steps=4` to preserve the same effective tokens/iter while reducing VRAM.

- **2026-04-30:** Resuming cloud training runs: `launch.py --resume` pulls `checkpoints/` from R2 and runs `train.py --init_from=resume` (resume from `out_dir/ckpt.pt`). To continue a finished run (e.g. v2 stopped at 150k but val loss still falling), bump `max_iters` / `lr_decay_iters` above the prior stop and relaunch on Vast with `--resume`.

- **2026-04-30:** W&B resume support: `patzer/train.py` now stores `wandb_run_id` inside `ckpt.pt` and uses `wandb.init(..., id=..., resume="must")` on `--init_from=resume`, plus logs with `step=iter_num` so charts continue from the right step instead of starting a new run.

- **2026-04-30:** Eval display-name fix: `eval/tournament.py` and `eval/model_tournament.py` now prefer the **model version** directory (`patzer_v*`) for the `Model` label instead of checkpoint filenames like `ckpt_150000`, so tables show stable model ids with iteration shown separately (or as `_best` when appropriate).