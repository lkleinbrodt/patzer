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

- **2026-04-30:** Lichess deploy lives in-repo: `bot/configs/patzer_v1.yml` & `patzer_v2.yml`, `bot/lichess_homemade.py` (`PatzerEngine` reads `homemade_options`: checkpoint path, `device`, `temperature`, `conditioning`), `bot/templates/homemade_shim.py` synced to external `lichess-bot/homemade.py`, and `bot/deploy_bot.py run v1|v2` (sets `PATZER_ROOT`, uses `LICHESS_BOT_TOKEN`). `bot/configs/*.local.yml` gitignored for overrides.
- **2026-04-30:** `eval/model_tournament.py` design choice: each run is **fresh** (no resume bookkeeping), but every match record is stamped with `prefix/temperature/top_k/conditioning/openings` so a separate aggregator combines results across runs sharing a settings bucket. Added `--analyze` (no games, just print combined leaderboard) and `--no-aggregate`. Aggregated Elo uses iterative replay of all games (order-independent), so reruns automatically tighten ratings instead of restarting them.
- **2026-04-30:** `eval/model_tournament.py` supports **multiple `--prefix` values** (concatenated R2 catalog + global indices), fair `--select N` split across prefixes, `--cross-best` (one `ckpt_best.pt` per prefix), and `settings.prefix` as a sorted `|` join for aggregation; baseline when multiple bests picks highest `iter_num`.
- **2026-04-30:** `patzer/r2.py` — added server-side `copy_object()` + CLI `python r2.py copy <src> <dst> [--force]`. Used it to duplicate R2 `checkpoints/patzer_v1/ckpt.pt` → `checkpoints/patzer_v1/ckpt_best.pt` so v1 has both keys for model tournament / eval defaults.

- **2026-04-29:** Model versioning reset after bad v2 run: set up **v2** = v1-sized (6L/6H/384d, ~12M) trained on the new **11.7M-game / 913M-token** dataset, and **v3** = 12L/8H/512d (~40M) on the same dataset. `train_patzer.py` now points at v3 by default; configs live in `patzer/config/train_patzer_v2.py` and `patzer/config/train_patzer_v3.py`.

- **2026-04-29:** Vast (RTX 3060 12GB) OOM guard: switched v2/v3 configs to `batch_size=32` with `gradient_accumulation_steps=4` to preserve the same effective tokens/iter while reducing VRAM.

- **2026-04-30:** Resuming cloud training runs: `launch.py --resume` pulls `checkpoints/` from R2 and runs `train.py --init_from=resume` (resume from `out_dir/ckpt.pt`). To continue a finished run (e.g. v2 stopped at 150k but val loss still falling), bump **`max_iters`** above the prior stop and relaunch with `--resume`. **Do not raise `lr_decay_iters` to match the new `max_iters` unless you intend to:** `get_lr()` only depends on `iter_num` and the *current* `lr_decay_iters` in the config, not on what schedule was used before the checkpoint. If the first job ended near the cosine floor (e.g. `lr_decay_iters=150k` → LR ≈ `min_lr` at 150k) and the second job sets `lr_decay_iters=250k`, then at 150k the cosine ratio moves back into the middle of the curve and **LR jumps ~35×** for v2 hyperparams—train loss recovers but **val loss can spike** right after resume until the schedule cools again. Fix: keep `lr_decay_iters` at the **original** total (stay at `min_lr` for the extension), or use a deliberate warm-restart / new schedule, or add checkpoint-persisted schedule state if we need something fancier.

- **2026-04-30:** Vast offer selection now accounts for **bandwidth costs**: `launch.py` surfaces `inet_up_cost` / `inet_down_cost` ($/GB) in the offer list and computes an estimated **all-in $/hr** using `--up-gb-per-hr` / `--down-gb-per-hr`, with optional filters `--max-inet-up-cost` / `--max-inet-down-cost`.

- **2026-04-30:** Reduced Vast/R2 **egress** by shrinking `ckpt_best.pt`: `patzer/train.py` now saves best checkpoints as **weights-only by default** (no optimizer state) and adds knobs `ckpt_best_min_delta` and `ckpt_best_cooldown_steps` to avoid uploading a new best on every tiny early-training improvement.
- **2026-04-30:** Checkpoint naming + egress redesign: `patzer/train.py` now writes `ckpt.pt` (resume, full optimizer), plus `weights_best.pt` (best weights for eval), and optional `weights_iter_*.pt` snapshots created by server-side R2 copy on best improvements (rate-limited by `weights_snapshot_interval`). Added `patzer/migrate_r2_checkpoint_names.py` to copy historical `ckpt_best.pt`/`ckpt_*.pt` to the new weights filenames without deleting old keys.

- **2026-04-30:** W&B resume support: `patzer/train.py` now stores `wandb_run_id` inside `ckpt.pt` and uses `wandb.init(..., id=..., resume="must")` on `--init_from=resume`, plus logs with `step=iter_num` so charts continue from the right step instead of starting a new run.

- **2026-04-30:** Resume hardening: `patzer/train.py` now fails with a clear error if `--init_from=resume` is used but `out_dir/ckpt.pt` is missing. `launch.py --resume` now pulls only the config’s `out_dir` prefix from R2 (instead of all `checkpoints/`) and fails fast if `ckpt.pt` is still missing after the pull attempt.

- **2026-04-30:** Eval display-name fix: `eval/tournament.py` and `eval/model_tournament.py` now prefer the **model version** directory (`patzer_v*`) for the `Model` label instead of checkpoint filenames like `ckpt_150000`, so tables show stable model ids with iteration shown separately (or as `_best` when appropriate).
- **2026-04-30:** `eval/model_tournament.py` aggregate leaderboard: re-derive display tags from `model_a`/`model_b` + `settings.prefix` so **historical** `model_results.json` rows (filename-only keys + old `label_*` strings) show `patzer_v2` + `Iter` (and `patzer_v2_best` for `ckpt_best.pt`) instead of `ckpt_050000.pt@...`.

- **2026-04-30:** **Eval system overhaul.** Deleted `tournament.py` (762 lines), `model_tournament.py` (1238 lines), `sweep.py` (broken), and all JSON result files. Replaced with three focused files:
  - `eval/db.py` — thin SQLite wrapper; one row per game, no aggregation
  - `eval/elo.py` — Bradley-Terry MLE; Stockfish anchored at configured Elo, Patzer models fitted; confidence intervals from Fisher information
  - `eval/evaluate.py` — single CLI (`stockfish`, `head2head`, `leaderboard`, `history`, `progress` subcommands)

  Key design wins: individual game records (re-analyzable), relative checkpoint paths (no `/Users/lando/...` baked in), no SPRT (was almost always "inconclusive" at 8–12 games), no R2 discovery baked into eval, unified leaderboard from all stored games. The adaptive Bayesian Elo estimation loop from `tournament.py` is preserved exactly in the `stockfish` subcommand. Results live in `eval/results.db` (gitignored). Prior JSON records discarded — cheap to re-run.

- **2026-04-30:** `eval/inspect_training_run.py` — fixed perceived “hang”: stopped materializing the full W&B history with `list(scan_history(...))` and stopped `print(f.read())` on the whole CSV (huge stdout). Now streams rows straight to disk, default `page_size=10000`, stderr progress every 50k rows, optional `--echo` for a short preview. CLI: positional `run_id` (path fixed to `lkleinbrodt-capital-group/patzer/runs/<id>`), `-o`, `--page-size`, `--progress-every`. Default CSV path: repo `data/wandb_runs/<run_id>.csv` (creates dir); `-o` still overrides.

- **2026-04-30:** Queried R2 (`r2.list_weights('checkpoints/patzer_v2/')`) after v2 training finished: **14** files (`weights_best.pt` + `weights_iter_*` from 230k–274k). Plan: `r2.py pull checkpoints/patzer_v2` then `eval/evaluate.py head2head … --round-robin` on a spread (best + early/mid/late iters) to see which checkpoint is strongest at chess vs best val loss alone.

- **2026-04-30:** R2 `checkpoints/patzer_v2/` also had legacy **`ckpt_<iter>.pt`** snapshots (10k–230k) plus `ckpt_best.pt` / `ckpt.pt`. Ran `python -m patzer.migrate_r2_checkpoint_names --prefix checkpoints/patzer_v2`: **23** server-side copies to `weights_iter_*.pt`, **2** skips (`weights_best.pt`, `weights_iter_230000.pt` already present). Docstring in `migrate_r2_checkpoint_names.py` updated for `ckpt_<iter>.pt`, skip-without-`--force` behavior, and that `ckpt.pt` is left alone.

- **2026-04-30:** R2 **`checkpoints/patzer_v1/`** migrated the same way: **9/9** copies (`ckpt_best.pt` → `weights_best.pt`, `ckpt_005000` … `040000` → `weights_iter_*`; `ckpt.pt` unchanged).

- **2026-04-30:** `bot/configs/patzer_v1.yml` & `patzer_v2.yml`: aligned **`challenge`** (briefly included classical) and **`matchmaking`** with the main `lichess-bot/config.yml` (matchmaking on, standard variant, 1‑minute idle timeout, 60/120/180 + 0/1/2 clocks, `opponent_max_rating: 2000`, rated + `coarse` filter). Deploy-specific bits (empty token, `engine.dir: "."`, homemade options) unchanged.

- **2026-04-30:** Bot configs: removed **`classical`** from `challenge.time_controls` in `patzer_v1.yml` / `patzer_v2.yml` (accept bullet/blitz/rapid only).