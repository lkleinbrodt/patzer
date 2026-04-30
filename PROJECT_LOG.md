I want to write one (or many) blog posts about this project. So we should keep a runningg log here of everything we do to make it easy to remember. Basically any time we do something meaningful or interesting, we shoud write a short note here. (for example, making the model better, expanding training data, fixing a pesky bug, changing our eval system, etc.)

- **2026-04-29:** `eval/inspect_training_run.py` now streams a proper CSV to stdout (`iter`, `train_loss`, `val_loss`, `lr`) via `run.scan_history()` so we get the full run, not a 500-row `history()` sample.

- **2026-04-29:** Training: generous val-based early stop (`early_stop_patience_evals`, `early_stop_min_iters` in `train.py`; enabled in `config/train_patzer.py` as 15 evals without improvement after 10k steps). R2 keeps `ckpt.pt` as latest eval for resume; added `ckpt_best.pt` (push on val improvement) for play/eval. DDP syncs the patience counter via `broadcast`.

- **2026-04-29:** Eval tooling defaults to `ckpt_best.pt`: `eval/tournament.py` auto-pick and labels, `eval/sweep.py` dedupes R2 listing when both `ckpt.pt` and `ckpt_best.pt` exist; play/uci docstrings updated.