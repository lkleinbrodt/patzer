# Patzer code review (living doc)

**Original review:** 2026-05-01 (full narrative archived in git history).

This file tracks **what we fixed** and a **short backlog**. Detailed write-ups were trimmed as fixes landed.

---

## Done

- **`pipeline/filter_games.py` (was §1.1):** `--allow-bots` / `--include-variants` (opt-in); defaults unchanged (exclude bots, standard chess only).
- **`pipeline/parse_pgn.py` (was §1.3):** Progress logging in `flush_game()` once per game.
- **`patzer/sample.py` (was §1.2):** Chess-aware sampler: `ChessTokenizer`, legal-move masking, `conditioning`, vocab check; context crop now matches engine (`GAME_START` + optional result, then move tail).
- **`patzer/configurator.py` (was §1.4):** CLI `--key=value` uses `split('=', 1)`; overrides allowed when the config default is `None` (type check only when the existing value is not `None`).
- **`patzer/train.py` (was §1.5–1.7, §2.13 warmup):** `torch.amp.GradScaler('cuda', …)` (enabled only on CUDA fp16); cosine LR uses safe denominator + clamped ratio; warmup floored at `min_lr`; resume / `weights_best` peek use shared `load_checkpoint`; eval-loop metrics use module-level `json`/`time`.
- **`patzer/checkpoint_util.py` + consumers:** Prefer `torch.load(..., weights_only=True)` with `GPTConfig` allow-listed; fall back for legacy pickles. Used by `train.py`, `sample.py`, `eval/engine.py`.

---

## Backlog (prioritized)

| Area | Summary |
|------|---------|
| `r2.py` | §1.9–1.10, §2.5, §2.15 — `pull_dir` freshness, client cache, retries, push sidecars |
| `bot/lichess_homemade.py` | §1.11 — configurable `top_k` |
| `eval/engine.py` | §2.1 — cache tokenized history; KV-cache |
| `eval/elo.py` | §2.2 — index games by player before BT loop |
| `eval/evaluate.py` | §2.3 — optional parallel games; §2.9 / §2.14 — progress plot label, quiet sync |
| `patzer/dataset.py` | §2.4 — use or delete |
| `requirements.txt` | §2.10 — pin torch, numpy, requests, matplotlib |
| `launch.py` | §2.11 — subprocess timeout, `shlex.quote` exports |
| `pipeline/scrape_lichess.py` | §2.7 — `--max-months` vs resumability |
| `pipeline/prepare.py` | §2.8 — `months` metadata helper |
| `eval/engine.py` | §2.12 — vectorized legal mask |
| Misc | §3.x — naming, tests, parse_pgn Elo `?`, etc. |

---

## Summary

Incremental hardening: PyTorch API migration, eval/train/sample parity for checkpoints and context cropping, R2 reliability, and engine speed (KV cache) for play-at-scale.
