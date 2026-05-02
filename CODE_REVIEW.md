# Patzer code review (living doc)

**Original review:** 2026-05-01 (full narrative archived in git history).

This file tracks **what we fixed** and a **short backlog**.

---

## Done

- **`pipeline/filter_games.py` (was §1.1):** Replaced broken `--no-bots` /
  `--standard-only` (`store_true` + `default=True`) with **`--allow-bots`** and
  **`--include-variants`** (`store_true`, default off). Default behaviour unchanged
  (exclude bots, standard chess only).

- **`pipeline/parse_pgn.py` (was §1.3):** Progress logging moved into
  `flush_game()` immediately after `total_games` increments, so it runs **once
  per game** instead of once per PGN line when `total_games` hits a multiple of
  `--log-every`.

---

## Backlog (prioritized)

| Area | Summary |
|------|---------|
| `patzer/sample.py` | §1.2 — chess-aware sampler (see open PR if not merged) |
| `configurator.py` | §1.4 — allow `None` defaults + `split('=', 1)` |
| `train.py` | §1.5–1.7 — `torch.amp.GradScaler`, safe cosine LR, `torch.load` / `weights_only` |
| `eval/engine.py` | §1.8 — preserve `GAME_START` + result when cropping context |
| `r2.py` | §1.9–1.10, §2.5, §2.15 — `pull_dir` freshness, client cache, retries, push sidecars |
| `bot/lichess_homemade.py` | §1.11 — configurable `top_k` |
| `eval/engine.py` | §2.1 — cache tokenized history; KV-cache |
| `eval/elo.py` | §2.2 — index games by player before BT loop |
| `eval/evaluate.py` | §2.3 — optional parallel games |
| `patzer/dataset.py` | §2.4 — use or delete |
| `requirements.txt` | §2.10 — pin torch, numpy, requests, matplotlib |
| `launch.py` | §2.11 — subprocess timeout, `shlex.quote` exports |
| `pipeline/scrape_lichess.py` | §2.7 — `--max-months` vs resumability |
| `pipeline/prepare.py` | §2.8 — `months` metadata helper |
| `eval/evaluate.py` | §2.9, §2.14 — progress plot label, quiet sync |
| `eval/engine.py` | §2.12 — vectorized legal mask |
| `train.py` | §2.13 — warmup `min_lr` floor |
| Misc | §3.x — naming, tests, parse_pgn Elo `?`, etc. |

---

## Summary

Incremental hardening: PyTorch API migration, eval/engine parity with training,
R2 reliability, and perf for play-at-scale.
