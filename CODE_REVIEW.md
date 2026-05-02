# Patzer code review (living doc)

**Original review:** 2026-05-01 (full narrative archived in git history).

This file tracks **what we fixed** and a **short backlog**. Detailed write-ups were trimmed as fixes landed.

---

## Done

- **`pipeline/filter_games.py` (was §1.1):** `--allow-bots` / `--include-variants` (opt-in); defaults unchanged (exclude bots, standard chess only).
- **`pipeline/parse_pgn.py` (was §1.3):** Progress logging in `flush_game()` once per game.
- **`patzer/sample.py` (was §1.2):** Chess-aware sampler: `ChessTokenizer`, legal-move masking, `conditioning`, vocab check. (Merged via separate PR; crop/utilities may be extended on `main`.)
- **`patzer/r2.py` (was §1.9–1.10, parts of §2.5):** Thread-safe cached boto3 client with adaptive retries; `pull_file` / `pull_dir` skip only when `is_fresh` (ETag sidecar); pushes write `.r2meta` after upload.

---

## Backlog (prioritized)

| Area | Summary |
|------|---------|
| `configurator.py` / `train.py` / `checkpoint_util` | §1.4–1.7 — CLI `None` overrides, GradScaler, LR safety, safe `torch.load` (see open PR if not merged) |
| `eval/engine.py` | §1.8 — prefix crop parity with training (if any gap remains); §2.1 — token cache / KV-cache; §2.12 — vectorized legal mask |
| `bot/lichess_homemade.py` | §1.11 — configurable `top_k` |
| `eval/elo.py` | §2.2 — index games by player before BT loop |
| `eval/evaluate.py` | §2.3 — optional parallel games; §2.9 / §2.14 — progress plot label, quiet sync |
| `patzer/dataset.py` | §2.4 — use or delete |
| `requirements.txt` | §2.10 — pin torch, numpy, requests, matplotlib |
| `launch.py` | §2.11 — subprocess timeout, `shlex.quote` exports |
| `pipeline/scrape_lichess.py` | §2.7 — `--max-months` vs resumability |
| `pipeline/prepare.py` | §2.8 — `months` metadata helper |
| `r2.py` | §2.15 — any remaining edge cases (multi-part uploads, copyObject sidecars on server-only paths) |
| Misc | §3.x — naming, tests, parse_pgn Elo `?`, etc. |

---

## Summary

Incremental hardening: PyTorch API migration, eval/train parity, R2 reliability (freshness + retries + upload sidecars), and engine throughput (KV cache, masking) for play-at-scale.
