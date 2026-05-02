# Patzer code review (living doc)

**Original review:** 2026-05-01 (full findings archived in git history if needed).

This file tracks **what we fixed** and a **short backlog**. Detailed write-ups of
each issue were trimmed as fixes landed.

---

## Done

- **`patzer/sample.py` (was §1.2):** Replaced nanoGPT/tiktoken stub with a chess
  sampler: `ChessTokenizer`, legal-move masking, optional UCI `start` / `FILE:`,
  `conditioning` aligned with `eval/engine.Patzer`, vocab_size check vs
  checkpoint. GPT-2 / `init_from=gpt2*` removed (use upstream nanoGPT for that).

---

## Backlog (prioritized)

| Area | Summary |
|------|---------|
| `filter_games.py` | §1.1 — `BooleanOptionalAction` for bots / variants |
| `parse_pgn.py` | §1.3 — progress log once per game |
| `configurator.py` | §1.4 — allow `None` defaults + `split('=', 1)` |
| `train.py` | §1.5–1.7 — `torch.amp.GradScaler`, safe cosine LR, `torch.load` plan |
| `eval/engine.py` | §1.8 — preserve `GAME_START` + result when cropping context |
| `r2.py` | §1.9–1.10, §2.5, §2.15 — `pull_dir` freshness, client cache, retries, push sidecars |
| `bot/lichess_homemade.py` | §1.11 — configurable `top_k` |
| `eval/engine.py` | §2.1 — cache tokenized history; KV-cache (larger) |
| `eval/elo.py` | §2.2 — index games by player before BT loop |
| `requirements.txt` | §2.10 — pin torch, numpy, requests, matplotlib |
| `launch.py` | §2.11 — subprocess timeout, `shlex.quote` exports |
| Misc | §2.7–2.9, §2.12–2.14, §3.x — scrape `max-months`, `prepare.py` months helper, progress plot labels, warmup floor, etc. |

---

## Summary

Training, R2, and eval are solid; remaining work is mostly correctness edges,
PyTorch deprecations, R2 hardening, and engine speed (KV cache).
