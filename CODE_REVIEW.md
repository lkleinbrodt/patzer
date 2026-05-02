# Patzer Code Review

**Reviewer:** Cursor agent (Opus 4.7)
**Date:** 2026-05-01
**Scope:** Full repo (training, pipeline, eval, bot, infra) — ~7.3k LOC of Python

This is a focused review aimed at finding genuine issues. I deliberately skip
nitpicks (style, formatting, naming preferences) and focus on:

1. **Bugs** — things that don't do what they say.
2. **Footguns** — code that works today but will silently break or mislead.
3. **High-impact improvements** — meaningful wins on correctness, speed, or
  maintainability.

Findings are grouped by severity. Each item links to a file and gives a concrete
fix where reasonable.

---

## 1. Bugs

### 1.1 `pipeline/filter_games.py` — `--no-bots` and `--standard-only` cannot be turned off

```122:131:pipeline/filter_games.py
    parser.add_argument("--no-bots", action="store_true", default=True,
                        help="Exclude games where either player is a BOT (default: True)")
    parser.add_argument("--standard-only", action="store_true", default=True,
                        help="Exclude variant games (default: True)")
```

`action="store_true"` only ever flips a flag from `False` → `True`. Combined
with `default=True`, the value is always `True`, regardless of CLI. There is no
way to disable bot/variant filtering from the command line.

Today this happens to do the right thing (we want to exclude both), but the help
text is misleading and any future "let bots through" experiment would silently
fail.

**Fix:** use `argparse.BooleanOptionalAction` (Py 3.9+) or invert the flag:

```python
parser.add_argument("--allow-bots", action="store_true", default=False,
                    help="Include games where either player is a BOT")
parser.add_argument("--include-variants", action="store_true", default=False,
                    help="Include non-standard chess variants")
```

---

### 1.2 `patzer/sample.py` — entirely stale; uses `tiktoken` and OpenWebText defaults

```1:25:patzer/sample.py
"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
```

This file is the unmodified nanoGPT sampler. It:

- Imports `tiktoken` (a GPT‑2 BPE encoder), which Patzer doesn't use and which
isn't in `requirements.txt`.
- Defaults `out_dir = 'out'` (Patzer uses `checkpoints/patzer_vN`).
- Looks for `meta.pkl` (Patzer writes `meta.json`).
- Falls back to `enc = tiktoken.get_encoding("gpt2")`, which would tokenize
prompt strings as English text and produce gibberish through a chess model.

`CLAUDE.md` documents this command (`python sample.py --out_dir=...`) as a
sanity check, but running it on a real Patzer checkpoint would either crash on
import or produce nonsense moves.

**Fix:** either delete `sample.py` (`eval/engine.py` already covers the use
case) or rewrite it to use `ChessTokenizer` and decode tokens as UCI moves.

---

### 1.3 `pipeline/parse_pgn.py` — progress log fires N times per game

```209:218:pipeline/parse_pgn.py
        if stats["total_games"] % args.log_every == 0 and stats["total_games"] > 0:
            elapsed = time.time() - start_time
            rate = stats["total_games"] / elapsed
            print(
                f"  {stats['total_games']:,} games processed, "
                f"{stats['kept_games']:,} kept "
                f"({100 * stats['kept_games'] / stats['total_games']:.1f}%) "
                f"| {rate:.0f} games/sec",
                file=sys.stderr
            )
```

This block lives in the per‑*line* loop, not the per‑game block. Each game
spans many lines, so when `total_games` is a multiple of `log_every` (default
10 000), the same log line is printed once per line of that game's PGN — easily
40+ duplicate lines on stderr.

**Fix:** move the progress block into `flush_game()`, right after
`stats["total_games"] += 1`.

---

### 1.4 `patzer/configurator.py` — type assert prevents setting `Optional[...]` fields

```33:45:patzer/configurator.py
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
```

`assert type(attempt) == type(globals()[key])` rejects any override whose type
differs from the *current* global. Concretely, in `train.py`:

```python
cooldown_start_iter = None  # NoneType
```

You cannot pass `--cooldown_start_iter=80000` from the CLI — the assert raises
`AssertionError: int != NoneType`. The only way to set it is via a config file
(or by editing `train.py`'s default to `0`).

Same issue for any field defaulting to `None`, e.g. `wandb_run_id = ""` — fine
today because it defaults to `""` (str), but the moment someone changes it to
`None` the CLI override breaks.

**Fix:** allow the override if the existing value is `None`, or compare type
classes:

```python
existing = globals()[key]
if existing is not None and type(attempt) is not type(existing):
    raise TypeError(f"Type mismatch for {key}: {type(attempt).__name__} vs {type(existing).__name__}")
```

Also: `arg.split('=')` will raise `ValueError: too many values to unpack` if
the value itself contains `=` (e.g. `--out_dir=foo=bar`). Use `split('=', 1)`.

---

### 1.5 `patzer/train.py` — `torch.cuda.amp.GradScaler` is deprecated

```243:244:patzer/train.py
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```

PyTorch 2.3+ deprecated `torch.cuda.amp.GradScaler` in favor of
`torch.amp.GradScaler('cuda', enabled=...)`. Today this prints a
`FutureWarning` on every run; in a future minor release it will be removed.

**Fix:**

```python
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
```

---

### 1.6 `torch.load(...)` without `weights_only=True` — will break with PyTorch 2.6+

Every checkpoint load in the repo (`train.py`, `eval/engine.py`,
`patzer/sample.py`) calls `torch.load(...)` with the historic default
`weights_only=False` (or the `weights_only=False` is set explicitly in
`eval/engine.py`). PyTorch is migrating the default to `True` for security, and
already issues a `FutureWarning` on every load.

The blocker for switching is that Patzer's checkpoints contain non-tensor
values (`model_args` dict, `config` dict, `wandb_run_id` str). Loading those
under `weights_only=True` requires explicitly allow-listing
`patzer.model.GPTConfig` (and possibly the dict types) via
`torch.serialization.add_safe_globals([...])`.

**Recommendation:** before this becomes mandatory, add a small loader helper:

```python
def load_checkpoint(path, map_location):
    import torch.serialization as ts
    from patzer.model import GPTConfig
    ts.add_safe_globals([GPTConfig])
    return torch.load(path, map_location=map_location, weights_only=True)
```

and use it everywhere. Otherwise a routine `pip install -U torch` will brick
every existing checkpoint.

---

### 1.7 `patzer/train.py` — cosine schedule division-by-zero / out-of-bounds

```331:338:patzer/train.py
    else:
        # cosine decay (original nanoGPT schedule)
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)
```

Two failure modes hidden by the assert:

1. If a config sets `lr_decay_iters == warmup_iters` (or less), the divisor is
  `0` (or negative) — `ZeroDivisionError` or a wildly negative ratio. Crashes
   training mid-run.
2. The `assert 0 <= decay_ratio <= 1` makes the schedule a binary contract:
  either it succeeds, or it crashes the entire training loop with no useful
   message. Train.py runs for hours before hitting this.

**Fix:** clamp and guard:

```python
denom = max(1, lr_decay_iters - warmup_iters)
decay_ratio = max(0.0, min(1.0, (it - warmup_iters) / denom))
```

---

### 1.8 `eval/engine.py` — `block_size` cropping silently drops the conditioning token

```112:117:eval/engine.py
        # Crop to model block size
        block_size = self.model.config.block_size
        if len(token_ids) > block_size:
            token_ids = token_ids[-block_size:]
```

The token sequence is `[<GAME_START>, <RESULT>, m1, m2, ...]`. When a game
exceeds `block_size` plies (256 for v1–v4), the cropping window keeps the
*last* `block_size` tokens and drops `<GAME_START>` and the result token.

The whole conceit of the model is "conditioned on outcome from position 0".
For long games, mid-game the model loses its conditioning and may shift policy
unexpectedly.

For v1–v4 with `block_size=256`, this triggers in any game past ~254 plies
(127 full moves) — uncommon in casual play, but absolutely happens in eval and
on Lichess (Patzer can play long endings).

**Fix:** preserve the prefix:

```python
if len(token_ids) > block_size:
    prefix = token_ids[:2]                     # <GAME_START>, <RESULT>
    tail = token_ids[2:][-(block_size - 2):]    # last (block-2) moves
    token_ids = prefix + tail
```

---

### 1.9 `patzer/r2.py` — `pull_dir(skip_existing=True)` doesn't validate ETag

```237:241:patzer/r2.py
            local_path = local_dir / rel
            if skip_existing and local_path.exists():
                print(f"[r2] skipping {key} (already local)")
                skipped += 1
                continue
```

`pull_dir` skips files based purely on existence, never compares ETags. If you
re-train and re-upload a checkpoint, then run
`python r2.py pull checkpoints/patzer_v3`, the *old* local copy is silently
preserved.

Combined with the fact that `pull_dir` doesn't write sidecar ETags either (only
`pull_file` does), `is_fresh()` will then return `False` (no sidecar = stale)
on any subsequent eval — confusing and inconsistent.

**Fix:** in `pull_dir`, factor the per-key download through `pull_file()` so
sidecars are written. Optionally, when `skip_existing=True`, prefer
`is_fresh()` over `exists()`:

```python
if skip_existing and is_fresh(key, local_path):
    ...
```

---

### 1.10 `patzer/r2.py` — `_client()` re-loads `.env` and rebuilds boto3 client on every call

```39:56:patzer/r2.py
def _client():
    import boto3
    from dotenv import load_dotenv
    load_dotenv()
    endpoint = os.environ.get("R2_ENDPOINT_URL", "").strip()
    ...
    client = boto3.client("s3", ...)
    return client, bucket
```

Every `push_file`, `push_async`, `pull_file`, `get_etag`, `is_fresh`,
`copy_object`, `checkpoint_exists` calls `_client()`. Each call re-reads
`.env` and constructs a fresh boto3 client (which has its own credential
provider chain, signer setup, etc.).

During `evaluate.py rr-leaderboard` over 10 checkpoints with ETag dedup, this
fires `_client()` ~30+ times. Cheap individually, but noticeable.

**Fix:** cache:

```python
_CLIENT_CACHE: tuple | None = None
def _client():
    global _CLIENT_CACHE
    if _CLIENT_CACHE is not None:
        return _CLIENT_CACHE
    # ...build...
    _CLIENT_CACHE = (client, bucket)
    return _CLIENT_CACHE
```

---

### 1.11 `bot/lichess_homemade.py` — `top_k` is hardcoded `None` despite being a known knob

```93:99:bot/lichess_homemade.py
        self.patzer = Patzer(
            checkpoint_path=ckpt_path,
            device=device,
            temperature=temperature,
            top_k=None,
            conditioning=conditioning,
        )
```

`temperature` and `conditioning` are read from `homemade_options`, but `top_k`
is hardcoded to `None`. If you wanted to experiment with top-k sampling on
Lichess (likely useful at higher temperature), you'd need to edit and redeploy
the shim.

**Fix:** add `top_k = _opt_float(options, "top_k")` (cast to int when not None).

---

## 2. Significant improvements

### 2.1 `eval/engine.py` — re-encodes the entire move history every move

```108:122:eval/engine.py
        token_ids = [self.tokenizer.game_start_id]
        if result_id is not None:
            token_ids.append(result_id)
        token_ids += [self.tokenizer.encode(m) for m in move_history]

        # Crop to model block size
        block_size = self.model.config.block_size
        if len(token_ids) > block_size:
            token_ids = token_ids[-block_size:]

        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            with self._ctx:
                logits, _ = self.model(x)
```

Two compounding inefficiencies, both relevant for Lichess:

1. **Re-tokenization every move** — the entire `move_history` is re-encoded on
  each call. O(n²) over a game.
2. **Full forward pass every move** — no KV cache. The model recomputes
  attention over the entire prefix each ply, even though only the last token
   is new.

Per-move cost on CPU/MPS goes from O(1) to O(plies). On CPU at move 100, this
is roughly 100× more compute than necessary.

**Fix (cheap):** cache `self._encoded_history: list[int]` keyed off the longest
common prefix of `move_history` so you only encode the diff.

**Fix (deep):** plumb a KV cache through `model.py`'s `CausalSelfAttention`.
Since this is a `nanoGPT` derivative without an existing cache, this is real
work — but for any serious bot deployment (especially CPU/MPS), it's the
single biggest speedup available.

---

### 2.2 `eval/elo.py` — Bradley-Terry inner loops scan all games per player per iter

```81:115:eval/elo.py
    for _ in range(1000):
        max_delta = 0.0
        for player in free:
            num = 0.0
            den = 0.0
            for g in games:
                w, b, r = g["white"], g["black"], g["result"]
                if player not in (w, b):
                    continue
                ...
```

Triple-nested loop: 1000 iterations × ~50 players × ~10 000 games = 500M
condition checks. Today a leaderboard build takes seconds, but this scales
poorly as the DB grows.

**Fix:** precompute `games_by_player: dict[str, list[GameRow]]` once before
the BT loop, then iterate `for g in games_by_player[player]:`. ~50× speedup
for typical mixes.

While there, the same fix applies to `_stderr` (line 121) and the win-prob
flip on line 95–97 can be expressed in one line:

```python
p_win = (_elo_win_prob(r_p, r_o) if is_white
         else 1.0 - _elo_win_prob(r_o, r_p))
```

---

### 2.3 `eval/evaluate.py` — games are sequential; round-robin is embarrassingly parallel

`cmd_head2head` and `cmd_stockfish` play games one-at-a-time. For a
round-robin across N=8 checkpoints with 10 games per pair (28 pairs × 10
games = 280 games), at ~30s/game that's 2.3 hours of wall time — most of
which is CPU/MPS-bound python-chess + Stockfish, fully parallelizable.

**Recommendation:** add `--parallel N` that runs an asyncio or multiprocessing
pool of game workers. Each worker holds its own `Patzer` and `StockfishPlayer`
process. The DB write is the only shared resource and SQLite handles
concurrent inserts fine with `INSERT OR ROLLBACK`.

The Bayesian Elo loop in `cmd_stockfish` is harder to parallelize because each
game's Stockfish target depends on the posterior mean — but a "play in batches
of K, update posterior, pick next K target Elos" works.

---

### 2.4 `patzer/dataset.py` — defined but unused; either consolidate or delete

`train.py` has its own inline `get_batch()` (lines 158–173) using `np.memmap`.
`patzer/dataset.py` defines a `ChessDataset` with the same logic plus a
`DataLoader`-friendly interface — but nothing imports it.

If the plan is to migrate `train.py` to a `DataLoader` (which would unlock
`num_workers > 0` for prefetching), great — `dataset.py` is a head start.
Otherwise it's dead code with subtle differences from the live loader (e.g.
samples by sequential index vs random `randint`), which is a maintenance trap.

**Fix:** either wire `train.py` to use `ChessDataset` + `DataLoader`, or delete
`patzer/dataset.py`.

---

### 2.5 `patzer/r2.py` — uploads have no retry logic

```108:122:patzer/r2.py
    def _do():
        try:
            print(f"[r2] pushing {local_path} → {r2_key} (async)")
            client.upload_file(str(tmp), bucket, r2_key)
            if then_copy_to:
                copy_object(r2_key, then_copy_to, overwrite=False)
        except Exception as exc:
            print(f"[r2] ERROR uploading {local_path} → {r2_key}: {exc}", file=sys.stderr)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
```

A single transient network failure (R2 5xx, TLS reset, DNS hiccup) drops the
upload silently after one stderr line. For a 10‑hour Vast.ai run that ends
with the best checkpoint never reaching R2, this is genuine pain.

**Fix:** wrap `client.upload_file` in `tenacity.retry` (or a small hand-rolled
backoff: 3 attempts, exponential with jitter). Same for `pull_file`.

Bonus: add a checksum verification post-upload using `head_object().ETag`
against an MD5 you compute pre-upload (boto3 transfer manager handles
multi-part ETags for you, but you can at least detect "uploaded zero bytes
silently").

---

### 2.6 `patzer/tokenizer.py` — no validation that vocab matches the trained model

`Patzer.__init__` always calls `ChessTokenizer()` — rebuilds the vocab
deterministically from `chess.SQUARES`. The model loaded from a checkpoint has
its own vocab size baked into the embedding matrix, but there is no runtime
check that `len(ChessTokenizer().token_to_id) == checkpoint_vocab_size`.

If a future python-chess release ever changes the iteration order of
`chess.SQUARES` (unlikely but possible), every old checkpoint would silently
emit nonsense moves with no error.

**Fix:** in `_load_model`, assert:

```python
assert checkpoint["model_args"]["vocab_size"] == self.tokenizer.vocab_size, (
    f"vocab mismatch: checkpoint={checkpoint['model_args']['vocab_size']} "
    f"vs runtime tokenizer={self.tokenizer.vocab_size}. "
    "Did the chess vocab generator change?"
)
```

The training pipeline already saves `data/vocab.json`. Persisting it next to
each checkpoint (or hashing it into `model_args`) would be even more robust.

---

### 2.7 `pipeline/scrape_lichess.py` — `--max-months` cap interacts oddly with resumability

```324:328:pipeline/scrape_lichess.py
    if args.max_months:
        all_files = all_files[:args.max_months]
        log.info(f"Capped to first {args.max_months} months (oldest first)")
```

The cap is applied to `all_files` *before* skipping completed months. So if you
ran `--max-months 12` once (downloading the first 12), then re-ran with
`--max-months 24` hoping to add 12 more, you'd get… nothing. The universe is
re-capped at 12, all 12 are completed, "Nothing to do."

The CLI hint suggests `--max-months 24`, but you actually need
`--max-months 24` *and* nothing-cap-aware extension logic. Right now the only
way to extend is to delete `progress.json` or pass `--months 2014-01 2014-02 …` explicitly.

**Fix:** apply cap *after* filtering to `todo`:

```python
todo = [f for f in all_files if extract_month(f) not in completed]
if args.max_months:
    todo = todo[:args.max_months]
    log.info(f"Capped run to next {args.max_months} months")
```

This makes `--max-months` mean "do up to N more months this run", which is
what users expect from a resumable scraper.

---

### 2.8 `pipeline/prepare.py` — `months` metadata builder is cryptic and fragile

```564:568:pipeline/prepare.py
        "months": sorted([
            m for f in input_files
            for m in [re.search(r"(\d{4}-\d{2})", Path(f).name)]
            if m for m in [m.group(1)]
        ]),
```

This shadows `m` three times in the same comprehension (regex Match → str →
str). Works today via Python's lexical-scoping coincidence, but it's
unreviewable and one tiny edit will silently break it.

**Fix:**

```python
def _extract_month(name: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2})", name)
    return m.group(1) if m else None

months = sorted({m for f in input_files
                 for m in [_extract_month(Path(f).name)]
                 if m})
"months": months,
```

---

### 2.9 `eval/evaluate.py cmd_progress` — Elo per iter is fitted in isolation, not comparable to leaderboard

```958:967:eval/evaluate.py
    iters = sorted(sf_games)
    elos = []
    for it in iters:
        sub_games = sf_games[it]
        ratings = compute_ratings(sub_games)
        patzer_ratings = [r for r in ratings if version in r.name and not r.name.startswith("stockfish:")]
        if patzer_ratings:
            elos.append(patzer_ratings[0].elo)
        else:
            elos.append(float("nan"))
```

Each iter's Elo is computed from *only* games at that iter, fitting a fresh BT
model with one Patzer + a few Stockfish anchors. That's a valid per-snapshot
Elo, but it is **on a different scale** than the unified leaderboard in
`cmd_leaderboard`, which fits all checkpoints jointly.

A user looking at "progress" expecting to compare with the leaderboard rank
will be silently misled — especially when only a handful of Stockfish anchors
overlap.

**Fix:** at minimum, document this in the plot title / axis label:

```python
ax.set_ylabel("Estimated Elo (per-snapshot, vs Stockfish only)")
```

Better: fit BT once on *all* games, then plot the iter‑indexed Patzer Elos
from that single fit.

---

### 2.10 `requirements.txt` is missing the most important deps

```1:5:requirements.txt
python-chess
boto3
python-dotenv
wandb
pygame
```

Missing: `torch`, `numpy`, `requests`, `tiktoken` (if `sample.py` is kept),
`matplotlib` (used by `evaluate.py progress`). `CLAUDE.md` documents the gap
informally, but a fresh clone on a new machine will fail at `python train.py`
because `numpy` and `torch` aren't installed.

**Fix:** pin a baseline:

```
torch>=2.3
numpy>=1.26
requests>=2.31
matplotlib>=3.8
python-chess>=1.999
boto3>=1.34
python-dotenv>=1.0
wandb>=0.17
pygame>=2.5     # only needed for play_gui
```

Consider splitting into `requirements.txt` (training/eval core) and
`requirements-extras.txt` (`pygame`, `matplotlib`).

---

### 2.11 `launch.py` — `subprocess.run(vastai, ...)` has no timeout

```48:53:launch.py
def vast(*args, raw=True):
    cmd = ["vastai"] + list(args) + (["--raw"] if raw else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return json.loads(r.stdout) if raw else r.stdout.strip()
```

If `vastai` ever hangs (network outage, API down, broken auth re-prompt),
`launch.py` hangs forever. `--list` and `--status` calls are interactive
debug commands and a hang isn't catastrophic, but `vast("show", "instance", str(instance_id))` is also called from `run_on_instance` *after* you've
already paid for the GPU.

**Fix:** add `timeout=60` (or a sensible default), catch
`subprocess.TimeoutExpired`, surface a useful error.

Same goes for `r2_export_lines()`'s shell quoting:

```83:84:launch.py
    lines = [f"export {k}='{os.environ[k]}'" for k in keys if os.environ.get(k)]
```

If any env var contains a `'`, the generated bash is malformed. Use
`shlex.quote(os.environ[k])` to be safe.

---

### 2.12 `eval/engine.py` — `for tid in legal_ids: mask[tid] = logits[tid]` is slow

```125:130:eval/engine.py
        # Legal move masking — zero out everything not in the legal set
        legal_ids = {self.tokenizer.token_to_id[m] for m in legal_moves}
        mask = torch.full_like(logits, float("-inf"))
        for tid in legal_ids:
            mask[tid] = logits[tid]
        logits = mask
```

A Python loop over 30 ints calling into PyTorch indexing is slow on every
device — and this happens once per move per game. Vectorize:

```python
legal_ids_t = torch.tensor(sorted(legal_ids), device=self.device, dtype=torch.long)
mask = torch.full_like(logits, float("-inf"))
mask.scatter_(0, legal_ids_t, logits.gather(0, legal_ids_t))
logits = mask
```

Saves ~50–200µs per move depending on device — small absolute, but with
thousands of moves in a tournament it adds up to a few minutes of wall time.

---

### 2.13 `patzer/train.py` — warmup ignores `min_lr` floor

```322:324:patzer/train.py
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
```

For typical configs (`learning_rate=6e-4`, `warmup_iters=3000`, `min_lr=1e-5`),
the first few hundred iters use LR ≪ `min_lr`. E.g. iter 0 → LR ≈ 2e-7. The
schedule technically ramps from "essentially zero" rather than from
`min_lr`.

In practice this doesn't matter much (the model is so under-trained at iter 0
that LR doesn't matter), but it's surprising and inconsistent with how cosine
explicitly floors at `min_lr` post-decay.

**Fix:**

```python
if it < warmup_iters:
    return max(min_lr, learning_rate * (it + 1) / (warmup_iters + 1))
```

---

### 2.14 `eval/evaluate.py` — `_sync_checkpoint` is loud and slow on round-robin

Each call prints "is up to date" / "stale — pulling from R2..." plus a
`head_object` call. For `rr-leaderboard` with 15 checkpoints, that's 15 R2
round trips + 15 noisy lines, even when everything is up-to-date.

**Fix:** batch — call `list_objects_v2` once to get all ETags under the prefix,
then check freshness in memory. Or at least gate the chatter behind a
`--quiet` flag.

---

### 2.15 `patzer/r2.py` — `push_async` doesn't write a sidecar

`pull_file` writes `<file>.r2meta` containing the ETag so future `is_fresh`
calls can short-circuit. `push_async` doesn't, which means after training
finishes:

- Local `weights_best.pt` exists.
- No sidecar.
- Next `evaluate.py` run calls `is_fresh()` → no sidecar → "stale" → re-pulls
from R2 (and the freshly pulled file is byte-identical to the local copy).

Wasted bandwidth and a confusing log line.

**Fix:** after `client.upload_file(...)` succeeds, fetch the ETag and write the
sidecar:

```python
etag = client.head_object(Bucket=bucket, Key=r2_key)["ETag"].strip('"')
_write_sidecar(local_path, etag)
```

---

## 3. Smaller things (still worth doing)

### 3.1 `eval/engine.py` — `illegal_move_count` is misnamed

The counter is incremented when probs are NaN or sum to ~0, not when an
illegal move is selected (legal masking already prevents that). Rename to
`degenerate_distribution_count` or `nan_fallback_count`.

### 3.2 `patzer/train.py` — local-import noise

```385:386:patzer/train.py
        import json as _json, time as _time
```

`json` and `time` are already imported at the top. The `_json` / `_time`
aliasing suggests they were renamed to avoid conflict with something — but
nothing in scope conflicts.

### 3.3 `eval/engine.py` — temperature comparison on a float

```136:139:eval/engine.py
        if self.temperature == 0:
            chosen_id = torch.argmax(logits).item()
        else:
            logits = logits / self.temperature
```

Comparing a float to `0` exactly. With CLI floats this is fine because users
pass `0` literally, but `== 0.0` is more honest. Also, the default is `0.01`,
so any user passing `--temperature 0` (and meaning it) gets greedy — fine,
just worth a comment.

### 3.4 `pipeline/parse_pgn.py` — ELO defaults to 0 silently on parse error

```132:136:pipeline/parse_pgn.py
        try:
            white_elo = int(current_headers.get("WhiteElo", 0))
            black_elo = int(current_headers.get("BlackElo", 0))
        except ValueError:
            white_elo, black_elo = 0, 0
```

A game with malformed Elo (e.g. `"?"`) silently gets recorded as
`0 0 e2e4 ...`. That game then passes through the prepared `.bin` and is
indistinguishable from a real 0-Elo game. No filter downstream uses Elo
(`prepare.py` discards it), but if/when we add Elo conditioning, this becomes
silent data corruption.

**Fix:** drop the game (`return`) instead of zero-defaulting.

### 3.5 `eval/evaluate.py cmd_head2head` — semicolons in branch bodies

```635:640:eval/evaluate.py
            if score_a == 1.0:
                w += 1; tag = f"{na} wins"
            elif score_a == 0.0:
                l += 1; tag = f"{nb} wins"
            else:
                d += 1; tag = "draw"
```

Just split onto two lines each — current form is harder to scan.

### 3.6 No tests anywhere

There is no `tests/` directory. The codebase has many natural unit-test
candidates: `tokenizer.py` round-trip, `elo.py` BT convergence on a synthetic
result set, `parse_game_line` edge cases, `_resolve_checkpoint` shorthand
parsing, `_pick_evenly_spaced_iters`. A small `pytest` suite would catch most
regressions cheaply.

### 3.7 Repo-root markdown clutter

`PROJECT_LOG.md`, `MODELS.md`, `PLAN.md`, `wsd_schedule_guide.md`,
`v2_v3_learnings.md`, `v2_v3_retrospective.md`, `CODE_REVIEW.md` (this file)
all live at the repo root. Consider moving everything except `README.md` and
`CLAUDE.md` into `docs/`.

### 3.8 `__pycache__` is checked into the work tree but not gitignored at the right level

`__pycache__/` *is* in `.gitignore`, but the directory shows up in the file
listing at the repo root. Probably a stale artifact — `git rm -rf --cached __pycache__/` once would clean it up.

---

## 4. Summary

The codebase is in genuinely good shape for a one-person research project of
this size. Strong points:

- **Configurator + per-version configs** keeps experiments tidy and
reproducible.
- **R2 sync is well-thought-out** with separation of latest vs best vs
snapshot, async upload throttling, and atexit drain.
- **Eval is unified and reproducible** — single CLI, SQLite source of truth,
one row per game. `PROJECT_LOG.md` captures the iteration well.
- **Pipeline is resumable and observable** — checksum verification, progress
bars, in-memory streaming through `zstdcat | filter | parse`.

The most impactful fixes (in rough ROI order):

1. **Fix or delete `patzer/sample.py*`* — it's broken bait.
2. **Fix `--no-bots` / `--standard-only` flags** in `filter_games.py`.
3. **Stop cropping the conditioning prefix** in `eval/engine.py` (1.8).
4. **Add KV-cache + history caching** to `Patzer.get_move` (2.1) — single
  biggest engine speedup.
5. **Future-proof `torch.load`** for `weights_only=True` (1.6) — looming
  PyTorch breakage.
6. **Migrate `GradScaler` to the new API** (1.5) — same.
7. **Fix `parse_pgn.py` log spam** (1.3) — annoyance, but cheap.
8. **Add upload retries + sidecar on push_async** (2.5, 2.15) — protects
  against losing best checkpoints.
9. **Speed up `elo.py` BT loop** (2.2) — needed before the DB grows much
  larger.
10. **Add a small test suite** (3.6) — even 50 lines of pytest would catch
  most of the bugs above.

Nothing here is critical-path-blocking. The model trains, checkpoints sync,
the bot plays. The recommendations are about hardening, not rescue.