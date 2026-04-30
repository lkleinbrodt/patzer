"""
eval/model_tournament.py

Run a "model league" tournament where Patzer checkpoints play each other (no Stockfish).

Goals:
- Compare checkpoints within one model dir, or across versions (e.g. v1 vs v2).
- Prune weak players in stage 1, then round-robin the rest with SPRT early-stop per pair.
- Append match rows to eval/model_results.json; optional aggregated leaderboard across runs
  that share the same settings bucket (temperature, conditioning, openings, and prefix set).

How to run (from repo root, venv active):

  # List checkpoints in R2 for one version
  .venv/bin/python eval/model_tournament.py --prefix checkpoints/patzer_v2 --list

  # Interactive pick (catalog is one prefix unless you pass --prefix more than once)
  .venv/bin/python eval/model_tournament.py --prefix checkpoints/patzer_v2

  # Non-interactive: N checkpoints (ckpt_best if present + evenly spaced iters)
  .venv/bin/python eval/model_tournament.py --prefix checkpoints/patzer_v2 --select 6 --device mps

  # Cross-version: combine catalogs; indices are global over the concatenated list
  .venv/bin/python eval/model_tournament.py \\
      --prefix checkpoints/patzer_v1 \\
      --prefix checkpoints/patzer_v2 \\
      --list

  # v1 best vs v2 best only (one ckpt_best.pt per prefix)
  .venv/bin/python eval/model_tournament.py \\
      --prefix checkpoints/patzer_v1 \\
      --prefix checkpoints/patzer_v2 \\
      --cross-best \\
      --device mps --keep 2 --max-games-per-pair 40

  # Re-print aggregated Elo from past runs (must match the same --prefix set and play settings)
  .venv/bin/python eval/model_tournament.py --analyze \\
      --prefix checkpoints/patzer_v1 \\
      --prefix checkpoints/patzer_v2 \\
      --temperature 0.0 --conditioning match_color

Defaults if you omit --prefix: checkpoints/patzer_v2. Shorthand --prefix patzer_v1 works
  (same as checkpoints/patzer_v1). Requires R2 credentials in .env for pull/list.

Results: eval/model_results.json (separate from eval/results.json used for Stockfish tournaments).
Aggregation bucket field settings.prefix is a sorted join of prefixes (e.g. a|b) when multiple.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.engine import Patzer
from patzer.r2 import _client, pull_file

MODEL_RESULTS_FILE = Path(__file__).parent / "model_results.json"


# ── Openings ───────────────────────────────────────────────────────────────────

# Short UCI opening lines (plies). Keep them simple and broadly legal.
# These are only used to reduce variance and avoid identical startpos games.
OPENING_LINES_UCI: list[list[str]] = [
    ["e2e4", "e7e5", "g1f3", "b8c6"],  # Italian-ish
    ["d2d4", "d7d5", "c2c4", "e7e6"],  # QGD
    ["e2e4", "c7c5", "g1f3", "d7d6"],  # Sicilian
    ["d2d4", "g8f6", "c2c4", "g7g6"],  # King's Indian
    ["c2c4", "e7e5", "b1c3", "g8f6"],  # English
    ["e2e4", "e7e6", "d2d4", "d7d5"],  # French
    ["e2e4", "c7c6", "d2d4", "d7d5"],  # Caro-Kann
    ["d2d4", "d7d5", "g1f3", "g8f6"],  # London-ish
]


def _apply_opening(board: chess.Board, move_history: list[str], opening_uci: list[str]) -> None:
    for uci in opening_uci:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            # If our small book ever becomes incompatible (shouldn't), just stop.
            break
        board.push(mv)
        move_history.append(uci)


# ── R2 discovery + caching ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class R2Checkpoint:
    r2_key: str
    filename: str
    iter_hint: int | None
    etag: str | None
    last_modified: str | None
    size: int | None


def _normalize_prefix(prefix: str) -> str:
    return prefix.strip().rstrip("/")


def _resolve_checkpoint_prefix(prefix: str) -> str:
    """
    R2 keys in this repo live under checkpoints/<dir>/....
    Accept shorthand: --prefix patzer_v1 → checkpoints/patzer_v1
    Full paths (contain '/') are left unchanged after strip.
    """
    p = _normalize_prefix(prefix)
    if not p:
        return p
    if "/" not in p:
        return f"checkpoints/{p}"
    return p


def _canonical_run_prefixes(prefixes: list[str]) -> str:
    """Stable bucket id for settings/aggregation when multiple R2 roots are used."""
    uniq = sorted({_normalize_prefix(p) for p in prefixes if p.strip()})
    return "|".join(uniq)


def _normalize_prefix_list(raw: list[str] | None) -> list[str]:
    """CLI prefixes: dedupe preserving order; bare names get checkpoints/ prepended."""
    if not raw:
        return ["checkpoints/patzer_v2"]
    out: list[str] = []
    seen: set[str] = set()
    for p in raw:
        n = _resolve_checkpoint_prefix(p)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out if out else ["checkpoints/patzer_v2"]


def _list_r2_checkpoints(prefix: str) -> list[R2Checkpoint]:
    """
    List .pt checkpoint objects under R2 prefix.
    """
    client, bucket = _client()
    if client is None:
        raise SystemExit("R2 not configured — check .env credentials")

    paginator = client.get_paginator("list_objects_v2")
    out: list[R2Checkpoint] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".pt"):
                continue
            filename = Path(key).name
            iter_hint = None
            # ckpt_005000.pt → 5000
            if filename.startswith("ckpt_") and filename.endswith(".pt"):
                tail = filename.removesuffix(".pt").removeprefix("ckpt_")
                if tail.isdigit():
                    iter_hint = int(tail)
            etag = obj.get("ETag")
            # boto3 ETag often includes quotes
            if isinstance(etag, str):
                etag = etag.strip('"')
            last_modified = None
            if obj.get("LastModified") is not None:
                last_modified = str(obj["LastModified"])
            size = obj.get("Size")
            out.append(
                R2Checkpoint(
                    r2_key=key,
                    filename=filename,
                    iter_hint=iter_hint,
                    etag=etag,
                    last_modified=last_modified,
                    size=int(size) if size is not None else None,
                )
            )

    # Prefer ckpt_best.pt over ckpt.pt if both exist in the same dir.
    parents_with_best = {str(Path(e.r2_key).parent) for e in out if e.filename == "ckpt_best.pt"}
    out = [
        e
        for e in out
        if not (e.filename == "ckpt.pt" and str(Path(e.r2_key).parent) in parents_with_best)
    ]

    def sort_key(e: R2Checkpoint):
        if e.filename == "ckpt_best.pt":
            return (-1, -1)
        if e.iter_hint is not None:
            return (0, e.iter_hint)
        # ckpt.pt (latest) last
        return (1, 10**18)

    return sorted(out, key=sort_key)


def _meta_path_for(local_path: Path) -> Path:
    return local_path.with_suffix(local_path.suffix + ".r2meta.json")


def _load_local_meta(local_path: Path) -> dict | None:
    mp = _meta_path_for(local_path)
    if not mp.exists():
        return None
    try:
        return json.loads(mp.read_text())
    except Exception:
        return None


def _write_local_meta(local_path: Path, meta: dict) -> None:
    mp = _meta_path_for(local_path)
    mp.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _ensure_local_checkpoint(entry: R2Checkpoint, force_refresh_best: bool = True) -> Path:
    """
    Ensure a checkpoint exists locally at path == r2_key.

    - For ckpt_best.pt: re-download if ETag differs (default).
    - For numbered ckpt_*.pt: skip download if file exists and (size matches if available).
    """
    local_path = Path(entry.r2_key)
    need = not local_path.exists()

    if not need:
        # Sanity check size when available.
        if entry.size is not None:
            try:
                if local_path.stat().st_size != int(entry.size):
                    need = True
            except OSError:
                need = True

        # ckpt_best can change while name stays same.
        if not need and force_refresh_best and entry.filename == "ckpt_best.pt":
            meta = _load_local_meta(local_path) or {}
            if meta.get("etag") and entry.etag and meta.get("etag") != entry.etag:
                need = True
            elif meta.get("etag") is None and entry.etag is not None:
                # No meta but remote has ETag; play safe and refresh once.
                need = True

    if need:
        ok = pull_file(entry.r2_key, local_path)
        if not ok:
            raise SystemExit(f"Failed to pull {entry.r2_key} from R2")
        _write_local_meta(
            local_path,
            {
                "r2_key": entry.r2_key,
                "etag": entry.etag,
                "last_modified": entry.last_modified,
                "size": entry.size,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    return local_path


# ── Tournament logic ───────────────────────────────────────────────────────────


@dataclass
class ModelSpec:
    checkpoint_key: str  # canonical id (R2 key)
    local_path: Path
    iter_num: int
    label: str


class NamedPlayer:
    def __init__(self, label: str, engine: Patzer):
        self._label = label
        self._engine = engine

    @property
    def name(self) -> str:
        return self._label

    def get_move(self, board: chess.Board, move_history: list[str]) -> str:
        return self._engine.get_move(board, move_history)


def play_model_game(
    white: NamedPlayer,
    black: NamedPlayer,
    rng: random.Random,
    max_moves: int = 300,
) -> str:
    board = chess.Board()
    move_history: list[str] = []
    opening = rng.choice(OPENING_LINES_UCI)
    _apply_opening(board, move_history, opening)

    while not board.is_game_over(claim_draw=True) and len(move_history) < max_moves * 2:
        player = white if board.turn == chess.WHITE else black
        try:
            uci = player.get_move(board, move_history)
            move = chess.Move.from_uci(uci)
        except Exception as e:
            print(f"  [warn] {player.name} error: {e} — random move")
            move = rng.choice(list(board.legal_moves))
        if move not in board.legal_moves:
            print(f"  [warn] {player.name} played illegal {move.uci()} — random move")
            move = rng.choice(list(board.legal_moves))
        board.push(move)
        move_history.append(move.uci())

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return "1/2-1/2"
    if outcome.winner is chess.WHITE:
        return "1-0"
    if outcome.winner is chess.BLACK:
        return "0-1"
    return "1/2-1/2"


def _score_from_a_perspective(result: str, a_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if a_is_white:
        return 1.0 if result == "1-0" else 0.0
    return 1.0 if result == "0-1" else 0.0


def _sprt_thresholds(alpha: float, beta: float) -> tuple[float, float]:
    """
    Returns (A, B) boundaries for log-likelihood ratio:
      accept H1 if llr >= A
      accept H0 if llr <= B
    """
    A = math.log((1.0 - beta) / alpha)
    B = math.log(beta / (1.0 - alpha))
    return A, B


def _sprt_llr_update(llr: float, score: float, p0: float, p1: float) -> float:
    # likelihood ∝ p^score * (1-p)^(1-score), score in {0,0.5,1}
    eps = 1e-12
    p0 = min(max(p0, eps), 1.0 - eps)
    p1 = min(max(p1, eps), 1.0 - eps)
    llr += score * math.log(p1 / p0) + (1.0 - score) * math.log((1.0 - p1) / (1.0 - p0))
    return llr


def run_pair_match_sprt(
    a: ModelSpec,
    b: ModelSpec,
    a_player: NamedPlayer,
    b_player: NamedPlayer,
    rng: random.Random,
    max_games: int,
    p0: float = 0.5,
    p1: float = 0.6,
    alpha: float = 0.05,
    beta: float = 0.05,
) -> dict:
    """
    Run A vs B games with early stopping using an SPRT on A's score vs B.

    Returns a summary dict from A's perspective.
    """
    A, B = _sprt_thresholds(alpha, beta)
    llr = 0.0
    w = l = d = 0
    games = 0

    for g in range(max_games):
        a_is_white = (g % 2 == 0)
        white = a_player if a_is_white else b_player
        black = b_player if a_is_white else a_player

        result = play_model_game(white, black, rng=rng)
        s = _score_from_a_perspective(result, a_is_white=a_is_white)
        if s == 1.0:
            w += 1
        elif s == 0.0:
            l += 1
        else:
            d += 1
        games += 1
        llr = _sprt_llr_update(llr, s, p0=p0, p1=p1)

        # Stop early when decisive.
        if llr >= A or llr <= B:
            break

    decision = "inconclusive"
    if llr >= A:
        decision = "A_better"
    elif llr <= B:
        decision = "A_not_better"

    return {
        "model_a": a.checkpoint_key,
        "model_b": b.checkpoint_key,
        "iter_a": a.iter_num,
        "iter_b": b.iter_num,
        "label_a": a.label,
        "label_b": b.label,
        "games": games,
        "W": w,
        "L": l,
        "D": d,
        "score": (w + 0.5 * d) / games if games else 0.0,
        "sprt": {
            "p0": p0,
            "p1": p1,
            "alpha": alpha,
            "beta": beta,
            "llr": llr,
            "decision": decision,
            "A": A,
            "B": B,
        },
    }


def _elo_expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def _elo_update(r: float, expected: float, score: float, k: float) -> float:
    return r + k * (score - expected)


def load_model_results() -> list[dict]:
    if not MODEL_RESULTS_FILE.exists():
        return []
    return json.loads(MODEL_RESULTS_FILE.read_text())


def save_model_results(records: list[dict]) -> None:
    existing = load_model_results()
    existing.extend(records)
    MODEL_RESULTS_FILE.write_text(json.dumps(existing, indent=2))
    print(f"\n[saved] {len(records)} record(s) → {MODEL_RESULTS_FILE}")


def _format_label(key: str) -> str:
    p = Path(key)
    # Prefer stable "model version" labels over checkpoint filenames.
    # Iteration is displayed separately (we append "@{iter_num}" elsewhere).
    if p.suffix == ".pt" and p.parent.name and p.parent.name.startswith("patzer_v") and p.name.startswith("ckpt"):
        if p.name == "ckpt_best.pt":
            return f"{p.parent.name}_best"
        return p.parent.name
    return p.name or key


# ── Aggregation: combine results across runs sharing the same settings bucket ─

# Settings fields that define an "aggregation bucket". A run's records contribute
# to the same bucket as another run if all of these match. `seed` is intentionally
# excluded so different seeds can stack into more data.
AGG_SETTINGS_KEYS = ("prefix", "temperature", "top_k", "conditioning", "openings")


def _settings_bucket(rec: dict) -> tuple:
    s = rec.get("settings", {}) or {}
    bucket = []
    for k in AGG_SETTINGS_KEYS:
        if k == "prefix" and s.get("prefix") is None:
            # Backfill: infer prefix from model_a path's parent directory.
            ma = rec.get("model_a", "")
            inferred = str(Path(ma).parent) if ma else None
            bucket.append(inferred)
        else:
            bucket.append(s.get(k))
    return tuple(bucket)


def _bucket_label(bucket: tuple) -> str:
    return ", ".join(f"{k}={v}" for k, v in zip(AGG_SETTINGS_KEYS, bucket))


def _aggregate_pair_results(
    records: list[dict],
    bucket: tuple | None = None,
) -> dict[tuple, dict]:
    """
    Sum W/L/D per ordered (model_a, model_b, iter_a, iter_b) across all matching records.
    If `bucket` is provided, only records in that aggregation bucket are counted.
    """
    pairs: dict[tuple, dict] = {}
    for r in records:
        if bucket is not None and _settings_bucket(r) != bucket:
            continue
        if "model_a" not in r or "model_b" not in r:
            continue
        key = (
            r["model_a"],
            r["model_b"],
            r.get("iter_a"),
            r.get("iter_b"),
        )
        slot = pairs.setdefault(
            key,
            {
                "W": 0,
                "L": 0,
                "D": 0,
                "games": 0,
                "label_a": r.get("label_a", _format_label(r["model_a"])),
                "label_b": r.get("label_b", _format_label(r["model_b"])),
                "iter_a": r.get("iter_a"),
                "iter_b": r.get("iter_b"),
            },
        )
        slot["W"] += int(r.get("W", 0))
        slot["L"] += int(r.get("L", 0))
        slot["D"] += int(r.get("D", 0))
        slot["games"] += int(r.get("games", 0))
    return pairs


def _symmetrize_pair_totals(pairs: dict[tuple, dict]) -> dict[tuple, dict]:
    """
    Merge ordered pairs (A,B) and (B,A) into a single canonical record per
    unordered pair, keyed by (model_lo, model_hi, iter_lo, iter_hi). The merged
    record stores W/L/D from the canonical (lo) side's perspective.
    """
    merged: dict[tuple, dict] = {}
    for (a, b, ia, ib), c in pairs.items():
        if (a, ia) <= (b, ib):
            lo, hi = (a, ia), (b, ib)
            w = int(c["W"])
            l = int(c["L"])
        else:
            lo, hi = (b, ib), (a, ia)
            w = int(c["L"])
            l = int(c["W"])
        d = int(c["D"])
        key = (lo[0], hi[0], lo[1], hi[1])
        slot = merged.setdefault(
            key,
            {
                "W": 0,
                "L": 0,
                "D": 0,
                "games": 0,
                "label_lo": c.get("label_a") if (a, ia) == lo else c.get("label_b"),
                "label_hi": c.get("label_b") if (a, ia) == lo else c.get("label_a"),
            },
        )
        slot["W"] += w
        slot["L"] += l
        slot["D"] += d
        slot["games"] += int(c["games"])
    return merged


def _fit_elo_iterative(
    pair_totals: dict[tuple, dict],
    iters: int = 200,
    k: float = 16.0,
    init: float = 1500.0,
) -> dict[tuple[str, int | None], float]:
    """
    Order-independent rating fit by iteratively replaying every aggregated game
    against the current ratings. Each pair's W/L/D acts as `games` independent
    score samples between the same two players.

    Returns a dict keyed by (checkpoint_key, iter_num) → rating.
    """
    players: set[tuple[str, int | None]] = set()
    for (a, b, ia, ib) in pair_totals.keys():
        players.add((a, ia))
        players.add((b, ib))

    ratings: dict[tuple[str, int | None], float] = {p: float(init) for p in players}
    if not players:
        return ratings

    samples: list[tuple[tuple[str, int | None], tuple[str, int | None], list[float]]] = []
    for (a, b, ia, ib), c in pair_totals.items():
        seq = [1.0] * int(c["W"]) + [0.0] * int(c["L"]) + [0.5] * int(c["D"])
        samples.append(((a, ia), (b, ib), seq))

    for _ in range(iters):
        for (pa, pb, seq) in samples:
            for s in seq:
                ea = _elo_expected(ratings[pa], ratings[pb])
                ratings[pa] = _elo_update(ratings[pa], ea, s, k)
                ratings[pb] = _elo_update(ratings[pb], 1.0 - ea, 1.0 - s, k)

    if ratings:
        mean = sum(ratings.values()) / len(ratings)
        shift = float(init) - mean
        ratings = {k: v + shift for k, v in ratings.items()}
    return ratings


def _print_aggregated_leaderboard(
    records: list[dict],
    bucket: tuple,
    label_overrides: dict[tuple[str, int | None], str] | None = None,
) -> None:
    pairs = _aggregate_pair_results(records, bucket=bucket)
    if not pairs:
        print(f"[aggregate] no records found for bucket: {_bucket_label(bucket)}")
        return
    sym = _symmetrize_pair_totals(pairs)
    ratings = _fit_elo_iterative(sym)

    per_player: dict[tuple[str, int | None], dict] = {}
    for (a, b, ia, ib), c in pairs.items():
        slot_a = per_player.setdefault((a, ia), {"games": 0, "score": 0.0, "label": c.get("label_a")})
        slot_b = per_player.setdefault((b, ib), {"games": 0, "score": 0.0, "label": c.get("label_b")})
        g = int(c["games"])
        if g <= 0:
            continue
        slot_a["games"] += g
        slot_a["score"] += float(c["W"]) + 0.5 * float(c["D"])
        slot_b["games"] += g
        slot_b["score"] += float(c["L"]) + 0.5 * float(c["D"])

    rows = []
    for player, rating in ratings.items():
        info = per_player.get(player, {"games": 0, "score": 0.0, "label": None})
        label = (label_overrides or {}).get(player) or info.get("label") or _format_label(player[0])
        if player[1] is not None and "@" not in label:
            label = f"{label}@{player[1]}"
        games = int(info["games"])
        pct = (info["score"] / games * 100.0) if games else 0.0
        rows.append((rating, label, player, games, pct))

    rows.sort(key=lambda r: r[0], reverse=True)
    print(f"\n[aggregate-leaderboard] bucket: {_bucket_label(bucket)}")
    print(f"  {'Elo':>7}  {'Model':<36}  {'Games':>6}  {'Score%':>6}")
    for rating, label, player, games, pct in rows:
        print(f"  {rating:7.1f}  {label:<36}  {games:>6d}  {pct:>5.1f}%")


def _choose_evenly_spaced(entries: list[R2Checkpoint], n: int) -> list[R2Checkpoint]:
    """
    Pick n entries with preference:
      - always include ckpt_best.pt if present
      - among numbered ckpt_*.pt, choose evenly across iteration range
      - if still short, include ckpt.pt
    """
    if n <= 0:
        return []
    best = [e for e in entries if e.filename == "ckpt_best.pt"]
    numbered = [e for e in entries if e.iter_hint is not None]
    latest = [e for e in entries if e.filename == "ckpt.pt"]

    picked: list[R2Checkpoint] = []
    if best:
        picked.append(best[0])
    remaining = n - len(picked)
    if remaining <= 0:
        return picked

    if numbered:
        numbered_sorted = sorted(numbered, key=lambda e: int(e.iter_hint))
        if remaining >= len(numbered_sorted):
            picked.extend(numbered_sorted)
            remaining = n - len(picked)
        else:
            # Quantile-based indices
            for i in range(remaining):
                q = (i + 0.5) / remaining
                idx = int(q * len(numbered_sorted))
                idx = min(max(0, idx), len(numbered_sorted) - 1)
                picked.append(numbered_sorted[idx])
            # de-dup while preserving order
            seen = set()
            dedup = []
            for e in picked:
                if e.r2_key in seen:
                    continue
                seen.add(e.r2_key)
                dedup.append(e)
            picked = dedup
            remaining = n - len(picked)

    if remaining > 0 and latest:
        for e in latest:
            if e.r2_key not in {p.r2_key for p in picked}:
                picked.append(e)
                remaining -= 1
                if remaining <= 0:
                    break

    return picked[:n]


def _entries_ordered_by_prefix(prefixes: list[str]) -> list[tuple[str, list[R2Checkpoint]]]:
    return [(p, _list_r2_checkpoints(p)) for p in prefixes]


def _flatten_catalog(ordered: list[tuple[str, list[R2Checkpoint]]]) -> tuple[list[R2Checkpoint], list[str]]:
    flat: list[R2Checkpoint] = []
    tags: list[str] = []
    for pfx, es in ordered:
        for e in es:
            flat.append(e)
            tags.append(pfx)
    return flat, tags


def _choose_evenly_spaced_multi(
    ordered: list[tuple[str, list[R2Checkpoint]]],
    n: int,
) -> list[R2Checkpoint]:
    """
    Pick n checkpoints across multiple prefixes: split budget across prefixes (fair),
    then run per-prefix evenly-spaced selection.
    If n < number of prefixes, take 1 checkpoint each from the first n prefixes (CLI order).
    """
    nonempty = [(p, es) for p, es in ordered if es]
    if not nonempty or n <= 0:
        return []
    p_count = len(nonempty)
    picked: list[R2Checkpoint] = []
    if n < p_count:
        for i in range(n):
            _, es = nonempty[i]
            picked.extend(_choose_evenly_spaced(es, 1))
        return picked[:n]

    base = n // p_count
    rem = n % p_count
    for i, (_, es) in enumerate(nonempty):
        ni = base + (1 if i < rem else 0)
        if ni <= 0:
            continue
        picked.extend(_choose_evenly_spaced(es, ni))

    seen: set[str] = set()
    dedup: list[R2Checkpoint] = []
    for e in picked:
        if e.r2_key in seen:
            continue
        seen.add(e.r2_key)
        dedup.append(e)
    return dedup[:n]


def _pick_ckpt_best_per_prefix(ordered: list[tuple[str, list[R2Checkpoint]]]) -> list[R2Checkpoint]:
    out: list[R2Checkpoint] = []
    for pfx, es in ordered:
        best = next((e for e in es if e.filename == "ckpt_best.pt"), None)
        if best is None:
            raise SystemExit(
                f"No ckpt_best.pt found under R2 prefix {pfx!r}. "
                "Upload best weights or pick checkpoints with --indices / manual selection."
            )
        out.append(best)
    return out


def _interactive_select_catalog(prefixes: list[str]) -> list[R2Checkpoint]:
    ordered = _entries_ordered_by_prefix(prefixes)
    flat, tags = _flatten_catalog(ordered)
    multi = len(prefixes) > 1

    print(f"Found {len(flat)} checkpoint file(s) across {len(prefixes)} R2 prefix(es):\n")
    prev_tag: str | None = None
    for i, e in enumerate(flat):
        tag = tags[i]
        if multi and tag != prev_tag:
            print(f"\n--- {tag} ---")
            prev_tag = tag
        hint = "" if e.iter_hint is None else f"iter~{e.iter_hint}"
        ftag = "best" if e.filename == "ckpt_best.pt" else ("latest" if e.filename == "ckpt.pt" else "ckpt")
        lm = e.last_modified or "?"
        sz = f"{(e.size/1e6):.1f}MB" if e.size else "?"
        print(f"  [{i:2d}] {ftag:<6} {hint:<10} {sz:>8}  {lm:<24}  {e.r2_key}")

    print(
        "\nSelection options:\n"
        "  - Enter a number N: per-prefix evenly spaced picks split across prefixes (fair)\n"
        "    (single-prefix: ckpt_best if present + spaced iters, same as before)\n"
        "  - Enter comma-separated global indices (e.g. 0,3,7) from the list above\n"
        "  - Enter empty to accept default (N=6)\n"
    )
    raw = input("Select checkpoints: ").strip()
    if raw == "":
        raw = "6"

    if raw.isdigit():
        n = int(raw)
        picked = _choose_evenly_spaced_multi(ordered, n) if multi else _choose_evenly_spaced(ordered[0][1], n)
        print(f"\nSelected {len(picked)} checkpoint(s) (evenly spaced):")
        for e in picked:
            print(f"  - {e.r2_key}")
        return picked

    idxs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise SystemExit(f"Invalid selection token: {part!r}")
        idxs.append(int(part))
    picked = []
    for i in idxs:
        if i < 0 or i >= len(flat):
            raise SystemExit(f"Index out of range: {i}")
        picked.append(flat[i])
    print(f"\nSelected {len(picked)} checkpoint(s) (manual):")
    for e in picked:
        print(f"  - {e.r2_key}")
    return picked


def _load_iter_num(local_path: Path, device: str) -> int:
    # Avoid importing torch at module load if user only uses --list.
    import torch
    ckpt = torch.load(local_path, map_location=device, weights_only=False)
    return int(ckpt.get("iter_num", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model-vs-model Patzer league tournament")
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        default=None,
        metavar="PREFIX",
        help=(
            "R2 prefix under the bucket (e.g. checkpoints/patzer_v2 or shorthand patzer_v2). "
            "Bare names get checkpoints/ prepended. Repeat for cross-version leagues."
        ),
    )
    parser.add_argument(
        "--cross-best",
        action="store_true",
        help="Select ckpt_best.pt from every --prefix (e.g. v1 best vs v2 best)",
    )
    parser.add_argument("--device", default="mps", help="torch device: cpu | mps | cuda")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--conditioning",
        default="match_color",
        help="Result conditioning token strategy",
    )
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed for openings/randomness")
    parser.add_argument("--list", action="store_true", help="List R2 checkpoints and exit")
    parser.add_argument(
        "--select",
        type=int,
        default=None,
        help="Non-interactive selection: pick N checkpoints (ckpt_best if present + evenly spaced iters)",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default=None,
        help="Non-interactive selection: comma-separated indices into the R2 listing (e.g. 0,3,7)",
    )

    # Pruning / match controls
    parser.add_argument("--stage1-games", type=int, default=12, help="Max games vs baseline in stage 1")
    parser.add_argument("--keep", type=int, default=6, help="How many models to keep after stage 1")
    parser.add_argument("--max-games-per-pair", type=int, default=30, help="Max games per pair in league")
    parser.add_argument("--sprt-p1", type=float, default=0.6, help="SPRT alternative winrate for A")
    parser.add_argument("--sprt-alpha", type=float, default=0.1)
    parser.add_argument("--sprt-beta", type=float, default=0.1)
    parser.add_argument("--elo-k", type=float, default=16.0, help="Online Elo K-factor for leaderboard")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Don't play any games; just print an aggregated leaderboard from existing model_results.json",
    )
    parser.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip the aggregated (cross-run) leaderboard at end of run",
    )
    args = parser.parse_args()

    prefixes = _normalize_prefix_list(args.prefixes)
    ordered = _entries_ordered_by_prefix(prefixes)
    flat_catalog, _flat_tags = _flatten_catalog(ordered)

    if args.analyze:
        records = load_model_results()
        if not records:
            print("[analyze] model_results.json is empty")
            return
        bucket = (
            _canonical_run_prefixes(prefixes),
            float(args.temperature),
            args.top_k,
            args.conditioning,
            "builtin_short_uci",
        )
        _print_aggregated_leaderboard(records, bucket=bucket)
        return

    if args.list:
        for pfx in prefixes:
            print(f"# {pfx}")
            for e in _list_r2_checkpoints(pfx):
                print(e.r2_key)
        return

    picked: list[R2Checkpoint]
    if args.cross_best:
        picked = _pick_ckpt_best_per_prefix(ordered)
        print(f"[select] --cross-best: picked {len(picked)} ckpt_best.pt (one per prefix)")
        for e in picked:
            print(f"  - {e.r2_key}")
    elif args.indices:
        idxs = []
        for part in str(args.indices).split(","):
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise SystemExit(f"--indices contains invalid token: {part!r}")
            idxs.append(int(part))
        picked = []
        for i in idxs:
            if i < 0 or i >= len(flat_catalog):
                raise SystemExit(f"--indices out of range: {i} (have {len(flat_catalog)} entries)")
            picked.append(flat_catalog[i])
        print(f"[select] picked {len(picked)} checkpoint(s) by --indices")
    elif args.select is not None:
        multi = len(prefixes) > 1
        picked = (
            _choose_evenly_spaced_multi(ordered, int(args.select))
            if multi
            else _choose_evenly_spaced(ordered[0][1], int(args.select))
        )
        print(f"[select] picked {len(picked)} checkpoint(s) by --select={args.select}")
    else:
        if sys.stdin is not None and sys.stdin.isatty():
            picked = _interactive_select_catalog(prefixes)
        else:
            multi = len(prefixes) > 1
            picked = (
                _choose_evenly_spaced_multi(ordered, 6)
                if multi
                else _choose_evenly_spaced(ordered[0][1], 6)
            )
            print("[select] stdin is not interactive; defaulting to --select=6 (split across prefixes if multiple)")
    if not picked:
        raise SystemExit("No checkpoints selected")

    # Pull/cache selected checkpoints.
    local_paths = []
    for e in picked:
        local_paths.append(_ensure_local_checkpoint(e, force_refresh_best=True))

    # Build ModelSpecs with iter numbers (read from checkpoint).
    specs: list[ModelSpec] = []
    for e, lp in zip(picked, local_paths):
        it = _load_iter_num(lp, device=args.device)
        specs.append(
            ModelSpec(
                checkpoint_key=e.r2_key,
                local_path=lp,
                iter_num=it,
                label=_format_label(e.r2_key) + f"@{it}",
            )
        )

    def _choose_baseline(specs_in: list[ModelSpec]) -> ModelSpec:
        bests = [s for s in specs_in if Path(s.checkpoint_key).name == "ckpt_best.pt"]
        if len(bests) == 1:
            return bests[0]
        if len(bests) > 1:
            # Multiple "best" snapshots (e.g. v1 vs v2): use strongest by recorded iter.
            return max(bests, key=lambda s: s.iter_num)
        return max(specs_in, key=lambda s: s.iter_num)

    baseline = _choose_baseline(specs)

    print(f"\n[baseline] {baseline.label} ({baseline.checkpoint_key})")

    # Load all engines once.
    engines: dict[str, NamedPlayer] = {}
    for s in specs:
        p = Patzer(
            s.local_path,
            device=args.device,
            temperature=args.temperature,
            top_k=args.top_k,
            conditioning=args.conditioning,
        )
        engines[s.checkpoint_key] = NamedPlayer(s.label, p)

    rng = random.Random(int(args.seed))
    now = datetime.now(timezone.utc).isoformat()
    settings = {
        "prefix": _canonical_run_prefixes(prefixes),
        "prefixes": list(prefixes),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "conditioning": args.conditioning,
        "seed": args.seed,
        "openings": "builtin_short_uci",
    }

    # ── Stage 1: prune obvious losers vs baseline ────────────────────────────
    print("\n[stage1] quick filter vs baseline (SPRT early-stop)")
    stage1_records: list[dict] = []
    stage1_scores: list[tuple[float, ModelSpec]] = []
    for s in specs:
        if s.checkpoint_key == baseline.checkpoint_key:
            continue
        rec = run_pair_match_sprt(
            a=s,
            b=baseline,
            a_player=engines[s.checkpoint_key],
            b_player=engines[baseline.checkpoint_key],
            rng=rng,
            max_games=int(args.stage1_games),
            p0=0.5,
            p1=float(args.sprt_p1),
            alpha=float(args.sprt_alpha),
            beta=float(args.sprt_beta),
        )
        rec.update(
            {
                "timestamp": now,
                "type": "stage1_vs_baseline",
                "settings": settings,
            }
        )
        stage1_records.append(rec)
        stage1_scores.append((float(rec["score"]), s))
        print(
            f"  {s.label:<28} vs {baseline.label:<28} "
            f"{rec['W']}-{rec['L']}-{rec['D']} score={rec['score']*100:5.1f}% "
            f"({rec['sprt']['decision']}, games={rec['games']})"
        )

    # Keep top-K plus baseline.
    keep_n = max(2, min(int(args.keep), len(specs)))
    stage1_scores.sort(key=lambda x: x[0], reverse=True)
    kept = [baseline]
    for _, s in stage1_scores[: max(0, keep_n - 1)]:
        kept.append(s)
    print(f"\n[stage1] keeping {len(kept)} model(s):")
    for s in kept:
        print(f"  - {s.label}")

    # ── Stage 2: league among kept models ────────────────────────────────────
    print("\n[stage2] league (per-pair SPRT early-stop)")
    ratings = {s.checkpoint_key: 1500.0 for s in kept}
    league_records: list[dict] = []
    for i in range(len(kept)):
        for j in range(i + 1, len(kept)):
            a = kept[i]
            b = kept[j]
            rec = run_pair_match_sprt(
                a=a,
                b=b,
                a_player=engines[a.checkpoint_key],
                b_player=engines[b.checkpoint_key],
                rng=rng,
                max_games=int(args.max_games_per_pair),
                p0=0.5,
                p1=float(args.sprt_p1),
                alpha=float(args.sprt_alpha),
                beta=float(args.sprt_beta),
            )
            rec.update(
                {
                    "timestamp": now,
                    "type": "league_pair",
                    "settings": settings,
                }
            )
            league_records.append(rec)

            # Update online Elo once per match (aggregate score).
            s = float(rec["score"])
            ea = _elo_expected(ratings[a.checkpoint_key], ratings[b.checkpoint_key])
            eb = 1.0 - ea
            k = float(args.elo_k)
            ratings[a.checkpoint_key] = _elo_update(ratings[a.checkpoint_key], ea, s, k)
            ratings[b.checkpoint_key] = _elo_update(ratings[b.checkpoint_key], eb, 1.0 - s, k)

            print(
                f"  {a.label:<28} vs {b.label:<28} "
                f"{rec['W']}-{rec['L']}-{rec['D']} score={rec['score']*100:5.1f}% "
                f"({rec['sprt']['decision']}, games={rec['games']})"
            )

    save_model_results(stage1_records + league_records)

    print("\n[leaderboard] (this run only, online Elo over stage2 matches)")
    rows = sorted(
        [(ratings[s.checkpoint_key], s) for s in kept],
        key=lambda x: x[0],
        reverse=True,
    )
    for r, s in rows:
        print(f"  {r:7.1f}  {s.label:<32}  {s.checkpoint_key}")

    if not args.no_aggregate:
        all_records = load_model_results()
        bucket = _settings_bucket({"settings": settings})
        # Override labels so aggregated rows show the human-friendly @iter labels
        # we used in this run (helpful when a checkpoint key like ckpt_best.pt
        # has been seen at multiple iter_nums historically).
        label_overrides = {
            (s.checkpoint_key, s.iter_num): s.label for s in specs
        }
        _print_aggregated_leaderboard(
            all_records, bucket=bucket, label_overrides=label_overrides
        )

    print(f"\n[note] results stored in {MODEL_RESULTS_FILE} (separate from eval/results.json)")


if __name__ == "__main__":
    main()

