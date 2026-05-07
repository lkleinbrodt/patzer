"""
scripts/build_blog_assets.py

Reads eval/results.db and dashboard/lichess_games.db, then writes static
JSON snapshots to scripts/blog_assets/ for use by the /patzer blog page.

Outputs
-------
blog_assets/models.json          per-version metadata + offline Elo
blog_assets/leaderboard.json     stockfish-ladder Elo for each checkpoint
blog_assets/loss_curves.json     val loss vs iter from checkpoints/*/metrics.jsonl
blog_assets/cpl_by_move.json     avg centipawn loss by move bucket (lichess games)
blog_assets/phase_stats.json     win/draw/loss rates by version (lichess games)

Usage
-----
  python scripts/build_blog_assets.py
  # outputs → scripts/blog_assets/*.json
"""

import json
import math
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
EVAL_DB = REPO_ROOT / "eval" / "results.db"
LICHESS_DB = REPO_ROOT / "dashboard" / "lichess_games.db"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
OUT_DIR = Path(__file__).parent / "blog_assets"


# ─── helpers ───────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(name: str, obj: object) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(obj, indent=2))
    print(f"  wrote {path.relative_to(REPO_ROOT)}  ({path.stat().st_size:,} bytes)")


# ─── Bradley-Terry Elo (same logic as eval/elo.py) ──────────────────────────

_SF_RE = re.compile(r"^stockfish:(\d+)$")


def _is_sf(name: str) -> bool:
    return bool(_SF_RE.match(name))


def _sf_elo(name: str) -> float:
    return float(_SF_RE.match(name).group(1))


def _win_prob(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def compute_bt_ratings(games: list[dict]) -> dict[str, dict]:
    """Return {player_name: {elo, stderr, games, wins, losses, draws}}."""
    if not games:
        return {}

    all_players: set[str] = set()
    for g in games:
        all_players.add(g["white"])
        all_players.add(g["black"])

    stats = {p: {"games": 0, "wins": 0, "losses": 0, "draws": 0} for p in all_players}
    for g in games:
        w, b, r = g["white"], g["black"], g["result"]
        stats[w]["games"] += 1
        stats[b]["games"] += 1
        if r == "1-0":
            stats[w]["wins"] += 1
            stats[b]["losses"] += 1
        elif r == "0-1":
            stats[b]["wins"] += 1
            stats[w]["losses"] += 1
        else:
            stats[w]["draws"] += 1
            stats[b]["draws"] += 1

    anchored = {p: _sf_elo(p) for p in all_players if _is_sf(p)}
    free = [p for p in all_players if not _is_sf(p)]
    ratings: dict[str, float] = {**anchored, **{p: 1500.0 for p in free}}

    for _ in range(1000):
        max_delta = 0.0
        for player in free:
            num = den = 0.0
            for g in games:
                w, b, r = g["white"], g["black"], g["result"]
                if player not in (w, b):
                    continue
                is_white = player == w
                opp = b if is_white else w
                p_win = _win_prob(ratings[player], ratings[opp]) if is_white else 1 - _win_prob(ratings[opp], ratings[player])
                obs = 0.5 if r == "1/2-1/2" else (1.0 if (r == "1-0") == is_white else 0.0)
                num += obs - p_win
                den += p_win * (1.0 - p_win)
            if den < 1e-9:
                continue
            delta = max(-100.0, min(100.0, (400.0 / math.log(10)) * (num / den)))
            ratings[player] += delta
            max_delta = max(max_delta, abs(delta))
        if max_delta < 0.01:
            break

    def stderr(player: str) -> float:
        info = 0.0
        for g in games:
            w, b = g["white"], g["black"]
            if player not in (w, b):
                continue
            is_white = player == w
            opp = b if is_white else w
            p_win = _win_prob(ratings[player], ratings[opp]) if is_white else 1 - _win_prob(ratings[opp], ratings[player])
            info += p_win * (1.0 - p_win)
        if info < 1e-9:
            return None
        return round((400.0 / math.log(10)) / math.sqrt(info), 1)

    return {
        p: {
            "elo": round(ratings[p], 1),
            "stderr": stderr(p) if not _is_sf(p) else 0.0,
            **stats[p],
        }
        for p in free
    }


# ─── eval/results.db ────────────────────────────────────────────────────────

def load_eval_games() -> list[dict]:
    if not EVAL_DB.exists():
        print(f"  [warn] {EVAL_DB} not found — skipping eval data")
        return []
    with sqlite3.connect(EVAL_DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM games ORDER BY id").fetchall()
    return [dict(r) for r in rows]


# ─── build leaderboard.json ─────────────────────────────────────────────────

def build_leaderboard(games: list[dict]) -> None:
    if not games:
        print("  [warn] no eval games found")
        _write("leaderboard.json", {"generated_at": _now(), "entries": []})
        return

    # Use ALL games for BT fit — same as `evaluate.py leaderboard`.
    # Stockfish players are anchored at their configured Elo; patzer models are free.
    ratings = compute_bt_ratings(games)

    # One entry per checkpoint
    entries = []
    for name, r in sorted(ratings.items(), key=lambda x: -x[1]["elo"]):
        # Parse version from "patzer_vN@iter"
        m = re.match(r"patzer_(v\d+)@(\d+)", name)
        version = m.group(1) if m else name
        entries.append({
            "checkpoint": name,
            "version": version,
            "elo": r["elo"],
            "stderr": r["stderr"],
            "games": r["games"],
            "wins": r["wins"],
            "losses": r["losses"],
            "draws": r["draws"],
        })

    _write("leaderboard.json", {"generated_at": _now(), "entries": entries})


# ─── build models.json ──────────────────────────────────────────────────────
# Architecture (layers / heads / dim) must match patzer/config/train_patzer_v*.py.
# Rounded training_games_m matches comment blocks in those configs (~M train games).

VERSION_META = {
    "v1": {
        "headline": "baseline",
        "sub": "12M · 1M games",
        "parameters_m": 12,
        "training_games_m": 1,
        "training_tokens_short": "~80M",
        "layers": 6, "heads": 6, "dim": 384,
        "what_changed": "baseline — 12M params, 1M games",
        "lichess_username": "patzer_v1",
    },
    "v2": {
        "headline": "10× data",
        "sub": "12M · 11M games",
        "parameters_m": 12,
        "training_games_m": 11,
        "training_tokens_short": "~868M",
        "layers": 6, "heads": 6, "dim": 384,
        "what_changed": "10× more training data, same architecture",
        "lichess_username": "patzer_v2b",
    },
    "v3": {
        "headline": "3× model",
        "sub": "40M · 11M games",
        "parameters_m": 40,
        "training_games_m": 11,
        "training_tokens_short": "~868M",
        "layers": 12, "heads": 8, "dim": 512,
        "what_changed": "3× larger model, same data",
        "lichess_username": "patzer_v3",
    },
    "v4": {
        "headline": "WSD cooldown",
        "sub": "40M · 36M games",
        "parameters_m": 40,
        "training_games_m": 36,
        "training_tokens_short": "~2.85B",
        "layers": 12, "heads": 8, "dim": 512,
        "what_changed": "3.3× more data + WSD lr schedule",
        "lichess_username": "patzer_v4",
    },
    "v5": {
        "headline": "116M scale",
        "sub": "116M · 36M games",
        "parameters_m": 116,
        "training_games_m": 36,
        "training_tokens_short": "~2.85B",
        "layers": 16, "heads": 16, "dim": 768,
        "what_changed": "~116M params, same 36M-game corpus as v4 · WSD auto-cooldown",
        "lichess_username": "patzer_v5",
    },
    "v6": {
        "headline": "2100+ data",
        "sub": "116M · 21M games",
        "parameters_m": 116,
        "training_games_m": 21,
        "training_tokens_short": "~1.73B",
        "layers": 16, "heads": 16, "dim": 768,
        "what_changed": "2100+ ELO filter · same architecture as v5",
        "lichess_username": "patzer_v6",
    },
}


def build_models(games: list[dict]) -> None:
    ratings = compute_bt_ratings(games) if games else {}

    # Best Elo per version = max across all checkpoints of that version
    best_by_version: dict[str, dict] = {}
    for name, r in ratings.items():
        m = re.match(r"patzer_(v\d+)@(\d+)", name)
        if not m:
            continue
        ver = m.group(1)
        if ver not in best_by_version or r["elo"] > best_by_version[ver]["elo"]:
            best_by_version[ver] = {"elo": r["elo"], "stderr": r["stderr"], "checkpoint": name}

    models = []
    prev_elo = None
    for ver in ["v1", "v2", "v3", "v4", "v5", "v6"]:
        meta = VERSION_META.get(ver, {})
        best = best_by_version.get(ver, {})
        elo = best.get("elo")
        delta = round(elo - prev_elo, 1) if (elo is not None and prev_elo is not None) else None
        prev_elo = elo if elo is not None else prev_elo
        models.append({
            "version": ver,
            **meta,
            "best_checkpoint": best.get("checkpoint"),
            "elo_stockfish": elo,
            "elo_stderr": best.get("stderr"),
            "elo_delta": delta,
        })

    _write("models.json", {"generated_at": _now(), "models": models})


# ─── build loss_curves.json ─────────────────────────────────────────────────

def build_loss_curves() -> None:
    curves = {}
    for ver_dir in sorted(CHECKPOINTS_DIR.glob("patzer_v*")):
        metrics_file = ver_dir / "metrics.jsonl"
        if not metrics_file.exists():
            continue
        ver = ver_dir.name.replace("patzer_", "")
        points = []
        for line in metrics_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                points.append({
                    "iter": rec["iter"],
                    "train_loss": round(rec["train_loss"], 4),
                    "val_loss": round(rec["val_loss"], 4),
                })
            except (json.JSONDecodeError, KeyError):
                continue
        if points:
            curves[ver] = points

    _write("loss_curves.json", {"generated_at": _now(), "versions": curves})


# ─── lichess_games.db ───────────────────────────────────────────────────────

def load_lichess_data() -> tuple[list[dict], list[dict], list[dict]]:
    if not LICHESS_DB.exists():
        print(f"  [warn] {LICHESS_DB} not found — skipping lichess data")
        return [], [], []
    with sqlite3.connect(LICHESS_DB) as con:
        con.row_factory = sqlite3.Row
        games = [dict(r) for r in con.execute("SELECT * FROM lichess_games").fetchall()]
        moves = [dict(r) for r in con.execute("SELECT * FROM lichess_moves").fetchall()]
        analysis = [dict(r) for r in con.execute("SELECT * FROM lichess_move_analysis").fetchall()]
    return games, moves, analysis


# ─── build cpl_by_move.json ─────────────────────────────────────────────────

def build_cpl_by_move(moves: list[dict], analysis: list[dict]) -> None:
    # Index analysis by move_id
    analysis_by_id = {r["move_id"]: r for r in analysis}

    # Bucket definition: (label, ply_min inclusive, ply_max inclusive)
    BUCKETS = [
        ("1–10",  1,  20),
        ("11–20", 21, 40),
        ("21–30", 41, 60),
        ("31–40", 61, 80),
        ("41–50", 81, 100),
        ("51+",   101, 9999),
    ]

    bucket_totals: dict[str, list[float]] = {b[0]: [] for b in BUCKETS}

    for mv in moves:
        if not mv["is_bot_move"]:
            continue
        ana = analysis_by_id.get(mv["id"])
        if ana is None:
            continue
        cpl = ana.get("eval_loss_cp")
        if cpl is None or cpl <= 0:
            continue
        # Cap at 500 to reduce outlier influence
        cpl = min(cpl, 500)
        ply = mv["ply"]
        for label, pmin, pmax in BUCKETS:
            if pmin <= ply <= pmax:
                bucket_totals[label].append(cpl)
                break

    buckets = []
    for label, pmin, pmax in BUCKETS:
        vals = bucket_totals[label]
        buckets.append({
            "label": label,
            "ply_min": pmin,
            "ply_max": pmax,
            "avg_cpl": round(sum(vals) / len(vals), 1) if vals else None,
            "n": len(vals),
        })

    _write("cpl_by_move.json", {"generated_at": _now(), "buckets": buckets})


# ─── build phase_stats.json ─────────────────────────────────────────────────

def build_phase_stats(lichess_games: list[dict]) -> None:
    by_version: dict[str, dict] = {}

    for g in lichess_games:
        ver = g.get("bot_version", "unknown")
        if ver not in by_version:
            by_version[ver] = {"games": 0, "wins": 0, "draws": 0, "losses": 0}
        s = by_version[ver]
        s["games"] += 1
        r = g.get("result", "")
        if r == "1-0":
            if g.get("bot_color") == "white":
                s["wins"] += 1
            else:
                s["losses"] += 1
        elif r == "0-1":
            if g.get("bot_color") == "black":
                s["wins"] += 1
            else:
                s["losses"] += 1
        else:
            s["draws"] += 1

    stats = []
    for ver in sorted(by_version):
        s = by_version[ver]
        n = s["games"]
        stats.append({
            "version": ver,
            "games": n,
            "wins": s["wins"],
            "draws": s["draws"],
            "losses": s["losses"],
            "win_rate": round(s["wins"] / n, 3) if n else None,
            "draw_rate": round(s["draws"] / n, 3) if n else None,
            "loss_rate": round(s["losses"] / n, 3) if n else None,
        })

    _write("phase_stats.json", {"generated_at": _now(), "by_version": stats})


# ─── main ────────────────────────────────────────────────────────────────────

HTML_PAGE = Path(__file__).parent / "patzer_page.html"
BEGIN_MARKER = "<!-- BEGIN_BLOG_DATA -->"
END_MARKER   = "<!-- END_BLOG_DATA -->"


def embed_data_in_html() -> None:
    """Read the five JSON assets and splice them into patzer_page.html."""
    if not HTML_PAGE.exists():
        print(f"  [skip] {HTML_PAGE.name} not found — skipping embed")
        return

    payload = {}
    for name in ("models", "leaderboard", "loss_curves", "cpl_by_move", "phase_stats"):
        p = OUT_DIR / f"{name}.json"
        if p.exists():
            payload[name] = json.loads(p.read_text())

    inline = (
        f"{BEGIN_MARKER}\n"
        "  <script>\n"
        f"  window.BLOG_DATA = {json.dumps(payload, separators=(',', ':'))};\n"
        "  </script>\n"
        f"  {END_MARKER}"
    )

    html = HTML_PAGE.read_text()
    start = html.find(BEGIN_MARKER)
    end   = html.find(END_MARKER) + len(END_MARKER)
    if start == -1 or end < len(END_MARKER):
        print(f"  [warn] markers not found in {HTML_PAGE.name} — skipping embed")
        return

    html = html[:start] + inline + html[end:]
    HTML_PAGE.write_text(html)
    print(f"  embedded data → {HTML_PAGE.relative_to(REPO_ROOT)}")


def main() -> None:
    print("build_blog_assets — reading databases…")

    eval_games = load_eval_games()
    print(f"  eval games loaded: {len(eval_games)}")

    lichess_games, lichess_moves, lichess_analysis = load_lichess_data()
    print(f"  lichess games: {len(lichess_games)}, moves: {len(lichess_moves)}, analysis rows: {len(lichess_analysis)}")

    print("\nwriting JSON assets…")
    build_leaderboard(eval_games)
    build_models(eval_games)
    build_loss_curves()
    build_cpl_by_move(lichess_moves, lichess_analysis)
    build_phase_stats(lichess_games)

    print("\nembedding data in HTML page…")
    embed_data_in_html()

    print(f"\ndone → {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
