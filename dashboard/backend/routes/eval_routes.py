"""
/api/eval/* — Bradley-Terry leaderboard, progress curves, H2H matrix, terminations.
Reads from eval/results.db using existing eval/db.py + eval/elo.py utilities.
"""

import re
from flask import Blueprint, jsonify

from eval.db import query_games
from eval.elo import compute_ratings

bp = Blueprint("eval", __name__, url_prefix="/api/eval")

_PATZER_RE = re.compile(r"^(patzer_v\d+)@(\d+)$")


def _is_patzer(name: str) -> bool:
    return bool(_PATZER_RE.match(name))


def _parse_patzer(name: str) -> tuple[str, int]:
    """Returns (version, iter_k) e.g. ('patzer_v3', 180)."""
    m = _PATZER_RE.match(name)
    return m.group(1), int(m.group(2))


@bp.route("/leaderboard")
def leaderboard():
    games = query_games()
    ratings = compute_ratings(games)

    # Summary stats
    patzer_ratings = [r for r in ratings if _is_patzer(r.name)]
    best = patzer_ratings[0] if patzer_ratings else None

    return jsonify({
        "ratings": [
            {
                "name": r.name,
                "elo": round(r.elo, 1),
                "stderr": round(r.stderr, 1) if r.stderr == r.stderr else None,
                "games": r.games,
                "wins": r.wins,
                "losses": r.losses,
                "draws": r.draws,
                "is_stockfish": False,
            }
            for r in ratings
            if _is_patzer(r.name)
        ],
        "total_games": len(games),
        "models_evaluated": len(patzer_ratings),
        "best_model": best.name if best else None,
        "best_elo": round(best.elo, 1) if best else None,
    })


@bp.route("/progress")
def progress():
    games = query_games()
    ratings = compute_ratings(games)

    # Group patzer ratings by version family
    by_version: dict[str, list[dict]] = {}
    for r in ratings:
        if not _is_patzer(r.name):
            continue
        version, iter_k = _parse_patzer(r.name)
        by_version.setdefault(version, []).append({
            "iter": iter_k,
            "elo": round(r.elo, 1),
            "stderr": round(r.stderr, 1) if r.stderr == r.stderr else None,
            "games": r.games,
        })

    series = []
    for version in sorted(by_version):
        points = sorted(by_version[version], key=lambda p: p["iter"])
        series.append({"version": version, "points": points})

    return jsonify({"series": series})


@bp.route("/h2h")
def h2h():
    games = query_games()
    patzer_games = [
        g for g in games
        if _is_patzer(g["white"]) and _is_patzer(g["black"])
    ]

    players: set[str] = set()
    for g in patzer_games:
        players.add(g["white"])
        players.add(g["black"])
    players_list = sorted(players, key=lambda p: _parse_patzer(p))

    matrix: dict[str, dict[str, dict]] = {p: {} for p in players_list}
    for g in patzer_games:
        w, b, r = g["white"], g["black"], g["result"]
        rec_w = matrix[w].setdefault(b, {"wins": 0, "losses": 0, "draws": 0, "games": 0})
        rec_b = matrix[b].setdefault(w, {"wins": 0, "losses": 0, "draws": 0, "games": 0})
        rec_w["games"] += 1
        rec_b["games"] += 1
        if r == "1-0":
            rec_w["wins"] += 1
            rec_b["losses"] += 1
        elif r == "0-1":
            rec_w["losses"] += 1
            rec_b["wins"] += 1
        else:
            rec_w["draws"] += 1
            rec_b["draws"] += 1

    # Add win_rate
    for src in matrix:
        for dst in matrix[src]:
            rec = matrix[src][dst]
            total = rec["games"]
            rec["win_rate"] = round(rec["wins"] / total, 3) if total else None

    return jsonify({"players": players_list, "matrix": matrix})


@bp.route("/terminations")
def terminations():
    games = query_games()
    counts: dict[str, int] = {}
    for g in games:
        t = g.get("termination") or "unknown"
        counts[t] = counts.get(t, 0) + 1

    distribution = sorted(
        [{"type": k, "count": v} for k, v in counts.items()],
        key=lambda x: -x["count"],
    )
    return jsonify({"distribution": distribution, "total": len(games)})
