"""
/api/lichess/* — sync, stats, game list, move-level performance.
"""

import json
import time
from flask import Blueprint, jsonify, request, Response, stream_with_context

from ..lichess_sync import (
    start_sync,
    get_sync_status,
    lichess_username_for_bot_version,
    get_public_bullet_blitz_ratings,
)
from ..analysis import trigger_analysis, is_analysis_running
from ..db import (
    get_lichess_stats,
    query_lichess_games,
    get_all_sync_states,
    get_lichess_performance_snapshot,
)

bp = Blueprint("lichess", __name__, url_prefix="/api/lichess")


@bp.route("/sync", methods=["POST"])
def sync():
    body = request.get_json(silent=True) or {}
    bot_versions = body.get("bot_versions") or None
    started = start_sync(bot_versions)
    if not started:
        return jsonify({"error": "Sync already running"}), 409
    return jsonify({"status": "started"}), 202


@bp.route("/sync/status")
def sync_status():
    return jsonify(get_sync_status())


@bp.route("/sync/stream")
def sync_stream():
    def generate():
        last_sent = -1
        while True:
            state = get_sync_status()
            line_count = len(state["lines"])
            if line_count != last_sent or state["status"] in ("done", "error", "idle"):
                yield f"data: {json.dumps(state)}\n\n"
                last_sent = line_count
            if state["status"] in ("done", "error", "idle"):
                break
            time.sleep(0.5)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@bp.route("/analyze", methods=["POST"])
def analyze():
    if is_analysis_running():
        return jsonify({"error": "Analysis already running"}), 409
    trigger_analysis()
    return jsonify({"status": "started"}), 202


@bp.route("/analyze/status", methods=["GET"])
def analyze_status():
    return jsonify({"running": is_analysis_running()})


@bp.route("/stats")
def stats():
    by_version = get_lichess_stats()
    total = sum(v["total_games"] for v in by_version.values())

    formatted = {}
    for version, s in by_version.items():
        total_g = s["total_games"] or 0
        wins = s["wins"] or 0
        losses = s["losses"] or 0
        draws = s["draws"] or 0
        analyzed = s["analyzed_games"] or 0
        total_blunders = s["total_blunders"] or 0
        analyzed_with_moves = s.get("analyzed_with_moves") or 0

        lichess_user = lichess_username_for_bot_version(version)
        ratings = get_public_bullet_blitz_ratings(lichess_user)
        formatted[version] = {
            "total_games": total_g,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": round(wins / total_g, 3) if total_g else None,
            "avg_cpl": round(s["avg_cpl"], 1) if s["avg_cpl"] is not None else None,
            "blunder_rate": round(total_blunders / analyzed_with_moves, 3) if analyzed_with_moves else None,
            "analyzed_games": analyzed,
            "lichess_username": lichess_user,
            "lichess_bullet_rating": ratings.get("bullet"),
            "lichess_blitz_rating": ratings.get("blitz"),
        }

    return jsonify({"by_version": formatted, "total_games": total})


@bp.route("/games")
def games():
    bot_version = request.args.get("bot_version") or None
    speed = request.args.get("speed") or None
    result = request.args.get("result") or None
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))

    rows, total = query_lichess_games(
        bot_version=bot_version,
        speed=speed,
        result=result,
        limit=limit,
        offset=offset,
    )

    return jsonify({
        "games": [dict(r) for r in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@bp.route("/performance")
def performance():
    """Bot decision quality vs move order + error rates by opening / mid / end (see db helper)."""
    bot_version = request.args.get("bot_version") or None
    max_idx = min(max(int(request.args.get("max_bot_move_index", 80)), 10), 120)
    min_bin = min(max(int(request.args.get("min_per_bin", 3)), 1), 50)
    o_end = min(max(int(request.args.get("opening_end", 12)), 4), 40)
    m_end = min(max(int(request.args.get("middlegame_end", 32)), o_end + 1), 80)
    snap = get_lichess_performance_snapshot(
        bot_version,
        max_bot_move_index=max_idx,
        min_per_bin=min_bin,
        opening_bot_moves_end=o_end,
        middlegame_bot_moves_end=m_end,
    )
    return jsonify(snap)
