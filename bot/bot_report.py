#!/usr/bin/env python3
"""Check Patzer bot stats on Lichess — one-off or incremental for cron."""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = "https://lichess.org/api"
BOTS = [f"patzer_v{i}" for i in range(1, 9)]


def fetch(url: str) -> dict | list:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"error": "non-json response", "raw": data[:500]}


def fetch_ndjson(url: str) -> list[dict]:
    """Fetch newline-delimited JSON (games endpoint)."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        raw = r.read().decode()
    results = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def user_stats(username: str) -> dict:
    data = fetch(f"{BASE}/user/{username}")
    if "error" in data:
        return {"error": data["error"], "username": username}
    perfs = data.get("perfs", {})
    counts = data.get("count", {})

    # Flatten per-format ELOs
    formats = {}
    wanted = ["bullet", "blitz", "rapid", "classical", "correspondence"]
    for f in wanted:
        p = perfs.get(f, {})
        prov = p.get("prov", False)
        formats[f] = {
            "rating": p.get("rating", "—"),
            "games": p.get("games", 0),
            "rd": p.get("rd", "—"),
            "prog": p.get("prog", 0),
            "prov": prov,
        }

    playing_url = data.get("playing")
    currently_playing = bool(playing_url)
    game_id = playing_url.split("/")[-1].split("/black")[0].split("/white")[0] if playing_url else None
    if game_id:
        game_id = game_id.split("?")[0]

    return {
        "username": username,
        "title": data.get("title", ""),
        "url": data["url"],
        "perfs": formats,
        "count": {
            "all": counts.get("all", 0),
            "rated": counts.get("rated", 0),
            "win": counts.get("win", 0),
            "loss": counts.get("loss", 0),
            "draw": counts.get("draw", 0),
        },
        "play_time_min": data.get("playTime", {}).get("total", 0) // 60,
        "currently_playing": currently_playing,
        "game_id": game_id,
    }


def recent_games(username: str, n: int = 50) -> list[dict]:
    url = f"{BASE}/games/user/{username}?max={n}&rated=true"
    games = fetch_ndjson(url)
    results = []
    for g in games:
        is_white = g["players"]["white"]["user"]["name"] == username
        my_color = "w" if is_white else "b"
        my_rating = g["players"]["white"]["rating"] if is_white else g["players"]["black"]["rating"]
        opp = g["players"]["black"]["user"]["name"] if is_white else g["players"]["white"]["user"]["name"]
        opp_rating = g["players"]["black"]["rating"] if is_white else g["players"]["white"]["rating"]
        winner = g.get("winner")
        if winner is None:
            result = "½"  # draw
        elif (winner == "white" and is_white) or (winner == "black" and not is_white):
            result = "W"
        else:
            result = "L"
        speed = g.get("speed", "?")
        ts = datetime.fromtimestamp(g["createdAt"] // 1000, tz=timezone.utc)
        results.append({
            "id": g["id"],
            "result": result,
            "speed": speed,
            "opponent": opp,
            "opp_rating": opp_rating,
            "my_rating": my_rating,
            "rating_diff": g["players"]["white"].get("ratingDiff") if is_white else g["players"]["black"].get("ratingDiff"),
            "date": ts.strftime("%Y-%m-%d %H:%M"),
        })
    return results


def fmt_rating(r: dict) -> str:
    if r["prov"]:
        return "   ?"
    return f"{r['rating']:>4d}"


def fmt_prog(prog: int) -> str:
    if prog > 0:
        return f"+{prog}"
    return str(prog)


def report_all(bots: list[str], show_recent: bool = True) -> str:
    lines = []
    lines.append(f"{'Bot':<12} {'Bullet':>10} {'Blitz':>10} {'Rapid':>10} {'All':>6} {'W':>4} {'L':>4} {'D':>4} {'Playing':>9}")
    lines.append("─" * 72)

    for name in bots:
        s = user_stats(name)
        if "error" in s:
            lines.append(f"{name:<12}  ERROR: {s['error']}")
            continue
        c = s["count"]
        p = s["perfs"]
        bullet = f"{fmt_rating(p['bullet'])} ({p['bullet']['games']})"
        blitz = f"{fmt_rating(p['blitz'])} ({p['blitz']['games']})"
        rapid = f"{fmt_rating(p['rapid'])} ({p['rapid']['games']})"
        playing = f"game {s['game_id'][:8]}" if s["currently_playing"] else "—"
        lines.append(
            f"{name:<12} {bullet:>10} {blitz:>10} {rapid:>10} "
            f"{c['all']:>5} {c['win']:>4} {c['loss']:>4} {c['draw']:>4} {playing:>9}"
        )

    lines.append("")

    if show_recent:
        for name in bots:
            s = user_stats(name)
            if "error" in s:
                continue
            lines.append(f"── {name} — last 50 games ──")
            games = recent_games(name, 50)
            if not games:
                lines.append("  (no rated games)")
            else:
                # Summary
                w = sum(1 for g in games if g["result"] == "W")
                l = sum(1 for g in games if g["result"] == "L")
                d = sum(1 for g in games if g["result"] == "½")
                lines.append(f"  Recent: {w}W {l}L {d}D  (score: {w + d/2}/{len(games)})")
                # Compact line of results
                # Group into chunks per speed
                chunks = {"bullet": [], "blitz": [], "rapid": [], "classical": [], "correspondence": []}
                for g in games:
                    sp = g["speed"]
                    if sp in chunks:
                        chunks[sp].append(g)
                for sp, gs in chunks.items():
                    if not gs:
                        continue
                    seq = "".join(g["result"] for g in gs)
                    low = gs[-1]["my_rating"]
                    high = gs[0]["my_rating"]
                    change = gs[0].get("rating_diff")
                    sign = "+" if change and change > 0 else ""
                    diff = f"{sign}{change}" if change is not None else ""
                    lines.append(f"  {sp:<7} {seq[:60]}{'…' if len(seq)>60 else ''}  ({low}→{high} {diff})")
            lines.append("")

    # Play time summary
    total_play = 0
    for name in bots:
        s = user_stats(name)
        if "error" not in s:
            total_play += s["play_time_min"]
    hr = total_play // 60
    min_ = total_play % 60
    lines.append(f"Total play time across all bots: {hr}h {min_}m")

    return "\n".join(lines)


def compute_snapshot(bots: list[str]) -> dict:
    """Build a snapshot dict keyed by bot with total games count + per-format ratings."""
    snap = {"_timestamp": int(time.time()), "bots": {}}
    for name in bots:
        s = user_stats(name)
        if "error" in s:
            continue
        snap["bots"][name] = {
            "total_games": s["count"]["all"],
            "perfs": {f: p["games"] for f, p in s["perfs"].items()},
        }
    return snap


def total_games(snap: dict) -> int:
    return sum(b["total_games"] for b in snap.get("bots", {}).values())


def main():
    ap = argparse.ArgumentParser(description="Check Patzer bot stats on Lichess")
    ap.add_argument("bots", nargs="*", metavar="BOT", help="Bot names (default: all v1-v8)")
    ap.add_argument("--no-recent", action="store_true", help="Skip recent games section")
    ap.add_argument("--json", action="store_true", help="Output raw JSON snapshot (for cron)")
    ap.add_argument("--since", type=Path, help="Compare against previous snapshot file; report only if ≥100 new games")
    args = ap.parse_args()

    bots = [f"patzer_{b}" if not b.startswith("patzer_") else b for b in (args.bots or [])]
    if not bots:
        bots = BOTS

    # --since mode: for cron job
    if args.since:
        current = compute_snapshot(bots)
        prev = json.loads(args.since.read_text()) if args.since.exists() else {"bots": {}}
        prev_total = sum(b["total_games"] for b in prev.get("bots", {}).values())
        cur_total = total_games(current)
        new_games = cur_total - prev_total
        # Save current as new baseline
        args.since.write_text(json.dumps(current, indent=2))
        if new_games >= 100:
            print(f"⚠  {new_games} new games played since last check (threshold: 100)")
            print()
            print(report_all(bots, show_recent=True))
        else:
            print(f"Only {new_games} new games — below threshold, skipping full report.")
        return

    if args.json:
        print(json.dumps(compute_snapshot(bots), indent=2))
        return

    print(report_all(bots, show_recent=not args.no_recent))


if __name__ == "__main__":
    main()
