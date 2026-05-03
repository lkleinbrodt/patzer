"""
Lichess game sync via berserk API client.

Fetches game history for each bot version and stores in lichess_games.db.
Bot tokens are read from env (PATZER_V1_TOKEN, etc.).
Lichess usernames default to the version id (e.g. patzer_v1) except **patzer_v2**
→ **patzer_v2b** (the live bot account). Override any bot with PATZER_VN_USERNAME.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import berserk
import chess
import chess.pgn
import io

from .db import (
    upsert_game_with_moves,
    get_sync_state,
    set_sync_state,
    get_all_sync_states,
)

# Module-level sync state
_lock = threading.Lock()
_state: dict = {
    "status": "idle",  # idle | running | done | error
    "bots": {},
    "lines": [],
    "error": None,
}


_DEFAULT_LICHESS_USERNAME: dict[str, str] = {
    "patzer_v2": "patzer_v2b",
}

_PUBLIC_RATINGS_LOCK = threading.Lock()
_public_ratings_cache: dict[str, tuple[float, dict[str, int | None]]] = {}


def lichess_username_for_bot_version(version: str) -> str:
    """Lichess account name for a Patzer bot id (sync env + PATZER_VN_USERNAME overrides)."""
    key = version.upper() + "_USERNAME"
    return os.environ.get(key) or _DEFAULT_LICHESS_USERNAME.get(version, version)


def _fetch_public_bullet_blitz_ratings(username: str) -> dict[str, int | None]:
    """HTTP GET Lichess public profile; no token. Returns bullet/blitz Glicko-2 display ratings."""
    slug = urllib.parse.quote(username, safe="")
    url = f"https://lichess.org/api/user/{slug}"
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "PatzerDashboard/1.0 (lichess.org/api)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {"bullet": None, "blitz": None}
    perfs = data.get("perfs") or {}
    out: dict[str, int | None] = {}
    for key in ("bullet", "blitz"):
        block = perfs.get(key) or {}
        r = block.get("rating")
        if isinstance(r, (int, float)) and not isinstance(r, bool):
            out[key] = int(round(r))
        else:
            out[key] = None
    return out


def get_public_bullet_blitz_ratings(username: str, *, cache_ttl_sec: float = 120.0) -> dict[str, int | None]:
    """Cached bullet/blitz ratings from Lichess public API (per username)."""
    uname = (username or "").strip()
    if not uname:
        return {"bullet": None, "blitz": None}
    key = uname.lower()
    now = time.monotonic()
    with _PUBLIC_RATINGS_LOCK:
        hit = _public_ratings_cache.get(key)
        if hit and now - hit[0] < cache_ttl_sec:
            return dict(hit[1])
    ratings = _fetch_public_bullet_blitz_ratings(uname)
    with _PUBLIC_RATINGS_LOCK:
        _public_ratings_cache[key] = (now, ratings)
    return dict(ratings)


def _get_bot_configs() -> dict[str, dict]:
    configs = {}
    for version in ["patzer_v1", "patzer_v2", "patzer_v3", "patzer_v4"]:
        token_key = version.upper() + "_TOKEN"
        token = os.environ.get(token_key)
        username = lichess_username_for_bot_version(version)
        if token:
            configs[version] = {"token": token, "username": username}
    return configs


def _log(line: str) -> None:
    with _lock:
        _state["lines"].append(line)
    print(f"[lichess_sync] {line}")


def _set_bot_status(version: str, status: str, fetched: int | None = None) -> None:
    with _lock:
        if version not in _state["bots"]:
            _state["bots"][version] = {"status": "idle", "fetched": 0, "last_sync_at": None}
        _state["bots"][version]["status"] = status
        if fetched is not None:
            _state["bots"][version]["fetched"] = fetched


def _lichess_player_id(player: dict | None) -> str:
    """Lichess NDJSON uses players.*.user.id (berserk may also expose userId)."""
    if not player:
        return ""
    u = player.get("user") or {}
    return str(u.get("id") or player.get("userId") or "").lower()


def _game_to_row(game_dict: dict, bot_version: str, bot_username: str) -> dict:
    """Convert berserk game dict to DB row dict."""
    players = game_dict.get("players", {})
    white_id = _lichess_player_id(players.get("white"))
    black_id = _lichess_player_id(players.get("black"))
    bot_lower = bot_username.lower()

    if white_id == bot_lower:
        bot_color = "white"
        opp_player = players.get("black") or {}
    else:
        bot_color = "black"
        opp_player = players.get("white") or {}

    bot_player = players.get(bot_color) or {}

    result_map = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}
    winner = game_dict.get("winner")
    status = game_dict.get("status", "")
    if winner:
        result = result_map.get(winner, "*")
    elif status in ("draw", "stalemate"):
        result = "1/2-1/2"
    else:
        result = "*"

    # Termination label
    termination_map = {
        "mate": "checkmate",
        "resign": "resign",
        "stalemate": "stalemate",
        "draw": "draw",
        "outoftime": "timeout",
        "timeout": "timeout",
        "repetition": "repetition",
        "noStart": "no_start",
        "aborted": "aborted",
    }
    termination = termination_map.get(status, status)

    clock = game_dict.get("clock") or {}
    initial = clock.get("initial")
    increment = clock.get("increment")
    time_control = f"{initial}+{increment}" if initial is not None else None

    opening = game_dict.get("opening") or {}
    played_ms = game_dict.get("createdAt")
    if played_ms:
        if isinstance(played_ms, int):
            played_at = datetime.fromtimestamp(played_ms / 1000, tz=timezone.utc).isoformat()
        elif isinstance(played_ms, datetime):
            played_at = played_ms.isoformat() if not played_ms.tzinfo else played_ms.isoformat()
        else:
            played_at = None
    else:
        played_at = None

    # Lichess `moves` is space-separated SAN, not UCI. Prefer API `pgn` when pgnInJson=true.
    moves_san = game_dict.get("moves", "") or ""
    pgn = game_dict.get("pgn") or _build_pgn_from_san(moves_san, game_dict)

    return {
        "id": game_dict["id"],
        "bot_version": bot_version,
        "bot_color": bot_color,
        "opponent": (opp_player.get("user") or {}).get("id") or opp_player.get("userId"),
        "opponent_rating": opp_player.get("rating"),
        "bot_rating": bot_player.get("rating"),
        "result": result,
        "termination": termination,
        "time_control": time_control,
        "speed": game_dict.get("speed"),
        "opening_eco": opening.get("eco"),
        "opening_name": opening.get("name"),
        "pgn": pgn,
        "played_at": played_at,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _parse_move(board: chess.Board, token: str) -> chess.Move:
    """Lichess usually sends SAN (`e4`, `Bxc3+`); some exports use UCI (`e2e4`)."""
    t = token.strip()
    if (
        4 <= len(t) <= 5
        and t[0] in "abcdefgh"
        and t[1] in "12345678"
        and t[2] in "abcdefgh"
        and t[3] in "12345678"
    ):
        try:
            mv = chess.Move.from_uci(t)
        except ValueError:
            pass
        else:
            if mv in board.legal_moves:
                return mv
    return board.parse_san(t)


def _board_for_game(game_dict: dict) -> chess.Board:
    """Standard board, or chess960 / from-position when the API supplies initialFen."""
    vkey = str(game_dict.get("variant") or "standard").lower().replace(" ", "")
    fen = game_dict.get("initialFen")
    if vkey == "chess960" and fen:
        try:
            return chess.Board(fen, chess960=True)
        except Exception:
            pass
    if vkey == "fromposition" and fen:
        try:
            return chess.Board(fen)
        except Exception:
            pass
    return chess.Board()


def _build_pgn_from_san(moves_san: str, game_dict: dict) -> str | None:
    """Build a PGN string from space-separated SAN tokens (Lichess export format)."""
    if not moves_san:
        return None
    try:
        board = _board_for_game(game_dict)
        game = chess.pgn.Game()
        node = game
        for token in moves_san.strip().split():
            move = _parse_move(board, token)
            node = node.add_variation(move)
            board.push(move)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return game.accept(exporter)
    except Exception:
        return None


def _extract_moves(game_id: str, moves_san: str, bot_color: str, game_dict: dict) -> list[dict]:
    """Parse Lichess space-separated SAN into per-ply rows (uci column stores UCI for analysis)."""
    if not moves_san:
        return []
    rows = []
    board = _board_for_game(game_dict)
    for ply, token in enumerate(moves_san.strip().split(), start=1):
        try:
            move = _parse_move(board, token)
            uci = move.uci()
            san = board.san(move)
            fen_before = board.fen()
            is_white_move = board.turn == chess.WHITE
            is_bot_move = (is_white_move and bot_color == "white") or (not is_white_move and bot_color == "black")
            board.push(move)
            rows.append({
                "game_id": game_id,
                "ply": ply,
                "uci": uci,
                "san": san,
                "fen_before": fen_before,
                "is_bot_move": int(is_bot_move),
            })
        except Exception:
            break
    return rows


def _sync_bot(bot_version: str, token: str, username: str) -> int:
    """Sync one bot version. Returns number of new games fetched."""
    _log(f"{bot_version}: starting sync (Lichess username: {username})")
    _set_bot_status(bot_version, "running", 0)

    session = berserk.TokenSession(token)
    client = berserk.Client(session)

    sync_state = get_sync_state(bot_version)
    last_fetched_at = sync_state.get("last_fetched_at")

    # Convert ISO timestamp to milliseconds for the since parameter
    since_ms = None
    if last_fetched_at:
        try:
            dt = datetime.fromisoformat(last_fetched_at)
            since_ms = int(dt.timestamp() * 1000) + 1
        except Exception:
            pass

    fetched = 0
    with_pgn = 0
    move_rows_total = 0
    latest_played_at: str | None = None

    try:
        kwargs: dict = {
            "as_pgn": False,
            "perf_type": None,
            "pgn_in_json": True,
            "opening": True,
        }
        if since_ms:
            kwargs["since"] = since_ms

        games = client.games.export_by_player(username, **kwargs)
        for game_dict in games:
            try:
                row = _game_to_row(game_dict, bot_version, username)
                moves_san = game_dict.get("moves", "") or ""
                move_rows = _extract_moves(row["id"], moves_san, row["bot_color"], game_dict)
                upsert_game_with_moves(row, move_rows)
                fetched += 1
                if row.get("pgn"):
                    with_pgn += 1
                move_rows_total += len(move_rows)
                if row["played_at"] and (latest_played_at is None or row["played_at"] > latest_played_at):
                    latest_played_at = row["played_at"]
                if fetched % 50 == 0:
                    _log(f"{bot_version}: fetched {fetched} games...")
                    _set_bot_status(bot_version, "running", fetched)
            except Exception as e:
                _log(f"{bot_version}: error processing game {game_dict.get('id', '?')}: {e}")

    except Exception as e:
        _log(f"{bot_version}: fetch error: {e}")
        _set_bot_status(bot_version, "error", fetched)
        raise

    set_sync_state(bot_version, last_fetched_at=latest_played_at or last_fetched_at, total_fetched=fetched)
    _log(
        f"{bot_version}: done — fetched {fetched} new games "
        f"(with PGN: {with_pgn}, move rows written: {move_rows_total})"
    )
    _set_bot_status(bot_version, "done", fetched)
    return fetched


def _run_sync(bot_versions: list[str] | None = None) -> None:
    """Background thread entry point."""
    configs = _get_bot_configs()
    targets = bot_versions or list(configs)
    targets = [v for v in targets if v in configs]

    with _lock:
        _state["status"] = "running"
        _state["lines"] = []
        _state["error"] = None
        for v in targets:
            _state["bots"][v] = {"status": "pending", "fetched": 0, "last_sync_at": None}

    try:
        total = 0
        for version in targets:
            cfg = configs[version]
            try:
                n = _sync_bot(version, cfg["token"], cfg["username"])
                total += n
            except Exception as e:
                _log(f"Error syncing {version}: {e}")

        _log(f"Sync complete — total {total} games fetched across {len(targets)} bot(s)")
        with _lock:
            _state["status"] = "done"

        # Trigger analysis in background (stdout from analysis module — see [lichess_analysis])
        from .analysis import trigger_analysis

        _log("Scheduling Stockfish analysis (background)…")
        trigger_analysis()

    except Exception as e:
        with _lock:
            _state["status"] = "error"
            _state["error"] = str(e)
        _log(f"Fatal sync error: {e}")


def start_sync(bot_versions: list[str] | None = None) -> bool:
    """Start sync in background thread. Returns False if already running."""
    with _lock:
        if _state["status"] == "running":
            return False
    thread = threading.Thread(target=_run_sync, args=(bot_versions,), daemon=True)
    thread.start()
    return True


def get_sync_status() -> dict:
    with _lock:
        # Merge DB sync state for last_sync_at info
        db_states = get_all_sync_states()
        bots_merged = {}
        all_bot_versions = set(list(_state["bots"].keys()) + list(db_states.keys()))
        for v in all_bot_versions:
            mem = _state["bots"].get(v, {})
            db = db_states.get(v, {})
            bots_merged[v] = {
                "status": mem.get("status", "idle"),
                "fetched": mem.get("fetched", db.get("total_fetched", 0)),
                "last_sync_at": db.get("last_sync_at"),
            }
        return {
            "status": _state["status"],
            "lines": _state["lines"].copy(),
            "error": _state["error"],
            "bots": bots_merged,
        }
