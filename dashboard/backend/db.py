"""
dashboard/backend/db.py — SQLite helpers for the Lichess game store.

Separate from eval/results.db; lives at dashboard/lichess_games.db.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone

# Always absolute — relative paths depend on process cwd and break sync vs HTTP reads.
DB_PATH = (Path(__file__).resolve().parent.parent / "lichess_games.db").resolve()


def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


@contextmanager
def _db():
    """Open connection, **commit** on success, rollback on error, always close.

    `contextlib.closing()` alone is not enough: sqlite3 rolls back uncommitted
    work when the connection closes, so every INSERT was being discarded.
    """
    con = _connect()
    try:
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def init_db():
    print(f"[dashboard] Lichess DB path: {DB_PATH}", flush=True)
    with _db() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS lichess_games (
                id              TEXT PRIMARY KEY,
                bot_version     TEXT NOT NULL,
                bot_color       TEXT NOT NULL,
                opponent        TEXT,
                opponent_rating INTEGER,
                bot_rating      INTEGER,
                result          TEXT,
                termination     TEXT,
                time_control    TEXT,
                speed           TEXT,
                opening_eco     TEXT,
                opening_name    TEXT,
                pgn             TEXT,
                played_at       TEXT,
                fetched_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS lichess_sync_state (
                bot_version     TEXT PRIMARY KEY,
                last_fetched_at TEXT,
                last_sync_at    TEXT,
                total_fetched   INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS lichess_moves (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id     TEXT NOT NULL REFERENCES lichess_games(id),
                ply         INTEGER NOT NULL,
                uci         TEXT NOT NULL,
                san         TEXT,
                fen_before  TEXT,
                is_bot_move INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS lichess_move_analysis (
                move_id        INTEGER PRIMARY KEY REFERENCES lichess_moves(id),
                depth          INTEGER,
                eval_before_cp INTEGER,
                eval_after_cp  INTEGER,
                best_move_uci  TEXT,
                eval_loss_cp   INTEGER,
                error_class    TEXT
            );

            CREATE TABLE IF NOT EXISTS lichess_game_aggregates (
                game_id          TEXT PRIMARY KEY REFERENCES lichess_games(id),
                total_bot_moves  INTEGER,
                avg_eval_loss_cp REAL,
                blunders         INTEGER DEFAULT 0,
                mistakes         INTEGER DEFAULT 0,
                inaccuracies     INTEGER DEFAULT 0,
                analysis_depth   INTEGER,
                analyzed_at      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_lichess_games_bot
                ON lichess_games(bot_version, played_at);

            CREATE INDEX IF NOT EXISTS idx_lichess_moves_game
                ON lichess_moves(game_id, ply);
        """)


def _exec_upsert_game(con: sqlite3.Connection, game: dict) -> None:
    con.execute("""
        INSERT OR REPLACE INTO lichess_games
            (id, bot_version, bot_color, opponent, opponent_rating, bot_rating,
             result, termination, time_control, speed, opening_eco, opening_name,
             pgn, played_at, fetched_at)
        VALUES
            (:id, :bot_version, :bot_color, :opponent, :opponent_rating, :bot_rating,
             :result, :termination, :time_control, :speed, :opening_eco, :opening_name,
             :pgn, :played_at, :fetched_at)
    """, game)


def _exec_insert_moves(con: sqlite3.Connection, game_id: str, moves: list[dict]) -> None:
    con.execute("DELETE FROM lichess_moves WHERE game_id = ?", (game_id,))
    con.executemany("""
        INSERT INTO lichess_moves (game_id, ply, uci, san, fen_before, is_bot_move)
        VALUES (:game_id, :ply, :uci, :san, :fen_before, :is_bot_move)
    """, moves)


def upsert_game(game: dict) -> None:
    with _db() as con:
        _exec_upsert_game(con, game)


def insert_moves(game_id: str, moves: list[dict]) -> None:
    """Insert move rows; silently skip if already present (game was re-synced)."""
    with _db() as con:
        _exec_insert_moves(con, game_id, moves)


def upsert_game_with_moves(game: dict, moves: list[dict]) -> None:
    """Write game + moves in one connection (used by Lichess sync).

    Delete child moves *before* INSERT OR REPLACE on lichess_games so foreign keys
    never block replacing an existing game row on re-sync.
    """
    gid = game["id"]
    with _db() as con:
        con.execute("DELETE FROM lichess_moves WHERE game_id = ?", (gid,))
        _exec_upsert_game(con, game)
        if moves:
            con.executemany("""
                INSERT INTO lichess_moves (game_id, ply, uci, san, fen_before, is_bot_move)
                VALUES (:game_id, :ply, :uci, :san, :fen_before, :is_bot_move)
            """, moves)


def lichess_store_stats() -> dict:
    """Counts from the same DB file HTTP handlers use (for diagnostics)."""
    path = str(DB_PATH)
    with _db() as con:
        games = con.execute("SELECT COUNT(*) FROM lichess_games").fetchone()[0]
        moves = con.execute("SELECT COUNT(*) FROM lichess_moves").fetchone()[0]
        nonempty_pgn = con.execute(
            "SELECT COUNT(*) FROM lichess_games WHERE pgn IS NOT NULL AND length(trim(pgn)) > 0"
        ).fetchone()[0]
        queue = con.execute("""
            SELECT COUNT(*) FROM lichess_games g
            LEFT JOIN lichess_game_aggregates a ON g.id = a.game_id
            WHERE a.analyzed_at IS NULL
              AND g.pgn IS NOT NULL
              AND length(trim(g.pgn)) > 0
        """).fetchone()[0]
    return {"db_path": path, "games": games, "moves": moves, "nonempty_pgn": nonempty_pgn, "analysis_queue": queue}


def get_sync_state(bot_version: str) -> dict:
    with _db() as con:
        row = con.execute(
            "SELECT * FROM lichess_sync_state WHERE bot_version = ?", (bot_version,)
        ).fetchone()
    return dict(row) if row else {}


def set_sync_state(bot_version: str, *, last_fetched_at: str | None = None, total_fetched: int = 0) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _db() as con:
        con.execute("""
            INSERT INTO lichess_sync_state (bot_version, last_fetched_at, last_sync_at, total_fetched)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(bot_version) DO UPDATE SET
                last_fetched_at = excluded.last_fetched_at,
                last_sync_at    = excluded.last_sync_at,
                total_fetched   = excluded.total_fetched
        """, (bot_version, last_fetched_at, now, total_fetched))


def get_games_without_analysis(limit: int = 200) -> list[dict]:
    with _db() as con:
        rows = con.execute("""
            SELECT g.id, g.pgn, g.bot_color
            FROM lichess_games g
            LEFT JOIN lichess_game_aggregates a ON g.id = a.game_id
            WHERE a.analyzed_at IS NULL
              AND g.pgn IS NOT NULL
              AND length(trim(g.pgn)) > 0
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_move_ids_for_game(game_id: str) -> list[dict]:
    with _db() as con:
        rows = con.execute(
            "SELECT id, fen_before, is_bot_move FROM lichess_moves WHERE game_id = ? ORDER BY ply",
            (game_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_move_analysis(move_id: int, data: dict) -> None:
    with _db() as con:
        con.execute("""
            INSERT OR REPLACE INTO lichess_move_analysis
                (move_id, depth, eval_before_cp, eval_after_cp, best_move_uci, eval_loss_cp, error_class)
            VALUES (:move_id, :depth, :eval_before_cp, :eval_after_cp, :best_move_uci, :eval_loss_cp, :error_class)
        """, {"move_id": move_id, **data})


def upsert_game_aggregate(game_id: str, data: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _db() as con:
        con.execute("""
            INSERT OR REPLACE INTO lichess_game_aggregates
                (game_id, total_bot_moves, avg_eval_loss_cp, blunders, mistakes, inaccuracies,
                 analysis_depth, analyzed_at)
            VALUES (:game_id, :total_bot_moves, :avg_eval_loss_cp, :blunders, :mistakes, :inaccuracies,
                    :analysis_depth, :analyzed_at)
        """, {"game_id": game_id, "analyzed_at": now, **data})


def query_lichess_games(
    bot_version: str | None = None,
    speed: str | None = None,
    result: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict], int]:
    clauses = []
    params: list = []
    if bot_version:
        clauses.append("g.bot_version = ?")
        params.append(bot_version)
    if speed:
        clauses.append("g.speed = ?")
        params.append(speed)
    if result:
        clauses.append("g.result = ?")
        params.append(result)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    with _db() as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM lichess_games g {where}", params
        ).fetchone()[0]

        rows = con.execute(f"""
            SELECT g.id, g.bot_version, g.bot_color, g.opponent, g.opponent_rating,
                   g.bot_rating, g.result, g.speed, g.opening_name, g.played_at,
                   a.avg_eval_loss_cp, a.blunders, a.analyzed_at IS NOT NULL AS analyzed
            FROM lichess_games g
            LEFT JOIN lichess_game_aggregates a ON g.id = a.game_id
            {where}
            ORDER BY g.played_at DESC
            LIMIT ? OFFSET ?
        """, params + [limit, offset]).fetchall()

    return [dict(r) for r in rows], total


def get_lichess_stats() -> dict:
    with _db() as con:
        rows = con.execute("""
            SELECT g.bot_version,
                   COUNT(*) AS total_games,
                   SUM(CASE WHEN (g.bot_color='white' AND g.result='1-0') OR
                                 (g.bot_color='black' AND g.result='0-1') THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN (g.bot_color='white' AND g.result='0-1') OR
                                 (g.bot_color='black' AND g.result='1-0') THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN g.result='1/2-1/2' THEN 1 ELSE 0 END) AS draws,
                   AVG(a.avg_eval_loss_cp) AS avg_cpl,
                   SUM(a.blunders) AS total_blunders,
                   COUNT(a.analyzed_at) AS analyzed_games,
                   COUNT(CASE WHEN a.analyzed_at IS NOT NULL AND a.total_bot_moves > 0 THEN 1 END) AS analyzed_with_moves
            FROM lichess_games g
            LEFT JOIN lichess_game_aggregates a ON g.id = a.game_id
            GROUP BY g.bot_version
            ORDER BY g.bot_version
        """).fetchall()
    return {r["bot_version"]: dict(r) for r in rows}


def get_all_sync_states() -> dict:
    with _db() as con:
        rows = con.execute("SELECT * FROM lichess_sync_state").fetchall()
    return {r["bot_version"]: dict(r) for r in rows}


def get_lichess_performance_snapshot(
    bot_version: str | None = None,
    *,
    max_bot_move_index: int = 80,
    min_per_bin: int = 3,
    opening_bot_moves_end: int = 12,
    middlegame_bot_moves_end: int = 32,
) -> dict:
    """Aggregate bot-move analysis for “performance over the game” charts.

    *bot_move_index* is the Nth time the bot chooses a move in that game (1-based),
    so it lines up for white and black. Each curve bin includes ``avg_eval_before_cp``
    (Stockfish before the bot moves) and ``avg_cpl`` from non-negative ``eval_loss_cp``.
    The dashboard plots ``avg_cpl`` vs move index for decision quality.

    Phases are coarse ply buckets on that same index: opening ≤ ``opening_bot_moves_end``,
    middlegame through ``middlegame_bot_moves_end``, then endgame.

    Response ``curve_series`` / ``phase_series`` hold one entry per ``patzer_v*`` bot (or a
    single bot when ``bot_version`` is filtered) so clients can plot separate lines/bars.
    """
    bv = (bot_version or "").strip()
    if bv and not bv.startswith("patzer_v"):
        bv = ""
    bv_param = bv or ""

    curve_sql = """
        WITH bot_moves AS (
            SELECT
                g.bot_version,
                ROW_NUMBER() OVER (PARTITION BY m.game_id ORDER BY m.ply) AS bot_idx,
                ma.eval_before_cp,
                ma.eval_loss_cp,
                ma.error_class
            FROM lichess_moves m
            INNER JOIN lichess_move_analysis ma ON ma.move_id = m.id
            INNER JOIN lichess_games g ON g.id = m.game_id
            WHERE m.is_bot_move = 1
              AND ma.eval_before_cp IS NOT NULL
              AND g.bot_version LIKE 'patzer_v%'
              AND ((? = '') OR (g.bot_version = ?))
        )
        SELECT
            bot_version,
            bot_idx AS bot_move_index,
            COUNT(*) AS n,
            ROUND(AVG(eval_before_cp), 2) AS avg_eval_before_cp,
            ROUND(AVG(
                CASE
                    WHEN eval_loss_cp IS NULL THEN NULL
                    WHEN eval_loss_cp < 0 THEN 0
                    ELSE eval_loss_cp
                END
            ), 2) AS avg_cpl,
            ROUND(SUM(CASE WHEN error_class = 'blunder' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 5) AS blunder_rate,
            ROUND(SUM(CASE WHEN error_class = 'mistake' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 5) AS mistake_rate,
            ROUND(SUM(CASE WHEN error_class = 'inaccuracy' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 5) AS inaccuracy_rate
        FROM bot_moves
        WHERE bot_idx <= ?
        GROUP BY bot_version, bot_idx
        HAVING n >= ?
        ORDER BY bot_version, bot_idx
    """

    phase_sql = """
        WITH bot_moves AS (
            SELECT
                g.bot_version,
                ROW_NUMBER() OVER (PARTITION BY m.game_id ORDER BY m.ply) AS bot_idx,
                ma.eval_loss_cp,
                ma.error_class
            FROM lichess_moves m
            INNER JOIN lichess_move_analysis ma ON ma.move_id = m.id
            INNER JOIN lichess_games g ON g.id = m.game_id
            WHERE m.is_bot_move = 1
              AND ma.eval_before_cp IS NOT NULL
              AND g.bot_version LIKE 'patzer_v%'
              AND ((? = '') OR (g.bot_version = ?))
        ),
        phased AS (
            SELECT
                bot_version,
                bot_idx,
                eval_loss_cp,
                error_class,
                CASE
                    WHEN bot_idx <= ? THEN 'opening'
                    WHEN bot_idx <= ? THEN 'middlegame'
                    ELSE 'endgame'
                END AS phase
            FROM bot_moves
        )
        SELECT
            bot_version,
            phase,
            COUNT(*) AS bot_moves,
            ROUND(AVG(
                CASE
                    WHEN eval_loss_cp IS NULL THEN NULL
                    WHEN eval_loss_cp < 0 THEN 0
                    ELSE eval_loss_cp
                END
            ), 2) AS avg_cpl,
            ROUND(SUM(CASE WHEN error_class = 'blunder' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 5) AS blunder_rate,
            ROUND(SUM(CASE WHEN error_class = 'mistake' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 5) AS mistake_rate,
            ROUND(SUM(CASE WHEN error_class = 'inaccuracy' THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 5) AS inaccuracy_rate
        FROM phased
        GROUP BY bot_version, phase
        ORDER BY bot_version,
                 CASE phase WHEN 'opening' THEN 1 WHEN 'middlegame' THEN 2 ELSE 3 END
    """

    curve_params = [bv_param, bv_param, max_bot_move_index, min_per_bin]
    phase_params = [bv_param, bv_param, opening_bot_moves_end, middlegame_bot_moves_end]

    with _db() as con:
        curve_rows = con.execute(curve_sql, curve_params).fetchall()
        phase_rows = con.execute(phase_sql, phase_params).fetchall()
        total_bot_moves = con.execute(
            """
            SELECT COUNT(*)
            FROM lichess_moves m
            INNER JOIN lichess_move_analysis ma ON ma.move_id = m.id
            INNER JOIN lichess_games g ON g.id = m.game_id
            WHERE m.is_bot_move = 1
              AND ma.eval_before_cp IS NOT NULL
              AND g.bot_version LIKE 'patzer_v%'
              AND ((? = '') OR (g.bot_version = ?))
            """,
            [bv_param, bv_param],
        ).fetchone()[0]

    curve_by_bv: dict[str, list] = {}
    for r in curve_rows:
        d = dict(r)
        bver = d.pop("bot_version")
        curve_by_bv.setdefault(bver, []).append(d)
    curve_series = [{"bot_version": bver, "points": pts} for bver, pts in sorted(curve_by_bv.items())]

    phase_by_bv: dict[str, list] = {}
    for r in phase_rows:
        d = dict(r)
        bver = d.pop("bot_version")
        phase_by_bv.setdefault(bver, []).append(d)
    phase_series = [{"bot_version": bver, "phases": pts} for bver, pts in sorted(phase_by_bv.items())]

    return {
        "bot_version": bv or None,
        "total_analyzed_bot_moves": total_bot_moves,
        "max_bot_move_index": max_bot_move_index,
        "min_per_bin": min_per_bin,
        "phase_boundaries": {
            "opening_bot_moves_end": opening_bot_moves_end,
            "middlegame_bot_moves_end": middlegame_bot_moves_end,
        },
        "curve_series": curve_series,
        "phase_series": phase_series,
    }
