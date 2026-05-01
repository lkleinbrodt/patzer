"""
eval/db.py — thin SQLite wrapper for game results.

Schema: one row per game. All queries and insertions go through here.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "results.db"


def iter_display_k(iter_num: int) -> int:
    """Training step in thousands (iter / 1000) for all user-facing model labels."""
    return int(iter_num) // 1000


def player_name(checkpoint_path: str | Path, iter_num: int) -> str:
    """
    Derive a short stable player identifier from a checkpoint path + iter.
    checkpoints/patzer_v2/weights_best.pt, 45000 -> "patzer_v2@45"
    """
    p = Path(checkpoint_path)
    version = p.parent.name if p.parent.name.startswith("patzer_v") else p.stem
    return f"{version}@{iter_display_k(iter_num)}"


def stockfish_name(elo: int) -> str:
    return f"stockfish:{elo}"


def init_db(db_path: Path = DB_PATH) -> None:
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS games (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT    NOT NULL,
                white            TEXT    NOT NULL,
                black            TEXT    NOT NULL,
                result           TEXT    NOT NULL,
                white_checkpoint TEXT,
                black_checkpoint TEXT,
                white_iter       INTEGER,
                black_iter       INTEGER,
                opening          TEXT,
                temperature      REAL    DEFAULT 0.0,
                top_k            INTEGER,
                conditioning     TEXT    DEFAULT 'match_color'
            )
        """)
        con.commit()


def insert_game(
    *,
    white: str,
    black: str,
    result: str,
    white_checkpoint: str | None = None,
    black_checkpoint: str | None = None,
    white_iter: int | None = None,
    black_iter: int | None = None,
    opening: str | None = None,
    temperature: float = 0.0,
    top_k: int | None = None,
    conditioning: str = "match_color",
    db_path: Path = DB_PATH,
) -> None:
    init_db(db_path)
    ts = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT INTO games
                (timestamp, white, black, result, white_checkpoint, black_checkpoint,
                 white_iter, black_iter, opening, temperature, top_k, conditioning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, white, black, result, white_checkpoint, black_checkpoint,
             white_iter, black_iter, opening, temperature, top_k, conditioning),
        )
        con.commit()


def query_games(
    player: str | None = None,
    db_path: Path = DB_PATH,
) -> list[dict]:
    """Return all games, optionally filtered by substring match on white or black."""
    if not db_path.exists():
        return []
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        if player is None:
            rows = con.execute("SELECT * FROM games ORDER BY id").fetchall()
        else:
            pattern = f"%{player}%"
            rows = con.execute(
                "SELECT * FROM games WHERE white LIKE ? OR black LIKE ? ORDER BY id",
                (pattern, pattern),
            ).fetchall()
    return [dict(r) for r in rows]
