"""
Stockfish per-move analysis for Lichess games.

Analyzes bot moves at depth 12, computes centipawn loss and error classification.
Triggered automatically after sync or on demand.
"""

import os
import threading
import logging

import chess
import chess.engine

from .db import (
    get_games_without_analysis,
    get_move_ids_for_game,
    lichess_store_stats,
    upsert_move_analysis,
    upsert_game_aggregate,
)

logger = logging.getLogger(__name__)


def _log(line: str) -> None:
    """Stdout so Flask dev server shows progress (root logger often INFO-off)."""
    print(f"[lichess_analysis] {line}", flush=True)


ANALYSIS_DEPTH = 12
MATE_CP_CLAMP = 10_000

_analysis_lock = threading.Lock()
_analysis_running = False


def _stockfish_path() -> str:
    return os.environ.get("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")


def _score_to_cp(score) -> int | None:
    if score is None:
        return None
    try:
        return score.relative.score(mate_score=MATE_CP_CLAMP)
    except (AttributeError, TypeError):
        return None


def _classify_error(eval_loss_cp: int | None) -> str | None:
    if eval_loss_cp is None:
        return None
    if eval_loss_cp >= 300:
        return "blunder"
    if eval_loss_cp >= 100:
        return "mistake"
    if eval_loss_cp >= 50:
        return "inaccuracy"
    return None


def _analyze_game(game_id: str, pgn: str | None, bot_color: str, engine) -> None:
    """Analyze bot moves in one game. Engine is already open."""
    if not pgn:
        return

    import chess.pgn
    import io
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return

    board = game.board()
    moves = list(game.mainline_moves())

    # Fetch move rows to get IDs and is_bot_move flags
    move_rows = get_move_ids_for_game(game_id)
    if not move_rows:
        logger.warning("Skip analysis %s: no lichess_moves rows (re-sync games)", game_id)
        return
    if len(move_rows) != len(moves):
        logger.warning(
            "Skip analysis %s: move count mismatch db=%d pgn=%d (do not mark analyzed)",
            game_id,
            len(move_rows),
            len(moves),
        )
        return

    blunders = mistakes = inaccuracies = 0
    total_loss = 0
    analyzed_moves = 0

    for i, (move_row, move) in enumerate(zip(move_rows, moves)):
        if not move_row["is_bot_move"]:
            board.push(move)
            continue

        try:
            info_before = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
            best_eval_cp = _score_to_cp(info_before["score"])
            best_move = info_before.get("pv", [None])[0]

            board.push(move)

            info_after = engine.analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH))
            eval_after_raw = _score_to_cp(info_after["score"])
            # After bot moves, engine gives opponent POV — negate
            eval_after_cp = -eval_after_raw if eval_after_raw is not None else None

            if best_eval_cp is not None and eval_after_cp is not None:
                eval_loss_cp = best_eval_cp - eval_after_cp
            else:
                eval_loss_cp = None

            error_class = _classify_error(eval_loss_cp)

            upsert_move_analysis(move_row["id"], {
                "depth": ANALYSIS_DEPTH,
                "eval_before_cp": best_eval_cp,
                "eval_after_cp": eval_after_cp,
                "best_move_uci": best_move.uci() if best_move else None,
                "eval_loss_cp": eval_loss_cp,
                "error_class": error_class,
            })

            if eval_loss_cp is not None:
                total_loss += max(0, eval_loss_cp)
                analyzed_moves += 1
            if error_class == "blunder":
                blunders += 1
            elif error_class == "mistake":
                mistakes += 1
            elif error_class == "inaccuracy":
                inaccuracies += 1

        except Exception as e:
            logger.warning("Analysis error game %s move %d: %s", game_id, i, e)
            try:
                board.push(move)
            except Exception:
                pass

    avg_cpl = round(total_loss / analyzed_moves, 2) if analyzed_moves else None
    upsert_game_aggregate(game_id, {
        "total_bot_moves": analyzed_moves,
        "avg_eval_loss_cp": avg_cpl,
        "blunders": blunders,
        "mistakes": mistakes,
        "inaccuracies": inaccuracies,
        "analysis_depth": ANALYSIS_DEPTH,
    })


def run_analysis(batch_size: int = 200) -> int:
    """Analyze all unanalyzed games. Returns count of games analyzed."""
    global _analysis_running
    with _analysis_lock:
        if _analysis_running:
            return 0
        _analysis_running = True

    analyzed = 0
    try:
        sf_path = _stockfish_path()
        if not os.path.exists(sf_path):
            _log(f"Stockfish not found at {sf_path} — set STOCKFISH_PATH or install Stockfish")
            logger.error("Stockfish not found at %s", sf_path)
            return 0

        games = get_games_without_analysis(limit=batch_size)
        if not games:
            stats = lichess_store_stats()
            _log(
                "No games in analysis queue — DB snapshot: "
                f"{stats['games']} games, {stats['moves']} move rows, "
                f"{stats['nonempty_pgn']} with PGN, queue={stats['analysis_queue']} | {stats['db_path']}"
            )
            logger.info("No games to analyze")
            return 0

        _log(f"started — {len(games)} game(s), Stockfish depth {ANALYSIS_DEPTH}")
        logger.info("Analyzing %d games with Stockfish depth %d", len(games), ANALYSIS_DEPTH)
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        try:
            for g in games:
                try:
                    _analyze_game(g["id"], g["pgn"], g["bot_color"], engine)
                    analyzed += 1
                except Exception as e:
                    logger.warning("Failed to analyze game %s: %s", g["id"], e)
        finally:
            engine.quit()

        _log(f"finished — {analyzed} game(s) written")
        logger.info("Analysis complete: %d games processed", analyzed)
    finally:
        with _analysis_lock:
            _analysis_running = False

    return analyzed


def trigger_analysis() -> None:
    """Start analysis in a background thread (non-blocking)."""
    thread = threading.Thread(target=run_analysis, daemon=True)
    thread.start()


def is_analysis_running() -> bool:
    return _analysis_running
