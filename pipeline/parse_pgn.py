"""
Parse filtered Lichess PGN into UCI move sequences.

Reads a PGN file (or stdin) and outputs one game per line as space-separated
UCI moves, with a metadata prefix. Output is a simple text format that the
tokenizer can consume directly.

Output format (one game per line):
    {result} {white_elo} {black_elo} {moves...}

    e.g: 1-0 2100 2000 e2e4 c7c5 g1f3 b8c6 ...

Usage:
    python parse_pgn.py --input filtered.pgn --output games.txt
    python parse_pgn.py --input filtered.pgn --output games.txt --min-moves 10 --max-moves 200
    cat filtered.pgn | python parse_pgn.py --output games.txt

    # Fast mode (default): skip chess.pgn tree building, use direct board.parse_san loop
    cat filtered.pgn | python parse_pgn.py --output games.txt --workers 4

    # Strict mode: full python-chess PGN validation (slower, for debugging)
    cat filtered.pgn | python parse_pgn.py --output games.txt --validate

Stats are written to a separate JSON file for inspection.
"""

import argparse
import gzip
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import io


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Input PGN file. Reads stdin if not provided.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output text file (one game per line)")
    parser.add_argument("--stats-output", type=str, default=None,
                        help="JSON file to write distribution stats to. "
                             "Defaults to <output>.stats.json")
    parser.add_argument("--min-moves", type=int, default=10,
                        help="Minimum number of moves to keep a game (default: 10)")
    parser.add_argument("--max-moves", type=int, default=200,
                        help="Maximum number of moves to keep a game (default: 200)")
    parser.add_argument("--log-every", type=int, default=10_000,
                        help="Log progress every N games (default: 10000)")
    parser.add_argument("--validate", action="store_true", default=False,
                        help="Use full chess.pgn validation (slower, for debugging). "
                             "Default is fast mode which skips PGN tree building.")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel worker processes for SAN→UCI conversion. "
                             "Default: cpu_count-1 (min 1). Set 1 to disable multiprocessing.")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Games per batch sent to workers (default: 2000)")
    return parser.parse_args()


# Strip PGN comments like { [%eval 0.17] [%clk 0:05:00] }
COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\d+\.+\s*")

VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def clean_pgn_moves(moves_text):
    """Remove comments and move numbers, return just the SAN move tokens."""
    text = COMMENT_RE.sub("", moves_text)
    text = MOVE_NUM_RE.sub("", text)
    for r in VALID_RESULTS:
        text = text.replace(r, "")
    text = text.replace("*", "")
    return text.split()


def fast_pgn_to_uci(game_lines):
    """
    Fast SAN→UCI conversion: extract SAN tokens from PGN lines and push
    through a chess.Board. Skips the heavy chess.pgn.read_game() tree builder.
    Returns (uci_moves_list, error_string_or_None).
    """
    moves_text_parts = []
    for line in game_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("["):
            moves_text_parts.append(stripped)

    if not moves_text_parts:
        return None, "no moves"

    san_tokens = clean_pgn_moves(" ".join(moves_text_parts))
    if not san_tokens:
        return None, "no moves"

    board = chess.Board()
    uci_moves = []
    for san in san_tokens:
        try:
            move = board.parse_san(san)
        except (chess.IllegalMoveError, chess.AmbiguousMoveError,
                chess.InvalidMoveError, ValueError):
            return None, "illegal move"
        uci_moves.append(move.uci())
        board.push(move)

    return uci_moves, None


def validated_pgn_to_uci(pgn_text):
    """
    Full chess.pgn validation (original behavior). Builds PGN game tree and
    checks for errors reported by the parser.
    """
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
    except Exception as e:
        return None, f"pgn parse error: {e}"

    if game is None:
        return None, "empty game"

    if game.errors:
        return None, f"illegal move: {str(game.errors[0])}"

    return [move.uci() for move in game.mainline_moves()], None


# ── Worker function for multiprocessing ───────────────────────────────────────

def _process_batch(args):
    """
    Worker: process a batch of games, return (output_lines, partial_stats).
    Each game is (headers_dict, game_lines_list).
    args is (games_batch, min_moves, max_moves, validate_mode).
    """
    games_batch, min_moves, max_moves, validate_mode = args

    partial_stats = {
        "total_games": 0,
        "kept_games": 0,
        "skipped_illegal_move": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_bad_result": 0,
        "skipped_parse_error": 0,
        "result_counts": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "move_length_histogram": {},
        "elo_histogram": {},
        "min_elo_histogram": {},
    }

    output_lines = []

    for headers, game_lines in games_batch:
        partial_stats["total_games"] += 1

        result = headers.get("Result", "")
        if result not in VALID_RESULTS:
            partial_stats["skipped_bad_result"] += 1
            continue

        try:
            white_elo = int(headers.get("WhiteElo", 0))
            black_elo = int(headers.get("BlackElo", 0))
        except ValueError:
            white_elo, black_elo = 0, 0

        if validate_mode:
            pgn_text = "".join(game_lines)
            uci_moves, error = validated_pgn_to_uci(pgn_text)
        else:
            uci_moves, error = fast_pgn_to_uci(game_lines)

        if uci_moves is None:
            if "illegal" in (error or ""):
                partial_stats["skipped_illegal_move"] += 1
            else:
                partial_stats["skipped_parse_error"] += 1
            continue

        n_moves = len(uci_moves)

        if n_moves < min_moves:
            partial_stats["skipped_too_short"] += 1
            continue

        if n_moves > max_moves:
            partial_stats["skipped_too_long"] += 1
            continue

        line = f"{result} {white_elo} {black_elo} {' '.join(uci_moves)}\n"
        output_lines.append(line)

        partial_stats["kept_games"] += 1
        partial_stats["result_counts"][result] = \
            partial_stats["result_counts"].get(result, 0) + 1

        bucket_moves = (n_moves // 10) * 10
        partial_stats["move_length_histogram"][bucket_moves] = \
            partial_stats["move_length_histogram"].get(bucket_moves, 0) + 1

        avg_elo = (white_elo + black_elo) // 2
        bucket_elo = (avg_elo // 100) * 100
        partial_stats["elo_histogram"][bucket_elo] = \
            partial_stats["elo_histogram"].get(bucket_elo, 0) + 1

        min_elo = min(white_elo, black_elo)
        bucket_min_elo = (min_elo // 100) * 100
        partial_stats["min_elo_histogram"][bucket_min_elo] = \
            partial_stats["min_elo_histogram"].get(bucket_min_elo, 0) + 1

    return output_lines, partial_stats


# ── Stats helpers ─────────────────────────────────────────────────────────────

def _empty_stats():
    return {
        "total_games": 0,
        "kept_games": 0,
        "skipped_illegal_move": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_bad_result": 0,
        "skipped_parse_error": 0,
        "result_counts": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "move_length_histogram": {},
        "elo_histogram": {},
        "min_elo_histogram": {},
    }


def _merge_stats(target, partial):
    """Merge partial_stats into target (mutates target)."""
    for key in ("total_games", "kept_games", "skipped_illegal_move",
                "skipped_too_short", "skipped_too_long",
                "skipped_bad_result", "skipped_parse_error"):
        target[key] += partial[key]
    for r in VALID_RESULTS:
        target["result_counts"][r] += partial["result_counts"].get(r, 0)
    for k, v in partial["move_length_histogram"].items():
        target["move_length_histogram"][k] = \
            target["move_length_histogram"].get(k, 0) + v
    for k, v in partial["elo_histogram"].items():
        target["elo_histogram"][k] = \
            target["elo_histogram"].get(k, 0) + v
    for k, v in partial.get("min_elo_histogram", {}).items():
        target["min_elo_histogram"][k] = \
            target["min_elo_histogram"].get(k, 0) + v


# ── Game reader (accumulates PGN stream into game tuples) ─────────────────────

def read_games(pgn_stream):
    """
    Yield (headers_dict, game_lines_list) from a PGN stream.
    Game boundaries are detected by [Event ...] lines.
    """
    current_game_lines = []
    current_headers = {}

    for raw_line in pgn_stream:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line

        stripped = line.strip()

        if stripped.startswith("[Event "):
            if current_game_lines:
                yield current_headers, current_game_lines
            current_game_lines = [line]
            current_headers = {}
        else:
            if current_game_lines:
                current_game_lines.append(line)
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    key = stripped[1:stripped.index(" ")]
                    value = stripped[stripped.index('"') + 1:stripped.rindex('"')]
                    current_headers[key] = value
                except Exception:
                    pass

    if current_game_lines:
        yield current_headers, current_game_lines


def batch_games(game_iter, batch_size):
    """Collect games from iterator into batches of batch_size."""
    batch = []
    for game in game_iter:
        batch.append(game)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ── Sequential (single-process) path ──────────────────────────────────────────

def parse_games_sequential(pgn_stream, output_file, args):
    """Process games sequentially in the main process. Used when workers=1."""
    stats = _empty_stats()
    start_time = time.time()

    for headers, game_lines in read_games(pgn_stream):
        stats["total_games"] += 1

        if stats["total_games"] % args.log_every == 0:
            elapsed = time.time() - start_time
            rate = stats["total_games"] / elapsed
            print(
                f"[parse] {stats['total_games']:,} games processed, "
                f"kept {stats['kept_games']:,} "
                f"({100 * stats['kept_games'] / stats['total_games']:.1f}%) "
                f"| {rate:.0f} games/sec",
                file=sys.stderr,
            )

        result = headers.get("Result", "")
        if result not in VALID_RESULTS:
            stats["skipped_bad_result"] += 1
            continue

        try:
            white_elo = int(headers.get("WhiteElo", 0))
            black_elo = int(headers.get("BlackElo", 0))
        except ValueError:
            white_elo, black_elo = 0, 0

        if args.validate:
            pgn_text = "".join(game_lines)
            uci_moves, error = validated_pgn_to_uci(pgn_text)
        else:
            uci_moves, error = fast_pgn_to_uci(game_lines)

        if uci_moves is None:
            if "illegal" in (error or ""):
                stats["skipped_illegal_move"] += 1
            else:
                stats["skipped_parse_error"] += 1
            continue

        n_moves = len(uci_moves)
        if n_moves < args.min_moves:
            stats["skipped_too_short"] += 1
            continue
        if n_moves > args.max_moves:
            stats["skipped_too_long"] += 1
            continue

        line = f"{result} {white_elo} {black_elo} {' '.join(uci_moves)}\n"
        output_file.write(line)

        stats["kept_games"] += 1
        stats["result_counts"][result] = stats["result_counts"].get(result, 0) + 1

        bucket_moves = (n_moves // 10) * 10
        stats["move_length_histogram"][bucket_moves] = \
            stats["move_length_histogram"].get(bucket_moves, 0) + 1

        avg_elo = (white_elo + black_elo) // 2
        bucket_elo = (avg_elo // 100) * 100
        stats["elo_histogram"][bucket_elo] = \
            stats["elo_histogram"].get(bucket_elo, 0) + 1

        min_elo = min(white_elo, black_elo)
        bucket_min_elo = (min_elo // 100) * 100
        stats["min_elo_histogram"][bucket_min_elo] = \
            stats["min_elo_histogram"].get(bucket_min_elo, 0) + 1

    stats["elapsed_seconds"] = round(time.time() - start_time, 1)
    return stats


# ── Parallel (multi-process) path ─────────────────────────────────────────────

def parse_games_parallel(pgn_stream, output_file, args, workers):
    """
    Process games in parallel using a pool of worker processes.

    Architecture:
      - Main thread reads PGN stream and accumulates games into batches
      - Batches are dispatched to workers via imap_unordered
      - Workers do the expensive SAN→UCI conversion
      - Main thread writes results and merges stats
    """
    stats = _empty_stats()
    start_time = time.time()
    batch_size = args.batch_size

    print(
        f"[parse] parallel mode: {workers} workers, batch_size={batch_size}",
        file=sys.stderr,
    )

    game_iter = read_games(pgn_stream)
    batches = batch_games(game_iter, batch_size)

    def make_work_items():
        for batch in batches:
            yield (batch, args.min_moves, args.max_moves, args.validate)

    batches_done = 0

    with mp.Pool(processes=workers) as pool:
        for output_lines, partial_stats in pool.imap_unordered(
            _process_batch, make_work_items(), chunksize=1
        ):
            _merge_stats(stats, partial_stats)
            for line in output_lines:
                output_file.write(line)

            batches_done += 1
            if (batches_done * batch_size) % args.log_every < batch_size:
                elapsed = time.time() - start_time
                rate = stats["total_games"] / max(elapsed, 0.001)
                print(
                    f"[parse] {stats['total_games']:,} games processed, "
                    f"kept {stats['kept_games']:,} "
                    f"({100 * stats['kept_games'] / max(stats['total_games'], 1):.1f}%) "
                    f"| {rate:.0f} games/sec | {workers} workers",
                    file=sys.stderr,
                )

    stats["elapsed_seconds"] = round(time.time() - start_time, 1)
    return stats


# ── Output ────────────────────────────────────────────────────────────────────

def print_stats_summary(stats):
    total = stats["total_games"]
    kept = stats["kept_games"]

    print("\n=== Parse Summary ===", file=sys.stderr)
    print(f"Total games:        {total:,}", file=sys.stderr)
    print(f"Kept:               {kept:,} ({100*kept/max(total,1):.1f}%)", file=sys.stderr)
    print(f"Skipped (too short):{stats['skipped_too_short']:,}", file=sys.stderr)
    print(f"Skipped (too long): {stats['skipped_too_long']:,}", file=sys.stderr)
    print(f"Skipped (illegal):  {stats['skipped_illegal_move']:,}", file=sys.stderr)
    print(f"Skipped (parse err):{stats['skipped_parse_error']:,}", file=sys.stderr)
    print(f"Skipped (bad result):{stats['skipped_bad_result']:,}", file=sys.stderr)
    print(f"Elapsed:            {stats['elapsed_seconds']}s", file=sys.stderr)

    print("\nResult distribution:", file=sys.stderr)
    for result, count in sorted(stats["result_counts"].items()):
        pct = 100 * count / max(kept, 1)
        print(f"  {result:>10}: {count:,} ({pct:.1f}%)", file=sys.stderr)

    print("\nMove length distribution (full moves):", file=sys.stderr)
    for bucket in sorted(stats["move_length_histogram"].keys()):
        count = stats["move_length_histogram"][bucket]
        bar = "█" * (count * 40 // max(stats["move_length_histogram"].values()))
        print(f"  {bucket:>4}-{bucket+9}: {bar} {count:,}", file=sys.stderr)

    print("\nAvg ELO distribution:", file=sys.stderr)
    for bucket in sorted(stats["elo_histogram"].keys()):
        count = stats["elo_histogram"][bucket]
        bar = "█" * (count * 40 // max(stats["elo_histogram"].values()))
        print(f"  {bucket:>4}-{bucket+99}: {bar} {count:,}", file=sys.stderr)

    if stats.get("min_elo_histogram"):
        print("\nMin(player) ELO distribution:", file=sys.stderr)
        for bucket in sorted(stats["min_elo_histogram"].keys()):
            count = stats["min_elo_histogram"][bucket]
            bar = "█" * (count * 40 // max(stats["min_elo_histogram"].values()))
            print(f"  {bucket:>4}-{bucket+99}: {bar} {count:,}", file=sys.stderr)


def main():
    args = parse_args()

    workers = args.workers
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)

    # Default stats path: keep stable name even if output is .gz
    if args.stats_output:
        stats_path = args.stats_output
    else:
        out_p = Path(args.output)
        if out_p.suffix == ".gz":
            stats_path = str(out_p.with_suffix("").with_suffix(".stats.json"))
        else:
            stats_path = args.output + ".stats.json"

    mode_str = "validate" if args.validate else "fast"
    print(f"Parsing PGN → UCI moves ({mode_str} mode, {workers} workers)", file=sys.stderr)
    print(f"  min_moves={args.min_moves}, max_moves={args.max_moves}", file=sys.stderr)
    print(f"  output={args.output}", file=sys.stderr)
    print(f"  stats={stats_path}\n", file=sys.stderr)

    if args.input:
        pgn_stream = open(args.input, "r", encoding="utf-8", errors="replace")
    else:
        pgn_stream = sys.stdin

    if str(args.output).endswith(".gz"):
        out_open = lambda p: gzip.open(p, "wt", encoding="utf-8", compresslevel=6)  # noqa: E731
    else:
        out_open = lambda p: open(p, "w", encoding="utf-8")  # noqa: E731

    with out_open(args.output) as out_f:
        if workers <= 1:
            stats = parse_games_sequential(pgn_stream, out_f, args)
        else:
            stats = parse_games_parallel(pgn_stream, out_f, args, workers)

    if args.input:
        pgn_stream.close()

    print_stats_summary(stats)

    stats["mode"] = mode_str
    stats["workers"] = workers
    stats["move_length_histogram"] = dict(
        sorted((int(k), v) for k, v in stats["move_length_histogram"].items()))
    stats["elo_histogram"] = dict(
        sorted((int(k), v) for k, v in stats["elo_histogram"].items()))
    stats["min_elo_histogram"] = dict(
        sorted((int(k), v) for k, v in stats.get("min_elo_histogram", {}).items()))

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written to {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
