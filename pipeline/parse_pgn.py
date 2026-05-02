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

Stats are written to a separate JSON file for inspection.
"""

import argparse
import json
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
    return parser.parse_args()


# Strip PGN comments like { [%eval 0.17] [%clk 0:05:00] }
COMMENT_RE = re.compile(r"\{[^}]*\}")

# Valid result strings
VALID_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def clean_pgn_moves(moves_text):
    """Remove comments and move numbers, return just the SAN move tokens."""
    # Remove comments
    text = COMMENT_RE.sub("", moves_text)
    # Remove move numbers (e.g. "1." "12." "5...")
    text = re.sub(r"\d+\.+", "", text)
    # Remove result string at end
    for r in VALID_RESULTS:
        text = text.replace(r, "")
    return text.split()


def pgn_to_uci(pgn_text):
    """
    Convert a single PGN game string to a list of UCI moves.
    Returns (uci_moves, error_message). On failure, uci_moves is None.
    Relies on python-chess PGN parsing for legality checks.
    Illegal or ambiguous SAN moves are captured in game.errors.
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


def parse_games(pgn_stream, output_file, args):
    """
    Stream through a PGN file, parse each game, write UCI sequences.
    Returns a stats dict.
    """
    stats = {
        "total_games": 0,
        "kept_games": 0,
        "skipped_illegal_move": 0,
        "skipped_too_short": 0,
        "skipped_too_long": 0,
        "skipped_bad_result": 0,
        "skipped_parse_error": 0,
        "result_counts": {"1-0": 0, "0-1": 0, "1/2-1/2": 0},
        "move_length_histogram": {},  # bucket by 10s
        "elo_histogram": {},           # bucket by 100s
    }

    start_time = time.time()

    # Buffer lines into individual game strings
    current_game_lines = []
    current_headers = {}

    def flush_game():
        nonlocal current_game_lines, current_headers

        if not current_game_lines:
            return

        stats["total_games"] += 1
        if stats["total_games"] % args.log_every == 0:
            elapsed = time.time() - start_time
            rate = stats["total_games"] / elapsed
            print(
                f"  {stats['total_games']:,} games processed, "
                f"{stats['kept_games']:,} kept "
                f"({100 * stats['kept_games'] / stats['total_games']:.1f}%) "
                f"| {rate:.0f} games/sec",
                file=sys.stderr,
            )

        pgn_text = "".join(current_game_lines)

        # Extract result and ELO from already-parsed headers
        result = current_headers.get("Result", "")
        if result not in VALID_RESULTS:
            stats["skipped_bad_result"] += 1
            current_game_lines = []
            current_headers = {}
            return

        try:
            white_elo = int(current_headers.get("WhiteElo", 0))
            black_elo = int(current_headers.get("BlackElo", 0))
        except ValueError:
            white_elo, black_elo = 0, 0

        # Parse moves
        uci_moves, error = pgn_to_uci(pgn_text)

        if uci_moves is None:
            if "illegal" in (error or ""):
                stats["skipped_illegal_move"] += 1
            else:
                stats["skipped_parse_error"] += 1
            current_game_lines = []
            current_headers = {}
            return

        n_moves = len(uci_moves)

        if n_moves < args.min_moves:
            stats["skipped_too_short"] += 1
            current_game_lines = []
            current_headers = {}
            return

        if n_moves > args.max_moves:
            stats["skipped_too_long"] += 1
            current_game_lines = []
            current_headers = {}
            return

        # Write output line: result white_elo black_elo move1 move2 ...
        line = f"{result} {white_elo} {black_elo} {' '.join(uci_moves)}\n"
        output_file.write(line)

        # Update stats
        stats["kept_games"] += 1
        stats["result_counts"][result] = stats["result_counts"].get(result, 0) + 1

        bucket_moves = (n_moves // 10) * 10
        stats["move_length_histogram"][bucket_moves] = \
            stats["move_length_histogram"].get(bucket_moves, 0) + 1

        avg_elo = (white_elo + black_elo) // 2
        bucket_elo = (avg_elo // 100) * 100
        stats["elo_histogram"][bucket_elo] = \
            stats["elo_histogram"].get(bucket_elo, 0) + 1

        current_game_lines = []
        current_headers = {}

    for raw_line in pgn_stream:
        if isinstance(raw_line, bytes):
            line = raw_line.decode("utf-8", errors="replace")
        else:
            line = raw_line

        stripped = line.strip()

        if stripped.startswith("[Event "):
            flush_game()
            current_game_lines = [line]
            current_headers = {}
        else:
            if current_game_lines:
                current_game_lines.append(line)

            # Parse headers as we go
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    key = stripped[1:stripped.index(" ")]
                    value = stripped[stripped.index('"') + 1:stripped.rindex('"')]
                    current_headers[key] = value
                except Exception:
                    pass

    # Flush final game
    flush_game()

    stats["elapsed_seconds"] = round(time.time() - start_time, 1)
    return stats


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


def main():
    args = parse_args()

    stats_path = args.stats_output or (args.output + ".stats.json")

    print(f"Parsing PGN → UCI moves", file=sys.stderr)
    print(f"  min_moves={args.min_moves}, max_moves={args.max_moves}", file=sys.stderr)
    print(f"  output={args.output}", file=sys.stderr)
    print(f"  stats={stats_path}\n", file=sys.stderr)

    if args.input:
        pgn_stream = open(args.input, "r", encoding="utf-8", errors="replace")
    else:
        pgn_stream = sys.stdin

    with open(args.output, "w", encoding="utf-8") as out_f:
        stats = parse_games(pgn_stream, out_f, args)

    if args.input:
        pgn_stream.close()

    print_stats_summary(stats)

    # Sort histogram keys for clean JSON output
    stats["move_length_histogram"] = dict(
        sorted((int(k), v) for k, v in stats["move_length_histogram"].items()))
    stats["elo_histogram"] = dict(
        sorted((int(k), v) for k, v in stats["elo_histogram"].items()))

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats written to {stats_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
