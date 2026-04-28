r"""
Filter high-ELO games from a Lichess PGN stream.

Usage:
    zstdcat lichess_db.pgn.zst | python filter_games.py > filtered.pgn
    zstdcat lichess_db.pgn.zst | python filter_games.py --min-elo 2000 --time-controls 600 180 > filtered.pgn
    zstdcat lichess_db.pgn.zst | python filter_games.py | sed 's/ { \[%[^}]*\] }//g' > filtered.pgn

No decompression needed. Reads stdin, writes stdout.
Progress is logged to stderr so it doesn't pollute the output file.
"""

import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-elo", type=int, default=1800,
                        help="Minimum ELO for both players (default: 1800)")
    parser.add_argument("--time-controls", type=int, nargs="+",
                        default=None,
                        help="Whitelist of base time controls in seconds e.g. 600 180 300. "
                             "If not set, bullet (<180s base) is excluded and everything else passes.")
    parser.add_argument("--no-bots", action="store_true", default=True,
                        help="Exclude games where either player is a BOT (default: True)")
    parser.add_argument("--standard-only", action="store_true", default=True,
                        help="Exclude variant games (default: True)")
    parser.add_argument("--log-every", type=int, default=100_000,
                        help="Log progress every N games (default: 100000)")
    return parser.parse_args()


def parse_time_control(tc_str):
    """
    Parse a PGN TimeControl string like '600+5' or '300+0'.
    Returns base time in seconds, or None if unparseable.
    """
    try:
        base = tc_str.split("+")[0]
        return int(base)
    except Exception:
        return None


def passes_filters(headers, args):
    # Standard chess only
    if args.standard_only and headers.get("Variant", "Standard") != "Standard":
        return False

    # No bots
    if args.no_bots:
        if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
            return False

    # ELO filter
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
    except ValueError:
        return False
    if white_elo < args.min_elo or black_elo < args.min_elo:
        return False

    # Time control filter
    tc_str = headers.get("TimeControl", "")
    base_time = parse_time_control(tc_str)
    if base_time is None:
        return False

    if args.time_controls is not None:
        # Explicit whitelist
        if base_time not in args.time_controls:
            return False
    else:
        # Default: exclude bullet (base < 180s)
        if base_time < 180:
            return False

    return True


def main():
    args = parse_args()

    total = 0
    kept = 0
    current_headers = {}
    current_lines = []
    in_game = False

    for raw_line in sys.stdin.buffer:
        try:
            line = raw_line.decode("utf-8", errors="replace")
        except Exception:
            continue

        stripped = line.strip()

        if stripped.startswith("[Event "):
            # Start of a new game — flush previous if keeping
            if in_game and passes_filters(current_headers, args):
                sys.stdout.write("".join(current_lines))
                kept += 1

            if in_game:
                total += 1
                if total % args.log_every == 0:
                    print(f"Processed {total:,} games, kept {kept:,} "
                          f"({100*kept/total:.1f}%)", file=sys.stderr)

            # Reset for new game
            current_headers = {}
            current_lines = [line]
            in_game = True

        elif in_game:
            current_lines.append(line)

            # Parse header tags
            if stripped.startswith("[") and stripped.endswith("]"):
                # e.g. [WhiteElo "2100"]
                try:
                    key = stripped[1:stripped.index(" ")]
                    value = stripped[stripped.index('"') + 1:stripped.rindex('"')]
                    current_headers[key] = value
                except Exception:
                    pass

    # Flush last game
    if in_game:
        total += 1
        if passes_filters(current_headers, args):
            sys.stdout.write("".join(current_lines))
            kept += 1

    print(f"\nDone. Processed {total:,} games, kept {kept:,} "
          f"({100*kept/total:.1f}% kept)", file=sys.stderr)


if __name__ == "__main__":
    main()
