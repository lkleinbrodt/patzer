"""
eval/elo.py — Bradley-Terry MLE Elo estimation from stored game records.

Stockfish players (name matching "stockfish:NNNN") are anchored at their
configured Elo and not fitted. Patzer models are free parameters initialized
at 1500 and updated iteratively until convergence.
"""

import math
import re
from typing import NamedTuple


class PlayerRating(NamedTuple):
    name: str
    elo: float
    stderr: float
    games: int
    wins: int
    losses: int
    draws: int


_SF_PATTERN = re.compile(r"^stockfish:(\d+)$")


def _is_stockfish(name: str) -> bool:
    return bool(_SF_PATTERN.match(name))


def _sf_elo(name: str) -> float:
    return float(_SF_PATTERN.match(name).group(1))


def _elo_win_prob(r_white: float, r_black: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_black - r_white) / 400.0))


def compute_ratings(games: list[dict]) -> list[PlayerRating]:
    """
    Fit Bradley-Terry ratings from game records.

    Returns ratings sorted best → worst. Players with no games are omitted.
    Stockfish players are anchored; only patzer models are fitted.
    """
    if not games:
        return []

    # Collect players and raw stats
    all_players: set[str] = set()
    for g in games:
        all_players.add(g["white"])
        all_players.add(g["black"])

    # Per-player W/L/D totals (from their own perspective)
    stats: dict[str, dict] = {
        p: {"games": 0, "wins": 0, "losses": 0, "draws": 0} for p in all_players
    }
    for g in games:
        w, b, r = g["white"], g["black"], g["result"]
        stats[w]["games"] += 1
        stats[b]["games"] += 1
        if r == "1-0":
            stats[w]["wins"] += 1
            stats[b]["losses"] += 1
        elif r == "0-1":
            stats[b]["wins"] += 1
            stats[w]["losses"] += 1
        else:
            stats[w]["draws"] += 1
            stats[b]["draws"] += 1

    # Separate anchored vs free players
    anchored = {p: _sf_elo(p) for p in all_players if _is_stockfish(p)}
    free = [p for p in all_players if not _is_stockfish(p)]

    # Initialize free player ratings
    ratings: dict[str, float] = {**anchored, **{p: 1500.0 for p in free}}

    # Iterative BT update (coordinate ascent, one player at a time)
    for _ in range(1000):
        max_delta = 0.0
        for player in free:
            # For each opponent, compute expected and observed score
            num = 0.0  # observed score
            den = 0.0  # sum of win-prob derivatives (Fisher info approximation)
            for g in games:
                w, b, r = g["white"], g["black"], g["result"]
                if player not in (w, b):
                    continue
                is_white = player == w
                opp = b if is_white else w
                r_p = ratings[player]
                r_o = ratings[opp]
                p_win = _elo_win_prob(r_p, r_o) if is_white else _elo_win_prob(r_o, r_p)
                if not is_white:
                    p_win = 1.0 - p_win
                # Observed score from player's perspective
                if r == "1/2-1/2":
                    obs = 0.5
                elif (r == "1-0" and is_white) or (r == "0-1" and not is_white):
                    obs = 1.0
                else:
                    obs = 0.0
                num += obs - p_win
                den += p_win * (1.0 - p_win)

            if den < 1e-9:
                continue
            # Newton step in Elo space (400/ln10 converts log-odds to Elo)
            delta = (400.0 / math.log(10)) * (num / den)
            # Dampen large jumps
            delta = max(-100.0, min(100.0, delta))
            ratings[player] += delta
            max_delta = max(max_delta, abs(delta))

        if max_delta < 0.01:
            break

    # Estimate stderr from Fisher information (diagonal approximation)
    def _stderr(player: str) -> float:
        info = 0.0
        for g in games:
            w, b = g["white"], g["black"]
            if player not in (w, b):
                continue
            is_white = player == w
            opp = b if is_white else w
            r_p, r_o = ratings[player], ratings[opp]
            p_win = _elo_win_prob(r_p, r_o) if is_white else 1.0 - _elo_win_prob(r_o, r_p)
            info += p_win * (1.0 - p_win)
        if info < 1e-9:
            return float("nan")
        # Variance in log-odds units → Elo units
        return (400.0 / math.log(10)) / math.sqrt(info)

    result = []
    for p in all_players:
        s = stats[p]
        elo = ratings[p]
        se = 0.0 if _is_stockfish(p) else _stderr(p)
        result.append(PlayerRating(
            name=p,
            elo=elo,
            stderr=se,
            games=s["games"],
            wins=s["wins"],
            losses=s["losses"],
            draws=s["draws"],
        ))

    result.sort(key=lambda r: -r.elo)
    return result
