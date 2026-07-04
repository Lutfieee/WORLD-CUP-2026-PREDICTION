"""Monte Carlo tournament simulation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

def simulate_knockout(predictions: pd.DataFrame, simulations: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Simulate knockout paths using predicted scoreline distributions."""

    rng = np.random.default_rng(seed)
    teams = sorted(set(predictions["home_team"]).union(predictions["away_team"]))
    counters = {team: {"round_of_16": 0, "quarter_final": 0, "semi_final": 0, "runner_up": 0, "champion": 0} for team in teams}
    match_rows = predictions.to_dict("records")
    for _ in range(simulations):
        winners = []
        for match in match_rows:
            counters[match["home_team"]]["round_of_16"] += 1
            counters[match["away_team"]]["round_of_16"] += 1
            probs = normalize_triplet(
                float(match.get("prob_home_win", 0.34)),
                float(match.get("prob_draw", 0.33)),
                float(match.get("prob_away_win", 0.33)),
            )
            sampled = rng.choice(["home", "draw", "away"], p=probs)
            if sampled == "home":
                winners.append(match["home_team"])
            elif sampled == "away":
                winners.append(match["away_team"])
            else:
                home_prob = float(match.get("prob_home_win", 0.34))
                away_prob = float(match.get("prob_away_win", 0.33))
                winners.append(rng.choice([match["home_team"], match["away_team"]], p=normalize_pair(home_prob, away_prob)))
        champion = run_bracket_rounds(winners, rng, counters)
        counters[champion]["champion"] += 1
    rows = []
    for team, counts in counters.items():
        rows.append({"team": team, **{key: value / simulations for key, value in counts.items()}})
    return pd.DataFrame(rows).sort_values("champion", ascending=False).reset_index(drop=True)


def normalize_pair(a: float, b: float) -> list[float]:
    """Normalize two non-negative values into a probability pair."""

    total = max(a + b, 1e-9)
    return [a / total, b / total]


def normalize_triplet(a: float, b: float, c: float) -> list[float]:
    """Normalize three non-negative values into probabilities."""

    total = max(a + b + c, 1e-9)
    return [a / total, b / total, c / total]


def run_bracket_rounds(winners: list[str], rng: np.random.Generator, counters: dict[str, dict[str, int]]) -> str:
    """Advance sampled teams through quarter-final, semi-final, and final rounds."""

    current = winners[:]
    stages = ["quarter_final", "semi_final", "runner_up"]
    for stage in stages:
        next_round = []
        for i in range(0, len(current), 2):
            team_a, team_b = current[i], current[i + 1]
            counters[team_a][stage] += 1
            counters[team_b][stage] += 1
            next_round.append(str(rng.choice([team_a, team_b])))
        current = next_round
    return current[0]
