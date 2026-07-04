"""Tournament bracket logic for full path prediction and round progression."""

from __future__ import annotations

import pandas as pd
from typing import Any
from datetime import datetime, timedelta

from feature_engineering.features import build_fixture_features
from models.predict import predict_fixtures


def predict_full_bracket(
    fixtures: pd.DataFrame,
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
    classifier: dict[str, Any],
    regressors: dict[str, Any],
) -> dict[str, Any]:
    """Predict the entire tournament bracket from the current fixtures to the final."""

    bracket: dict[str, Any] = {}
    current_fixtures = fixtures.copy()

    for _ in range(4):  # Max 4 knockout rounds
        if current_fixtures.empty:
            break

        stage_name = current_fixtures.iloc[0].get("stage", "Unknown")
        features = build_fixture_features(current_fixtures, matches, teams, players)
        preds = predict_fixtures(features, classifier, regressors)

        round_matches = []
        winners = []
        for _, row in preds.iterrows():
            # In knockouts, team with higher win prob advances regardless of draw prob
            home_prob = float(row.get("prob_home_win", 0))
            away_prob = float(row.get("prob_away_win", 0))
            winner = row["home_team"] if home_prob >= away_prob else row["away_team"]
            winners.append(winner)

            round_matches.append(
                {
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "prob_home": home_prob,
                    "prob_away": away_prob,
                    "predicted_score": row["predicted_score"],
                    "winner": winner,
                    "stage": stage_name,
                }
            )

        bracket[stage_name] = round_matches

        # Generate next round fixtures
        if len(winners) >= 2:
            next_fixtures = []
            if stage_name == "Round of 16":
                next_stage = "Quarter-final"
            elif stage_name == "Quarter-final":
                next_stage = "Semi-final"
            else:
                next_stage = "Final"

            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    next_fixtures.append(
                        {
                            "date": "TBD",
                            "home_team": winners[i],
                            "away_team": winners[i + 1],
                            "stage": next_stage,
                            "neutral_venue": True,
                        }
                    )
            current_fixtures = pd.DataFrame(next_fixtures)
        else:
            # We have a champion!
            bracket["Champion"] = winners[0]
            break

    return bracket


def advance_tournament_round(
    fixtures: pd.DataFrame,
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    players: pd.DataFrame,
    classifier: dict[str, Any],
    regressors: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate the results of the current fixtures and return updated matches and new fixtures.
    """
    if fixtures.empty:
        return matches, fixtures

    stage_name = fixtures.iloc[0].get("stage", "Unknown")
    features = build_fixture_features(fixtures, matches, teams, players)
    preds = predict_fixtures(features, classifier, regressors)

    # 1. Create simulated completed matches
    completed = []
    winners = []
    for _, row in preds.iterrows():
        home_prob = float(row.get("prob_home_win", 0))
        away_prob = float(row.get("prob_away_win", 0))
        home_win = home_prob >= away_prob
        winner = row["home_team"] if home_win else row["away_team"]
        winners.append(winner)

        # Parse scoreline
        score = str(row["predicted_score"]).split("-")
        home_goals = int(score[0]) if len(score) == 2 else (1 if home_win else 0)
        away_goals = int(score[1]) if len(score) == 2 else (0 if home_win else 1)
        
        # Ensure winner actually has more goals (no draws in knockouts)
        if home_goals == away_goals:
            if home_win:
                home_goals += 1
            else:
                away_goals += 1
        elif (home_goals > away_goals) and not home_win:
            # Swap if score contradicts prob winner
            home_goals, away_goals = away_goals, home_goals

        completed.append(
            {
                "date": row.get("date", datetime.now().isoformat()),
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg": row.get("expected_home_goals", 1.0),
                "away_xg": row.get("expected_away_goals", 1.0),
                "home_shots": 10,
                "away_shots": 10,
                "home_possession": 50.0,
                "away_possession": 50.0,
                "home_pass_accuracy": 80.0,
                "away_pass_accuracy": 80.0,
                "home_corners": 5,
                "away_corners": 5,
                "home_cards": 1,
                "away_cards": 1,
                "neutral_venue": True,
                "stage": stage_name,
                "status": "FINISHED",
                "result": "home_win" if home_win else "away_win",
            }
        )

    new_matches = pd.concat([matches, pd.DataFrame(completed)], ignore_index=True)

    # 2. Create next round fixtures
    next_fixtures_data = []
    if len(winners) >= 2:
        if stage_name == "Round of 16":
            next_stage = "Quarter-final"
        elif stage_name == "Quarter-final":
            next_stage = "Semi-final"
        else:
            next_stage = "Final"

        for i in range(0, len(winners), 2):
            if i + 1 < len(winners):
                # Fake date + 4 days
                date_str = str(fixtures.iloc[0].get("date", datetime.now().isoformat()))
                try:
                    match_date = datetime.fromisoformat(date_str) + timedelta(days=4)
                    new_date_str = match_date.date().isoformat()
                except:
                    new_date_str = "TBD"

                next_fixtures_data.append(
                    {
                        "date": new_date_str,
                        "home_team": winners[i],
                        "away_team": winners[i + 1],
                        "stage": next_stage,
                        "neutral_venue": True,
                    }
                )

    new_fixtures = pd.DataFrame(next_fixtures_data)
    return new_matches, new_fixtures
