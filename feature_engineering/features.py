"""Feature engineering for team, match, and fixture prediction datasets."""

from __future__ import annotations

import pandas as pd


BASE_TEAM_COLUMNS = [
    "fifa_rank",
    "elo",
    "attack",
    "defense",
    "possession",
    "pass_accuracy",
    "tournament_experience",
]


def build_team_form(matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rolling team form from historical match data."""

    rows = []
    for _, row in matches.iterrows():
        rows.extend(
            [
                {
                    "date": row["date"],
                    "team": row["home_team"],
                    "goals_for": row["home_goals"],
                    "goals_against": row["away_goals"],
                    "xg_for": row["home_xg"],
                    "xg_against": row["away_xg"],
                    "shots": row["home_shots"],
                    "possession_match": row["home_possession"],
                    "pass_accuracy_match": row["home_pass_accuracy"],
                    "win": int(row["result"] == "home_win"),
                    "draw": int(row["result"] == "draw"),
                    "clean_sheet": int(row["away_goals"] == 0),
                },
                {
                    "date": row["date"],
                    "team": row["away_team"],
                    "goals_for": row["away_goals"],
                    "goals_against": row["home_goals"],
                    "xg_for": row["away_xg"],
                    "xg_against": row["home_xg"],
                    "shots": row["away_shots"],
                    "possession_match": row["away_possession"],
                    "pass_accuracy_match": row["away_pass_accuracy"],
                    "win": int(row["result"] == "away_win"),
                    "draw": int(row["result"] == "draw"),
                    "clean_sheet": int(row["home_goals"] == 0),
                },
            ]
        )
    long = pd.DataFrame(rows).sort_values(["team", "date"])
    agg = (
        long.groupby("team")
        .tail(12)
        .groupby("team")
        .agg(
            avg_goals=("goals_for", "mean"),
            avg_goals_against=("goals_against", "mean"),
            avg_xg=("xg_for", "mean"),
            avg_xga=("xg_against", "mean"),
            shot_accuracy=("goals_for", lambda x: x.sum()),
            win_rate=("win", "mean"),
            draw_rate=("draw", "mean"),
            recent_form=("win", lambda x: x.tail(5).mean()),
            clean_sheet_rate=("clean_sheet", "mean"),
        )
        .reset_index()
    )
    shots = long.groupby("team").tail(12).groupby("team")["shots"].sum()
    goals = long.groupby("team").tail(12).groupby("team")["goals_for"].sum()
    agg["conversion_rate"] = agg["team"].map((goals / shots.clip(lower=1)).fillna(0)).fillna(0)
    return agg


def build_feature_table(matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Build supervised learning rows from historical matches."""

    form = build_team_form(matches)
    team_features = teams.merge(form, on="team", how="left").fillna(0)
    rows = []
    for _, row in matches.iterrows():
        home = team_features.loc[team_features["team"] == row["home_team"]].iloc[0]
        away = team_features.loc[team_features["team"] == row["away_team"]].iloc[0]
        features = build_match_features(home, away, bool(row["neutral_venue"]))
        features.update(
            {
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "result": row["result"],
                "home_goals": row["home_goals"],
                "away_goals": row["away_goals"],
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def build_match_features(home: pd.Series, away: pd.Series, neutral_venue: bool = True) -> dict[str, float | int | bool]:
    """Create model features for a single match from two team profiles."""

    feature: dict[str, float | int | bool] = {"neutral_venue": int(neutral_venue)}
    for col in BASE_TEAM_COLUMNS + [
        "avg_goals",
        "avg_goals_against",
        "avg_xg",
        "avg_xga",
        "win_rate",
        "recent_form",
        "clean_sheet_rate",
        "conversion_rate",
    ]:
        feature[f"home_{col}"] = float(home.get(col, 0))
        feature[f"away_{col}"] = float(away.get(col, 0))
    feature["elo_difference"] = feature["home_elo"] - feature["away_elo"]
    feature["ranking_difference"] = feature["away_fifa_rank"] - feature["home_fifa_rank"]
    feature["xg_difference"] = feature["home_avg_xg"] - feature["away_avg_xg"]
    feature["form_difference"] = feature["home_recent_form"] - feature["away_recent_form"]
    feature["possession_difference"] = feature["home_possession"] - feature["away_possession"]
    feature["passing_difference"] = feature["home_pass_accuracy"] - feature["away_pass_accuracy"]
    feature["defense_difference"] = feature["away_defense"] - feature["home_defense"]
    return feature


def build_fixture_features(fixtures: pd.DataFrame, matches: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Build prediction features for scheduled fixtures."""

    form = build_team_form(matches)
    team_features = teams.merge(form, on="team", how="left").fillna(0)
    rows = []
    for _, fixture in fixtures.iterrows():
        home = team_features.loc[team_features["team"] == fixture["home_team"]].iloc[0]
        away = team_features.loc[team_features["team"] == fixture["away_team"]].iloc[0]
        features = build_match_features(home, away, bool(fixture["neutral_venue"]))
        features.update(
            {
                "date": fixture["date"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "stage": fixture.get("stage", "Scheduled"),
            }
        )
        rows.append(features)
    return pd.DataFrame(rows)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric model columns excluding identifiers and targets."""

    excluded = {"date", "home_team", "away_team", "result", "home_goals", "away_goals", "stage"}
    return [col for col in frame.columns if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])]
