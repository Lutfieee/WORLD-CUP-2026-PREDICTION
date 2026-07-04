"""Data cleaning, validation, and storage utilities."""

from __future__ import annotations

import logging
import sqlite3

import numpy as np
import pandas as pd

from config.settings import settings

LOGGER = logging.getLogger(__name__)


def clean_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Clean match-level data and create the target outcome."""

    frame = matches.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "home_team", "away_team"]).drop_duplicates()
    numeric_cols = [
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "home_shots",
        "away_shots",
        "home_possession",
        "away_possession",
        "home_pass_accuracy",
        "away_pass_accuracy",
        "home_corners",
        "away_corners",
        "home_cards",
        "away_cards",
    ]
    for col in numeric_cols:
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
            frame[col] = frame[col].fillna(frame[col].median())
            low, high = frame[col].quantile([0.01, 0.99])
            frame[col] = frame[col].clip(low, high)
    frame["neutral_venue"] = frame.get("neutral_venue", True).astype(bool)
    frame["goal_difference"] = frame["home_goals"] - frame["away_goals"]
    frame["result"] = np.select(
        [frame["goal_difference"] > 0, frame["goal_difference"] < 0],
        ["home_win", "away_win"],
        default="draw",
    )
    return frame.sort_values("date").reset_index(drop=True)


def clean_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Clean scheduled fixture data."""

    frame = fixtures.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date", "home_team", "away_team"])
    frame["neutral_venue"] = frame.get("neutral_venue", True).astype(bool)
    return frame.reset_index(drop=True)


def store_processed_tables(tables: dict[str, pd.DataFrame]) -> None:
    """Persist processed tables to CSV and SQLite."""

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = settings.data_dir / "worldcup2026.db"
    with sqlite3.connect(sqlite_path) as conn:
        for name, frame in tables.items():
            csv_path = settings.processed_dir / f"{name}.csv"
            frame.to_csv(csv_path, index=False)
            frame.to_sql(name, conn, if_exists="replace", index=False)
            LOGGER.info("Stored table %s in CSV and SQLite", name)
