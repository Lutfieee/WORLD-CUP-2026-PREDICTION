"""Unit tests for feature engineering and prediction math."""

from __future__ import annotations

from feature_engineering.features import build_feature_table, model_feature_columns
from models.predict import scoreline_distribution
from preprocessing.cleaning import clean_matches
from scraping.sources import generate_seed_matches, team_profiles_frame


def test_feature_table_contains_derived_columns() -> None:
    """Feature table should include core derived differences."""

    matches = clean_matches(generate_seed_matches(seed=1, seasons=1))
    features = build_feature_table(matches, team_profiles_frame())
    columns = model_feature_columns(features)
    assert "elo_difference" in columns
    assert "xg_difference" in columns
    assert not features.empty


def test_scoreline_distribution_is_normalized() -> None:
    """Scoreline probabilities should sum to one after truncation normalization."""

    dist = scoreline_distribution(1.8, 1.1)
    assert abs(dist["probability"].sum() - 1.0) < 1e-9
    assert dist.iloc[0]["probability"] >= dist.iloc[-1]["probability"]
