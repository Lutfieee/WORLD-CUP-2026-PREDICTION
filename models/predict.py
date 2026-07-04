"""Prediction utilities for match outcome and scoreline probabilities."""

from __future__ import annotations

from math import exp, factorial
from typing import Any

import numpy as np
import pandas as pd


def poisson_pmf(k: int, lam: float) -> float:
    """Return Poisson probability mass for scoreline calculations."""

    lam = max(float(lam), 0.05)
    return (lam**k * exp(-lam)) / factorial(k)


def predict_fixtures(feature_frame: pd.DataFrame, classifier: dict[str, Any], regressors: dict[str, Any]) -> pd.DataFrame:
    """Predict outcome probabilities, expected goals, and likely scorelines."""

    feature_cols = classifier["feature_cols"]
    x = feature_frame[feature_cols]
    probs = classifier["model"].predict_proba(x)
    classes = classifier["label_encoder"].inverse_transform(np.arange(probs.shape[1]))
    predictions = feature_frame[["date", "home_team", "away_team", "stage"]].copy()
    for idx, class_name in enumerate(classes):
        predictions[f"prob_{class_name}"] = probs[:, idx]
    home_xg = np.clip(regressors["targets"]["home_goals"]["model"].predict(x), 0.05, 5)
    away_xg = np.clip(regressors["targets"]["away_goals"]["model"].predict(x), 0.05, 5)
    predictions["expected_home_goals"] = home_xg.round(2)
    predictions["expected_away_goals"] = away_xg.round(2)
    score_rows = [scoreline_distribution(h, a).iloc[0].to_dict() for h, a in zip(home_xg, away_xg)]
    score_frame = pd.DataFrame(score_rows)
    predictions["predicted_score"] = score_frame["scoreline"]
    predictions["scoreline_probability"] = score_frame["probability"]
    predictions["confidence"] = predictions[[c for c in predictions.columns if c.startswith("prob_")]].max(axis=1)
    return predictions


def scoreline_distribution(home_xg: float, away_xg: float, max_goals: int = 6) -> pd.DataFrame:
    """Return a ranked probability distribution of scorelines."""

    rows = []
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            rows.append(
                {
                    "scoreline": f"{home_goals}-{away_goals}",
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "probability": poisson_pmf(home_goals, home_xg) * poisson_pmf(away_goals, away_xg),
                }
            )
    frame = pd.DataFrame(rows).sort_values("probability", ascending=False)
    frame["probability"] = frame["probability"] / frame["probability"].sum()
    return frame.reset_index(drop=True)
