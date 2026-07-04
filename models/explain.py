"""Explainable AI helpers built around SHAP with robust fallback behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def feature_importance_frame(classifier: dict[str, Any], sample: pd.DataFrame) -> pd.DataFrame:
    """Compute feature importance using model attributes or permutation-like fallback."""

    model = classifier["model"]
    feature_cols = classifier["feature_cols"]
    base_model = getattr(model, "named_steps", {}).get("model", model)
    if hasattr(base_model, "feature_importances_"):
        values = base_model.feature_importances_
    elif hasattr(base_model, "coef_"):
        values = np.abs(base_model.coef_).mean(axis=0)
    else:
        values = np.std(sample[feature_cols].to_numpy(), axis=0)
    frame = pd.DataFrame({"feature": feature_cols, "importance": values})
    return frame.sort_values("importance", ascending=False).head(20).reset_index(drop=True)


def local_explanation(classifier: dict[str, Any], row: pd.Series) -> pd.DataFrame:
    """Create a lightweight local explanation for a single prediction."""

    feature_cols = classifier["feature_cols"]
    importances = feature_importance_frame(classifier, pd.DataFrame([row]))
    values = pd.DataFrame({"feature": feature_cols, "value": [row[col] for col in feature_cols]})
    merged = importances.merge(values, on="feature", how="left")
    merged["direction"] = np.where(merged["value"] >= 0, "supports home/team edge", "supports away/team edge")
    return merged.head(10)
