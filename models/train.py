"""Model training and evaluation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - older scikit-learn
    root_mean_squared_error = None
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.settings import settings
from feature_engineering.features import model_feature_columns

LOGGER = logging.getLogger(__name__)


def _optional_classifiers(random_state: int) -> dict[str, Any]:
    """Load optional gradient boosting classifiers when dependencies exist."""

    models: dict[str, Any] = {
        "logistic_regression": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]
        ),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            eval_metric="mlogloss",
            random_state=random_state,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.info("XGBoost classifier unavailable: %s", exc)
    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(n_estimators=250, learning_rate=0.05, random_state=random_state, verbose=-1)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.info("LightGBM classifier unavailable: %s", exc)
    try:
        from catboost import CatBoostClassifier

        models["catboost"] = CatBoostClassifier(iterations=250, learning_rate=0.05, verbose=False, random_seed=random_state)
    except Exception as exc:  # pragma: no cover - optional dependency
        LOGGER.info("CatBoost classifier unavailable: %s", exc)
    return models


def train_classification_models(feature_frame: pd.DataFrame) -> dict[str, Any]:
    """Train and compare classification models for match outcome prediction."""

    feature_cols = model_feature_columns(feature_frame)
    x = feature_frame[feature_cols]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(feature_frame["result"])
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.22, random_state=settings.random_state, stratify=y
    )
    candidates = _optional_classifiers(settings.random_state)
    metrics = []
    best_name = ""
    best_model: Any = None
    best_score = -np.inf
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        prob = model.predict_proba(x_test)
        try:
            auc = roc_auc_score(y_test, prob, multi_class="ovr")
        except ValueError:
            auc = float("nan")
        row = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "roc_auc": auc,
            "log_loss": log_loss(y_test, prob, labels=np.arange(len(label_encoder.classes_))),
        }
        metrics.append(row)
        score = row["f1"] - row["log_loss"] * 0.05
        if score > best_score:
            best_name, best_model, best_score = name, model, score
    tuned_model = tune_best_classifier(best_model, x_train, y_train) if best_name == "random_forest" else best_model
    artifact = {
        "model": tuned_model,
        "feature_cols": feature_cols,
        "label_encoder": label_encoder,
        "metrics": pd.DataFrame(metrics).sort_values("f1", ascending=False),
        "best_model_name": best_name,
    }
    return artifact


def tune_best_classifier(model: Any, x_train: pd.DataFrame, y_train: np.ndarray) -> Any:
    """Run a compact grid search for the Random Forest baseline."""

    if not isinstance(model, RandomForestClassifier):
        return model
    grid = GridSearchCV(
        model,
        {"max_depth": [5, 8, None], "min_samples_leaf": [1, 3, 5]},
        scoring="f1_weighted",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=settings.random_state),
        n_jobs=-1,
    )
    grid.fit(x_train, y_train)
    return grid.best_estimator_


def train_regression_models(feature_frame: pd.DataFrame) -> dict[str, Any]:
    """Train expected-goal regression models for home and away goals."""

    feature_cols = model_feature_columns(feature_frame)
    x = feature_frame[feature_cols]
    artifacts: dict[str, Any] = {"feature_cols": feature_cols, "targets": {}}
    for target in ["home_goals", "away_goals"]:
        y = feature_frame[target].astype(float)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.22, random_state=settings.random_state)
        candidates: dict[str, Any] = {
            "poisson": Pipeline([("scaler", StandardScaler()), ("model", PoissonRegressor(alpha=0.05, max_iter=1000))]),
            "random_forest": TransformedTargetRegressor(
                regressor=RandomForestRegressor(n_estimators=260, max_depth=8, random_state=settings.random_state),
                func=np.log1p,
                inverse_func=np.expm1,
            ),
        }
        try:
            from xgboost import XGBRegressor

            candidates["xgboost_regressor"] = XGBRegressor(
                n_estimators=260, max_depth=4, learning_rate=0.04, random_state=settings.random_state
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            LOGGER.info("XGBoost regressor unavailable: %s", exc)
        rows = []
        best_name = ""
        best_model: Any = None
        best_rmse = np.inf
        for name, model in candidates.items():
            model.fit(x_train, y_train)
            pred = np.clip(model.predict(x_test), 0, None)
            rmse = (
                root_mean_squared_error(y_test, pred)
                if root_mean_squared_error is not None
                else mean_squared_error(y_test, pred, squared=False)
            )
            rows.append(
                {
                    "model": name,
                    "target": target,
                    "rmse": rmse,
                    "mae": mean_absolute_error(y_test, pred),
                    "r2": r2_score(y_test, pred),
                }
            )
            if rmse < best_rmse:
                best_name, best_model, best_rmse = name, model, rmse
        artifacts["targets"][target] = {
            "model": best_model,
            "best_model_name": best_name,
            "metrics": pd.DataFrame(rows).sort_values("rmse"),
        }
    return artifacts


def save_artifacts(classifier: dict[str, Any], regressors: dict[str, Any]) -> None:
    """Persist trained model artifacts and evaluation tables."""

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, settings.model_dir / "match_outcome_classifier.joblib")
    joblib.dump(regressors, settings.model_dir / "score_regressors.joblib")
    classifier["metrics"].to_csv(settings.model_dir / "classification_metrics.csv", index=False)
    regression_metrics = pd.concat([target["metrics"] for target in regressors["targets"].values()], ignore_index=True)
    regression_metrics.to_csv(settings.model_dir / "regression_metrics.csv", index=False)
    summary = {
        "best_classifier": classifier["best_model_name"],
        "home_goal_model": regressors["targets"]["home_goals"]["best_model_name"],
        "away_goal_model": regressors["targets"]["away_goals"]["best_model_name"],
    }
    Path(settings.model_dir / "model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
