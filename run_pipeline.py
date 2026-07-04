"""Run the full data, feature, model, and prediction pipeline."""

from __future__ import annotations

import logging

from config.settings import settings
from feature_engineering.features import build_feature_table, build_fixture_features
from models.predict import predict_fixtures
from models.simulate import simulate_knockout
from models.train import save_artifacts, train_classification_models, train_regression_models
from preprocessing.cleaning import clean_fixtures, clean_matches, store_processed_tables
from scraping.pipeline import run_ingestion


def main() -> None:
    """Execute the production-style local pipeline."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    raw = run_ingestion(use_live=True)
    matches = clean_matches(raw["matches"])
    fixtures = clean_fixtures(raw["fixtures"])
    feature_frame = build_feature_table(matches, raw["teams"])
    fixture_features = build_fixture_features(fixtures, matches, raw["teams"])
    classifier = train_classification_models(feature_frame)
    regressors = train_regression_models(feature_frame)
    predictions = predict_fixtures(fixture_features, classifier, regressors)
    simulation = simulate_knockout(predictions, simulations=settings.default_simulations, seed=settings.random_state)
    save_artifacts(classifier, regressors)
    store_processed_tables(
        {
            "matches": matches,
            "fixtures": fixtures,
            "teams": raw["teams"],
            "players": raw["players"],
            "features": feature_frame,
            "fixture_features": fixture_features,
            "predictions": predictions,
            "simulation": simulation,
        }
    )


if __name__ == "__main__":
    main()
