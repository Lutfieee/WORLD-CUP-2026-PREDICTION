# FIFA World Cup 2026 Remaining Match Predictor

Professional football analytics portfolio project for match prediction, score prediction, tournament simulation, and interactive Streamlit reporting.

## Highlights

- Automated ingestion pipeline with API/HTML hooks and deterministic portfolio seed data.
- Cleaning, validation, duplicate handling, outlier clipping, CSV storage, and SQLite storage.
- Feature engineering for Elo, FIFA ranking, xG, xGA, recent form, possession, passing, conversion, clean sheets, and derived differences.
- Multi-model classification: Logistic Regression, Random Forest, plus optional XGBoost, LightGBM, and CatBoost.
- Regression score prediction with Poisson and tree-based regressors.
- Expected score, scoreline probability distribution, and match confidence.
- Monte Carlo knockout simulation for champion, runner-up, semi-final, and quarter-final probabilities.
- Streamlit multipage dashboard with dark, light, and auto theme modes.
- Optional live World Cup feed for current goals, remaining matches, qualified teams, and match status.
- Analytics pages for match preview, shot maps, xG flow, player analytics, player comparison, team comparison, tournament analytics, simulation, and explainable AI.

## Project Structure

```text
worldcup2026_predictor/
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── scraping/
├── preprocessing/
├── feature_engineering/
├── models/
├── notebooks/
├── saved_models/
├── streamlit_app/
│   ├── pages/
│   ├── assets/
│   ├── components/
│   ├── utils/
│   └── styles/
├── config/
├── docs/
├── tests/
├── requirements.txt
├── run_pipeline.py
├── app.py
└── LICENSE
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python run_pipeline.py
streamlit run app.py
```

## Data Sources

The project is designed for Football-data.org, FBref, Understat, StatsBomb Open Data, Kaggle historical World Cup data, and official FIFA feeds. The default runnable version uses realistic seed data so the dashboard works without private keys or fragile scraping assumptions.

The dashboard also includes an optional public World Cup 2026 live feed adapter in `scraping/live_worldcup.py`. If the endpoint is reachable, KPI cards use current tournament goals, remaining matches, qualified teams, and match status. If it fails, the app falls back to local processed ML data.

Add provider credentials in `.env`:

```text
FOOTBALL_DATA_API_KEY=your_key_here
DEFAULT_SIMULATIONS=10000
```

## Model Outputs

The pipeline writes:

- `data/processed/features.csv`
- `data/processed/predictions.csv`
- `data/processed/simulation.csv`
- `saved_models/match_outcome_classifier.joblib`
- `saved_models/score_regressors.joblib`
- `saved_models/classification_metrics.csv`
- `saved_models/regression_metrics.csv`

## Testing

```bash
pytest
```

## Deployment

See `docs/DEPLOYMENT.md`. The repository includes `.streamlit/config.toml` and a `Dockerfile` for Streamlit Cloud or Docker-based hosting.

## Portfolio Positioning

This project demonstrates sports data engineering, ML experimentation, explainable AI, Streamlit product design, and football analytics storytelling in one deployable application.
