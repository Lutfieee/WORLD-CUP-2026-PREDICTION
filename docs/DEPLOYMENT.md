# Deployment Guide

## Streamlit Community Cloud

1. Push this project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to `app.py`.
5. Add optional secrets:

```toml
FOOTBALL_DATA_API_KEY = ""
DEFAULT_SIMULATIONS = "10000"
```

6. Deploy.

## Render

Use the included `Dockerfile`.

1. Create a new Web Service.
2. Connect the GitHub repository.
3. Select Docker runtime.
4. Expose port `8501`.
5. Deploy.

## Local Production Run

```bash
python run_pipeline.py
python -m streamlit run app.py --server.port 8501
```

## Live Data Behavior

The app attempts to read the public World Cup 2026 live feed at runtime. If the endpoint is unreachable, KPI cards and tables fall back to locally generated processed data.
