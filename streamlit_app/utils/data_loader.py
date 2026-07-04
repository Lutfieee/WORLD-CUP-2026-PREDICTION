"""Data loading utilities for Streamlit pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from config.settings import settings


@st.cache_data(show_spinner=False, ttl=1)
def load_table(name: str) -> pd.DataFrame:
    """Load a processed table with graceful empty fallback."""

    path = settings.processed_dir / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def load_artifact(path: str) -> Any | None:
    """Load a joblib artifact from disk."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)


def data_ready() -> bool:
    """Return whether the main pipeline outputs exist."""

    required = ["matches.csv", "predictions.csv", "simulation.csv", "players.csv", "teams.csv"]
    return all((settings.processed_dir / name).exists() for name in required)
