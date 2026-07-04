"""End-to-end ingestion pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from config.settings import settings
from scraping.sources import (
    fetch_football_data_matches,
    generate_player_seed,
    generate_remaining_fixtures,
    generate_seed_matches,
    team_profiles_frame,
)

LOGGER = logging.getLogger(__name__)


def run_ingestion(use_live: bool = True) -> dict[str, pd.DataFrame]:
    """Ingest live data when available and fall back to deterministic seed data."""

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    live = fetch_football_data_matches() if use_live else pd.DataFrame()
    matches = live if not live.empty else generate_seed_matches(settings.random_state)
    fixtures = generate_remaining_fixtures()
    teams = team_profiles_frame()
    players = generate_player_seed(settings.random_state)

    outputs = {
        "matches": matches,
        "fixtures": fixtures,
        "teams": teams,
        "players": players,
    }
    for name, frame in outputs.items():
        path = settings.raw_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        LOGGER.info("Wrote %s rows to %s", len(frame), path)
    return outputs
