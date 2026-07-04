"""Source adapters for football data ingestion.

The adapters are intentionally small and composable. Live public websites often
change markup or require API keys, so the project ships with deterministic seed
data while keeping production-grade hooks for API and HTML sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None
try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

from config.settings import settings

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamProfile:
    """Static team profile used for feature bootstrapping."""

    team: str
    confederation: str
    fifa_rank: int
    elo: int
    attack: float
    defense: float
    possession: float
    pass_accuracy: float
    tournament_experience: int


TEAMS: tuple[TeamProfile, ...] = (
    TeamProfile("Argentina", "CONMEBOL", 1, 2148, 1.86, 0.78, 59.2, 86.1, 18),
    TeamProfile("France", "UEFA", 2, 2130, 1.88, 0.82, 56.8, 85.0, 17),
    TeamProfile("Spain", "UEFA", 3, 2124, 1.82, 0.74, 64.0, 89.0, 16),
    TeamProfile("England", "UEFA", 4, 2094, 1.76, 0.86, 57.9, 86.4, 16),
    TeamProfile("Portugal", "UEFA", 5, 2058, 1.72, 0.88, 56.2, 85.2, 9),
    TeamProfile("Brazil", "CONMEBOL", 6, 2112, 1.94, 0.84, 58.6, 85.5, 22),
    TeamProfile("Netherlands", "UEFA", 7, 2022, 1.66, 0.92, 55.4, 84.6, 11),
    TeamProfile("Belgium", "UEFA", 8, 1968, 1.54, 1.02, 54.5, 84.1, 14),
    TeamProfile("Germany", "UEFA", 9, 2034, 1.70, 0.96, 60.1, 87.0, 20),
    TeamProfile("Croatia", "UEFA", 10, 1958, 1.44, 1.04, 55.8, 85.8, 7),
    TeamProfile("Uruguay", "CONMEBOL", 12, 1988, 1.52, 0.98, 51.6, 81.7, 14),
    TeamProfile("Colombia", "CONMEBOL", 13, 1960, 1.46, 1.00, 52.0, 82.0, 7),
    TeamProfile("Mexico", "CONCACAF", 14, 1878, 1.30, 1.14, 53.1, 82.8, 17),
    TeamProfile("Japan", "AFC", 16, 1884, 1.32, 1.10, 54.2, 83.6, 8),
    TeamProfile("USA", "CONCACAF", 17, 1892, 1.34, 1.12, 52.8, 81.4, 12),
    TeamProfile("Morocco", "CAF", 18, 1890, 1.26, 0.96, 48.7, 79.8, 7),
)


def fetch_football_data_matches(competition: str = "WC") -> pd.DataFrame:
    """Fetch matches from football-data.org when an API key is configured."""

    if not settings.football_data_api_key:
        LOGGER.info("FOOTBALL_DATA_API_KEY is empty; skipping live API pull.")
        return pd.DataFrame()
    if requests is None:
        LOGGER.info("requests is unavailable; skipping live API pull.")
        return pd.DataFrame()
    url = f"https://api.football-data.org/v4/competitions/{competition}/matches"
    response = requests.get(url, headers={"X-Auth-Token": settings.football_data_api_key}, timeout=30)
    response.raise_for_status()
    rows = []
    for match in response.json().get("matches", []):
        rows.append(
            {
                "date": match.get("utcDate", "")[:10],
                "home_team": match.get("homeTeam", {}).get("name"),
                "away_team": match.get("awayTeam", {}).get("name"),
                "home_goals": match.get("score", {}).get("fullTime", {}).get("home"),
                "away_goals": match.get("score", {}).get("fullTime", {}).get("away"),
                "stage": match.get("stage"),
                "status": match.get("status"),
            }
        )
    return pd.DataFrame(rows)


def scrape_table(url: str, table_css: str = "table") -> pd.DataFrame:
    """Scrape the first matching HTML table from a public page."""

    if requests is None or BeautifulSoup is None:
        raise ImportError("Install requests and beautifulsoup4 to use HTML scraping.")
    response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.select_one(table_css)
    if table is None:
        raise ValueError(f"No table found for selector: {table_css}")
    return pd.read_html(str(table))[0]


def generate_seed_matches(seed: int = 42, seasons: int = 6) -> pd.DataFrame:
    """Generate a realistic historical match dataset for portfolio demos."""

    rng = np.random.default_rng(seed)
    profiles = {team.team: team for team in TEAMS}
    rows: list[dict[str, object]] = []
    start = date(2018, 6, 1)
    for season in range(seasons):
        for i, home in enumerate(TEAMS):
            opponents = [team for team in TEAMS if team.team != home.team]
            for away in rng.choice(opponents, size=5, replace=False):
                neutral = bool(rng.integers(0, 2))
                home_strength = home.attack + (home.elo - away.elo) / 800
                away_strength = away.attack + (away.elo - home.elo) / 900
                home_xg = max(0.15, rng.normal(home_strength, 0.35))
                away_xg = max(0.15, rng.normal(away_strength, 0.35))
                home_goals = int(rng.poisson(home_xg))
                away_goals = int(rng.poisson(away_xg))
                match_date = start + timedelta(days=season * 160 + i * 3 + int(rng.integers(0, 35)))
                rows.append(
                    {
                        "date": match_date.isoformat(),
                        "home_team": home.team,
                        "away_team": away.team,
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "home_xg": round(home_xg, 2),
                        "away_xg": round(away_xg, 2),
                        "home_shots": int(max(3, rng.normal(home_xg * 7.8, 2.4))),
                        "away_shots": int(max(3, rng.normal(away_xg * 7.8, 2.4))),
                        "home_possession": round(np.clip(rng.normal(home.possession, 5), 35, 72), 1),
                        "away_possession": round(np.clip(rng.normal(away.possession, 5), 28, 65), 1),
                        "home_pass_accuracy": round(np.clip(rng.normal(home.pass_accuracy, 2.4), 68, 93), 1),
                        "away_pass_accuracy": round(np.clip(rng.normal(away.pass_accuracy, 2.4), 68, 93), 1),
                        "home_corners": int(max(0, rng.normal(home_xg * 2.7, 1.5))),
                        "away_corners": int(max(0, rng.normal(away_xg * 2.7, 1.5))),
                        "home_cards": int(max(0, rng.poisson(1.6))),
                        "away_cards": int(max(0, rng.poisson(1.7))),
                        "neutral_venue": neutral,
                        "stage": "Historical",
                        "status": "FINISHED",
                    }
                )
    return pd.DataFrame(rows)


def generate_remaining_fixtures() -> pd.DataFrame:
    """Create a seed remaining-fixtures table for the 2026 tournament scenario."""

    pairings: Iterable[tuple[str, str, str]] = (
        ("Spain", "Portugal", "Round of 16"),
        ("France", "Morocco", "Round of 16"),
        ("Argentina", "USA", "Round of 16"),
        ("Brazil", "Japan", "Round of 16"),
        ("England", "Mexico", "Round of 16"),
        ("Germany", "Colombia", "Round of 16"),
        ("Netherlands", "Uruguay", "Round of 16"),
        ("Belgium", "Croatia", "Round of 16"),
    )
    base = date(2026, 7, 5)
    return pd.DataFrame(
        [
            {
                "date": (base + timedelta(days=i)).isoformat(),
                "home_team": home,
                "away_team": away,
                "stage": stage,
                "status": "SCHEDULED",
                "neutral_venue": True,
            }
            for i, (home, away, stage) in enumerate(pairings)
        ]
    )


def team_profiles_frame() -> pd.DataFrame:
    """Return team profile data as a dataframe."""

    return pd.DataFrame([profile.__dict__ for profile in TEAMS])


def generate_player_seed(seed: int = 42) -> pd.DataFrame:
    """Generate player-level analytics data for dashboard pages."""

    rng = np.random.default_rng(seed)
    rows = []
    roles = ("Forward", "Midfielder", "Defender", "Goalkeeper")
    for profile in TEAMS:
        for idx in range(1, 12):
            role = roles[min(idx // 4, 3)]
            attacking_boost = 1.3 if role == "Forward" else 0.75 if role == "Midfielder" else 0.25
            rows.append(
                {
                    "player": f"{profile.team} Player {idx}",
                    "team": profile.team,
                    "position": role,
                    "minutes": int(rng.integers(180, 620)),
                    "goals": int(max(0, rng.poisson(profile.attack * attacking_boost))),
                    "assists": int(max(0, rng.poisson(attacking_boost))),
                    "xg": round(max(0, rng.normal(profile.attack * attacking_boost, 0.35)), 2),
                    "xa": round(max(0, rng.normal(attacking_boost, 0.25)), 2),
                    "touches": int(rng.integers(90, 460)),
                    "passes": int(rng.integers(55, 390)),
                    "pass_accuracy": round(np.clip(rng.normal(profile.pass_accuracy, 4), 62, 95), 1),
                    "key_passes": int(rng.poisson(2.5 * attacking_boost)),
                    "progressive_passes": int(rng.poisson(4.0 if role != "Goalkeeper" else 0.4)),
                    "tackles": int(rng.poisson(5.0 if role in {"Defender", "Midfielder"} else 1.2)),
                    "interceptions": int(rng.poisson(4.0 if role in {"Defender", "Midfielder"} else 0.8)),
                    "rating": round(np.clip(rng.normal(7.05 + profile.attack / 8, 0.38), 5.8, 8.9), 2),
                }
            )
    return pd.DataFrame(rows)
