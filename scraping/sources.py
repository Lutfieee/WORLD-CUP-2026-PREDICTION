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
    TeamProfile("Argentina", "CONMEBOL", 2, 1914, 1.86, 0.78, 59.2, 86.1, 18),
    TeamProfile("France", "UEFA", 1, 1916, 1.88, 0.82, 56.8, 85.0, 17),
    TeamProfile("Spain", "UEFA", 3, 1892, 1.82, 0.74, 64.0, 89.0, 16),
    TeamProfile("England", "UEFA", 4, 1851, 1.76, 0.86, 57.9, 86.4, 16),
    TeamProfile("Portugal", "UEFA", 7, 1788, 1.72, 0.88, 56.2, 85.2, 9),
    TeamProfile("Brazil", "CONMEBOL", 5, 1805, 1.94, 0.84, 58.6, 85.5, 22),
    TeamProfile("Belgium", "UEFA", 9, 1968, 1.54, 1.02, 54.5, 84.1, 14),
    TeamProfile("Colombia", "CONMEBOL", 13, 1960, 1.46, 1.00, 52.0, 82.0, 7),
    TeamProfile("Mexico", "CONCACAF", 14, 1878, 1.30, 1.14, 53.1, 82.8, 17),
    TeamProfile("Switzerland", "UEFA", 19, 1908, 1.34, 1.05, 53.4, 83.8, 13),
    TeamProfile("USA", "CONCACAF", 17, 1892, 1.34, 1.12, 52.8, 81.4, 12),
    TeamProfile("Morocco", "CAF", 6, 1789, 1.26, 0.96, 48.7, 79.8, 7),
    TeamProfile("Norway", "UEFA", 31, 1848, 1.42, 1.08, 50.8, 81.5, 4),
    TeamProfile("Canada", "CONCACAF", 30, 1788, 1.18, 1.18, 49.6, 79.4, 3),
    TeamProfile("Egypt", "CAF", 29, 1766, 1.20, 1.16, 48.8, 78.9, 4),
    TeamProfile("Paraguay", "CONMEBOL", 41, 1748, 1.08, 1.10, 46.9, 77.6, 9),
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

    pairings: Iterable[tuple[str, str, str, str]] = (
        ("Canada", "Morocco", "Round of 16", "2026-07-05"),
        ("Paraguay", "France", "Round of 16", "2026-07-05"),
        ("Brazil", "Norway", "Round of 16", "2026-07-06"),
        ("Mexico", "England", "Round of 16", "2026-07-06"),
        ("Portugal", "Spain", "Round of 16", "2026-07-07"),
        ("USA", "Belgium", "Round of 16", "2026-07-07"),
        ("Argentina", "Egypt", "Round of 16", "2026-07-07"),
        ("Switzerland", "Colombia", "Round of 16", "2026-07-08"),
    )
    return pd.DataFrame(
        [
            {
                "date": match_date,
                "home_team": home,
                "away_team": away,
                "stage": stage,
                "status": "SCHEDULED",
                "neutral_venue": True,
            }
            for home, away, stage, match_date in pairings
        ]
    )


def team_profiles_frame() -> pd.DataFrame:
    """Return team profile data as a dataframe."""

    return pd.DataFrame([profile.__dict__ for profile in TEAMS])


SQUAD_LISTS = {
    "Argentina": [
        "L. Messi", "J. Alvarez", "A. Di Maria", "L. Martinez", "N. Gonzalez", "A. Garnacho",
        "E. Fernandez", "A. Mac Allister", "R. De Paul", "G. Lo Celso", "L. Paredes", "E. Palacios",
        "C. Romero", "N. Otamendi", "L. Martinez Quarta", "G. Pezzella", "N. Tagliafico", "M. Acuna", "N. Molina", "G. Montiel",
        "E. Martinez", "G. Rulli", "F. Armani"
    ],
    "France": [
        "K. Mbappe", "O. Dembele", "A. Griezmann", "M. Thuram", "O. Giroud", "R. Kolo Muani", "B. Barcola", "K. Coman",
        "A. Rabiot", "N. Kante", "A. Tchouameni", "E. Camavinga", "Y. Fofana", "W. Zaire-Emery",
        "W. Saliba", "D. Upamecano", "I. Konate", "B. Pavard", "J. Kounde", "T. Hernandez", "F. Mendy",
        "M. Maignan", "A. Areola", "B. Samba"
    ],
    "Spain": ["A. Morata", "L. Yamal", "N. Williams", "Pedri", "Fabian Ruiz", "Rodri", "M. Cucurella", "A. Laporte", "R. Le Normand", "D. Carvajal", "U. Simon"],
    "England": [
        "H. Kane", "P. Foden", "B. Saka", "O. Watkins", "I. Toney", "A. Gordon", "E. Eze", "C. Palmer",
        "J. Bellingham", "D. Rice", "K. Mainoo", "C. Gallagher", "A. Wharton",
        "J. Stones", "M. Guehi", "L. Dunk", "E. Konsa", "L. Shaw", "K. Trippier", "K. Walker", "T. Alexander-Arnold",
        "J. Pickford", "A. Ramsdale", "D. Henderson"
    ],
    "Portugal": [
        "C. Ronaldo", "R. Leao", "J. Felix", "G. Ramos", "D. Jota", "P. Neto", "F. Conceicao",
        "B. Silva", "B. Fernandes", "Vitinha", "J. Palhinha", "R. Neves", "J. Neves", "M. Nunes",
        "Pepe", "Ruben Dias", "A. Silva", "G. Inacio", "J. Cancelo", "N. Mendes", "N. Semedo", "D. Dalot",
        "D. Costa", "R. Patricio", "J. Sa"
    ],
    "Brazil": [
        "Vinicius Jr", "Rodrygo", "Raphinha", "Endrick", "Martinelli", "Savinho", "Evanilson",
        "L. Paqueta", "B. Guimaraes", "J. Gomes", "A. Pereira", "D. Luiz", "Ederson",
        "Marquinhos", "Eder Militao", "Beraldo", "Bremer", "Danilo", "Yan Couto", "Arana", "Wendell",
        "Alisson", "Ederson", "Bento"
    ],
    "Belgium": ["R. Lukaku", "J. Doku", "L. Trossard", "K. De Bruyne", "Y. Tielemans", "A. Onana", "A. Theate", "J. Vertonghen", "W. Faes", "T. Castagne", "K. Casteels"],
    "Colombia": ["J. Cordoba", "L. Diaz", "J. Rodriguez", "J. Arias", "J. Lerma", "R. Rios", "J. Mojica", "J. Lucumi", "D. Sanchez", "D. Munoz", "C. Vargas"],
    "Mexico": ["S. Gimenez", "H. Lozano", "U. Antuna", "O. Pineda", "L. Chavez", "E. Alvarez", "J. Gallardo", "J. Vasquez", "C. Montes", "J. Sanchez", "G. Ochoa"],
    "Switzerland": ["B. Embolo", "D. Ndoye", "X. Shaqiri", "G. Xhaka", "R. Freuler", "M. Aebischer", "R. Rodriguez", "M. Akanji", "F. Schar", "S. Widmer", "Y. Sommer"],
    "USA": ["F. Balogun", "C. Pulisic", "T. Weah", "W. McKennie", "Y. Musah", "T. Adams", "A. Robinson", "T. Ream", "C. Richards", "S. Dest", "M. Turner"],
    "Morocco": ["Y. En-Nesyri", "S. Boufal", "H. Ziyech", "S. Amallah", "A. Ounahi", "S. Amrabat", "N. Mazraoui", "R. Saiss", "N. Aguerd", "A. Hakimi", "Y. Bounou"],
    "Norway": ["E. Haaland", "A. Sorloth", "O. Bobb", "M. Odegaard", "S. Berge", "P. Berg", "B. Meling", "L. Ostigard", "K. Ajer", "J. Ryerson", "O. Nyland"],
    "Canada": ["J. David", "C. Larin", "T. Buchanan", "I. Kone", "S. Eustaquio", "A. Davies", "L. Millar", "D. Cornelius", "M. Bombito", "A. Johnston", "M. Crepeau"],
    "Egypt": ["M. Mohamed", "M. Salah", "Trezeguet", "E. Ashour", "M. Elneny", "M. Attia", "M. Hamdi", "M. Abdelmonem", "R. Rabia", "M. Hany", "M. El Shenawy"],
    "Paraguay": ["A. Sanabria", "J. Enciso", "M. Almiron", "R. Sosa", "M. Villasanti", "A. Cubas", "M. Espinoza", "O. Alderete", "G. Gomez", "I. Ramirez", "C. Coronel"],
}

PLAYER_STATS_OVERRIDES = {
    "L. Messi": {"goals": 20, "assists": 8, "xg": 15.3, "xa": 9.1, "rating": 9.3},
    "K. Mbappe": {"goals": 18, "assists": 4, "xg": 13.2, "xa": 4.0, "rating": 9.1},
    "C. Ronaldo": {"goals": 8, "assists": 3, "xg": 11.5, "xa": 2.3, "rating": 8.7},
    "H. Kane": {"goals": 13, "assists": 4, "xg": 11.3, "xa": 3.0, "rating": 8.6},
    "A. Griezmann": {"goals": 6, "assists": 7, "xg": 5.3, "xa": 7.4, "rating": 8.8},
    "B. Saka": {"goals": 5, "assists": 4, "xg": 3.8, "xa": 3.6, "rating": 8.4},
    "B. Fernandes": {"goals": 4, "assists": 8, "xg": 3.3, "xa": 6.5, "rating": 8.8},
    "J. Bellingham": {"goals": 4, "assists": 3, "xg": 3.4, "xa": 2.6, "rating": 8.6},
    "Vinicius Jr": {"goals": 4, "assists": 6, "xg": 4.7, "xa": 4.6, "rating": 8.7},
    "Rodrygo": {"goals": 3, "assists": 3, "xg": 3.3, "xa": 2.6, "rating": 8.3},
    "Raphinha": {"goals": 2, "assists": 2, "xg": 2.4, "xa": 2.3, "rating": 8.1},
    "L. Martinez": {"goals": 4, "assists": 1, "xg": 5.6, "xa": 0.9, "rating": 8.0},
    "E. Martinez": {"goals": 0, "assists": 0, "xg": 0.0, "xa": 0.0, "rating": 8.6, "tackles": 0},
    "M. Maignan": {"goals": 0, "assists": 0, "xg": 0.0, "xa": 0.0, "rating": 8.2, "tackles": 0},
    "Alisson": {"goals": 0, "assists": 0, "xg": 0.0, "xa": 0.0, "rating": 8.1, "tackles": 0},
}

def generate_player_seed(seed: int = 42) -> pd.DataFrame:
    """Generate player-level analytics data for dashboard pages using realistic 2026 rosters."""

    rng = np.random.default_rng(seed)
    rows = []
    
    for profile in TEAMS:
        squad = SQUAD_LISTS.get(profile.team, [f"{profile.team} Player {i+1}" for i in range(11)])
        squad_size = len(squad)
        for idx, player_name in enumerate(squad):
            # Proportional roles for larger squads
            if idx < squad_size * 0.25: role = "Forward"
            elif idx < squad_size * 0.60: role = "Midfielder"
            elif idx < squad_size * 0.90: role = "Defender"
            else: role = "Goalkeeper"
            
            attacking_boost = 1.3 if role == "Forward" else 0.75 if role == "Midfielder" else 0.25 if role == "Defender" else 0.05
            
            # Base generated stats
            goals = int(max(0, rng.poisson(profile.attack * attacking_boost)))
            assists = int(max(0, rng.poisson(attacking_boost)))
            xg = round(max(0, rng.normal(profile.attack * attacking_boost, 0.35)), 2)
            xa = round(max(0, rng.normal(attacking_boost, 0.25)), 2)
            rating = round(np.clip(rng.normal(7.05 + profile.attack / 8, 0.38), 5.8, 8.9), 2)
            tackles = int(rng.poisson(5.0 if role in {"Defender", "Midfielder"} else 1.2))
            
            # Apply overrides if available
            overrides = PLAYER_STATS_OVERRIDES.get(player_name, {})
            if "goals" in overrides: goals = overrides["goals"]
            if "assists" in overrides: assists = overrides["assists"]
            if "xg" in overrides: xg = overrides["xg"]
            if "xa" in overrides: xa = overrides["xa"]
            if "rating" in overrides: rating = overrides["rating"]
            if "tackles" in overrides: tackles = overrides["tackles"]

            rows.append(
                {
                    "player": player_name,
                    "team": profile.team,
                    "position": role,
                    "minutes": int(rng.integers(180, 620)),
                    "goals": goals,
                    "assists": assists,
                    "xg": xg,
                    "xa": xa,
                    "touches": int(rng.integers(90, 460)),
                    "passes": int(rng.integers(55, 390)),
                    "pass_accuracy": round(np.clip(rng.normal(profile.pass_accuracy, 4), 62, 95), 1),
                    "key_passes": int(rng.poisson(2.5 * attacking_boost)),
                    "progressive_passes": int(rng.poisson(4.0 if role != "Goalkeeper" else 0.4)),
                    "tackles": tackles,
                    "interceptions": int(rng.poisson(4.0 if role in {"Defender", "Midfielder"} else 0.8)),
                    "rating": rating,
                }
            )
    return pd.DataFrame(rows)
