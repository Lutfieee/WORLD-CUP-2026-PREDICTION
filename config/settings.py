"""Central configuration for the World Cup 2026 predictor project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional at runtime
    def load_dotenv(*_: object, **__: object) -> bool:
        """No-op fallback when python-dotenv is not installed yet."""

        return False


ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    """Typed project settings loaded from environment variables."""

    football_data_api_key: str = os.getenv("FOOTBALL_DATA_API_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{ROOT_DIR / 'data' / 'worldcup2026.db'}")
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))
    default_simulations: int = int(os.getenv("DEFAULT_SIMULATIONS", "10000"))
    data_dir: Path = ROOT_DIR / "data"
    raw_dir: Path = ROOT_DIR / "data" / "raw"
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    model_dir: Path = ROOT_DIR / "saved_models"


settings = Settings()
