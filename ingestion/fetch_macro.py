"""Fetch macroeconomic indicators from FRED -> long Parquet.

Pulls four core indicators (fed funds rate, CPI, unemployment, GDP) via the FRED
REST API and lands them in tidy long form (one row per series+date). These become
macro context features in the Phase 3 mart. Requires FRED_KEY in .env.
FRED's free tier is unlimited, so no rate limiting is needed.
"""

import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_KEY = os.getenv("FRED_KEY")
ENDPOINT = "https://api.stlouisfed.org/fred/series/observations"

# FRED series id -> friendly name.
SERIES = {
    "FEDFUNDS": "fed_funds_rate",   # effective federal funds rate (monthly, %)
    "CPIAUCSL": "cpi",              # CPI, all urban consumers (monthly, index)
    "UNRATE": "unemployment_rate",  # unemployment rate (monthly, %)
    "GDP": "gdp",                   # gross domestic product (quarterly, $B)
}

OBSERVATION_START = "2018-01-01"  # a few years of history for context

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "macro.parquet"


def fetch_series(series_id: str, name: str) -> list[dict]:
    """Fetch one FRED series as a list of {series_id, series_name, date, value}."""
    params = {
        "series_id": series_id,
        "api_key": API_KEY,
        "file_type": "json",
        "observation_start": OBSERVATION_START,
    }
    resp = requests.get(ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    observations = resp.json()["observations"]

    rows = []
    for obs in observations:
        raw_value = obs["value"]
        # FRED uses "." for missing values.
        value = None if raw_value == "." else float(raw_value)
        rows.append({
            "series_id": series_id,
            "series_name": name,
            "date": obs["date"],
            "value": value,
        })
    logger.info("{} ({}): {} observations", name, series_id, len(rows))
    return rows


def main() -> None:
    if not API_KEY:
        raise RuntimeError("FRED_KEY not set in .env")

    all_rows: list[dict] = []
    for series_id, name in SERIES.items():
        all_rows.extend(fetch_series(series_id, name))

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    logger.success(
        "Wrote {} observations across {} series to {}",
        len(df), df["series_id"].nunique(), OUTPUT_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
