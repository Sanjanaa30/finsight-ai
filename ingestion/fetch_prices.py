"""Fetch daily OHLCV price history for all 25 tracked assets via yfinance.

One yf.download() call pulls every ticker, then the wide multi-index frame is
reshaped to a tidy long format (one row per date+ticker) and written to Parquet.
Raw landing only — no cleaning or feature engineering happens here (that is dbt's
job in Phase 3).
"""

from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

# 25 assets across 5 categories. Strings are the exact yfinance ticker symbols.
TICKERS = [
    # 12 US stocks
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "TSLA", "JPM", "XOM", "NFLX", "AMD", "BRK-B",
    # 3 crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # 4 commodities (futures)
    "CL=F", "GC=F", "SI=F", "NG=F",
    # 4 country indices
    "^NSEI", "^FTSE", "^GDAXI", "^N225",
    # 2 benchmarks
    "^GSPC", "^IXIC",
]

START_DATE = "2022-01-01"

# Resolve paths relative to the project root so the script works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "prices.parquet"


def fetch_prices(tickers: list[str] = TICKERS, start: str = START_DATE) -> pd.DataFrame:
    """Download OHLCV for all tickers and return a tidy long DataFrame."""
    logger.info("Downloading {} tickers from {} ...", len(tickers), start)
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,   # adjust OHLC for splits/dividends
        progress=False,
        group_by="column",  # columns = (field, ticker)
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data - check connectivity/tickers")

    # Columns are a MultiIndex (field, ticker). Stack the ticker level so each
    # row is a (date, ticker) pair with one column per OHLCV field.
    long = (
        raw.stack(level=1, future_stack=True)
        .rename_axis(index=["date", "ticker"])
        .reset_index()
    )
    long.columns = [str(c).lower() for c in long.columns]

    # Drop rows with no close price (markets closed / asset not yet listed).
    before = len(long)
    long = long.dropna(subset=["close"]).reset_index(drop=True)
    logger.info("Dropped {} empty rows; {} rows remain", before - len(long), len(long))

    long = long.sort_values(["ticker", "date"]).reset_index(drop=True)
    return long


def main() -> None:
    df = fetch_prices()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    logger.success(
        "Wrote {} rows ({} tickers, {} -> {}) to {}",
        len(df),
        df["ticker"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
        OUTPUT_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
