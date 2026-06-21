"""Volatility forecasting with a HAR-RV model -- the forecast that actually beats naive.

Price LEVELS are a random walk (persistence wins), but VOLATILITY clusters and is
genuinely predictable. We forecast next-week realized volatility per ticker with a
HAR model (Corsi 2009): regress future 5-day return-volatility on realized vol over
short/medium/long windows (5d / 22d / 66d).

For every ticker we benchmark HAR against the honest naive baseline (next week's vol
= this week's vol). A model is only worth shipping if it beats that.
"""

import os
import sys
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "volatility_forecasts.parquet"

HORIZON = 5            # forecast next-5-trading-day realized vol
WINDOWS = (5, 22, 66)  # HAR lags: weekly, monthly, quarterly
TRAIN_FRACTION = 0.8   # time-ordered split
MIN_ROWS = 150

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "finsight-volatility"


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 1e-9
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def build_features(returns: pd.Series) -> pd.DataFrame:
    """HAR features (realized vol over each window) + forward-vol target."""
    df = pd.DataFrame({"ret": returns})
    for w in WINDOWS:
        df[f"rv_{w}"] = df["ret"].rolling(w).std()
    # target = realized vol over the NEXT HORIZON days (no leakage: uses future only)
    df["target"] = df["ret"].rolling(HORIZON).std().shift(-HORIZON)
    return df.dropna()


def main() -> None:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    data = con.sql("SELECT date, ticker, daily_return FROM int_price_features ORDER BY ticker, date").df()
    con.close()

    feat_cols = [f"rv_{w}" for w in WINDOWS]
    rows, forecasts = [], []

    for ticker in sorted(data["ticker"].unique()):
        s = data[data["ticker"] == ticker].set_index("date")["daily_return"].dropna()
        if len(s) < MIN_ROWS:
            continue
        df = build_features(s)
        if len(df) < MIN_ROWS:
            continue

        split = int(len(df) * TRAIN_FRACTION)
        train, test = df.iloc[:split], df.iloc[split:]

        model = LinearRegression()
        model.fit(train[feat_cols], train["target"])
        pred = model.predict(test[feat_cols])

        har = mape(test["target"].to_numpy(), pred)
        naive = mape(test["target"].to_numpy(), test["rv_5"].to_numpy())  # vol persistence
        rows.append((ticker, har, naive))

        # forward forecast: latest realized-vol features -> next-week vol
        latest = pd.DataFrame({c: [s.rolling(w).std().iloc[-1]] for c, w in zip(feat_cols, WINDOWS)})
        forecasts.append({"ticker": ticker, "predicted_volatility": float(model.predict(latest)[0]),
                          "current_volatility": float(s.rolling(5).std().iloc[-1])})
        logger.info("{}: HAR MAPE={:.1f}%  naive={:.1f}%", ticker, har, naive)

    r = pd.DataFrame(rows, columns=["ticker", "har", "naive"])
    r["har_wins"] = r["har"] < r["naive"]
    win_rate = r["har_wins"].mean()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="har-rv"):
        mlflow.log_params({"horizon": HORIZON, "windows": WINDOWS, "train_fraction": TRAIN_FRACTION})
        mlflow.log_metric("har_median_mape", float(r["har"].median()))
        mlflow.log_metric("naive_median_mape", float(r["naive"].median()))
        mlflow.log_metric("har_win_rate", float(win_rate))

    pd.DataFrame(forecasts).to_parquet(OUTPUT_PATH, index=False)

    print("\n" + "=" * 64)
    print("VOLATILITY FORECAST -- HAR-RV vs naive (vol persistence)")
    print("=" * 64)
    print("  HAR median MAPE   : %.1f%%" % r["har"].median())
    print("  Naive median MAPE : %.1f%%" % r["naive"].median())
    print("  HAR beats naive   : %d / %d assets (%.0f%%)" % (r["har_wins"].sum(), len(r), win_rate * 100))
    print("=" * 64)
    logger.success("Wrote {} volatility forecasts to {}", len(forecasts), OUTPUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
