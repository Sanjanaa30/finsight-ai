"""Prophet price forecasting -- one model per ticker.

For each of the 25 tickers:
  1. Split price history on date (train = all but the last HOLDOUT_DAYS; test =
     the rest). Time-series splits NEVER shuffle -- training must not see the future.
  2. Fit Prophet on the training window.
  3. Forecast through the test window + FORECAST_DAYS ahead.
  4. Score MAPE on the held-out test window and log it to MLflow.
  5. Register the model (prophet_<ticker>) and save the 7-day-ahead forecast.

MAPE is measured over the SAME horizon we actually serve (FORECAST_DAYS), so the
reported metric reflects the real 7-day product -- not a longer horizon that
unfairly inflates the error. evaluate.py (Step 5) adds a fuller rolling backtest.
"""

import logging
import os
import sys
import warnings
from pathlib import Path

import duckdb
import mlflow
import mlflow.prophet
import numpy as np
import pandas as pd
from loguru import logger
from prophet import Prophet

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
# Prophet/Stan are very chatty; quiet them down.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"
FORECAST_OUT = PROJECT_ROOT / "data" / "processed" / "forecasts.parquet"

FORECAST_DAYS = 7         # days ahead to forecast for the dashboard
HOLDOUT_DAYS = 7          # held-out window for MAPE -- matches the forecast horizon
MIN_ROWS = 100            # skip tickers with too little history
HIGH_UNCERTAINTY_MAPE = 15.0  # above this, flag the forecast as low-confidence

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "finsight-forecast"


def load_prices() -> pd.DataFrame:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df = con.sql("SELECT date, ticker, close FROM stg_prices ORDER BY ticker, date").df()
    con.close()
    return df


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _new_model() -> Prophet:
    return Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)


def evaluate_mape(tdf: pd.DataFrame) -> float:
    """Holdout MAPE: train on all but the last HOLDOUT_DAYS, score on that window."""
    cutoff = tdf["ds"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train = tdf[tdf["ds"] <= cutoff]
    test = tdf[tdf["ds"] > cutoff]
    if train.empty or test.empty:
        return float("nan")

    model = _new_model()
    model.fit(train)
    forecast = model.predict(model.make_future_dataframe(periods=HOLDOUT_DAYS + FORECAST_DAYS))
    merged = test.merge(forecast[["ds", "yhat"]], on="ds", how="inner")
    return mape(merged["y"].to_numpy(), merged["yhat"].to_numpy()) if len(merged) else float("nan")


def fit_and_forecast(tdf: pd.DataFrame):
    """Train the PRODUCTION model on ALL data and forecast FORECAST_DAYS ahead.

    Returns (model, future_forecast_df, mape). MAPE comes from a separate holdout
    model (evaluate_mape); the forecast we ship must use every available day so it
    anchors to the latest price -- never a model that excludes the last week.
    """
    score = evaluate_mape(tdf)

    model = _new_model()
    model.fit(tdf)  # full history
    forecast = model.predict(model.make_future_dataframe(periods=FORECAST_DAYS))

    future_only = forecast[forecast["ds"] > tdf["ds"].max()][
        ["ds", "yhat", "yhat_lower", "yhat_upper"]
    ].copy()
    return model, future_only, score


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    df = load_prices()
    tickers = sorted(df["ticker"].unique())
    logger.info("Training Prophet for {} tickers", len(tickers))

    all_forecasts, scores = [], {}
    for ticker in tickers:
        tdf = (
            df[df["ticker"] == ticker][["date", "close"]]
            .rename(columns={"date": "ds", "close": "y"})
        )
        tdf["ds"] = pd.to_datetime(tdf["ds"])
        if len(tdf) < MIN_ROWS:
            logger.warning("{}: only {} rows, skipping", ticker, len(tdf))
            continue

        with mlflow.start_run(run_name=f"prophet-{ticker}"):
            model, future_only, score = fit_and_forecast(tdf)
            mlflow.log_param("ticker", ticker)
            mlflow.log_param("train_rows", len(tdf))
            mlflow.log_param("holdout_days", HOLDOUT_DAYS)
            mlflow.log_metric("mape", score)
            mlflow.prophet.log_model(model, name="model", registered_model_name=f"prophet_{ticker}")

        # Attach quality metadata so the forecast travels with its reliability:
        # the dashboard can label/suppress low-confidence (volatile) assets.
        future_only["ticker"] = ticker
        future_only["mape"] = round(score, 2)
        future_only["high_uncertainty"] = bool(score > HIGH_UNCERTAINTY_MAPE)
        all_forecasts.append(future_only)
        scores[ticker] = score
        flag = " [LOW CONFIDENCE]" if score > HIGH_UNCERTAINTY_MAPE else ""
        logger.success("{}: MAPE={:.2f}%{}", ticker, score, flag)

    out = pd.concat(all_forecasts, ignore_index=True)
    FORECAST_OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(FORECAST_OUT, index=False)

    valid = {k: v for k, v in scores.items() if not np.isnan(v)}
    logger.success(
        "Done. {} models | mean MAPE {:.2f}% | wrote {} forecast rows to {}",
        len(valid), np.mean(list(valid.values())), len(out),
        FORECAST_OUT.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
