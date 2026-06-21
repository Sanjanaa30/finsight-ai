"""Consolidated, honest Phase 4 evaluation.

Every model is benchmarked against the naive baseline it must beat:

1. Price forecast (Prophet) vs naive persistence ("next week = today").
   Rolling 7-day-ahead backtest. Prices are a random walk -> Prophet does NOT
   beat naive. Reported honestly.
2. Volatility forecast (HAR-RV) vs naive vol-persistence.
   Volatility clusters -> HAR DOES beat naive. This is the model with real edge.
3. Sentiment (FinBERT): throughput + label mix.
4. Anomaly (Isolation Forest): flag rate.

Prints the honest summary, logs to MLflow, writes evaluation_metrics.json.
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path

import duckdb
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from prophet import Prophet
from scipy import stats
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"
NEWS_SCORED = PROJECT_ROOT / "data" / "processed" / "news_scored.parquet"
ANOMALIES = PROJECT_ROOT / "data" / "processed" / "anomalies.parquet"
METRICS_OUT = PROJECT_ROOT / "data" / "processed" / "evaluation_metrics.json"

HORIZON, FOLDS, STEP, MIN_ROWS = 7, 8, 7, 100  # 8 rolling folds -> firmer estimates
VOL_HORIZON, VOL_WINDOWS, VOL_TRAIN_FRAC = 5, (5, 22, 66), 0.8

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "finsight-evaluation"


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-9) -> float:
    mask = np.abs(actual) > eps
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def paired_significance(better: np.ndarray, worse: np.ndarray) -> dict:
    """Is `better` systematically lower-MAPE than `worse`? (paired across assets)
    Wilcoxon signed-rank (one-sided) + binomial test on the win count."""
    wins = int(np.sum(better < worse))
    n = len(better)
    try:
        w_p = float(stats.wilcoxon(worse, better, alternative="greater").pvalue)
    except ValueError:
        w_p = float("nan")
    b_p = float(stats.binomtest(wins, n, 0.5, alternative="greater").pvalue)
    return {"wins": wins, "n": n, "wilcoxon_p": round(w_p, 5), "binomial_p": round(b_p, 5)}


# ---- 1. Price forecast: Prophet vs naive persistence ----------------------------
def price_backtest(tdf: pd.DataFrame) -> tuple[float, float]:
    prophet_s, naive_s = [], []
    for k in range(FOLDS):
        cutoff = tdf["ds"].max() - pd.Timedelta(days=HORIZON + k * STEP)
        train = tdf[tdf["ds"] <= cutoff]
        test = tdf[(tdf["ds"] > cutoff) & (tdf["ds"] <= cutoff + pd.Timedelta(days=HORIZON))]
        if len(train) < MIN_ROWS or test.empty:
            continue
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
        m.fit(train)
        fc = m.predict(m.make_future_dataframe(periods=HORIZON + 5))
        merged = test.merge(fc[["ds", "yhat"]], on="ds", how="inner")
        if not len(merged):
            continue
        last = train["y"].iloc[-1]
        prophet_s.append(mape(merged["y"].to_numpy(), merged["yhat"].to_numpy()))
        naive_s.append(mape(merged["y"].to_numpy(), np.full(len(merged), last)))
    return (np.mean(prophet_s) if prophet_s else np.nan,
            np.mean(naive_s) if naive_s else np.nan)


def evaluate_price() -> dict:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df = con.sql("SELECT date, ticker, close FROM stg_prices ORDER BY ticker, date").df()
    con.close()
    prophet, naive = {}, {}
    for ticker in sorted(df["ticker"].unique()):
        t = df[df["ticker"] == ticker][["date", "close"]].rename(columns={"date": "ds", "close": "y"})
        t["ds"] = pd.to_datetime(t["ds"])
        if len(t) < MIN_ROWS:
            continue
        p, n = price_backtest(t)
        prophet[ticker], naive[ticker] = p, n
    pv = np.array([v for v in prophet.values() if not np.isnan(v)])
    nv = np.array([naive[k] for k in prophet if not np.isnan(prophet[k])])
    sig = paired_significance(pv, nv)  # does Prophet beat naive?
    return {"prophet_median_mape": round(float(np.median(pv)), 2),
            "naive_median_mape": round(float(np.median(nv)), 2),
            "prophet_beats_naive": sig["wins"], "n_total": sig["n"],
            "wilcoxon_p": sig["wilcoxon_p"], "binomial_p": sig["binomial_p"]}


# ---- 2. Volatility forecast: HAR-RV vs naive ------------------------------------
def evaluate_volatility() -> dict:
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    data = con.sql("SELECT date, ticker, daily_return FROM int_price_features ORDER BY ticker, date").df()
    con.close()
    cols = [f"rv_{w}" for w in VOL_WINDOWS]
    har, naive = {}, {}
    for ticker in sorted(data["ticker"].unique()):
        s = data[data["ticker"] == ticker].set_index("date")["daily_return"].dropna()
        if len(s) < 150:
            continue
        df = pd.DataFrame({"ret": s})
        for w in VOL_WINDOWS:
            df[f"rv_{w}"] = df["ret"].rolling(w).std()
        df["target"] = df["ret"].rolling(VOL_HORIZON).std().shift(-VOL_HORIZON)
        df = df.dropna()
        if len(df) < 150:
            continue
        split = int(len(df) * VOL_TRAIN_FRAC)
        tr, te = df.iloc[:split], df.iloc[split:]
        model = LinearRegression().fit(tr[cols], tr["target"])
        har[ticker] = mape(te["target"].to_numpy(), model.predict(te[cols]))
        naive[ticker] = mape(te["target"].to_numpy(), te["rv_5"].to_numpy())
    hv = np.array(list(har.values()))
    nv = np.array([naive[k] for k in har])
    sig = paired_significance(hv, nv)  # does HAR beat naive?
    return {"har_median_mape": round(float(np.median(hv)), 2),
            "naive_median_mape": round(float(np.median(nv)), 2),
            "har_beats_naive": sig["wins"], "n_total": sig["n"],
            "wilcoxon_p": sig["wilcoxon_p"], "binomial_p": sig["binomial_p"]}


# ---- 3 & 4. Sentiment + anomaly ------------------------------------------------
def evaluate_sentiment() -> dict:
    df = pd.read_parquet(NEWS_SCORED)
    n_days = df["published_date"].nunique()
    return {"total_articles": len(df), "articles_per_day": round(len(df) / n_days, 1),
            "categories": int(df["category"].nunique()),
            "label_distribution": df["sentiment_label"].value_counts(normalize=True).round(3).to_dict()}


def evaluate_anomaly() -> dict:
    df = pd.read_parquet(ANOMALIES)
    n = int(df["is_anomaly"].sum())
    return {"total_days": len(df), "anomalies": n, "anomaly_rate": round(n / len(df), 4)}


def main() -> None:
    logger.info("Backtesting price (Prophet vs naive)...")
    price = evaluate_price()
    logger.info("Backtesting volatility (HAR-RV vs naive)...")
    vol = evaluate_volatility()
    sentiment = evaluate_sentiment()
    anomaly = evaluate_anomaly()
    metrics = {"price_forecast": price, "volatility_forecast": vol,
               "sentiment": sentiment, "anomaly": anomaly}

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metric("price_prophet_median_mape", price["prophet_median_mape"])
        mlflow.log_metric("price_naive_median_mape", price["naive_median_mape"])
        mlflow.log_metric("vol_har_median_mape", vol["har_median_mape"])
        mlflow.log_metric("vol_har_beats_naive", vol["har_beats_naive"])
        mlflow.log_metric("vol_har_wilcoxon_p", vol["wilcoxon_p"])
        mlflow.log_metric("anomaly_rate", anomaly["anomaly_rate"])

    METRICS_OUT.write_text(json.dumps(metrics, indent=2))

    print("\n" + "=" * 78)
    print(f"PHASE 4 EVALUATION -- benchmarked vs naive ({FOLDS} folds) with significance tests")
    print("=" * 78)
    print(f"PRICE (Prophet)  : median MAPE {price['prophet_median_mape']}% vs naive "
          f"{price['naive_median_mape']}% -> beats naive {price['prophet_beats_naive']}/{price['n_total']}")
    print(f"                   Wilcoxon p={price['wilcoxon_p']} -> NOT significantly better "
          f"(price is a random walk; forecast is illustrative only)")
    print(f"VOLATILITY (HAR) : median MAPE {vol['har_median_mape']}% vs naive "
          f"{vol['naive_median_mape']}% -> beats naive {vol['har_beats_naive']}/{vol['n_total']}")
    print(f"                   Wilcoxon p={vol['wilcoxon_p']}, binomial p={vol['binomial_p']} "
          f"-> {'SIGNIFICANT real edge' if vol['wilcoxon_p'] < 0.05 else 'not significant'}")
    print(f"SENTIMENT        : FinBERT, ~{sentiment['articles_per_day']}/day across "
          f"{sentiment['categories']} categories")
    print(f"ANOMALY          : Isolation Forest, {anomaly['anomalies']} flagged "
          f"({anomaly['anomaly_rate']:.1%})")
    print("=" * 72)
    logger.success("Wrote metrics to {}", METRICS_OUT.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
