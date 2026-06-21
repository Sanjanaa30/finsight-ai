"""Isolation Forest anomaly detection on price behaviour.

Flags unusual trading days across all 25 assets using scale-free price features
(so one global model works regardless of price level):
    daily_return, volatility_20d, deviation from MA20, deviation from MA50.

Isolation Forest is unsupervised -- no labels needed. It isolates outliers by how
few splits it takes to separate a point. We log the contamination rate and the
number of anomalies (true precision would need labelled events, which we don't
have). Output: data/processed/anomalies.parquet, one row per ticker per day.
"""

import os
import sys
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import pandas as pd
from loguru import logger
from sklearn.ensemble import IsolationForest

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = PROJECT_ROOT / "data" / "processed" / "finsight.duckdb"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "anomalies.parquet"

CONTAMINATION = 0.02      # expected ~2% of days are anomalous
N_ESTIMATORS = 200
RANDOM_STATE = 42
FEATURES = ["daily_return", "volatility_20d", "ma20_dev", "ma50_dev"]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "finsight-anomaly"


def load_features() -> pd.DataFrame:
    """Build scale-free features from the mart; drop rows with null inputs."""
    con = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    df = con.sql("""
        SELECT date, ticker, close, daily_return, volatility_20d,
               close / ma20 - 1 AS ma20_dev,
               close / ma50 - 1 AS ma50_dev
        FROM fct_daily_signals
    """).df()
    con.close()
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def main() -> None:
    df = load_features()
    X = df[FEATURES].astype("float32")
    logger.info("Training Isolation Forest on {} rows x {} features", len(X), len(FEATURES))

    model = IsolationForest(
        n_estimators=N_ESTIMATORS, contamination=CONTAMINATION, random_state=RANDOM_STATE,
    )
    model.fit(X)

    # decision_function: lower = more anomalous. predict: -1 anomaly, 1 normal.
    df["anomaly_score"] = model.decision_function(X)
    df["is_anomaly"] = model.predict(X) == -1

    n_anom = int(df["is_anomaly"].sum())
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="isolation-forest"):
        mlflow.log_params({
            "contamination": CONTAMINATION, "n_estimators": N_ESTIMATORS,
            "features": ",".join(FEATURES), "n_rows": len(X),
        })
        mlflow.log_metric("n_anomalies", n_anom)
        mlflow.log_metric("anomaly_rate", n_anom / len(X))
        mlflow.sklearn.log_model(model, name="model", registered_model_name="anomaly_isolation_forest")

    out = df[["date", "ticker", "close", *FEATURES, "anomaly_score", "is_anomaly"]]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)

    logger.success(
        "Done. {} anomalies / {} rows ({:.1%}) -> {}",
        n_anom, len(X), n_anom / len(X), OUTPUT_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
