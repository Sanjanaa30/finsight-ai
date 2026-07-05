"""Live on-demand recompute of the price-derived analytics from fresh market data.

Each function pulls recent daily history from yfinance and re-runs the SAME model
logic as the ml/ training scripts -- Prophet (train_prophet.py), HAR-RV
(train_volatility.py), Isolation Forest (train_anomaly.py) -- so the dashboard can
show analytics anchored to TODAY rather than the last pipeline run.

Results are cached per (kind, ticker) for LIVE_TTL. The API endpoints call these
first and fall back to the stored DuckDB/parquet outputs if a live fetch or fit
fails, so a yfinance hiccup never blanks the dashboard.

Honesty note carried over from the offline models: the PRICE forecast is
illustrative only (prices ~ random walk); the VOLATILITY forecast (HAR-RV) is the
one with real predictive edge. Live data re-anchors these -- it does not make the
price forecast reliable.
"""

import logging
import os
import time
import warnings

import numpy as np
import pandas as pd

# Prophet / cmdstanpy are extremely chatty -- silence them for on-demand use.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

LIVE_TTL = int(os.getenv("FINSIGHT_LIVE_TTL", str(3 * 3600)))  # match the /live cadence
DEMO = os.getenv("FINSIGHT_DEMO", "").lower() in ("1", "true", "yes")  # serve stored data only

FORECAST_DAYS = 7
HOLDOUT_DAYS = 7
HIGH_UNCERTAINTY_MAPE = 15.0
HAR_HORIZON = 5
HAR_WINDOWS = (5, 22, 66)
ANOM_CONTAMINATION = 0.02
ANOM_FEATURES = ["daily_return", "volatility_20d", "ma20_dev", "ma50_dev"]

_cache: dict = {}  # (kind, ticker) -> {"t": epoch, "data": ...}


def _cached(kind: str, ticker: str, builder):
    """Return a cached result for (kind, ticker) or (re)build and store it."""
    key = (kind, ticker)
    hit = _cache.get(key)
    if hit and time.time() - hit["t"] < LIVE_TTL:
        return hit["data"]
    data = builder()
    _cache[key] = {"t": time.time(), "data": data}
    return data


def _history(ticker: str, period: str = "3y") -> pd.DataFrame:
    """Fresh daily OHLCV + the same features the mart computes. Raises on empty."""
    if DEMO:  # demo mode -> skip the live fetch so callers fall back to stored data
        raise RuntimeError("demo mode: using stored data")
    import yfinance as yf

    raw = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    if raw is None or raw.empty:
        raise ValueError(f"no live history for {ticker}")
    df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["daily_return"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["volatility_20d"] = df["daily_return"].rolling(20).std()
    return df


def _history_cached(ticker: str) -> pd.DataFrame:
    """One cached yfinance download per ticker -- shared by prices/forecast/anomalies."""
    return _cached("history", ticker, lambda: _history(ticker))


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            df[c] = df[c].astype(str)
    return df.where(df.notna(), None).to_dict(orient="records")


# ---- Prices -------------------------------------------------------------------
def live_prices(ticker: str, days: int = 180) -> dict:
    ticker = ticker.upper()
    df = _history_cached(ticker)  # slicing by `days` is cheap -> no separate cache
    cols = ["date", "open", "high", "low", "close", "volume",
            "ma20", "ma50", "daily_return", "volatility_20d"]
    hist = df[cols].tail(days).reset_index(drop=True)
    latest = _records(hist.tail(1))[0]
    return {"ticker": ticker, "latest": latest, "history": _records(hist), "live": True}


# ---- Forecast (Prophet price + HAR-RV volatility) -----------------------------
def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def _prophet_forecast(df: pd.DataFrame):
    """Holdout MAPE + naive baseline + full-history 7-day forecast (mirrors train_prophet.py).

    Returns (future_df, model_mape, naive_mape, trend_per_day, last_close) so the
    dashboard can show a transparent 'how this was predicted' scorecard.
    """
    from prophet import Prophet

    tdf = df[["date", "close"]].rename(columns={"date": "ds", "close": "y"}).dropna()
    last_close = float(tdf["y"].iloc[-1])

    def _fresh() -> "Prophet":
        return Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)

    cutoff = tdf["ds"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train, test = tdf[tdf["ds"] <= cutoff], tdf[tdf["ds"] > cutoff]
    score = naive = float("nan")
    if not train.empty and not test.empty:
        m = _fresh(); m.fit(train)
        hc = m.predict(m.make_future_dataframe(periods=HOLDOUT_DAYS + FORECAST_DAYS))
        merged = test.merge(hc[["ds", "yhat"]], on="ds", how="inner")
        if len(merged):
            actual = merged["y"].to_numpy()
            score = _mape(actual, merged["yhat"].to_numpy())
            # Naive baseline: "next week = last known price" (persistence).
            naive = _mape(actual, np.full(len(actual), float(train["y"].iloc[-1])))

    m = _fresh(); m.fit(tdf)  # production model on ALL history
    # Match the asset's real trading cadence: crypto trades 7 days/week (its history has
    # weekend dates) -> forecast every day; stocks/indices are weekdays only -> business days.
    freq = "D" if (tdf["ds"].dt.weekday >= 5).mean() > 0.1 else "B"
    fc = m.predict(m.make_future_dataframe(periods=FORECAST_DAYS, freq=freq))
    future = fc[fc["ds"] > tdf["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    # Prophet's own trend component: average slope over the forecast horizon.
    trend = fc["trend"].to_numpy()
    trend_per_day = float((trend[-1] - trend[len(tdf) - 1]) / FORECAST_DAYS) if len(trend) > len(tdf) else 0.0
    return future, score, naive, trend_per_day, last_close


def _har_vol(df: pd.DataFrame) -> dict | None:
    """Next-week realized-vol forecast (HAR-RV) -- mirrors train_volatility.py."""
    from sklearn.linear_model import LinearRegression

    s = df.set_index("date")["daily_return"].dropna()
    if len(s) < 150:
        return None
    feat = pd.DataFrame({"ret": s})
    cols = [f"rv_{w}" for w in HAR_WINDOWS]
    for w in HAR_WINDOWS:
        feat[f"rv_{w}"] = feat["ret"].rolling(w).std()
    feat["target"] = feat["ret"].rolling(HAR_HORIZON).std().shift(-HAR_HORIZON)
    feat = feat.dropna()
    if len(feat) < 150:
        return None
    model = LinearRegression().fit(feat[cols], feat["target"])
    latest = pd.DataFrame({c: [s.rolling(w).std().iloc[-1]] for c, w in zip(cols, HAR_WINDOWS)})
    cur = float(s.rolling(5).std().iloc[-1])
    pred = float(model.predict(latest)[0])
    return {"current": round(cur, 4), "predicted_next_week": round(pred, 4),
            "direction": "rising" if pred > cur else "falling",
            "note": "HAR-RV model; beats naive baseline on 88% of assets (real edge)."}


def live_forecast(ticker: str) -> dict:
    ticker = ticker.upper()

    def build():
        df = _history_cached(ticker)
        future, score, naive, trend_per_day, last_close = _prophet_forecast(df)
        central = float(future["yhat"].iloc[-1]) if len(future) else last_close
        out = {
            "ticker": ticker,
            "price_forecast": [
                {"date": str(pd.Timestamp(r["ds"]).date()),
                 "forecast": round(float(r["yhat"]), 2),
                 "low": round(float(r["yhat_lower"]), 2),
                 "high": round(float(r["yhat_upper"]), 2)}
                for _, r in future.iterrows()
            ],
            "price_mape_pct": None if np.isnan(score) else round(score, 1),
            "price_high_uncertainty": bool(score > HIGH_UNCERTAINTY_MAPE) if not np.isnan(score) else False,
            "price_note": "Illustrative only; price forecasting does not beat naive persistence.",
            # Transparent 'how this was predicted' scorecard for the dashboard.
            "backtest": {
                "model_mape_pct": None if np.isnan(score) else round(score, 1),
                "naive_mape_pct": None if np.isnan(naive) else round(naive, 1),
                "beats_naive": bool(not np.isnan(score) and not np.isnan(naive) and score < naive),
                "holdout_days": HOLDOUT_DAYS,
            },
            "drivers": {
                "last_close": round(last_close, 2),
                "trend_per_day": round(trend_per_day, 2),
                "trend_direction": ("rising" if trend_per_day > 0.01
                                    else "falling" if trend_per_day < -0.01 else "flat"),
                "forecast_vs_last_pct": round((central / last_close - 1) * 100, 1) if last_close else None,
                "history_days": int(len(df)),
            },
            "live": True,
        }
        vol = _har_vol(df)
        if vol:
            out["volatility_forecast"] = vol
        return out

    return _cached("forecast", ticker, build)


# ---- Anomalies (Isolation Forest on this ticker's live history) ---------------
def _anomaly_frame(ticker: str) -> pd.DataFrame:
    """Cached: full 3y history with is_anomaly flags. Fit once; windowing is then cheap."""

    def build():
        from sklearn.ensemble import IsolationForest

        df = _history_cached(ticker).copy()  # copy: we add feature columns below
        df["ma20_dev"] = df["close"] / df["ma20"] - 1
        df["ma50_dev"] = df["close"] / df["ma50"] - 1
        df = df.dropna(subset=ANOM_FEATURES).reset_index(drop=True)
        if len(df) < 60:
            return df.iloc[0:0]
        X = df[ANOM_FEATURES].astype("float32")
        model = IsolationForest(n_estimators=200, contamination=ANOM_CONTAMINATION, random_state=42)
        model.fit(X)
        df["anomaly_score"] = model.decision_function(X)
        df["is_anomaly"] = model.predict(X) == -1
        return df[["date", "close", "daily_return", "anomaly_score", "is_anomaly"]]

    return _cached("anomaly_frame", ticker, build)


def live_anomalies(ticker: str, days: int = 90) -> dict:
    """Anomalies flagged in the last `days` calendar days (the model still fits on 3y)."""
    ticker = ticker.upper()
    df = _anomaly_frame(ticker)
    if df.empty:
        return {"ticker": ticker, "anomalies": [], "live": True, "window_days": days}
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    hit = df[df["is_anomaly"] & (df["date"] >= cutoff)].iloc[::-1]
    return {"ticker": ticker, "anomalies": _records(hit), "live": True, "window_days": days}
