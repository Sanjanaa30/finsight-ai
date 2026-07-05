"""FinSight AI -- FastAPI backend.

Exposes the platform's data layers as REST endpoints the Streamlit dashboard
calls. Reuses the Phase 5 MCP tools (run_forecast, get_sentiment_score) and the
LangGraph agent where they fit, and adds history/feed/heatmap endpoints the
charts need.

Run:  uvicorn serving.api:app --reload --port 8000   (docs at /docs)
"""

import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.agent import analyst                                  # noqa: E402
from ai.assistant import ask as assistant_ask                 # noqa: E402
from ai.mcp_server import get_sentiment_score, run_forecast, search_filings  # noqa: E402
from serving import live_compute, live_news                   # noqa: E402

DUCKDB = str(PROJECT_ROOT / "data" / "processed" / "finsight.duckdb")
NEWS_SCORED = str(PROJECT_ROOT / "data" / "processed" / "news_scored.parquet")
ANOMALIES = str(PROJECT_ROOT / "data" / "processed" / "anomalies.parquet")

TICKERS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM",
    "NFLX", "AMD", "BRK-B", "BTC-USD", "ETH-USD", "SOL-USD", "CL=F", "GC=F",
    "SI=F", "NG=F", "^NSEI", "^FTSE", "^GDAXI", "^N225", "^GSPC", "^IXIC",
]

# ---- Live quotes + background refresher (runs even with zero viewers) ---------
TAPE = ["NVDA", "AAPL", "BTC-USD", "GC=F", "^NSEI", "TSLA", "^GSPC"]
LIVE_TTL = int(os.getenv("FINSIGHT_LIVE_TTL", str(3 * 3600)))  # refresh every 3h (override via env)
# Demo mode: serve real HISTORICAL data only -- no yfinance/NewsAPI/Claude calls, no scheduler.
# Safe for a public live demo (no external cost or rate limits). Enable with FINSIGHT_DEMO=1.
DEMO = os.getenv("FINSIGHT_DEMO", "").lower() in ("1", "true", "yes")
DEMO_REPORTS = PROJECT_ROOT / "data" / "processed" / "demo_reports"
_live_cache: dict = {"t": 0.0, "data": None}
_quote_cache: dict = {}  # per-ticker live-quote cache for the Overview price card


def _stored_quote(ticker: str) -> dict:
    """A quote built from the latest STORED close (used in demo mode / when yfinance is down)."""
    con = _con()
    df = con.execute("""SELECT close, daily_return, date FROM fct_daily_signals
                        WHERE ticker = ? ORDER BY date DESC LIMIT 1""", [ticker]).fetchdf()
    con.close()
    if df.empty:
        raise HTTPException(404, f"No stored data for {ticker}")
    r = df.iloc[0]
    return {"ticker": ticker, "price": float(r["close"]),
            "change_pct": float(r["daily_return"] or 0) * 100,
            "fetched_at": f"historical ({str(r['date'])[:10]})"}


def _stored_live() -> dict:
    """The ticker-tape built from stored closes (demo mode)."""
    con = _con()
    quotes = []
    for s in TAPE:
        df = con.execute("""SELECT close, daily_return FROM fct_daily_signals
                            WHERE ticker = ? ORDER BY date DESC LIMIT 1""", [s]).fetchdf()
        if not df.empty:
            r = df.iloc[0]
            quotes.append({"ticker": s, "price": float(r["close"]),
                           "change_pct": float(r["daily_return"] or 0) * 100})
    con.close()
    return {"quotes": quotes, "source": "historical snapshot (demo mode)",
            "fetched_at": "historical", "previous_fetched_at": None}


def _fetch_live_quotes() -> dict:
    import yfinance as yf
    quotes = []
    for s in TAPE:
        try:
            fi = yf.Ticker(s).fast_info
            last, prev = float(fi["last_price"]), float(fi["previous_close"])
            quotes.append({"ticker": s, "price": last, "change_pct": (last / prev - 1) * 100})
        except Exception:  # noqa: BLE001
            continue
    return {"quotes": quotes, "source": "yfinance (near-real-time, ~15m delayed)",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def _refresh_live_cache() -> None:
    prev = _live_cache["data"]["fetched_at"] if _live_cache["data"] else None
    data = _fetch_live_quotes()
    data["previous_fetched_at"] = prev  # None on the very first fetch since startup
    _live_cache["data"], _live_cache["t"] = data, time.time()


# Tickers whose full analytics (price + Prophet + HAR-RV + anomalies) the scheduler
# warms every cycle. Bounded to the tape/default set so background compute stays
# small; any other ticker is computed on first view and then cached for LIVE_TTL.
WARM_TICKERS = TAPE


def _refresh_all() -> None:
    """One full refresh of every live layer -- runs in the background scheduler."""
    _refresh_live_cache()                       # 1) ticker-tape quotes
    for t in WARM_TICKERS:                       # 2) price analytics (prices/forecast/anomalies)
        try:
            live_compute.live_prices(t)
            live_compute.live_forecast(t)
            live_compute.live_anomalies(t)
        except Exception:  # noqa: BLE001
            continue
    try:
        live_news.refresh()                      # 3) news + sentiment (NewsAPI + FinBERT)
    except Exception:  # noqa: BLE001
        pass


async def _live_scheduler() -> None:
    """Refresh ALL live layers every LIVE_TTL -- independent of any dashboard viewer."""
    while True:
        try:
            await asyncio.to_thread(_refresh_all)
        except Exception:  # noqa: BLE001
            pass
        await asyncio.sleep(LIVE_TTL)


@asynccontextmanager
async def lifespan(_app):
    task = None if DEMO else asyncio.create_task(_live_scheduler())  # no refresher in demo
    try:
        yield
    finally:
        if task is not None:
            task.cancel()


app = FastAPI(title="FinSight AI API", version="1.0", lifespan=lifespan)

_finbert = None  # lazy-loaded FinBERT for the manual analyzer


def _records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe list of dicts (dates to str, NaN to None)."""
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype).startswith("datetime"):
            df[c] = df[c].astype(str)
    return df.where(df.notna(), None).to_dict(orient="records")


def _con():
    return duckdb.connect(DUCKDB, read_only=True)


class AgentRequest(BaseModel):
    ticker: str


class AskRequest(BaseModel):
    question: str
    history: list = []


class TextRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"service": "FinSight AI API", "tickers": len(TICKERS),
            "docs": "/docs", "demo": DEMO}


@app.get("/tickers")
def tickers() -> list[str]:
    return TICKERS


@app.get("/prices/{ticker}")
def prices(ticker: str, days: int = Query(180, ge=10, le=2000)):
    """OHLCV + moving averages for charting. Live (yfinance) with stored fallback."""
    ticker = ticker.upper()
    try:
        return live_compute.live_prices(ticker, days)
    except Exception:  # noqa: BLE001 -- yfinance hiccup: fall back to the stored mart
        pass
    con = _con()
    df = con.execute(
        """SELECT p.date, p.open, p.high, p.low, p.close, p.volume,
                  f.ma20, f.ma50, f.daily_return, f.volatility_20d
           FROM stg_prices p JOIN fct_daily_signals f
             ON p.ticker = f.ticker AND p.date = f.date
           WHERE p.ticker = ? ORDER BY p.date DESC LIMIT ?""",
        [ticker, days],
    ).fetchdf()
    con.close()
    if df.empty:
        raise HTTPException(404, f"No price data for {ticker}")
    df = df.iloc[::-1]  # back to ascending
    latest = _records(df.tail(1))[0]
    return {"ticker": ticker, "latest": latest, "history": _records(df)}


@app.get("/forecast/{ticker}")
def forecast(ticker: str):
    """7-day price + next-week volatility forecast. Live recompute with stored fallback."""
    try:
        return live_compute.live_forecast(ticker)
    except Exception:  # noqa: BLE001 -- fall back to the pre-computed forecast
        pass
    out = run_forecast(ticker)
    if "error" in out:
        raise HTTPException(404, out["error"])
    return out


@app.get("/anomalies/{ticker}")
def anomalies(ticker: str, days: int = Query(548, ge=7, le=2000)):
    """Anomalies flagged in the last `days` days. Live Isolation Forest, stored fallback."""
    ticker = ticker.upper()
    try:
        return live_compute.live_anomalies(ticker, days)
    except Exception:  # noqa: BLE001 -- fall back to the stored anomalies
        pass
    df = duckdb.execute(
        f"""WITH mx AS (SELECT max(date) m FROM read_parquet('{ANOMALIES}') WHERE ticker = ?)
            SELECT a.date, a.close, a.daily_return, a.anomaly_score, a.is_anomaly
            FROM read_parquet('{ANOMALIES}') a, mx
            WHERE a.ticker = ? AND a.is_anomaly AND a.date >= mx.m - INTERVAL '{days}' DAY
            ORDER BY a.date DESC""",
        [ticker, ticker],
    ).fetchdf()
    return {"ticker": ticker, "anomalies": _records(df)}


@app.get("/sentiment")
def sentiment():
    """Latest sentiment per category. Live (NewsAPI+FinBERT) with stored fallback."""
    try:
        return live_news.live_sentiment()
    except Exception:  # noqa: BLE001
        return get_sentiment_score()


@app.get("/sentiment/trend")
def sentiment_trend():
    """Daily avg sentiment + article counts per category. Live with stored fallback."""
    try:
        return live_news.live_sentiment_trend()
    except Exception:  # noqa: BLE001
        pass
    con = _con()
    df = con.execute(
        """SELECT date, category, avg_sentiment, article_count
           FROM int_sentiment_daily ORDER BY date, category"""
    ).fetchdf()
    con.close()
    return {"trend": _records(df)}


@app.get("/sentiment/drivers")
def sentiment_drivers():
    """Top positive & negative headline per category ('what moved it'). Live-only."""
    try:
        return live_news.live_sentiment_drivers()
    except Exception:  # noqa: BLE001
        return {"drivers": {}, "live": False}


@app.get("/news")
def news(category: str | None = None, label: str | None = None,
         limit: int = Query(50, ge=1, le=300)):
    """Filterable scored-news feed. Live (NewsAPI+FinBERT) with stored fallback."""
    try:
        return live_news.live_news(category, label, limit)
    except Exception:  # noqa: BLE001
        pass
    con = _con()
    clauses, params = [], []
    if category:
        clauses.append("category = ?"); params.append(category.lower())
    if label:
        clauses.append("sentiment_label = ?"); params.append(label.lower())
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)
    df = con.execute(
        f"""SELECT category, source_name, title, url, published_date,
                   sentiment_score, sentiment_label
            FROM stg_news_scored {where}
            ORDER BY published_date DESC LIMIT ?""",
        params,
    ).fetchdf()
    con.close()
    return {"news": _records(df)}


@app.get("/heatmap")
def heatmap():
    """Latest daily return + volatility for every ticker (cross-asset grid)."""
    con = _con()
    df = con.execute(
        """SELECT ticker, daily_return, volatility_20d, close
           FROM fct_daily_signals
           QUALIFY row_number() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
           ORDER BY ticker"""
    ).fetchdf()
    con.close()
    return {"assets": _records(df)}


@app.get("/live")
def live():
    """Ticker-tape quotes, served from the cache the background scheduler keeps fresh (3h)."""
    if DEMO:
        return _stored_live()
    if _live_cache["data"] is None:
        _refresh_live_cache()  # cold start, before the scheduler's first tick lands
    return _live_cache["data"]


@app.get("/quote/{ticker}")
def quote(ticker: str):
    """One live quote for any ticker (Overview price card). Cached per-ticker for LIVE_TTL (3h)."""
    ticker = ticker.upper()
    if DEMO:
        return _stored_quote(ticker)
    c = _quote_cache.get(ticker)
    if c and time.time() - c["t"] < LIVE_TTL:
        return c["data"]
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        last, prev = float(fi["last_price"]), float(fi["previous_close"])
        data = {"ticker": ticker, "price": last, "change_pct": (last / prev - 1) * 100,
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    except Exception:  # noqa: BLE001
        if c:
            return c["data"]  # serve the last good quote rather than fail
        return _stored_quote(ticker)  # last resort: the stored close
    _quote_cache[ticker] = {"t": time.time(), "data": data}
    return data


@app.post("/sentiment/score")
def score_text(req: TextRequest):
    """Score arbitrary text with FinBERT (the manual analyzer)."""
    global _finbert
    if _finbert is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        mdl = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        mdl.eval()
        _finbert = (tok, mdl, mdl.config.id2label)
    import torch
    tok, mdl, id2label = _finbert
    enc = tok(req.text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        probs = torch.softmax(mdl(**enc).logits, dim=1)[0]
    d = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
    return {
        "text": req.text,
        "score": round(d.get("positive", 0) - d.get("negative", 0), 3),
        "label": max(d, key=d.get),
        "probabilities": {k: round(v, 3) for k, v in d.items()},
    }


@app.get("/filings/search")
def filings_search(query: str, ticker: str | None = None,
                   top_k: int = Query(5, ge=1, le=20)):
    """Semantic search over SEC filings (RAG)."""
    return search_filings(query, ticker=ticker, top_k=top_k)


@app.post("/agent")
def agent(req: AgentRequest):
    """Run the LangGraph analyst agent and return the report (Claude Haiku).

    In demo mode, serve a PRE-GENERATED report (no Claude call).
    """
    tk = req.ticker.upper()
    if DEMO:
        path = DEMO_REPORTS / f"{tk}.md"
        if path.exists():
            return {"ticker": tk, "report": path.read_text(encoding="utf-8")}
        return {"ticker": tk, "report": (
            f"### Demo mode\n\nA pre-generated report isn't available for **{tk}** in this demo. "
            "Pre-built reports exist for the featured tickers (e.g. NVDA, AAPL, BTC-USD). To generate "
            "a live report for any asset, run FinSight locally with an `ANTHROPIC_API_KEY`.")}
    result = analyst.invoke({"ticker": req.ticker})
    return {"ticker": tk, "report": result["report"]}


@app.post("/ask")
def ask(req: AskRequest):
    """Conversational analyst -- answers any finance/dashboard question using live-data tools."""
    if DEMO:
        return {"answer": (
            "💡 **Demo mode.** The live chat assistant needs a Claude API connection, which is turned "
            "off in this public demo (to avoid API cost/abuse). Everything else runs on real historical "
            "data — explore the **Overview / Forecast / News** tabs and the pre-built **AI report** on "
            "the Overview tab. To chat freely about any ticker, run FinSight locally with your own "
            "`ANTHROPIC_API_KEY`.")}
    try:
        return {"answer": assistant_ask(req.question, req.history)}
    except Exception as exc:  # noqa: BLE001
        return {"answer": f"Sorry — I ran into an error: {exc}"}
