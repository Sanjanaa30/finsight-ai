"""FinSight MCP server -- exposes the platform's data as 4 Model Context Protocol tools.

Built with the official MCP Python SDK (FastMCP). Each tool reads from the data
layers we built in earlier phases:
    get_stock_price     -> fct_daily_signals  (DuckDB mart)
    get_sentiment_score -> int_sentiment_daily (DuckDB mart)
    run_forecast        -> forecasts.parquet + volatility_forecasts.parquet
    search_filings      -> Qdrant 'filings' collection (RAG index)

Run as a stdio MCP server:  python ai/mcp_server.py
The Phase 5 LangGraph agent calls these tools; any MCP client (e.g. Claude
Desktop) can use them too.
"""

import sys
from pathlib import Path
from typing import Optional

import duckdb
from mcp.server.fastmcp import FastMCP

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUCKDB_PATH = str(PROJECT_ROOT / "data" / "processed" / "finsight.duckdb")
FORECASTS = str(PROJECT_ROOT / "data" / "processed" / "forecasts.parquet")
VOL_FORECASTS = str(PROJECT_ROOT / "data" / "processed" / "volatility_forecasts.parquet")

QDRANT_URL = "http://localhost:6333"
COLLECTION = "filings"
EMBED_MODEL = "all-MiniLM-L6-v2"

mcp = FastMCP("finsight")

# Lazy singletons -- the embedding model / Qdrant client load on first filing search.
_model = None
_qdrant = None


def _embed(text: str) -> list[float]:
    global _model
    if _model is None:
        from transformers.utils import logging as hl
        hl.set_verbosity_error()
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model.encode(text).tolist()


def _qclient():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(url=QDRANT_URL)
    return _qdrant


@mcp.tool()
def get_stock_price(ticker: str) -> dict:
    """Get the latest price and technical features for a ticker.

    Args:
        ticker: Asset symbol exactly as tracked (e.g. NVDA, BTC-USD, ^GSPC).
    Returns the most recent close, daily return, moving averages, and volatility.
    """
    ticker = ticker.upper()
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    df = con.execute(
        """SELECT date, close, daily_return, ma20, ma50, volatility_20d
           FROM fct_daily_signals WHERE ticker = ? ORDER BY date DESC LIMIT 1""",
        [ticker],
    ).fetchdf()
    con.close()
    if df.empty:
        return {"error": f"No price data for ticker '{ticker}'"}
    r = df.iloc[0]
    return {
        "ticker": ticker,
        "date": str(r["date"].date()),
        "close": round(float(r["close"]), 2),
        "daily_return": round(float(r["daily_return"]), 4),
        "ma20": round(float(r["ma20"]), 2),
        "ma50": round(float(r["ma50"]), 2),
        "volatility_20d": round(float(r["volatility_20d"]), 4),
    }


@mcp.tool()
def get_sentiment_score(category: Optional[str] = None) -> dict:
    """Get the latest daily news sentiment (FinBERT) by category.

    Sentiment is market-wide and grouped by news category (not per-ticker).
    Args:
        category: Optional one of geopolitical, commodities, crypto, country,
                  macro, market. Omit for all categories.
    Returns avg_sentiment in [-1, 1] and article_count for the most recent day.
    """
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    where = "WHERE category = ?" if category else ""
    params = [category.lower()] if category else []
    df = con.execute(
        f"""WITH latest AS (SELECT max(date) d FROM int_sentiment_daily)
            SELECT category, article_count, avg_sentiment
            FROM int_sentiment_daily, latest
            WHERE date = latest.d {('AND category = ?' if category else '')}
            ORDER BY category""",
        params,
    ).fetchdf()
    con.close()
    if df.empty:
        return {"error": f"No sentiment data" + (f" for category '{category}'" if category else "")}
    cats = {
        row["category"]: {
            "avg_sentiment": round(float(row["avg_sentiment"]), 3),
            "article_count": int(row["article_count"]),
        }
        for _, row in df.iterrows()
    }
    return {"categories": cats}


@mcp.tool()
def run_forecast(ticker: str) -> dict:
    """Get the 7-day price forecast and the next-week volatility forecast.

    Args:
        ticker: Asset symbol (e.g. NVDA).
    The price forecast is illustrative only (it does not beat a naive baseline --
    prices are a random walk). The volatility forecast (HAR-RV) is the model with
    real predictive edge. high_uncertainty flags assets with MAPE > 15%.
    """
    ticker = ticker.upper()
    price = duckdb.execute(
        f"""SELECT ds, yhat, yhat_lower, yhat_upper, mape, high_uncertainty
            FROM read_parquet('{FORECASTS}') WHERE ticker = ? ORDER BY ds""",
        [ticker],
    ).fetchdf()
    if price.empty:
        return {"error": f"No forecast for ticker '{ticker}'"}

    vol = duckdb.execute(
        f"""SELECT current_volatility, predicted_volatility
            FROM read_parquet('{VOL_FORECASTS}') WHERE ticker = ?""",
        [ticker],
    ).fetchdf()

    out = {
        "ticker": ticker,
        "price_forecast": [
            {
                "date": str(row["ds"].date()),
                "forecast": round(float(row["yhat"]), 2),
                "low": round(float(row["yhat_lower"]), 2),
                "high": round(float(row["yhat_upper"]), 2),
            }
            for _, row in price.iterrows()
        ],
        "price_mape_pct": round(float(price.iloc[0]["mape"]), 1),
        "price_high_uncertainty": bool(price.iloc[0]["high_uncertainty"]),
        "price_note": "Illustrative only; price forecasting does not beat naive persistence.",
    }
    if not vol.empty:
        cur = float(vol.iloc[0]["current_volatility"])
        pred = float(vol.iloc[0]["predicted_volatility"])
        out["volatility_forecast"] = {
            "current": round(cur, 4),
            "predicted_next_week": round(pred, 4),
            "direction": "rising" if pred > cur else "falling",
            "note": "HAR-RV model; beats naive baseline on 88% of assets (real edge).",
        }
    return out


@mcp.tool()
def search_filings(query: str, ticker: Optional[str] = None, top_k: int = 5) -> dict:
    """Semantic search over SEC 10-K filings (RAG).

    Args:
        query: Natural-language question (e.g. "main risk factors").
        ticker: Optional ticker to restrict the search to one company's filing.
        top_k: Number of passages to return (default 5).
    Returns the most relevant filing passages with a similarity score.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    flt = None
    if ticker:
        flt = Filter(must=[FieldCondition(key="ticker", match=MatchValue(value=ticker.upper()))])
    hits = _qclient().query_points(
        COLLECTION, query=_embed(query), limit=top_k, query_filter=flt
    ).points
    return {
        "query": query,
        "results": [
            {
                "ticker": h.payload["ticker"],
                "score": round(float(h.score), 3),
                "text": h.payload["text"],
            }
            for h in hits
        ],
    }


if __name__ == "__main__":
    mcp.run()
