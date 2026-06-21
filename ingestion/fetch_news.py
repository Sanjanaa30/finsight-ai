"""Fetch news articles across 6 global categories via NewsAPI -> tagged Parquet.

Each category is one query against NewsAPI's /v2/everything endpoint. Every
article is tagged with its category so Phase 4 can compute per-category FinBERT
sentiment. Raw landing only.

Free 'Developer' tier: 100 requests/day, articles up to ~1 month old. Six
queries per run sits comfortably under the limit. Requires NEWSAPI_KEY in .env.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_KEY = os.getenv("NEWSAPI_KEY")
ENDPOINT = "https://newsapi.org/v2/everything"
RATE_LIMIT_SECONDS = 1.0  # be polite between queries

# 6 global news categories -> search query.
# Queries are anchored to a finance/market context (AND clause) so off-topic
# matches (e.g. a "war" TV drama) are excluded. Combined with SEARCH_IN below,
# matching is restricted to the title+description, which kills most body-text
# false positives. Tune these freely; they trade recall for precision.
CATEGORY_QUERIES = {
    "geopolitical": '(sanctions OR tariffs OR "trade war" OR "geopolitical") AND (economy OR market OR markets OR oil OR trade)',
    "commodities": '("crude oil" OR "oil prices" OR "gold prices" OR "natural gas" OR commodities) AND (price OR market OR futures OR supply)',
    "crypto": '(bitcoin OR ethereum OR solana OR cryptocurrency OR crypto) AND (price OR market OR trading OR token OR blockchain)',
    "country": '"Indian economy" OR "UK economy" OR "German economy" OR "Japanese economy" OR "Nifty 50" OR "FTSE 100" OR "Nikkei 225"',
    "macro": '(inflation OR "interest rates" OR "Federal Reserve" OR "GDP growth" OR unemployment) AND (economy OR market OR policy OR rate)',
    "market": '("stock market" OR "S&P 500" OR Nasdaq OR "Dow Jones" OR "corporate earnings") AND (stocks OR shares OR investors OR trading)',
}

# Restrict keyword matching to these fields (NewsAPI default searches the body
# too, which is the main source of off-topic matches).
SEARCH_IN = "title,description"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "news.parquet"


def fetch_category(category: str, query: str) -> list[dict]:
    """Fetch up to 100 recent English articles for one category."""
    params = {
        "q": query,
        "searchIn": SEARCH_IN,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "apiKey": API_KEY,
    }
    resp = requests.get(ENDPOINT, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error for '{category}': {payload}")

    rows = []
    for art in payload["articles"]:
        rows.append({
            "category": category,
            "query": query,
            "source": (art.get("source") or {}).get("name"),
            "author": art.get("author"),
            "title": art.get("title"),
            "description": art.get("description"),
            "url": art.get("url"),
            "published_at": art.get("publishedAt"),
            "content": art.get("content"),
        })
    logger.info("{}: {} articles", category, len(rows))
    return rows


def main() -> None:
    if not API_KEY:
        raise RuntimeError("NEWSAPI_KEY not set in .env")

    all_rows: list[dict] = []
    for category, query in CATEGORY_QUERIES.items():
        all_rows.extend(fetch_category(category, query))
        time.sleep(RATE_LIMIT_SECONDS)

    df = pd.DataFrame(all_rows)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df["ingested_at"] = pd.Timestamp.now(tz="UTC")  # when this run fetched the data

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    logger.success(
        "Wrote {} articles across {} categories to {}",
        len(df), df["category"].nunique(), OUTPUT_PATH.relative_to(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
