"""Live news + sentiment -- re-runs the news pipeline on demand.

Mirrors ingestion/fetch_news.py (6 NewsAPI category queries) + ml/train_sentiment.py
(FinBERT scoring, sentiment = P(positive) - P(negative)) so the dashboard's News tab
can show fresh, freshly-scored articles instead of the last pipeline run.

One cached fetch+score (all 6 categories) backs all three shapes -- /news,
/sentiment, /sentiment/trend -- and is refreshed at most every LIVE_TTL. The API
endpoints fall back to the stored DuckDB tables if NewsAPI or FinBERT is
unavailable, so the dashboard never blanks.
"""

import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

LIVE_TTL = int(os.getenv("FINSIGHT_LIVE_TTL", str(3 * 3600)))
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
ENDPOINT = "https://newsapi.org/v2/everything"
SEARCH_IN = "title,description"

# Exactly the queries from ingestion/fetch_news.py so live == pipeline semantics.
EXCLUDE = (
    'NOT (football OR soccer OR "World Cup" OR esports OR NBA OR NFL OR cricket '
    'OR "Premier League" OR Bundesliga OR Olympics OR movie OR film OR celebrity '
    'OR Bayern OR "Real Madrid")'
)
_BASE_QUERIES = {
    "geopolitical": '(sanctions OR tariffs OR "trade war" OR "geopolitical") AND (economy OR market OR markets OR oil OR trade)',
    "commodities": '("crude oil" OR "oil prices" OR "gold prices" OR "natural gas" OR commodities) AND (price OR market OR futures OR supply)',
    "crypto": '(bitcoin OR ethereum OR solana OR cryptocurrency OR crypto) AND (price OR market OR trading OR token OR blockchain)',
    "country": '"Indian economy" OR "UK economy" OR "German economy" OR "Japanese economy" OR "Nifty 50" OR "FTSE 100" OR "Nikkei 225"',
    "macro": '(inflation OR "interest rates" OR "Federal Reserve" OR "GDP growth" OR unemployment) AND (economy OR market OR policy OR rate)',
    "market": '("stock market" OR "S&P 500" OR Nasdaq OR "Dow Jones" OR "corporate earnings") AND (stocks OR shares OR investors OR trading)',
}
CATEGORY_QUERIES = {cat: f"({q}) AND {EXCLUDE}" for cat, q in _BASE_QUERIES.items()}

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32
MAX_TOKENS = 128

_cache: dict = {"t": 0.0, "df": None}
_finbert = None


def _load_finbert():
    global _finbert
    if _finbert is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        mdl.eval()
        _finbert = (tok, mdl, mdl.config.id2label)
    return _finbert


def score_texts(texts: list[str]) -> tuple[list[float], list[str]]:
    """FinBERT scores + labels, batched -- shared by the live feed and /sentiment/score."""
    import torch

    tok, mdl, id2label = _load_finbert()
    scores, labels = [], []
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tok(texts[i:i + BATCH_SIZE], padding=True, truncation=True,
                  max_length=MAX_TOKENS, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(mdl(**enc).logits, dim=1).numpy()
        for p in probs:
            d = {id2label[j].lower(): float(p[j]) for j in range(len(p))}
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
            labels.append(max(d, key=d.get))
    return scores, labels


def _fetch_category(category: str, query: str) -> list[dict]:
    params = {"q": query, "searchIn": SEARCH_IN, "language": "en",
              "sortBy": "publishedAt", "pageSize": 100, "apiKey": NEWSAPI_KEY}
    r = requests.get(ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "ok":
        raise RuntimeError(f"NewsAPI error for {category}: {payload.get('message')}")
    return [{"category": category,
             "source_name": (a.get("source") or {}).get("name"),
             "title": a.get("title"), "url": a.get("url"),
             "published_at": a.get("publishedAt")} for a in payload["articles"]]


def _articles() -> pd.DataFrame:
    """Cached scored-article frame for all 6 categories. Raises if news/FinBERT down."""
    if _cache["df"] is not None and time.time() - _cache["t"] < LIVE_TTL:
        return _cache["df"]
    if not NEWSAPI_KEY:
        raise RuntimeError("NEWSAPI_KEY not set")
    rows = []
    for cat, q in CATEGORY_QUERIES.items():
        rows.extend(_fetch_category(cat, q))
    df = pd.DataFrame(rows)
    df["title"] = df["title"].fillna("").astype(str)
    df = df[df["title"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("NewsAPI returned no scorable articles")
    df["sentiment_score"], df["sentiment_label"] = score_texts(df["title"].tolist())
    df["published_at_dt"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_at_dt"]).reset_index(drop=True)
    df["published_date"] = df["published_at_dt"].dt.date.astype(str)
    _cache["df"], _cache["t"] = df, time.time()
    return df


def refresh() -> int:
    """Force a rebuild (used by the background scheduler). Returns article count."""
    _cache["t"] = 0.0
    return len(_articles())


def _daily(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["published_date", "category"])
              .agg(avg_sentiment=("sentiment_score", "mean"),
                   article_count=("sentiment_score", "size"))
              .reset_index())


def live_news(category: str | None = None, label: str | None = None, limit: int = 50) -> dict:
    df = _articles()
    if category:
        df = df[df["category"] == category.lower()]
    if label:
        df = df[df["sentiment_label"] == label.lower()]
    # Same article can match >1 category query -> show each headline once in the feed.
    df = (df.sort_values("published_at_dt", ascending=False)
            .drop_duplicates(subset="title", keep="first").head(limit))
    cols = ["category", "source_name", "title", "url", "published_date", "published_at",
            "sentiment_score", "sentiment_label"]
    out = df[cols].where(df[cols].notna(), None).to_dict(orient="records")
    return {"news": out, "live": True}


def live_sentiment(category: str | None = None) -> dict:
    g = _daily(_articles())
    latest = g.sort_values("published_date").groupby("category").tail(1)  # freshest day per cat
    if category:
        latest = latest[latest["category"] == category.lower()]
    cats = {r["category"]: {"avg_sentiment": round(float(r["avg_sentiment"]), 3),
                            "article_count": int(r["article_count"])}
            for _, r in latest.iterrows()}
    return {"categories": cats, "live": True}


def live_sentiment_trend() -> dict:
    g = _daily(_articles()).sort_values(["published_date", "category"])
    rows = [{"date": r["published_date"], "category": r["category"],
             "avg_sentiment": round(float(r["avg_sentiment"]), 4),
             "article_count": int(r["article_count"])} for _, r in g.iterrows()]
    return {"trend": rows, "live": True}


def live_sentiment_drivers() -> dict:
    """The single most-positive and most-negative headline per category ('what moved it')."""
    df = _articles().dropna(subset=["sentiment_score"])
    out = {}
    for cat, g in df.groupby("category"):
        if g.empty:
            continue
        pos = g.loc[g["sentiment_score"].idxmax()]
        neg = g.loc[g["sentiment_score"].idxmin()]
        out[cat] = {
            "top_pos": {"title": pos["title"], "source": pos["source_name"] or "—",
                        "score": round(float(pos["sentiment_score"]), 3), "url": pos["url"]},
            "top_neg": {"title": neg["title"], "source": neg["source_name"] or "—",
                        "score": round(float(neg["sentiment_score"]), 3), "url": neg["url"]},
        }
    return {"drivers": out, "live": True}
