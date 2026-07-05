"""FinSight conversational analyst -- ask anything about finance or this dashboard.

A tool-using Claude agent (LangGraph ReAct). When a question is about a tracked
asset it calls the platform's LIVE data tools (price, forecast, anomalies,
sentiment, news, SEC filings, cross-asset compare); for general finance questions
it answers directly. Honesty rules match the rest of the app: the 7-day PRICE
forecast is illustrative only; the VOLATILITY forecast is the reliable signal.

Used by the /ask endpoint in serving/api.py.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv                                          # noqa: E402
from langchain_anthropic import ChatAnthropic                          # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage           # noqa: E402
from langchain_core.tools import tool                                  # noqa: E402
from langgraph.prebuilt import create_react_agent                      # noqa: E402

load_dotenv()

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1200
ANN = 252 ** 0.5  # trading days -> annualization factor for volatility

TICKERS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "JPM", "XOM", "NFLX",
           "AMD", "BRK-B", "BTC-USD", "ETH-USD", "SOL-USD", "CL=F", "GC=F", "SI=F", "NG=F",
           "^NSEI", "^FTSE", "^GDAXI", "^N225", "^GSPC", "^IXIC"]


@tool
def list_assets() -> list:
    """List the 25 assets this dashboard tracks (their ticker symbols)."""
    return TICKERS


@tool
def get_price(ticker: str) -> dict:
    """Latest LIVE price, daily return %, 20/50-day moving averages and annualized 20-day
    volatility for one tracked ticker (e.g. NVDA, BTC-USD, ^GSPC)."""
    try:
        from serving.live_compute import live_prices
        lt = live_prices(ticker)["latest"]
        return {"ticker": ticker.upper(), "as_of": str(lt["date"])[:10],
                "price": round(lt["close"], 2), "daily_return_pct": round(lt["daily_return"] * 100, 2),
                "ma20": round(lt["ma20"], 2), "ma50": round(lt["ma50"], 2),
                "volatility_annualized_pct": round(lt["volatility_20d"] * ANN * 100, 1)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"no live data for '{ticker}': {e}"}


@tool
def get_forecast(ticker: str) -> dict:
    """7-day PRICE forecast (ILLUSTRATIVE ONLY -- not reliable) plus the next-week VOLATILITY
    forecast (the trustworthy signal) and the model's backtest vs a naive baseline, for one ticker."""
    try:
        from serving.live_compute import live_forecast
        f = live_forecast(ticker)
        pf = f["price_forecast"]
        out = {"ticker": ticker.upper(),
               "price_forecast_7d": round(pf[-1]["forecast"], 2),
               "price_range": [round(min(p["low"] for p in pf), 2), round(max(p["high"] for p in pf), 2)],
               "price_forecast_note": "ILLUSTRATIVE ONLY; does not beat a naive baseline",
               "model_error_mape_pct": f.get("price_mape_pct"),
               "beats_naive_baseline": f.get("backtest", {}).get("beats_naive")}
        if f.get("volatility_forecast"):
            v = f["volatility_forecast"]
            out["volatility_forecast"] = {"current_pct": round(v["current"] * 100, 2),
                                          "predicted_next_week_pct": round(v["predicted_next_week"] * 100, 2),
                                          "direction": v["direction"],
                                          "note": "RELIABLE signal (HAR-RV beats naive on ~88% of assets)"}
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": f"no forecast for '{ticker}': {e}"}


@tool
def get_anomalies(ticker: str) -> dict:
    """Statistically unusual trading days for a ticker in the last ~1.5 years (big/odd moves
    flagged by an Isolation Forest)."""
    try:
        from serving.live_compute import live_anomalies
        a = live_anomalies(ticker)["anomalies"]
        latest = ({"date": a[0]["date"][:10], "return_pct": round(a[0]["daily_return"] * 100, 1)}
                  if a else None)
        return {"ticker": ticker.upper(), "unusual_days_last_18mo": len(a), "most_recent": latest}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@tool
def get_market_sentiment() -> dict:
    """Current news-based market sentiment (0-100, 50 = neutral, higher = more positive) for the
    6 categories: market (US stocks), geopolitical, crypto, macro, commodities, country."""
    try:
        from serving.live_news import live_sentiment
        cats = live_sentiment()["categories"]
        return {k: {"score_0_100": round((v["avg_sentiment"] + 1) / 2 * 100), "articles": v["article_count"]}
                for k, v in cats.items()}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@tool
def get_news(category: str = "", sentiment: str = "") -> list:
    """Recent scored news headlines. category: one of market, geopolitical, crypto, macro,
    commodities, country (or empty = all). sentiment: positive/negative/neutral (or empty = all).
    Returns up to 8 headlines with source and sentiment score (-1..+1)."""
    try:
        from serving.live_news import live_news
        feed = live_news(category or None, sentiment or None, 8)["news"]
        return [{"title": n["title"], "source": n["source_name"], "date": n["published_date"],
                 "category": n["category"], "sentiment": round(n["sentiment_score"], 2)} for n in feed]
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]


@tool
def search_filings(query: str, ticker: str = "") -> list:
    """Search SEC 10-K annual filings for relevant passages (e.g. risk factors, business overview,
    competition). Optional ticker restricts to one US company. Only US stocks have filings."""
    try:
        from ai.mcp_server import search_filings as sf
        res = sf(query, ticker=ticker or None, top_k=3)
        return [{"ticker": r["ticker"], "score": r["score"], "text": r["text"][:400]}
                for r in res.get("results", [])]
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]


@tool
def compare_assets() -> list:
    """Latest daily return % and annualized volatility for EVERY tracked asset (from the latest
    daily snapshot) -- use to answer 'which asset is most volatile / best or worst performer' questions."""
    try:
        import duckdb
        db = str(Path(__file__).resolve().parents[1] / "data" / "processed" / "finsight.duckdb")
        con = duckdb.connect(db, read_only=True)
        rows = con.execute(
            """SELECT ticker, daily_return, volatility_20d FROM fct_daily_signals
               QUALIFY row_number() OVER (PARTITION BY ticker ORDER BY date DESC) = 1
               ORDER BY ticker""").fetchdf()
        con.close()
        return [{"ticker": r.ticker, "daily_return_pct": round(r.daily_return * 100, 2),
                 "volatility_annualized_pct": round(r.volatility_20d * ANN * 100, 1)}
                for r in rows.itertuples()]
    except Exception as e:  # noqa: BLE001
        return [{"error": str(e)}]


TOOLS = [list_assets, get_price, get_forecast, get_anomalies, get_market_sentiment,
         get_news, search_filings, compare_assets]

SYSTEM = """You are FinSight's analyst assistant, embedded in a financial-intelligence dashboard.
Your domain is FINANCE ONLY: markets, assets, investing, the economy, financial concepts, and this
dashboard's data.

SCOPE -- what you will and won't answer:
- Finance / markets / economics / investing questions, and questions about this dashboard's data: ANSWER.
- Anything clearly OUTSIDE finance (general trivia, geography, coding help, chit-chat, homework, etc.):
  politely DECLINE in ONE sentence and redirect -- e.g. "I'm FinSight's finance assistant, so I can only
  help with markets, assets, and finance questions -- try asking about a ticker, forecast, or sentiment."
  Do NOT answer the off-topic question itself.

WHAT YOU CAN LOOK UP:
- get_price / get_forecast / get_anomalies work for ANY real market ticker via live data -- not just the
  dashboard's featured set -- so you can answer about e.g. KO, DIS, or ^VIX too. If a symbol is invalid
  the tool returns an error; then say you couldn't find that ticker.
- compare_assets (rankings like "most volatile") covers only the 25 featured assets -- say so if asked to
  rank "all" assets. SEC-filing search covers only the featured US companies that were indexed.
- Sentiment and news are market-wide by category (market, geopolitical, crypto, macro, commodities,
  country), not per-asset.

RULES:
- Use the tools for any data question; ground every data answer in tool results -- never invent prices,
  scores, or headlines.
- HONESTY: the 7-day PRICE forecast is ILLUSTRATIVE ONLY (it does not beat a naive baseline); never
  present it as a reliable target. The VOLATILITY forecast is the genuinely reliable one -- say so.
- Be concise and plain-English so a non-finance person understands; show the actual numbers you used.
- If a tool errors or data is missing, say so honestly instead of guessing.
- You are not a financial adviser; don't give buy/sell recommendations -- explain the data instead.
"""

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
        _agent = create_react_agent(llm, TOOLS, prompt=SYSTEM)
    return _agent


def ask(question: str, history: list | None = None) -> str:
    """Answer a free-form finance/dashboard question, calling tools as needed."""
    messages = []
    for m in (history or [])[-8:]:
        content = m.get("content", "")
        if m.get("role") == "user":
            messages.append(HumanMessage(content=content))
        elif m.get("role") == "assistant":
            messages.append(AIMessage(content=content))
    messages.append(HumanMessage(content=question))
    result = _get_agent().invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which tracked asset is most volatile right now, and what is NVDA's volatility?"
    print(ask(q))
