"""FinSight analyst agent -- a LangGraph StateGraph over the 4 MCP tools.

Flow (matches the guide):
    fetch_price -> fetch_sentiment -> fetch_forecast -> search_filings -> generate_report

The first four nodes gather data via the MCP tools; the final node calls Claude
Haiku to synthesize a grounded, honesty-constrained report (see ai/prompts.py).

The tool functions are imported directly from mcp_server (they ARE the MCP tools);
the standalone stdio server in mcp_server.py still works for external MCP clients.

Run:  python ai/agent.py NVDA
"""

import os
import sys

# Load the cached embedding model offline -> no HuggingFace Hub chatter.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from typing import TypedDict

# Make the project root importable when run directly (python ai/agent.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

from ai.mcp_server import get_sentiment_score, get_stock_price, run_forecast, search_filings
from ai.prompts import ANALYST_SYSTEM_PROMPT, build_analysis_prompt

load_dotenv()

MODEL = "claude-haiku-4-5"
MAX_TOKENS = 1536

# Sentiment is market/category-level -> pick the category that fits the asset.
CRYPTO = {"BTC-USD", "ETH-USD", "SOL-USD"}
COMMODITIES = {"CL=F", "GC=F", "SI=F", "NG=F"}


def category_for_ticker(ticker: str) -> str:
    t = ticker.upper()
    if t in CRYPTO:
        return "crypto"
    if t in COMMODITIES:
        return "commodities"
    return "market"


class AnalystState(TypedDict, total=False):
    ticker: str
    price: dict
    sentiment: dict
    forecast: dict
    filings: dict
    report: str


def node_fetch_price(state: AnalystState) -> dict:
    """Live price/features (yfinance) so the report matches the dashboard; stored fallback."""
    ticker = state["ticker"]
    try:
        from serving.live_compute import live_prices
        lt = live_prices(ticker)["latest"]
        rnd = lambda v, d: (round(float(v), d) if v is not None else None)  # noqa: E731
        return {"price": {"ticker": ticker.upper(), "date": str(lt["date"])[:10],
                          "close": rnd(lt["close"], 2), "daily_return": rnd(lt["daily_return"], 4),
                          "ma20": rnd(lt["ma20"], 2), "ma50": rnd(lt["ma50"], 2),
                          "volatility_20d": rnd(lt["volatility_20d"], 4)}}
    except Exception:  # noqa: BLE001
        return {"price": get_stock_price(ticker)}


def node_fetch_sentiment(state: AnalystState) -> dict:
    cat = category_for_ticker(state["ticker"])
    try:
        from serving.live_news import live_sentiment
        return {"sentiment": live_sentiment(cat)}
    except Exception:  # noqa: BLE001
        return {"sentiment": get_sentiment_score(cat)}


def node_fetch_forecast(state: AnalystState) -> dict:
    try:
        from serving.live_compute import live_forecast
        return {"forecast": live_forecast(state["ticker"])}
    except Exception:  # noqa: BLE001
        return {"forecast": run_forecast(state["ticker"])}


def node_search_filings(state: AnalystState) -> dict:
    res = search_filings(
        "business overview, competitive position, and principal risks",
        ticker=state["ticker"], top_k=3,
    )
    # Only assets with a filed 10-K (US stocks) return hits; otherwise omit.
    return {"filings": res if res.get("results") else {}}


def node_generate_report(state: AnalystState) -> dict:
    data = {k: state.get(k) for k in ("price", "sentiment", "forecast", "filings")}
    # Drop tool errors so they don't reach the prompt.
    for key in ("price", "forecast"):
        if isinstance(data.get(key), dict) and "error" in data[key]:
            data[key] = None
    user_prompt = build_analysis_prompt(state["ticker"], {k: v for k, v in data.items() if v})

    llm = ChatAnthropic(model=MODEL, max_tokens=MAX_TOKENS)
    response = llm.invoke([
        SystemMessage(content=ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    return {"report": response.content}


def build_agent():
    g = StateGraph(AnalystState)
    g.add_node("fetch_price", node_fetch_price)
    g.add_node("fetch_sentiment", node_fetch_sentiment)
    g.add_node("fetch_forecast", node_fetch_forecast)
    g.add_node("search_filings", node_search_filings)
    g.add_node("generate_report", node_generate_report)

    g.add_edge(START, "fetch_price")
    g.add_edge("fetch_price", "fetch_sentiment")
    g.add_edge("fetch_sentiment", "fetch_forecast")
    g.add_edge("fetch_forecast", "search_filings")
    g.add_edge("search_filings", "generate_report")
    g.add_edge("generate_report", END)
    return g.compile()


analyst = build_agent()


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    logger.info("Analyzing {} with the LangGraph analyst agent ...", ticker)
    result = analyst.invoke({"ticker": ticker})
    print("\n" + "=" * 74)
    print(result["report"])
    print("=" * 74)


if __name__ == "__main__":
    main()
