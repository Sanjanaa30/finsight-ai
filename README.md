---
title: FinSight AI
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
---

# FinSight AI

[![Live Demo](https://img.shields.io/badge/Live_Demo-Hugging_Face_Spaces-FFD21E?style=for-the-badge)](https://sanjanaaa28-finsight-ai.hf.space)

**▶️ Live demo:** https://sanjanaaa28-finsight-ai.hf.space

## Overview

FinSight AI is an end-to-end financial-intelligence platform that combines market
data, machine-learning models, and LLM reasoning in a single web app. It tracks 25
assets across five classes — US equities, cryptocurrencies, commodities, international
indices, and benchmark indices — turning raw market and news data into something clear
and readable.

For any asset, the platform shows its current price, a short-term price forecast, a
volatility outlook, any trading days flagged as unusual, and the sentiment of the day's
financial news. On demand it writes a full analysis grounded in the asset's own market
data and SEC filings, and a built-in assistant answers open-ended finance questions by
querying the platform's data through tools.

I built it in six phases — environment setup, data ingestion, data engineering, machine
learning, AI engineering, and serving — which together form a complete pipeline from
public data sources to a deployed app. Figures are computed on live data, with an
offline pipeline kept as a fallback.

Every forecasting model is benchmarked against a simple baseline, and the interface is upfront about how far to trust each output.
Short-term price direction has no genuine edge, so it's labelled illustrative;
volatility measurably beats its baseline, so that's the signal the platform actually
stands behind. The honesty framework below explains the approach.

---

## Status — all 6 phases complete ✅

Every phase below is built and working end to end.

| Phase | In plain words | State |
|-------|----------------|-------|
| **1 — Setup** | Repo, folders, environment, dependencies, Docker | ✅ |
| **2 — Data ingestion** | Pull raw data from four public sources into files | ✅ |
| **3 — Data engineering** | Clean & shape the data with dbt on DuckDB; orchestrate with Prefect | ✅ |
| **4 — Machine learning** | Volatility, price, sentiment & anomaly models — each measured against a simple baseline | ✅ |
| **5 — AI engineering** | Search SEC filings (RAG), an MCP server, and a LangGraph analyst agent | ✅ |
| **6 — Serving (live)** | FastAPI backend + Streamlit dashboard + conversational assistant, all on **live** data | ✅ |

Figures are computed on live data, with the offline pipeline kept as a fallback.

---

## What you can do (the dashboard)

A four-tab web app. Everything you see is computed on live data, refreshed from the
latest pipeline run rather than a stale snapshot.

- **Overview** — a ticker tape, then the selected asset's price, 7-day forecast, recent
  anomaly count, and volatility. A price-and-forecast chart shows where the asset sits
  in its 90-day range, and a market-mood panel pulls together a sentiment gauge,
  per-topic cards, trend arrows, and the headlines driving them. One click generates an
  AI report [you can download as a PDF].
- **Forecast** — the volatility outlook, which is the forecast worth trusting, alongside the
  7-day price forecast, which is labelled illustrative for a reason. An error gauge and a
  "behind the scenes" panel show how each forecast was made and how it scored against a
  naive baseline.
- **News** — sentiment by category and over time, a live FinBERT analyzer (paste any
  headline, get a score), and a filterable news feed.
- **Analyst** — a chat assistant. Ask a general finance question ("what is a P/E ratio?") or
  something about the data ("which asset is most volatile?", "compare AAPL and MSFT") and
  it pulls live data through tools to answer. It declines non-finance questions.

---

## Tech stack

- **Data ingestion:** yfinance · NewsAPI · SEC EDGAR · FRED
- **ML:** HAR-RV (volatility) · Prophet · FinBERT · Isolation Forest · scikit-learn · MLflow
- **Engineering:** dbt (dbt-duckdb) · DuckDB · Parquet (PyArrow) · pandas · Prefect · PostgreSQL
- **AI:** LangGraph · LangChain · MCP (official SDK) · sentence-transformers · Qdrant (vector store) · Anthropic Claude Haiku 4.5
- **Serving:** FastAPI · Uvicorn · Streamlit · Plotly · xhtml2pdf (PDF export)
- **Infra:** Docker · docker-compose (PostgreSQL + Qdrant)
- **Tooling:** Python 3.13 · uv · loguru · pytest

---

## How the live system fits together

```
                       ┌──────────────────────  BROWSER  ──────────────────────┐
                       │              Streamlit dashboard (port 8501)          │
                       │        Overview · Forecast · News · Analyst           │
                       └───────────────────────────┬───────────────────────────┘
                                                   │  HTTP (JSON)
                                                   ▼
                       ┌──────────────────────  FastAPI (port 8000)  ─────────── ┐
                       │  /prices /forecast /anomalies /quote /live  (prices)    │
                       │  /sentiment /sentiment/trend /news /sentiment/drivers   │
                       │  /agent (report)   /ask (chat assistant)                │
                       │                                                         │
                       │  Background scheduler: every 3h it refreshes all of     │
                       │  the above on its own (works even with no one watching) │
                       └───────┬─────────────────────────────┬───────────────────┘
                               │ live compute (cached 3h)    │ falls back to ↓
             ┌─────────────────▼──────────────┐   ┌──────────▼───────────────────┐
             │ live_compute.py                │   │  Stored pipeline outputs     │
             │  yfinance history → re-run     │   │  (Phases 2–5): DuckDB marts, │
             │  Prophet + HAR-RV + IsoForest  │   │  ML parquets, Qdrant filings │
             │ live_news.py                   │   │  — used if a live fetch fails│
             │  NewsAPI → FinBERT scoring     │   └──────────────────────────────┘
             └────────────────────────────────┘
```

The offline pipeline (Phases 2–5) still exists and is the **safety net**: if a live
fetch ever fails (no internet, no API key), each endpoint quietly serves the last
stored result instead of blanking.

---

## Project structure

```
finsight-ai/
├── ingestion/                      # Phase 2 — source → raw files
│   ├── fetch_prices.py             # 25 assets via yfinance → prices.parquet
│   ├── fetch_filings.py            # SEC EDGAR 10-K text (12 US stocks) → filings/*.txt
│   ├── fetch_news.py               # 6 news categories via NewsAPI → news.parquet
│   └── fetch_macro.py              # 4 FRED macro indicators → macro.parquet
├── dbt/                            # Phase 3 — transformations on DuckDB
│   └── models/  staging/ · intermediate/ · marts/   (+ _sources.yml, tests)
├── pipelines/flows/
│   └── daily_pipeline.py           # Phase 3 — Prefect flow: ingest → sentiment → dbt run → test
├── ml/                             # Phase 4 — model training + evaluation
│   ├── train_volatility.py         # HAR-RV volatility forecast (the reliable one)
│   ├── train_prophet.py            # Prophet price forecast (illustrative) → forecasts.parquet
│   ├── train_sentiment.py          # FinBERT scoring → news_scored.parquet
│   ├── train_anomaly.py            # Isolation Forest → anomalies.parquet
│   └── evaluate.py                 # honest benchmarks vs naive → evaluation_metrics.json
├── ai/                             # Phase 5 + assistant
│   ├── rag_pipeline.py             # clean + chunk + embed SEC filings → Qdrant
│   ├── mcp_server.py               # MCP server: 4 tools over the data layers
│   ├── prompts.py                  # analyst system prompt + honesty rules
│   ├── agent.py                    # LangGraph analyst agent (fixed-flow report, now live)
│   └── assistant.py                # conversational tool-using assistant (powers the /ask chat)
├── serving/                        # Phase 6 — the live app
│   ├── api.py                      # FastAPI backend + 3-hourly background scheduler
│   ├── dashboard.py                # Streamlit 4-tab dashboard
│   ├── live_compute.py             # live price / forecast / anomalies (yfinance, cached, fallback)
│   └── live_news.py                # live news / sentiment (NewsAPI + FinBERT, cached, fallback)
├── .streamlit/config.toml          # dashboard theme
├── data/                           # (gitignored) raw + processed data
│   ├── raw/                        # Parquet files + filings/ text
│   └── processed/                  # finsight.duckdb + ML output parquets
├── docker-compose.yml              # Postgres + Qdrant
├── tests/                          # pytest unit tests
├── .env / .env.example             # API keys (.env is gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/), and Docker Desktop (for Qdrant, used by the Analyst/filings features).

```powershell
# 1. Clone and enter
git clone https://github.com/Sanjanaa30/finsight-ai.git
cd finsight-ai

# 2. Create the virtual environment and install deps
uv venv
.venv\Scripts\activate          # PowerShell  (mac/linux: source .venv/bin/activate)
uv pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env            # then fill in real values in .env

# 4. (once) fetch the dbt package + embed SEC filings for the Analyst
dbt deps --project-dir dbt --profiles-dir dbt
docker compose up -d qdrant
python ai/rag_pipeline.py
```

### API keys

| Key (`.env`) | Source | Needed for | Free tier |
|--------------|--------|-----------|-----------|
| `NEWSAPI_KEY` | newsapi.org | live news & sentiment | 100 req/day |
| `FRED_KEY` | fred.stlouisfed.org | macro ingestion | unlimited |
| `SEC_USER_AGENT` | your name + email | SEC EDGAR etiquette | header only |
| `ANTHROPIC_API_KEY` | console.anthropic.com | the AI analyst + chat (Claude Haiku) | ~$1/1M in, $5/1M out |

`yfinance` and `SEC EDGAR` need no key. Without `NEWSAPI_KEY` the app still runs —
news/sentiment just fall back to the last stored data.

---

## ▶️ Running the dashboard (the main way to use it)

You need **two** processes: the **API** (does the live fetching) and the **dashboard**
(the display). Open two terminals:

```powershell
# Terminal 1 — the API (also runs the 3-hourly background refresh)
.venv\Scripts\python.exe -m uvicorn serving.api:app --port 8000

# Terminal 2 — the dashboard
.venv\Scripts\streamlit.exe run serving/dashboard.py --server.port 8501
```

Then open **http://localhost:8501**.

- The dashboard **only displays**; the API **does the fetching** — so both must be running.
- Data caches are **in-memory**, so a fresh start always fetches **live** (never stale).
- The **Analyst tab and filings search** also need Docker/Qdrant up (`docker compose up -d qdrant`).
- The refresh interval is configurable: set `FINSIGHT_LIVE_TTL` (seconds) before starting the API.

### Demo mode

Start the API with **`FINSIGHT_DEMO=1`** to run on stored historical data only — no external API
calls (yfinance / NewsAPI / Claude), so it's safe for a public demo. The AI report is served from
pre-generated files (`serving/pregenerate_reports.py`) and the free-form chat is disabled.

---

## Phase 2 — Data ingestion (getting the raw data)

Scripts pull raw data from public APIs and save it **as-is** (cleaning is dbt's job).
Everything lands in `data/raw/` (gitignored).

| Script | Source | Output |
|--------|--------|--------|
| `fetch_prices.py` | yfinance | daily OHLCV for 25 tickers since 2022 → `prices.parquet` |
| `fetch_filings.py` | SEC EDGAR | 12 latest 10-K annual reports → `filings/*.txt` |
| `fetch_news.py` | NewsAPI | ~580 articles across 6 categories → `news.parquet` |
| `fetch_macro.py` | FRED | fed rate, CPI, unemployment, GDP → `macro.parquet` |

**The 25 assets:** 12 US stocks (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, JPM, XOM,
NFLX, AMD, BRK-B) · 3 crypto (BTC, ETH, SOL) · 4 commodities (oil, gold, silver, natural
gas) · 4 country indices (Nifty 50, FTSE 100, DAX, Nikkei 225) · 2 benchmarks (S&P 500, Nasdaq).

**The 6 news categories:** geopolitical, commodities, crypto, country, macro, general market.

---

## Phase 3 — Data engineering (cleaning & shaping)

**dbt** turns the raw Parquet into clean, tested, ML-ready tables, using **DuckDB**
as the engine (it reads Parquet directly — no load step). Output: one database at
`data/processed/finsight.duckdb`.

| Layer | Models | Purpose (plain words) |
|-------|--------|-----------------------|
| **staging** | `stg_prices`, `stg_news`, `stg_macro` | fix types, drop bad rows, de-duplicate |
| **intermediate** | `int_price_features`, `int_sentiment_daily` | build features (returns, moving averages, 20-day volatility); daily news averages |
| **marts** | `fct_daily_signals` | one wide, ML-ready row per ticker per day |

**21 data-quality tests** (not-null, unique `(ticker, date)`, valid categories) run
after every build. **Prefect** ties it together into one daily flow: *ingest → score
sentiment → dbt run → dbt test*. **Docker** provides PostgreSQL + Qdrant.

```powershell
python pipelines/flows/daily_pipeline.py        # run the whole pipeline once
```

---

## Phase 4 — Machine learning (the models)

Four models, each tracked in **MLflow**, and each **benchmarked against the simple
baseline it must beat**. This is where the project's honesty comes from.

| Model | What it does | Honest result |
|-------|--------------|---------------|
| **HAR-RV volatility** (`train_volatility.py`) | forecasts next-week "jumpiness" | ✅ **beats naive on 22/25 assets** — real edge |
| **Prophet price** (`train_prophet.py`) | 7-day price forecast + range | ⚠️ **illustrative only** — does *not* beat naive |
| **FinBERT sentiment** (`train_sentiment.py`) | scores each news headline −1…+1 | powers all sentiment |
| **Isolation Forest** (`train_anomaly.py`) | flags unusual trading days | ~2% flagged; catches real shocks |

**The honest forecasting story (important):** short-term prices are basically a
*random walk*, so "next week = today" is very hard to beat — and Prophet **doesn't
beat it**. So the price forecast is kept only to *visualise* trend/seasonality and is
labelled **illustrative** everywhere. **Volatility**, on the other hand, *clusters* and
**is** predictable — so the HAR-RV model genuinely beats the baseline. Across the app,
volatility is presented as the trustworthy signal and price as illustrative.

---

## Phase 5 — AI engineering (grounded analysis)

An analyst that grounds its answers in **primary-source SEC filings** and the
platform's own models.

1. **RAG over SEC filings** (`ai/rag_pipeline.py`) — cleans the 10-Ks, splits them into
   240-token chunks, embeds with `all-MiniLM-L6-v2`, and stores ~5K chunks in **Qdrant**
   for semantic search.
2. **MCP server** (`ai/mcp_server.py`) — 4 tools (`get_stock_price`, `get_sentiment_score`,
   `run_forecast`, `search_filings`) usable by the agent *and* any MCP client (e.g. Claude Desktop).
3. **LangGraph agent** (`ai/agent.py`) — a fixed flow (price → sentiment → forecast →
   filings → write-up) that ends with **Claude Haiku** writing a structured report. In
   Phase 6 its data was switched to the **live** layer, so reports match the dashboard.

The report obeys the same **honesty rules**: price forecast = illustrative, volatility =
the real signal, sentiment = market-wide (not per-asset), every filing claim quoted from
retrieved text.

---

## Phase 6 — Serving: the live app

This phase makes everything **live** and puts it behind a real UI.

**The live compute layer** (`serving/live_compute.py`, `serving/live_news.py`) re-runs
the Phase-4/5 logic on **fresh** data on demand:
- Pulls the latest history from **yfinance** and recomputes price, moving averages,
  20-day volatility, the **Prophet** 7-day forecast, the **HAR-RV** volatility forecast,
  and **Isolation-Forest** anomalies — anchored to *today*.
- Pulls fresh **NewsAPI** headlines and re-scores them with **FinBERT** for live sentiment.
- Everything is **cached** (default 3 hours) and **falls back to the stored pipeline
  output** if a live fetch fails — so the app never blanks.
- The price forecast respects each asset's real trading calendar (stocks skip weekends;
  crypto trades 7 days a week).

**The FastAPI backend** (`serving/api.py`) exposes it all as simple JSON endpoints
(prices, forecast, anomalies, live quote, ticker tape, sentiment, news, the AI report,
and the chat assistant). It also runs a **background scheduler**: on startup and then
every 3 hours it refreshes all the live data **on its own** — so it stays fresh even
when nobody has the dashboard open.

**The Streamlit dashboard** (`serving/dashboard.py`) is the 4-tab UI described in
*"What you can do"* above. Design goals throughout: **plain-English** labels, tooltips,
a consistent **0–100 sentiment scale**, **low-sample data faded** so noisy readings can't
masquerade as solid signal, and the **illustrative-vs-reliable** distinction shown
everywhere.

**The conversational assistant** (`ai/assistant.py`) powers the Analyst tab: a
tool-using Claude agent that can call the live tools (price, forecast, anomalies,
sentiment, news, filings, cross-asset compare) to answer free-form questions. It works
for **any** real ticker, answers **any finance** question, and politely **declines
off-topic** questions.

**Extras:** the AI report exports to **PDF** (via xhtml2pdf); the ticker tape and KPIs
show live timestamps; the whole UI is theme-styled and responsive.

---

## The honesty framework

Most finance demos over-promise. This one is built the other way: the interface is
designed to tell you how much to trust each number.

- Price forecasts are labelled *illustrative* wherever they appear, because they don't
  beat a naive baseline. A price target is never dressed up as reliable.
- Volatility is the highlighted signal, because it measurably does beat its baseline.
- Every forecast shows its own error (MAPE) next to a model-vs-naive scorecard, so you
  can see the evidence rather than take the number on faith.
- Sentiment shows its sample size. Scores built on a handful of articles are faded and
  flagged as less reliable.
- Live and stored data are always distinguished, and the app degrades to stored data
  rather than failing.
- The assistant won't give buy or sell advice — it explains the data instead.

## What it doesn't do

Being clear about the boundaries is part of the same principle.

**Modelling.** Short-term price forecasting has no real edge — prices behave like a random
walk over days, so the Prophet forecast doesn't beat a "tomorrow = today" baseline. It's
kept only to picture the recent trend, never as a target. Volatility does beat its
baseline (that's the point of HAR-RV), but it isn't precise: realised volatility is noisy,
and absolute error stays high (~50% MAPE). Anomaly detection is unsupervised, so a flag
means "worth a look," not a confirmed event — there's no labelled ground truth.

**Data.** Prices come from yfinance, an unofficial feed delayed about 15 minutes — fine for
analysis, not for trading. News is shallow: NewsAPI's free tier caps out around 30 days of
history and 100 requests a day across a limited set of outlets, sentiment is scored on
headlines rather than full articles, and it's market-wide by category rather than
per-asset. FinBERT can also miss sarcasm or nuance. SEC coverage is annual and US-only —
RAG runs over twelve US companies' 10-Ks, which go stale between yearly filings, with no
quarterly reports or international filings.

**System.** It runs locally: caches live in memory and clear on restart, and the 3-hour
refresher runs inside the API process, so it isn't a true background job if the API is
down. The first request for a new asset is slow — fitting Prophet takes a few seconds and
the first news refresh runs FinBERT for around thirty — but the cache and scheduler hide
that for most views. The cross-asset "most volatile" ranking reads the stored daily
snapshot for speed, while per-asset views are fully live. And the forecast calendar skips
weekends but not exchange holidays, so a holiday can still show up as a forecast day.

**Scope.** This is not financial advice and it isn't connected to any brokerage — it
explains data, it doesn't trade or recommend trades. The assistant is grounded in tools,
but like any LLM it can still be wrong.

## Future scope

Rough order of value.

**Deployment.** Move off localhost — Streamlit Cloud or a container host, a hosted API, and
Qdrant Cloud. Swap the in-memory cache for Redis and the in-process refresher for a real
scheduled job (Celery or cron) that survives the API going down. Add monitoring for live
model error and data drift.

**Modelling.** Reframe forecasting around something predictable — return *direction* or a
*probability* rather than a price — and try gradient-boosted or sequence models (LightGBM,
N-BEATS, Temporal Fusion Transformer) with macro and sentiment features. The honest
benchmarking stays either way. Add per-exchange holiday calendars for exact trading days,
and wire automated retraining into MLflow with versioning.

**Data & AI.** Deeper sentiment — score full articles, add sources (RSS, GDELT, Reddit),
and move to per-asset sentiment with entity linking. Richer RAG — more companies,
quarterly 10-Qs, earnings-call transcripts, international filings. And intraday streaming
over WebSockets instead of the delayed daily feed.

**Product.** Portfolios, watchlists, and alerts on anomalies or sentiment shifts; user
accounts with saved conversations; and a backtesting or paper-trading module to test ideas
without real money.

---

## Disclaimer & data attribution

**FinSight AI is a personal, educational project — not financial advice.** Nothing here is a
recommendation to buy, sell, or hold any asset; figures may be delayed or historical and must
not be used for real trading decisions.

**Data sources** (all used under their respective terms; not affiliated with or endorsed by any
provider):
- **Prices** — Yahoo Finance, via the `yfinance` library.
- **News** — NewsAPI (`newsapi.org`).
- **Filings** — U.S. SEC EDGAR (public domain).
- **Macro** — FRED, Federal Reserve Bank of St. Louis.

All trademarks belong to their respective owners. The free tiers of these APIs are intended for
development; a production/commercial deployment should switch to officially licensed data feeds.

