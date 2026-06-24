# FinSight AI

AI-powered global financial intelligence platform — stock forecasting, global
news sentiment, RAG over SEC filings, an MCP server, and an autonomous analyst
agent. Built fresh, phase by phase.

You type a stock, crypto symbol, commodity, or country index, and the platform
tells you the current price, a short-term forecast, how the news feels today,
any unusual market behaviour, and a full AI-written analysis on demand —
covering **25 global assets** across 5 categories and **6 global news
categories**.

---

## Status

| Phase | Scope | State |
|-------|-------|-------|
| **1 — Setup** | Repo, structure, env, deps, Docker | ✅ Complete |
| **2 — Data ingestion** | Multi-source ingestion → Parquet / text | ✅ 4 of 5 sources (Reddit deferred) |
| **3 — Data engineering** | dbt models on DuckDB + Prefect orchestration | ✅ Complete |
| **4 — Machine learning** | Volatility (HAR-RV) + Prophet, FinBERT sentiment, anomaly detection, MLflow | ✅ Complete |
| **5 — AI engineering** | RAG over SEC filings, MCP server, LangGraph analyst agent (Claude Haiku) | ✅ Complete |
| 6 — Dashboard + deploy | FastAPI + Streamlit 4-tab dashboard | ⬜ Planned |

---

## Tech stack

**Data ingestion:** yfinance · NewsAPI · SEC EDGAR · FRED · (Reddit/PRAW planned)
**Engineering:** dbt (dbt-duckdb) · DuckDB · Parquet (PyArrow) · Prefect · PostgreSQL · Qdrant (Docker)
**ML:** HAR-RV (volatility) · Prophet · FinBERT · Isolation Forest · scikit-learn · MLflow
**AI:** LangGraph · LangChain · MCP (official SDK) · sentence-transformers · Qdrant · Claude Haiku 4.5
**Serving (planned):** FastAPI · Streamlit
**Tooling:** Python 3.13 · uv · loguru · pytest

---

## Architecture (so far)

```
External APIs ──► ingestion/*.py ──► data/raw/*.parquet ─┐
 (yfinance,          (Phase 2)         (+ filings/*.txt)  │
  NewsAPI,                                                 │  dbt (DuckDB)
  SEC, FRED)                                               ▼  (Phase 3)
                                       staging  ─►  intermediate  ─►  marts
                                       stg_*          int_*            fct_daily_signals
                                                                         │
                            all orchestrated by Prefect (daily flow)     ▼
                            ingest → dbt run → dbt test              fct_daily_signals
                                                                         │
   ┌─────────────────────── Phase 4 ML ───────────────────────┐         ▼
   HAR-RV volatility · Prophet price · FinBERT sentiment · Isolation Forest anomaly
   (tracked in MLflow; volatility/anomaly outputs → parquet)            │
                                                                         ▼
   ┌─────────────────────── Phase 5 AI ───────────────────────┐
   SEC filings → RAG (Qdrant)        4 MCP tools         LangGraph agent (Claude Haiku)
   get_stock_price · get_sentiment_score · run_forecast · search_filings → grounded report
```

---

## Project structure

```
finsight-ai/
├── ingestion/                      # Phase 2 — source → raw landing
│   ├── fetch_prices.py             # 25 assets via yfinance → prices.parquet
│   ├── fetch_filings.py            # SEC EDGAR 10-K text (12 US stocks) → filings/*.txt
│   ├── fetch_news.py               # 6 news categories via NewsAPI → news.parquet
│   ├── fetch_macro.py              # 4 FRED macro indicators → macro.parquet
│   └── fetch_reddit.py             # (deferred — Reddit API access pending)
├── dbt/                            # Phase 3 — transformations on DuckDB
│   ├── dbt_project.yml
│   ├── profiles.yml                # DuckDB connection (no secrets)
│   ├── packages.yml                # dbt_utils
│   └── models/
│       ├── staging/                # stg_prices, stg_news, stg_macro (+ _sources.yml, tests)
│       ├── intermediate/           # int_price_features, int_sentiment_daily (+ tests)
│       └── marts/                  # fct_daily_signals (+ tests)
├── pipelines/flows/
│   └── daily_pipeline.py           # Phase 3 — Prefect flow: ingest → sentiment → dbt run → test
├── ml/                             # Phase 4 — model training + evaluation
│   ├── train_volatility.py         # HAR-RV volatility forecast (beats naive)
│   ├── train_prophet.py            # Prophet price forecast (illustrative) → forecasts.parquet
│   ├── train_sentiment.py          # FinBERT scoring → news_scored.parquet
│   ├── train_anomaly.py            # Isolation Forest → anomalies.parquet
│   └── evaluate.py                 # benchmarks vs naive → evaluation_metrics.json
├── ai/                             # Phase 5 — RAG, MCP server, agent
│   ├── rag_pipeline.py             # clean + chunk + embed SEC filings → Qdrant
│   ├── mcp_server.py               # MCP server: 4 tools over the data layers
│   ├── prompts.py                  # analyst system prompt + honesty rules
│   └── agent.py                    # LangGraph analyst agent (Claude Haiku)
├── serving/                        # Phase 6 — FastAPI + Streamlit
├── data/                           # (gitignored) raw + processed data
│   ├── raw/                        # Parquet files + filings/ text
│   └── processed/                  # finsight.duckdb + ML output parquets
├── docker-compose.yml              # Postgres + Qdrant (used Phase 5+)
├── tests/                          # pytest unit tests
├── .env / .env.example             # API keys (.env is gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/), and Docker Desktop with WSL 2.

```powershell
# 1. Clone and enter
git clone https://github.com/Sanjanaa30/finsight-ai.git
cd finsight-ai

# 2. Create the virtual environment and install deps
uv venv
.venv\Scripts\activate          # PowerShell  (use: source .venv/bin/activate on mac/linux)
uv pip install -r requirements.txt

# 3. Configure API keys
cp .env.example .env            # then fill in real values in .env

# 4. Fetch the dbt package dependency
dbt deps --project-dir dbt --profiles-dir dbt
```

### API keys

| Key (`.env`) | Source | Needed for | Free tier |
|--------------|--------|-----------|-----------|
| `NEWSAPI_KEY` | newsapi.org | news ingestion | 100 req/day |
| `FRED_KEY` | fred.stlouisfed.org | macro ingestion | unlimited |
| `SEC_USER_AGENT` | (your name + email) | SEC EDGAR etiquette | n/a — header only |
| `REDDIT_CLIENT_ID` / `REDDIT_SECRET` / `REDDIT_USER_AGENT` | reddit.com/prefs/apps | (deferred) Reddit | 100 req/min |
| `ANTHROPIC_API_KEY` | console.anthropic.com | Phase 5 analyst agent (Claude Haiku) | ~$1/1M in, $5/1M out |

`yfinance`, `SEC EDGAR`, and `GDELT` need no key. `OPENAI_API_KEY` is unused (the
guide's default; this build uses Claude instead). `.env` is gitignored — keys are never committed.

---

## Phase 2 — Data ingestion

Ingestion scripts pull raw data from external APIs and land it **as-is** (no
cleaning — that is dbt's job). Tabular sources are written in **tidy long format**
(one row per entity + date) with an `ingested_at` audit timestamp. All outputs go
to `data/raw/` (gitignored).

| Script | Source | Output | Shape (approx.) |
|--------|--------|--------|-----------------|
| `fetch_prices.py` | yfinance | `data/raw/prices.parquet` | ~29K rows · 25 tickers · daily OHLCV since 2022-01-01 |
| `fetch_filings.py` | SEC EDGAR | `data/raw/filings/*.txt` | 12 latest 10-K filings (~6 MB) |
| `fetch_news.py` | NewsAPI | `data/raw/news.parquet` | ~580 articles · 6 categories · last ~30 days |
| `fetch_macro.py` | FRED | `data/raw/macro.parquet` | ~340 obs · 4 series (fed rate, CPI, unemployment, GDP) since 2018 |

**The 25 assets:** 12 US stocks (NVDA, AAPL, MSFT, AMZN, GOOGL, META, TSLA, JPM,
XOM, NFLX, AMD, BRK-B) · 3 crypto (BTC, ETH, SOL) · 4 commodities (WTI oil, gold,
silver, natural gas) · 4 country indices (Nifty 50, FTSE 100, DAX, Nikkei 225) ·
2 benchmarks (S&P 500, Nasdaq Composite).

**The 6 news categories:** geopolitical, commodities, crypto, country-specific,
macro, general market. Queries are anchored to a finance context and matched on
title + description to keep off-topic articles out.

---

## Phase 3 — Data engineering

dbt transforms the raw Parquet into clean, tested, ML-ready tables, using DuckDB
as the query engine (it reads Parquet directly — no load step). Output is a single
DuckDB database at `data/processed/finsight.duckdb`.

**Model layers:**

| Layer | Models | Materialization | Purpose |
|-------|--------|-----------------|---------|
| **staging** | `stg_prices`, `stg_news`, `stg_macro` | view | 1:1 cleaning — cast types, drop nulls, de-duplicate |
| **intermediate** | `int_price_features`, `int_sentiment_daily` | view | feature engineering (window functions); daily news aggregates |
| **marts** | `fct_daily_signals` | table | wide ML-ready table, one row per ticker per day |

**`int_price_features`** computes per-ticker `daily_return`, `ma20`, `ma50`, and
`volatility_20d` via SQL window functions.

**`fct_daily_signals`** joins price features with macro context (attached via an
**ASOF join** so each daily row carries the latest macro value released on or
before that date) and daily news context, including `news_avg_sentiment` (filled
by the Phase 4 FinBERT model).

**Data-quality tests (21, all passing):** `not_null` on key columns, composite
`unique` on `(ticker, date)` / `(series_id, date)`, and `accepted_values` on the
news category. Run automatically after every build.

**Infrastructure:** `docker-compose.yml` defines PostgreSQL + Qdrant. They are not
needed for the dbt work (which runs on DuckDB) and are used from Phase 5 onward:

```powershell
docker compose up -d        # start
docker compose ps           # status
docker compose stop         # stop (data is kept)
```

---

## Phase 4 — Machine learning

Four models, each tracked in MLflow. Every forecasting model is **benchmarked
against the naive baseline it must beat** — accuracy is reported honestly.

| Model | Script | What it does | Result |
|-------|--------|--------------|--------|
| **HAR-RV volatility** | `ml/train_volatility.py` | forecasts next-week realized volatility | **beats naive on 22/25 assets** — real predictive edge |
| **Prophet price** | `ml/train_prophet.py` | 7-day price forecast + confidence bands | illustrative only (see below) |
| **FinBERT sentiment** | `ml/train_sentiment.py` | scores news sentiment (batched, CPU) | fills `news_avg_sentiment` |
| **Isolation Forest** | `ml/train_anomaly.py` | flags unusual market behaviour | 2% flagged; rediscovers FTX crash, earnings spikes |

**The honest forecasting story.** Price *levels* are close to a random walk, so a
naive "next week = today" baseline is hard to beat — and in a rolling backtest
Prophet **does not beat it** (median MAPE 4.6% vs naive 2.9%, 0/25 wins). Prophet
is kept for the dashboard's price *visualisation* (trend/seasonality + confidence
bands), not as a predictive edge. Forecasts carry a `high_uncertainty` flag
(MAPE > 15%) so volatile assets are never shown as trustworthy.

Volatility, by contrast, **clusters and is genuinely predictable**, so the
**HAR-RV** model (realized vol over 5/22/66-day windows) beats naive vol-persistence
on **22/25 assets** — the project's forecast with real edge. (Absolute MAPE on
volatility is high for everyone, ~50%, because realized vol is intrinsically noisy;
the meaningful result is beating the baseline.)

`ml/evaluate.py` reruns all benchmarks and writes `evaluation_metrics.json`.
Models train separately from the daily data pipeline (retraining on every refresh
would be wasteful), and never go in git — only the MLflow registry.

---

## Phase 5 — AI engineering

A conversational analyst that grounds its answers in primary-source SEC filings
and the platform's own models.

**1. RAG over SEC filings** (`ai/rag_pipeline.py`) — cleans the inline-XBRL noise
out of the 10-Ks, token-chunks the narrative (240 tokens to fit the embedder),
embeds with `all-MiniLM-L6-v2` (384-dim, CPU), and upserts ~5K chunks into Qdrant.

**2. MCP server** (`ai/mcp_server.py`) — an official-SDK MCP server exposing 4 tools,
each reading a different data layer. Works with the agent below *and* any MCP client
(e.g. Claude Desktop):

| Tool | Source |
|------|--------|
| `get_stock_price` | `fct_daily_signals` (DuckDB) |
| `get_sentiment_score` | `int_sentiment_daily` (market/category-level) |
| `run_forecast` | price + volatility forecast parquets |
| `search_filings` | Qdrant RAG index |

**3. LangGraph agent** (`ai/agent.py` + `ai/prompts.py`) — a `StateGraph`
(`fetch_price → fetch_sentiment → fetch_forecast → search_filings → generate_report`)
whose final node calls **Claude Haiku** to synthesize a structured report.

**The honesty constraints carry through to the report.** The system prompt forces
the agent to present the price forecast as *illustrative only* (it doesn't beat
naive), lead with the *volatility* forecast (the model with real edge), flag
high-uncertainty assets, treat sentiment as market-level, and ground every filing
claim in retrieved text — verified on both a US stock (NVDA) and a no-filing crypto
asset (SOL-USD). Requires `ANTHROPIC_API_KEY` and a running Qdrant.

---

## Running the pipeline

**Everything at once (recommended)** — the Prefect flow runs all ingestion in
parallel, then `dbt run`, then `dbt test`:

```powershell
python pipelines/flows/daily_pipeline.py
```

To run on a schedule (6pm weekdays), see the `__main__` block in
`pipelines/flows/daily_pipeline.py` (`daily_pipeline.serve(cron="0 18 * * 1-5")`).

**Or run pieces manually:**

```powershell
# Ingest a single source
python ingestion/fetch_prices.py

# Build / test the dbt models
dbt run  --project-dir dbt --profiles-dir dbt
dbt test --project-dir dbt --profiles-dir dbt

# Train models (needs MLflow running: mlflow ui --backend-store-uri sqlite:///mlflow.db)
python ml/train_sentiment.py     # FinBERT (run before dbt if rebuilding sentiment)
python ml/train_volatility.py    # HAR-RV volatility
python ml/train_prophet.py       # Prophet price forecast
python ml/train_anomaly.py       # Isolation Forest
python ml/evaluate.py            # honest benchmarks vs naive

# AI layer (needs Qdrant running: docker compose up -d qdrant, and ANTHROPIC_API_KEY)
python ai/rag_pipeline.py        # embed SEC filings into Qdrant (run once)
python ai/agent.py NVDA          # run the analyst agent on a ticker
```

### Inspect the data

```python
import duckdb
# Raw Parquet:
duckdb.sql("SELECT * FROM 'data/raw/prices.parquet' LIMIT 10").df()
# Modeled tables (the dbt output):
con = duckdb.connect("data/processed/finsight.duckdb", read_only=True)
con.sql("SELECT * FROM fct_daily_signals WHERE ticker='NVDA' ORDER BY date DESC LIMIT 5").df()
```

Filings are plain text — open any `data/raw/filings/*.txt` directly.

---

## Known follow-ups (tracked for later phases)

- **Reddit ingestion** is deferred pending Reddit Data API access; `fetch_reddit.py`
  plugs in without touching other code once available.
- **Price forecasting** (Prophet) is illustrative only — it does not beat naive
  persistence (prices are a random walk). The model with real edge is the HAR-RV
  **volatility** forecast. A future option is forecasting return *direction* or
  switching Prophet to a damped/log-return formulation.
- **News history** is limited to ~30 days by NewsAPI's free tier; the Prefect flow
  refreshes it on each run.
- **The agent imports the MCP tools directly** (they are the same functions the MCP
  server registers) rather than connecting over stdio — simpler and fully testable;
  the standalone `mcp_server.py` still runs as a real MCP server for external clients.
- **Phase 6 (dashboard + deploy)** is next: a Streamlit 4-tab UI (Overview / Forecast
  / News / Analyst) plus a FastAPI backend.
