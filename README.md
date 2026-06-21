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
| 4 — Machine learning | Prophet forecast, FinBERT sentiment, anomaly detection, MLflow | ⬜ Planned |
| 5 — AI engineering | RAG over SEC filings, MCP server, LangGraph agent | ⬜ Planned |
| 6 — Dashboard + deploy | FastAPI + Streamlit 4-tab dashboard | ⬜ Planned |

---

## Tech stack

**Data ingestion:** yfinance · NewsAPI · SEC EDGAR · FRED · (Reddit/PRAW planned)
**Engineering:** dbt (dbt-duckdb) · DuckDB · Parquet (PyArrow) · Prefect · PostgreSQL · Qdrant (Docker)
**ML (planned):** Prophet · FinBERT · Isolation Forest · MLflow
**AI (planned):** LangChain/LangGraph · MCP · sentence-transformers · OpenAI gpt-4o-mini
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
                            ingest → dbt run → dbt test            (Phase 4 ML input)
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
│   └── daily_pipeline.py           # Phase 3 — Prefect flow: ingest → dbt run → dbt test
├── ml/                             # Phase 4 — training + evaluation
├── ai/                             # Phase 5 — RAG, MCP server, agent
├── serving/                        # Phase 6 — FastAPI + Streamlit
├── data/                           # (gitignored) raw + processed data
│   ├── raw/                        # Parquet files + filings/ text
│   └── processed/                  # finsight.duckdb (dbt output)
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
| `OPENAI_API_KEY` | platform.openai.com | (Phase 5) analyst agent | ~$5 total dev |

`yfinance`, `SEC EDGAR`, and `GDELT` need no key. `.env` is gitignored — keys are never committed.

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
before that date) and daily news context. The `news_avg_sentiment` column is a
placeholder filled in Phase 4 once FinBERT scores the news.

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
- **News sentiment** (`int_sentiment_daily.avg_sentiment`, `fct_daily_signals.news_avg_sentiment`)
  is a placeholder until Phase 4's FinBERT model scores each article.
- **SEC filing text** is raw inline-XBRL HTML stripped to text, so it includes XBRL
  tagging noise alongside the narrative. The Phase 5 RAG pipeline will clean and
  chunk it (full content is preserved in the raw files).
- **News history** is limited to ~30 days by NewsAPI's free tier; the Prefect flow
  refreshes it on each run.
