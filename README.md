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
| **2 — Data ingestion** | 5 source ingestion scripts → Parquet / text | ✅ 4 of 5 sources (Reddit deferred) |
| 3 — Data engineering | dbt models on DuckDB + Prefect orchestration | ⬜ Planned |
| 4 — Machine learning | Prophet forecast, FinBERT sentiment, anomaly detection, MLflow | ⬜ Planned |
| 5 — AI engineering | RAG over SEC filings, MCP server, LangGraph agent | ⬜ Planned |
| 6 — Dashboard + deploy | FastAPI + Streamlit 4-tab dashboard | ⬜ Planned |

---

## Tech stack

**Data:** yfinance · NewsAPI · SEC EDGAR · FRED · (Reddit/PRAW planned) · DuckDB · Parquet (PyArrow)
**Engineering (planned):** dbt · Prefect · PostgreSQL · Qdrant
**ML (planned):** Prophet · FinBERT · Isolation Forest · MLflow
**AI (planned):** LangChain/LangGraph · MCP · sentence-transformers · OpenAI gpt-4o-mini
**Serving (planned):** FastAPI · Streamlit · Docker Compose
**Tooling:** Python 3.13 · uv · loguru · pytest

---

## Project structure

```
finsight-ai/
├── ingestion/                 # Phase 2 — source → raw landing
│   ├── fetch_prices.py        # 25 assets via yfinance → prices.parquet
│   ├── fetch_filings.py       # SEC EDGAR 10-K text (12 US stocks) → filings/*.txt
│   ├── fetch_news.py          # 6 news categories via NewsAPI → news.parquet
│   ├── fetch_macro.py         # 4 FRED macro indicators → macro.parquet
│   └── fetch_reddit.py        # (deferred — Reddit API access pending)
├── pipelines/flows/           # Phase 3 — Prefect flows
├── dbt/                       # Phase 3 — staging / intermediate / marts + tests
├── ml/                        # Phase 4 — training + evaluation
├── ai/                        # Phase 5 — RAG, MCP server, agent
├── serving/                   # Phase 6 — FastAPI + Streamlit
├── data/                      # (gitignored) raw + processed data
│   └── raw/                   # Parquet files + filings/ text
├── tests/                     # pytest unit tests
├── .env / .env.example        # API keys (.env is gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

**Requirements:** Python 3.13, [uv](https://docs.astral.sh/uv/), and (for Phase 3+) Docker Desktop with WSL 2.

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

## Phase 2 — Data ingestion (what's built)

Four ingestion scripts pull raw data from external APIs and land it **as-is**
(no cleaning yet — that's Phase 3's job). Tabular sources are written in **tidy
long format** (one row per entity + date), which is what dbt expects downstream.
All outputs go to `data/raw/` (gitignored).

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
macro, and general market. Queries are anchored to a finance context and matched
on title + description to keep off-topic articles out.

### Run ingestion

```powershell
python ingestion/fetch_prices.py
python ingestion/fetch_filings.py
python ingestion/fetch_news.py
python ingestion/fetch_macro.py
```

### Inspect the data

Parquet is binary — read it with DuckDB:

```python
import duckdb
duckdb.sql("SELECT * FROM 'data/raw/prices.parquet' LIMIT 10")
duckdb.sql("SELECT category, COUNT(*) FROM 'data/raw/news.parquet' GROUP BY category")
```

Filings are plain text — open any `data/raw/filings/*.txt` directly.

---

## Known follow-ups (tracked for later phases)

- **Reddit ingestion** is deferred pending Reddit Data API access; `fetch_reddit.py`
  plugs in without touching other code once available.
- **SEC filing text** is raw inline-XBRL HTML stripped to text, so it includes XBRL
  tagging noise alongside the narrative. The Phase 5 RAG pipeline will clean and
  chunk it (full content is preserved in the raw files).
- **News history** is limited to ~30 days by NewsAPI's free tier; the Phase 3
  Prefect flow will collect it on a daily schedule.
