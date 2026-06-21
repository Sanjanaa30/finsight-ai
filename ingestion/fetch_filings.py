"""Fetch the latest 10-K (annual report) text from SEC EDGAR for the US stocks.

Only US-listed companies file 10-Ks, so this covers the 12 US stocks (crypto,
commodities, and foreign indices have no SEC filings). Each filing's primary
document is downloaded, stripped of HTML, and saved as plain text in
data/raw/filings/ -- this is the corpus the RAG pipeline embeds in Phase 5.

SEC requires no API key, but it does require a descriptive User-Agent header
that includes a contact email (set SEC_USER_AGENT in .env to your own).
"""

import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Only the 12 US stocks file 10-Ks with the SEC.
US_STOCKS = [
    "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "TSLA", "JPM", "XOM", "NFLX", "AMD", "BRK-B",
]

# SEC asks for a UA like "Sample Company AdminContact@example.com". Override via
# .env so requests are attributed to you; the fallback keeps the script runnable.
USER_AGENT = os.getenv("SEC_USER_AGENT", "finsight-ai research bot contact@finsight-ai.dev")
HEADERS = {"User-Agent": USER_AGENT}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"

SEC_RATE_LIMIT_SECONDS = 0.3  # SEC allows <10 req/s; stay well under

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "filings"


def load_ticker_to_cik() -> dict[str, int]:
    """Return a {ticker: CIK} map from SEC's master ticker file."""
    resp = requests.get(TICKER_MAP_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    # File is keyed by row index: {"0": {"cik_str":..., "ticker":..., "title":...}}
    return {row["ticker"].upper(): int(row["cik_str"]) for row in resp.json().values()}


def latest_10k(cik: int) -> tuple[str, str, str] | None:
    """Return (accession_no_dashes, primary_document, filing_date) for newest 10-K."""
    resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=HEADERS, timeout=30)
    resp.raise_for_status()
    recent = resp.json()["filings"]["recent"]
    for form, accession, doc, filed in zip(
        recent["form"], recent["accessionNumber"],
        recent["primaryDocument"], recent["filingDate"],
    ):
        if form == "10-K":
            return accession.replace("-", ""), doc, filed
    return None


def download_filing_text(cik: int, accession: str, doc: str) -> str:
    """Download a filing's primary document and strip it to plain text."""
    url = ARCHIVE_URL.format(cik=cik, accession=accession, doc=doc)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()  # collapse whitespace


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker_to_cik = load_ticker_to_cik()
    time.sleep(SEC_RATE_LIMIT_SECONDS)

    saved = 0
    for ticker in US_STOCKS:
        cik = ticker_to_cik.get(ticker.upper())
        if cik is None:
            logger.warning("{}: no CIK found in SEC mapping, skipping", ticker)
            continue

        meta = latest_10k(cik)
        time.sleep(SEC_RATE_LIMIT_SECONDS)
        if meta is None:
            logger.warning("{}: no 10-K found, skipping", ticker)
            continue

        accession, doc, filed = meta
        text = download_filing_text(cik, accession, doc)
        time.sleep(SEC_RATE_LIMIT_SECONDS)

        out_path = OUTPUT_DIR / f"{ticker}_10K_{filed}.txt"
        out_path.write_text(text, encoding="utf-8")
        saved += 1
        logger.success("{}: saved 10-K filed {} ({:,} chars)", ticker, filed, len(text))

    logger.info("Done. Saved {}/{} filings to {}", saved, len(US_STOCKS),
                OUTPUT_DIR.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
