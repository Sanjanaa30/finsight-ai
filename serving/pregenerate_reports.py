"""Pre-generate analyst reports for DEMO mode.

Runs the real LangGraph agent once per ticker and saves the markdown to
`data/processed/demo_reports/{TICKER}.md`. Run this ONCE (needs `ANTHROPIC_API_KEY`
and internet); then serve the public demo with `FINSIGHT_DEMO=1`, and the `/agent`
endpoint reads these files instead of calling Claude — so the demo shows a real AI
report with zero live API cost.

Usage:
    python serving/pregenerate_reports.py                       # featured tickers
    python serving/pregenerate_reports.py NVDA AAPL BTC-USD     # specific ones
"""

import os
import sys
from pathlib import Path

os.environ.pop("FINSIGHT_DEMO", None)  # ensure LIVE data is used while generating
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from ai.agent import analyst  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "demo_reports"
FEATURED = ["NVDA", "AAPL", "MSFT", "TSLA", "AMD", "META",
            "BTC-USD", "ETH-USD", "GC=F", "^GSPC", "^NSEI"]


def main() -> None:
    tickers = [t.upper() for t in sys.argv[1:]] or FEATURED
    OUT.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        try:
            print(f"generating {t} ...", flush=True)
            report = analyst.invoke({"ticker": t})["report"]
            (OUT / f"{t}.md").write_text(report, encoding="utf-8")
            print(f"  saved {t}.md ({len(report)} chars)")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {t}: {exc}")
    print(f"Done -> {OUT}")


if __name__ == "__main__":
    main()
