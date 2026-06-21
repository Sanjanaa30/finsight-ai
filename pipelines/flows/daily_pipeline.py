"""Prefect flow: the daily FinSight ELT pipeline.

Orchestrates the whole data layer end to end:
    1. Ingest all sources (prices, news, macro, filings) -> data/raw  [parallel]
    2. dbt run   -> rebuild staging / intermediate / mart models
    3. dbt test  -> run all data-quality tests

Each task runs as a subprocess using the project's own venv, so the flow behaves
exactly like running the scripts by hand. Ingestion tasks get retries to absorb
transient API failures (the resilience we deliberately left out of the scripts).

Run once (manual):       python pipelines/flows/daily_pipeline.py
Serve on a schedule:     see the __main__ block (6pm weekdays).
"""

import os
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
DBT = str(Path(sys.executable).parent / ("dbt.exe" if os.name == "nt" else "dbt"))

# Reddit is deferred (API access pending), so it is not in the pipeline yet.
INGEST_SCRIPTS = ["fetch_prices.py", "fetch_news.py", "fetch_macro.py", "fetch_filings.py"]


def _run(cmd: list[str]) -> str:
    """Run a command from the project root; raise with output on failure."""
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return result.stdout


@task(retries=2, retry_delay_seconds=30, log_prints=True)
def ingest(script: str) -> None:
    """Run one ingestion script."""
    get_run_logger().info("Ingesting: %s", script)
    _run([PYTHON, f"ingestion/{script}"])


@task(log_prints=True)
def dbt_command(args: list[str]) -> None:
    """Run a dbt command against the project."""
    get_run_logger().info("dbt %s", " ".join(args))
    _run([DBT, *args, "--project-dir", "dbt", "--profiles-dir", "dbt"])


@flow(name="finsight-daily-pipeline")
def daily_pipeline() -> None:
    logger = get_run_logger()

    # 1. Ingest all sources concurrently, then wait for all to finish.
    logger.info("Starting ingestion of %d sources", len(INGEST_SCRIPTS))
    futures = [ingest.submit(script) for script in INGEST_SCRIPTS]
    for f in futures:
        f.result()  # raises if any ingestion task ultimately failed

    # 2. Transform: rebuild models. 3. Validate: run data-quality tests.
    run_future = dbt_command.submit(["run"])
    run_future.result()
    dbt_command.submit(["test"], wait_for=[run_future]).result()

    logger.info("Pipeline complete: raw -> staging -> intermediate -> mart, tested.")


if __name__ == "__main__":
    # Run once for testing:
    daily_pipeline()

    # To run on a schedule (6pm weekdays), comment out the line above and use:
    #   daily_pipeline.serve(name="daily", cron="0 18 * * 1-5")
