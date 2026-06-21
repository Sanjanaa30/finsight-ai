"""FinBERT sentiment scoring over all news articles.

Reads raw news, scores each headline with FinBERT (ProsusAI/finbert) in batches
of 32 (CPU-friendly -- never score everything at once), and writes a scored
Parquet that dbt consumes to fill the avg_sentiment placeholder. The run is
logged to MLflow.

sentiment_score is a single number in [-1, 1] = P(positive) - P(negative).
"""

import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Force UTF-8 stdout/stderr so emoji in library output (e.g. MLflow) don't crash
# on Windows' default cp1252 console.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "news.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "news_scored.parquet"

MODEL_NAME = "ProsusAI/finbert"
BATCH_SIZE = 32          # NEVER score all headlines at once on a laptop
MAX_TOKENS = 128         # headlines are short; keeps inference fast

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "finsight-sentiment"


def score_texts(texts: list[str], tokenizer, model, id2label: dict) -> tuple[list, list, list]:
    """Return (sentiment_score, label, prob_dicts) for a list of texts, batched."""
    scores, labels, prob_rows = [], [], []
    n = len(texts)
    for start in range(0, n, BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_TOKENS, return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()

        for p in probs:
            d = {id2label[i].lower(): float(p[i]) for i in range(len(p))}
            scores.append(d.get("positive", 0.0) - d.get("negative", 0.0))
            labels.append(max(d, key=d.get))
            prob_rows.append(d)
        logger.info("scored {}/{}", min(start + BATCH_SIZE, n), n)
    return scores, labels, prob_rows


def main() -> None:
    df = pd.read_parquet(INPUT_PATH)
    df["published_date"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).dt.date
    titles = df["title"].fillna("").astype(str).tolist()
    logger.info("Loaded {} articles to score", len(titles))

    logger.info("Loading FinBERT ({}) -- first run downloads ~440MB ...", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    id2label = model.config.id2label

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="finbert-scoring"):
        mlflow.log_params({
            "model": MODEL_NAME, "batch_size": BATCH_SIZE,
            "max_tokens": MAX_TOKENS, "n_articles": len(titles),
        })

        scores, labels, prob_rows = score_texts(titles, tokenizer, model, id2label)
        df["sentiment_score"] = scores
        df["sentiment_label"] = labels
        probs_df = pd.DataFrame(prob_rows).add_prefix("prob_")
        out = pd.concat([df.reset_index(drop=True), probs_df], axis=1)

        mlflow.log_metric("mean_sentiment", float(pd.Series(scores).mean()))
        for label, count in pd.Series(labels).value_counts().items():
            mlflow.log_metric(f"count_{label}", int(count))

        keep = [
            "category", "published_at", "published_date", "source", "title", "url",
            "sentiment_score", "sentiment_label",
            "prob_positive", "prob_negative", "prob_neutral",
        ]
        out = out[[c for c in keep if c in out.columns]]
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUTPUT_PATH, index=False)
        mlflow.log_metric("n_scored", len(out))

    logger.success("Wrote {} scored articles to {}", len(out),
                   OUTPUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
