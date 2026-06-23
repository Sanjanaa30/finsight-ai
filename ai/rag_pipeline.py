"""RAG pipeline: SEC filings -> cleaned chunks -> embeddings -> Qdrant.

1. Read each 10-K text file.
2. Token-chunk it (sliding window) using the embedding model's own tokenizer.
3. Drop XBRL tag-soup / numeric-table chunks with a prose-quality filter
   (this is the filing cleanup deferred from Phase 2 -- done at chunk granularity
   so we keep narrative and drop machine-tagging noise).
4. Embed the surviving chunks with all-MiniLM-L6-v2 (384-dim, CPU, batched).
5. Upsert into a Qdrant collection the analyst agent searches in Phase 5.

Note on chunk size: the guide says 512 tokens, but all-MiniLM-L6-v2 only encodes
the first 256 tokens -- a 512-token chunk would lose its second half in the
embedding. We chunk at 256 so every chunk is fully represented.
"""

import re
import sys
from pathlib import Path

from loguru import logger
from transformers.utils import logging as hf_logging

# Silence the cosmetic "sequence longer than 256" notice: it fires when we
# tokenize the FULL document to slice it into windows -- the model never sees
# the full sequence, only the <=256-token chunks.
hf_logging.set_verbosity_error()
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILINGS_DIR = PROJECT_ROOT / "data" / "raw" / "filings"

MODEL_NAME = "all-MiniLM-L6-v2"
# Target slightly under the model's 256 cap: decoding a 256-token window back to
# text and re-tokenizing can inflate by a few tokens, so 240 guarantees every
# chunk re-encodes to <=256 and is embedded in full (no silent tail truncation).
CHUNK_TOKENS = 240
CHUNK_OVERLAP = 64       # keeps sentences from being cut across chunk borders
BATCH_SIZE = 32          # CPU-friendly embedding batch

QDRANT_URL = "http://localhost:6333"
COLLECTION = "filings"

# Common English stopwords -- prose is dense with these; XBRL/number tables are not.
STOPWORDS = {
    "the", "and", "of", "to", "in", "a", "that", "for", "is", "are", "with",
    "as", "on", "by", "this", "be", "or", "an", "we", "our", "its", "from",
    "which", "their", "has", "have", "was", "were", "not", "may", "such",
}


def is_prose(text: str) -> bool:
    """Keep narrative; drop XBRL tag-soup and dense numeric tables."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 20:
        return False
    stop_ratio = sum(w in STOPWORDS for w in words) / len(words)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return stop_ratio > 0.12 and digit_ratio < 0.12


def chunk_tokens(text: str, tokenizer) -> list[str]:
    """Sliding-window chunks of CHUNK_TOKENS using the model's own tokenizer."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    chunks = []
    for start in range(0, len(ids), step):
        window = ids[start:start + CHUNK_TOKENS]
        if not window:
            break
        chunks.append(tokenizer.decode(window).strip())
    return chunks


def parse_filename(name: str) -> tuple[str, str]:
    """AAPL_10K_2025-10-31.txt -> ("AAPL", "2025-10-31")."""
    stem = name.replace(".txt", "")
    parts = stem.split("_")
    ticker = parts[0]
    filing_date = parts[-1] if len(parts) >= 3 else ""
    return ticker, filing_date


def main() -> None:
    files = sorted(FILINGS_DIR.glob("*.txt"))
    if not files:
        raise RuntimeError(f"No filings found in {FILINGS_DIR}")
    logger.info("Loading embedding model {} ...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(url=QDRANT_URL)
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    logger.info("Created Qdrant collection '{}' ({}-dim, cosine)", COLLECTION, dim)

    point_id = 0
    total_kept = 0
    for path in files:
        ticker, filing_date = parse_filename(path.name)
        text = path.read_text(encoding="utf-8")

        raw_chunks = chunk_tokens(text, model.tokenizer)
        kept = [c for c in raw_chunks if is_prose(c)]
        if not kept:
            logger.warning("{}: no prose chunks survived filter, skipping", ticker)
            continue

        vectors = model.encode(kept, batch_size=BATCH_SIZE, show_progress_bar=False)
        points = [
            PointStruct(
                id=point_id + i,
                vector=vec.tolist(),
                payload={
                    "ticker": ticker,
                    "filing_date": filing_date,
                    "source": path.name,
                    "chunk_index": i,
                    "text": chunk,
                },
            )
            for i, (chunk, vec) in enumerate(zip(kept, vectors))
        ]
        client.upsert(COLLECTION, points=points)
        point_id += len(points)
        total_kept += len(kept)
        logger.success("{}: {}/{} chunks kept (prose) and embedded",
                       ticker, len(kept), len(raw_chunks))

    logger.success("Done. {} chunks across {} filings in Qdrant collection '{}'",
                   total_kept, len(files), COLLECTION)


if __name__ == "__main__":
    main()
