# FinSight AI — demo container for Hugging Face Spaces (Docker SDK).
# Runs the FastAPI backend (internal :8000) + the Streamlit dashboard (public :7860)
# in DEMO mode: real historical data only, no external API calls.

FROM python:3.13-slim

# Build tools a few wheels need + curl for the readiness check
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# The app code defaults HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE to "1"; we set them to "0"
# so FinBERT (the News-tab analyzer) can download on first use inside the Space.
ENV FINSIGHT_DEMO=1 \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

EXPOSE 7860
CMD ["bash", "serving/start.sh"]
