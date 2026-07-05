#!/usr/bin/env bash
# Start the API (internal) then the dashboard (public). Used by the demo Docker image.
set -e

# FastAPI backend on :8000 in the background
uvicorn serving.api:app --host 0.0.0.0 --port 8000 --log-level warning &

# Wait for the API to answer before starting the dashboard (avoids a first-load error)
for _ in $(seq 1 60); do
    curl -sf http://localhost:8000/ >/dev/null 2>&1 && break
    sleep 1
done

# Streamlit dashboard on the public port (HF Spaces expects 7860)
exec streamlit run serving/dashboard.py \
    --server.port "${PORT:-7860}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
