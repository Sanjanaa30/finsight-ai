# Deploying the demo (Hugging Face Spaces · Docker)

The demo runs on **stored historical data only** — no API keys, no external calls, no cost.
The included `Dockerfile` runs the API + dashboard together on port 7860 with `FINSIGHT_DEMO=1`.

### 1. Ship the demo data (it's gitignored — force-add the ~5 MB snapshot)
```bash
python serving/pregenerate_reports.py     # once: builds data/processed/demo_reports/ (needs ANTHROPIC_API_KEY + internet)
git add -f data/processed/                 # force-add DuckDB + ML parquets + demo reports
git commit -m "Add demo data snapshot for deploy"
```

### 2. Create the Space
On huggingface.co → **New Space → SDK: Docker → blank**. Its `README.md` must begin with:

```yaml
---
title: FinSight AI
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
---
```
(Add this to the top of the README on the *Space* side — you can keep your GitHub README clean.)

### 3. Push the code
```bash
git remote add space https://huggingface.co/spaces/<your-username>/finsight-ai
git push space main
```

HF builds the Dockerfile and boots it; in a few minutes you get a public URL. To update, push again.

**Notes:** no secrets needed at runtime (don't add `.env`). First build is slow (~5–10 min) because
of torch/transformers. The FinBERT analyzer works in the demo; the free-form chat is disabled.
