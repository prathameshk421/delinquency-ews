#!/usr/bin/env bash
# One-command demo: builds the UI (first time) and serves UI + API on http://localhost:8000
set -e
cd "$(dirname "$0")"
if [ ! -f artifacts/bundle.joblib ]; then
  echo "No trained models found - training first (python scripts/train.py)..."
  python3 scripts/train.py
fi
if [ ! -d frontend/dist ]; then
  (cd frontend && npm install && npm run build)
fi
echo "Open http://localhost:8000"
python3 -m uvicorn api.main:app --port 8000
