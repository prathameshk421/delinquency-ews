#!/bin/bash
# Double-click to run the Early Signal demo on macOS.
cd "$(dirname "$0")" || exit 1
echo "== Early Signal demo =="

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then echo "Python 3.10+ not found. Install it from python.org and re-run."; read -r; exit 1; fi
echo "Using $($PY --version)"

if [ ! -d .venv ]; then
  echo "Creating virtual environment..."
  "$PY" -m venv .venv || { echo "venv failed"; read -r; exit 1; }
fi
source .venv/bin/activate
if [ ! -f .venv/.deps_ok ]; then
  echo "Installing Python packages (first run only, a few minutes)..."
  pip install --upgrade pip -q && pip install -r requirements.txt -q && touch .venv/.deps_ok || { echo "pip install failed"; read -r; exit 1; }
fi

# Serving scores one record at a time: single-threaded avoids the macOS OpenMP clash
# between LightGBM and PyTorch.
export OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

# Use the shipped models if they load with this machine's library versions, otherwise retrain.
echo "Checking the trained models load..."
if ! python -c "import sys; sys.path.insert(0,'src'); from ews.inference import Scorer; Scorer()" >/dev/null 2>&1; then
  echo "Training models on this machine (~8 min)..."
  OMP_NUM_THREADS= python scripts/train.py || { echo "training failed"; read -r; exit 1; }
fi

if command -v npm >/dev/null 2>&1; then
  if [ ! -d frontend/node_modules ]; then
    echo "Installing frontend packages (first run only)..."
    (cd frontend && npm install --no-audit --no-fund) || { echo "npm install failed"; read -r; exit 1; }
  fi
  echo "Building the frontend..."
  (cd frontend && npm run build >/dev/null) || { echo "frontend build failed"; read -r; exit 1; }
elif [ ! -d frontend/dist ]; then
  echo "Node.js/npm not found - install from nodejs.org for the UI. API will still run on :8000/docs"
fi

(sleep 4; open "http://localhost:8000") &
echo "Serving on http://localhost:8000  (close this window or press Ctrl+C to stop)"
python -m uvicorn api.main:app --port 8000
