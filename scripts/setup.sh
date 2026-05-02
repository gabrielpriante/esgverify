#!/usr/bin/env bash
# scripts/setup.sh
#
# One-shot development environment setup.
# Run this once after cloning the repo.
#
# Usage:
#   chmod +x scripts/setup.sh
#   ./scripts/setup.sh

set -euo pipefail

echo ""
echo "=== ESGVerify — development setup ==="
echo ""

# --- Check prerequisites ---
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not found."; exit 1; }
command -v node    >/dev/null 2>&1 || { echo "ERROR: node not found."; exit 1; }
command -v ollama  >/dev/null 2>&1 || { echo "ERROR: ollama not found. Install from https://ollama.com/download"; exit 1; }

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python $PYTHON_VERSION detected"

# --- Backend ---
echo ""
echo "--- Setting up Python backend ---"
cd backend
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "Backend dependencies installed."
deactivate
cd ..

# --- Frontend ---
echo ""
echo "--- Setting up Node frontend ---"
cd frontend
npm install --silent
echo "Frontend dependencies installed."
cd ..

# --- Ollama model ---
echo ""
echo "--- Pulling Ollama model (llama3.1:8b) ---"
echo "This may take a few minutes on first run..."
ollama pull llama3.1:8b

# --- Environment file ---
if [ ! -f .env ]; then
    cp .env.example .env
    echo ".env file created from .env.example"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Start the backend:   cd backend && source venv/bin/activate && uvicorn api.main:app --reload"
echo "Start the frontend:  cd frontend && npm run dev"
echo "API docs:            http://localhost:8000/docs"
echo ""
