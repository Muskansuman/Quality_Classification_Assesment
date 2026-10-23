#!/usr/bin/env bash
# Verify Ollama is running and pull the default open-source assessment model.
set -euo pipefail

MODEL="${OLLAMA_MODEL:-qwen2.5:7b-instruct}"
BASE="${OLLAMA_BASE_URL:-http://localhost:11434}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not installed. Get it from https://ollama.com and re-run this script."
  exit 1
fi

echo "Checking Ollama at ${BASE} ..."
if ! curl -sf "${BASE%/v1}/api/tags" >/dev/null 2>&1; then
  echo "Ollama API is not reachable. Start it with: ollama serve"
  echo "  (On many installs, the Ollama app starts the service automatically.)"
  exit 1
fi

if ollama show "${MODEL}" >/dev/null 2>&1; then
  echo "Model already present: ${MODEL}"
else
  echo "Pulling ${MODEL} (one-time download, may take a few minutes) ..."
  ollama pull "${MODEL}"
fi

echo ""
echo "Ready. Default in .env:"
echo "  LLM_PROVIDER=ollama"
echo "  OLLAMA_MODEL=${MODEL}"
echo ""
echo "Run the app:  source .venv/bin/activate && streamlit run app.py"
