#!/bin/sh
# Fetch missing models and build the index on first start; both are kept in volumes.
set -e
cd /app
if [ ! -d models/multilingual-e5-large ] || [ ! -d models/bge-reranker-v2-m3 ] || \
   { [ "${RAG_LLM_PROVIDER:-llamacpp}" = "llamacpp" ] && [ ! -f "models/${RAG_LLM_MODEL:-qwen2.5-3b-instruct-q4_k_m.gguf}" ]; }; then
  echo "Fetching models (first start only)â€¦"
  python scripts/fetch_models.py embedder reranker llm
fi
if [ ! -f data/index/chunks.jsonl ] || [ ! -f data/index/vectors.npy ]; then
  echo "Building the index (first start only)â€¦"
  python scripts/build_index.py
fi
exec "$@"
