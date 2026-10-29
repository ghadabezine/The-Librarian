# Backend image: FastAPI + retrieval + local LLM (CPU).
# Build from the repository root:  docker build -f backend/Dockerfile .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# Compilers are needed to build llama-cpp-python; curl for the health check.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY backend/requirements.txt backend/requirements.txt
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON" pip install -r backend/requirements.txt

COPY backend backend
COPY scripts scripts
COPY data/sources.yaml data/sources.yaml
COPY data/raw data/raw
COPY data/ocr data/ocr
COPY backend/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# models/ and data/index/ are volumes (see docker-compose.yml).
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:8000/health || exit 1
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
