# One-command setup and run. Requires python3 (3.11+), node/npm and a C compiler.
PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup models index run backend frontend test eval-retrieval eval-generation docker

setup: $(BIN)/uvicorn models index frontend/node_modules

$(BIN)/uvicorn:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
	CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX2=ON" $(BIN)/pip install -r backend/requirements-dev.txt

models:
	$(BIN)/python scripts/fetch_models.py embedder reranker llm

index:
	$(BIN)/python scripts/build_index.py

frontend/node_modules:
	cd frontend && npm ci

run:
	@echo "backend: http://localhost:8000   frontend: http://localhost:5173"
	$(BIN)/uvicorn app.main:app --app-dir backend --port 8000 & \
	cd frontend && npm run dev

backend:
	$(BIN)/uvicorn app.main:app --app-dir backend --port 8000

frontend:
	cd frontend && npm run dev

test:
	$(BIN)/pytest -q tests

eval-retrieval:
	$(BIN)/python eval/retrieval_eval.py --tag ablation

eval-generation:
	$(BIN)/python eval/generation_eval.py --tag baseline

docker:
	docker compose up --build
