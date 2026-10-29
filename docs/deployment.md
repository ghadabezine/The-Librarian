# Deployment

## What runs

| Service | Image | Port | State |
|---|---|---|---|
| `backend` | `backend/Dockerfile` (Python 3.13, CPU torch, llama.cpp) | 8000 | `./models` (weights, ~5 GB), `./data/index` (index), `./data/feedback` |
| `frontend` | `frontend/Dockerfile` (Vite build served by nginx) | 8080 → 80 | none |

nginx proxies `/api/*` to `backend:8000`, so the browser talks to one
origin and no CORS configuration is needed in production.

## First start

```bash
cp .env.example .env        # optional: change provider, top-k, optimisations
docker compose up --build -d
docker compose logs -f backend
```

On first start `backend/docker-entrypoint.sh`:

1. runs `scripts/fetch_models.py` if `./models` is empty, pulling the
   e5-large embedder, the bge-m3 reranker and the Qwen2.5-3B GGUF from
   Docker Hub;
2. runs `scripts/build_index.py` if `./data/index` is empty, building it
   from the committed OCR cache and the official PDFs (about 10 minutes
   on 4 vCPUs).

Both directories are volumes, so later restarts take seconds. `GET /health`
reports `degraded` with the reason until the index has loaded.

## Sizing

| Resource | Minimum | Notes |
|---|---|---|
| CPU | 4 vCPUs | latency figures in `optimisation.md` were measured on 4 vCPUs |
| RAM | 12 GB | embedder ~2.2 GB, reranker ~2.3 GB, LLM ~2.5 GB, plus the index |
| Disk | 15 GB | models 5 GB, images 6 GB |

A GPU is not required. With a GPU, or with `RAG_LLM_PROVIDER=anthropic`,
generation latency falls sharply.

## Public deployment

The stack runs on any VM with Docker (for example a 4 vCPU / 16 GB cloud
VM at about $0.20/h, the price used in the cost model). Steps:

1. Create the VM and open port 80/443.
2. `git clone https://github.com/ghadabezine/project2 && cd project2`
3. `docker compose up --build -d`
4. Put a TLS terminator in front (Caddy, or nginx with certbot), pointing
   to `localhost:8080`.

**Status:** no public URL has been deployed yet. The build environment used
for development had no cloud account or credentials. The team needs to run
the steps above on a VM of their own; nothing in the code depends on the
host.
