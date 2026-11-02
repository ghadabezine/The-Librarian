# Deployment

## Live Production Environment

The application is fully deployed and publicly accessible on the internet:

- **Frontend:** (https://the-librarian-iota.vercel.app/)
- **Backend API:** https://almurshid-4f2a.style.dev/health

### Architecture
- **Backend:** Hosted on a Freestyle cloud VM (4 vCPU, 8 GiB RAM + 4 GiB Swap). It runs the `seif70/almurshid-backend:latest` Docker image pulled from Docker Hub.
- **Frontend:** Hosted on Vercel as a static Vite build. It is configured with `VITE_API_URL=https://almurshid-4f2a.style.dev` and communicates directly with the backend via HTTPS.
- **Generation:** The deployed service uses OpenAI `gpt-4o-mini` for generation to minimize local memory usage and maximize answer correctness (see `docs/evaluation.md` §2).
- **Retrieval:** Runs locally inside the VM using the `multilingual-e5-large` embedder and `bge-reranker-v2-m3` reranker.

## What runs

| Service | Image | Port | State |
|---|---|---|---|
| `backend` | `seif70/almurshid-backend` (Python 3.13, CPU torch, e5-large, bge-reranker) | 7860 | `~/models` (weights), `data/index` (index baked into image) |
| `frontend` | Vite static build (hosted on Vercel) | 443 | none |

## Local / Self-Hosted Setup (Docker Compose)

To run the full stack locally:
```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f backend