"""FastAPI application entry point: ``uvicorn app.main:app``."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api.routes import router
from .core.config import get_settings
from .core.logging import setup_logging

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    app.state.service = None
    app.state.startup_error = None
    try:
        from .generation.generator import RAGService
        from .retrieval.retriever import Retriever

        retriever = Retriever(settings)
        if settings.retrieval_mode != "bm25":
            retriever.vindex  # fail fast if the vector index is missing
        app.state.service = RAGService(settings, retriever)
        log.info("index loaded: %d chunks", len(retriever.chunks))
    except Exception as e:  # keep /health up so the problem is visible
        app.state.startup_error = f"{type(e).__name__}: {e}"
        log.exception("startup failed")
    yield


app = FastAPI(title="Tunisian Personal Status Law RAG", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})
