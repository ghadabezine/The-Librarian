"""HTTP routes. Business logic lives in retrieval/ and generation/."""
from __future__ import annotations

import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from ..core.config import Settings, get_settings
from ..generation.generator import RAGService, _source_view
from ..models.schemas import (ArticleVersionsResponse, ChatRequest, ChatResponse, DocumentsResponse, FeedbackRequest, FeedbackResponse,
                              HealthResponse, SearchHit, SearchRequest, SearchResponse, Source)

log = logging.getLogger(__name__)
router = APIRouter()


def get_service(request: Request) -> RAGService:
    svc = getattr(request.app.state, "service", None)
    if svc is None:
        raise HTTPException(503, detail=getattr(request.app.state, "startup_error", None)
                            or "index not loaded; run scripts/build_index.py")
    return svc


@router.get("/health", response_model=HealthResponse)
def health(request: Request, settings: Settings = Depends(get_settings)) -> HealthResponse:
    svc = getattr(request.app.state, "service", None)
    return HealthResponse(
        status="ok" if svc else "degraded", index_loaded=svc is not None,
        chunks=len(svc.retriever.chunks) if svc else 0, llm_provider=settings.llm_provider,
        retrieval_mode=settings.retrieval_mode, detail=getattr(request.app.state, "startup_error", None))


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, svc: RAGService = Depends(get_service)) -> ChatResponse:
    # Each question is answered from retrieval alone; history is accepted so
    # clients can send it, but it is not fed to the model (keeps every answer
    # grounded in its own citations and avoids carrying injected text forward).
    try:
        ans = svc.answer(req.question, mode=req.mode, top_k=req.top_k,
                         filters=req.filters.as_dict() if req.filters else None)
    except RuntimeError as e:
        log.exception("generation failed")
        raise HTTPException(503, detail=str(e)) from e
    v = ans.verification or {}
    warnings = []
    if v.get("unverified_articles"):
        warnings.append("unverified_article_numbers:" + ",".join(v["unverified_articles"]))
    if v.get("unsupported_numbers"):
        warnings.append("numbers_not_found_in_sources:" + ",".join(v["unsupported_numbers"]))
    if v.get("invalid_ids"):
        warnings.append("invalid_citations_removed")
    if any(c.get("legal_status") == "historical" for c in ans.citations):
        warnings.append("cites_historical_wording")
    return ChatResponse(
        question=ans.question, language=ans.language, answer=ans.answer, status=ans.status,
        abstain_reason=ans.abstain_reason, citations=[Source(**c) for c in ans.citations],
        sources=[Source(**s) for s in ans.sources], disclaimer=ans.disclaimer, warnings=warnings,
        retrieval_mode=ans.retrieval_mode, model=ans.model, timings_ms=ans.timings_ms,
        usage={k: float(x) for k, x in ans.usage.items()})


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, svc: RAGService = Depends(get_service)) -> SearchResponse:
    predicate = None if req.include_historical else (lambda c: c.get("legal_status") != "historical")
    ret = svc.retriever.retrieve(req.query, mode=req.mode, top_k=req.top_k,
                                 filters=req.filters.as_dict() if req.filters else None, predicate=predicate)
    hits = [SearchHit(rank=i, score=h.score, bm25_rank=h.bm25_rank, dense_rank=h.dense_rank,
                      rerank_score=h.rerank_score, source=Source(**_source_view(h.chunk, i, h.rerank_score)))
            for i, h in enumerate(ret.hits, 1)]
    return SearchResponse(query=req.query, language=ret.query.language, mode=ret.mode, hits=hits,
                          timings_ms=ret.timings_ms)


@router.get("/articles/{work_id}/{label}", response_model=ArticleVersionsResponse)
def article_versions(work_id: str, label: str, svc: RAGService = Depends(get_service)) -> ArticleVersionsResponse:
    """Every in-force language version of one article (e.g. Arabic + French),
    so the interface can show a cited article next to its translation."""
    parts: dict[str, list[dict]] = {}
    for c in svc.retriever.chunks:
        if (c.get("chunk_type") == "article" and c.get("article_label") == label
                and (c.get("work_id") or c["doc_id"]) == work_id and c.get("legal_status") != "historical"):
            parts.setdefault(c.get("language") or "ar", []).append(c)
    if not parts:
        raise HTTPException(404, detail="article not found")
    versions = []
    for lang in sorted(parts, key=lambda x: (x != "ar", x)):
        chunks = sorted(parts[lang], key=lambda c: c.get("part") or 1)
        merged = {**chunks[0], "text": "\n".join(c["text"] for c in chunks),
                  "page_end": chunks[-1].get("page_end")}
        versions.append(Source(**_source_view(merged, len(versions) + 1)))
    return ArticleVersionsResponse(work_id=work_id, article_label=label, versions=versions)


@router.get("/documents", response_model=DocumentsResponse)
def documents(settings: Settings = Depends(get_settings)) -> DocumentsResponse:
    path = settings.index_dir / "manifest.json"
    if not path.exists():
        raise HTTPException(503, detail="index manifest missing; run scripts/build_index.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    from ..ingestion.loader import load_sources
    meta = {s.doc_id: s for s in load_sources(settings.sources_file)}
    docs = []
    for d in manifest["documents"]:
        s = meta.get(d["doc_id"])
        docs.append({**d, "title_fr": s.title_fr if s else None, "collection": s.collection if s else None,
                     "source_url": s.source_url if s else None, "edition": s.edition if s else None,
                     "publication_date": s.publication_date if s else None,
                     "version_note": s.version_note if s else None})
    return DocumentsResponse(documents=docs, total_pages=manifest["total_pages"],
                             total_chunks=manifest["total_chunks"], embedding_model=manifest.get("embedding_model"),
                             built_at=manifest.get("built_at"))


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(req: FeedbackRequest, settings: Settings = Depends(get_settings)) -> FeedbackResponse:
    fid = uuid.uuid4().hex[:12]
    settings.feedback_file.parent.mkdir(parents=True, exist_ok=True)
    record = {"id": fid, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **req.model_dump()}
    with open(settings.feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return FeedbackResponse(ok=True, id=fid)
