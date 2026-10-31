"""Request and response models for the HTTP API (see api/routes.py)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RetrievalMode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]
StrOrList = str | list[str] | None


class Filters(BaseModel):
    """Restrict retrieval to chunks whose metadata matches every given field.
    A list means "any of these values"."""
    doc_id: StrOrList = None
    work_id: StrOrList = None
    language: StrOrList = None
    legal_status: StrOrList = None
    chunk_type: StrOrList = None

    def as_dict(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v not in (None, [], "")}


class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


def _strip_nonempty(v: str) -> str:
    v = v.strip()
    if not v:
        raise ValueError("must not be empty")
    return v


class ChatRequest(BaseModel):
    question: str = Field(max_length=1000)
    # Accepted for client convenience; not sent to the model (see routes.chat).
    history: list[HistoryTurn] = Field(default_factory=list, max_length=20)
    mode: RetrievalMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: Filters | None = None

    _q = field_validator("question")(_strip_nonempty)


class SearchRequest(BaseModel):
    query: str = Field(max_length=1000)
    mode: RetrievalMode | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    filters: Filters | None = None
    include_historical: bool = False

    _q = field_validator("query")(_strip_nonempty)


class Source(BaseModel):
    """One retrieved passage, as shown in the source panel. `id` is the
    citation marker used in the answer ("S1", "S2", ...)."""
    id: str
    chunk_id: str
    doc_id: str
    doc_title: str
    doc_title_fr: str | None = None
    article_label: str | None = None
    article_label_display: str | None = None
    language: str | None = None
    work_id: str | None = None
    chunk_type: str | None = None
    context: str | None = None
    text: str
    page_start: int | None = None
    page_end: int | None = None
    source_url: str | None = None
    legal_status: str | None = None
    authority: str | None = None
    amending_laws: list[str] = Field(default_factory=list)
    repealed: bool = False
    edition: str | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    question: str
    language: str
    answer: str
    status: Literal["answered", "abstained"]
    abstain_reason: str | None = None
    citations: list[Source]
    sources: list[Source]
    disclaimer: str
    warnings: list[str] = Field(default_factory=list)
    retrieval_mode: str
    model: str
    timings_ms: dict[str, float] = Field(default_factory=dict)
    usage: dict[str, float] = Field(default_factory=dict)


class ArticleVersionsResponse(BaseModel):
    work_id: str
    article_label: str
    versions: list[Source]


class SearchHit(BaseModel):
    rank: int
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rerank_score: float | None = None
    source: Source


class SearchResponse(BaseModel):
    query: str
    language: str
    mode: str
    hits: list[SearchHit]
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentsResponse(BaseModel):
    documents: list[dict]
    total_pages: int
    total_chunks: int
    embedding_model: str | None = None
    built_at: str | None = None


class FeedbackRequest(BaseModel):
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=20000)
    rating: Literal["up", "down"]
    issue: Literal["wrong", "incomplete", "bad_citation", "outdated", "other"] | None = None
    comment: str | None = Field(default=None, max_length=2000)
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=50)


class FeedbackResponse(BaseModel):
    ok: bool
    id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_loaded: bool
    chunks: int
    llm_provider: str
    retrieval_mode: str
    detail: str | None = None
