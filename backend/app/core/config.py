"""Application settings, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="RAG_", extra="ignore")

    # Paths
    data_dir: Path = ROOT / "data"
    models_dir: Path = ROOT / "models"

    @property
    def sources_file(self) -> Path:
        return self.data_dir / "sources.yaml"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def feedback_file(self) -> Path:
        return self.data_dir / "feedback" / "feedback.jsonl"

    # Chunking
    chunk_strategy: Literal["article", "fixed"] = "article"
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150

    # Models
    embedding_model: str = "multilingual-e5-large"
    reranker_model: str = "bge-reranker-v2-m3"
    device: str = "cpu"

    # Retrieval
    retrieval_mode: Literal["bm25", "dense", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    top_k: int = 5
    candidate_k: int = 20
    rerank_k: int = 20
    rrf_k: int = 60
    min_rerank_score: float = 0.01
    skip_rerank_when_confident: bool = False
    collapse_translations: bool = True

    # Generation
    llm_provider: Literal["llamacpp", "anthropic", "openai", "extractive"] = "llamacpp"
    llm_model: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0
    llm_threads: int = 4
    llm_context: int = 8192
    llm_prompt_cache: bool = False
    max_context_chars: int = 6000

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "medium"

    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None  # for Azure / proxies; None = official API

    # Other
    include_historical: bool = False
    query_cache_size: int = 0
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:3000",
    ]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()