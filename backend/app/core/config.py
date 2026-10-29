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
    top_k: int = 5                 # chunks passed to the generator
    candidate_k: int = 20          # candidates per retriever before fusion
    rerank_k: int = 20             # fused candidates sent to the reranker
    rrf_k: int = 60
    # Abstain without calling the LLM when the best reranker score is below this.
    min_rerank_score: float = 0.01   # calibrated in eval/results/rerank_simulation.md
    # Skip the reranker when BM25 and dense agree strongly on the top hit (optimisation).
    skip_rerank_when_confident: bool = False
    # Keep only the best-ranked language version of an article in the results,
    # so the top-k holds k distinct provisions instead of translations.
    collapse_translations: bool = True

    # Generation
    llm_provider: Literal["llamacpp", "anthropic", "extractive"] = "llamacpp"
    llm_model: str = "qwen2.5-3b-instruct-q4_k_m.gguf"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.0
    llm_threads: int = 4
    llm_context: int = 8192
    # Keep KV states of earlier prompts in RAM so a shared prefix (the fixed
    # system prompt) is not re-evaluated. llama.cpp already reuses the prefix of
    # the immediately preceding call; this extends it across interleaved calls.
    llm_prompt_cache: bool = False
    max_context_chars: int = 6000
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "medium"
    # Optional: include only answers about these collections by default.
    include_historical: bool = False

    # Caching (optimisation)
    query_cache_size: int = 0

    # API
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080", "http://localhost:3000"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
