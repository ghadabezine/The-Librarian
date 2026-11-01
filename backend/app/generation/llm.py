"""LLM back-ends behind one small interface.

* ``llamacpp``   – local GGUF model (default: Qwen2.5-3B-Instruct Q4_K_M) via
                   llama-cpp-python. No API key, no per-token cost; runs on CPU.
* ``anthropic``  – Claude through the official SDK (needs RAG_ANTHROPIC_API_KEY).
* ``openai``     – OpenAI Chat Completions (needs RAG_OPENAI_API_KEY).
* ``extractive`` – no generation: returns the best passages verbatim. Used as a
                   fallback and as a lower bound in evaluation.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..core.config import Settings

log = logging.getLogger(__name__)


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str


class LlamaCppLLM:
    def __init__(self, model_path: Path, n_ctx: int, n_threads: int, prompt_cache: bool = False):
        from llama_cpp import Llama, LlamaRAMCache

        self.name = model_path.name
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_batch=512,
            verbose=False,
        )
        if prompt_cache:
            self.llm.set_cache(LlamaRAMCache(capacity_bytes=2 << 30))
        self._lock = threading.Lock()  # llama.cpp contexts are not thread-safe

    def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResult:
        t = time.perf_counter()
        with self._lock:
            out = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=0,
            )
        usage = out.get("usage", {})
        return LLMResult(
            out["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            (time.perf_counter() - t) * 1000,
            self.name,
        )


class AnthropicLLM:
    def __init__(self, api_key: str, model: str, effort: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com")
        self.name = model
        self.effort = effort

    def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResult:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        t = time.perf_counter()
        resp = self.client.messages.create(
            model=self.name,
            max_tokens=max(max_tokens, 2048),
            system=system,
            messages=convo,
            output_config={"effort": self.effort},
        )
        if resp.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMResult(
            text,
            resp.usage.input_tokens,
            resp.usage.output_tokens,
            (time.perf_counter() - t) * 1000,
            self.name,
        )


class OpenAILLM:
    """OpenAI Chat Completions back-end.

    Messages format is identical to llama.cpp: system / user / assistant with
    ``content`` strings. Token usage comes from ``response.usage``.
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import OpenAI

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.name = model

    def complete(self, messages: list[dict], max_tokens: int, temperature: float) -> LLMResult:
        t = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        return LLMResult(
            text,
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
            (time.perf_counter() - t) * 1000,
            resp.model or self.name,
        )


@lru_cache(maxsize=4)
def _llamacpp(path: str, n_ctx: int, n_threads: int, prompt_cache: bool) -> LlamaCppLLM:
    log.info("loading local LLM %s", path)
    return LlamaCppLLM(Path(path), n_ctx, n_threads, prompt_cache)


def get_llm(settings: Settings, model: str | None = None):
    if settings.llm_provider == "extractive":
        return None

    if settings.llm_provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("RAG_ANTHROPIC_API_KEY is not set")
        return AnthropicLLM(
            settings.anthropic_api_key,
            model or settings.anthropic_model,
            settings.anthropic_effort,
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("RAG_OPENAI_API_KEY is not set")
        return OpenAILLM(
            settings.openai_api_key,
            model or settings.openai_model,
            settings.openai_base_url,
        )

    return _llamacpp(
        str(settings.models_dir / (model or settings.llm_model)),
        settings.llm_context,
        settings.llm_threads,
        settings.llm_prompt_cache,
    )