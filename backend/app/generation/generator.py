"""Retrieval-augmented answering with citations and abstention."""
from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from ..core.config import Settings
from ..core.costs import api_cost, compute_cost
from ..retrieval.retriever import Retriever
from .llm import get_llm
from .prompts import ABSTAIN_TOKEN, build_messages
from .verify import CITATION, Verification, normalize_citations, verify_answer

log = logging.getLogger(__name__)

DISCLAIMER = {
    "ar": "تنبيه: هذه معلومات قانونية عامة مستخرجة من النصوص الرسمية وليست استشارة قانونية. "
          "للحالات الخاصة يرجى استشارة محام أو عدل إشهاد.",
    "fr": "Avertissement : il s'agit d'une information juridique tirée des textes officiels, et non d'un "
          "conseil juridique. Pour un cas particulier, consultez un avocat.",
    "en": "Disclaimer: this is legal information drawn from the official texts, not legal advice. "
          "For a specific case, consult a lawyer.",
}
ABSTAIN_MESSAGE = {
    "ar": "لم أجد في النصوص المتوفرة لدي ما يسمح بالإجابة عن هذا السؤال بشكل موثوق.",
    "fr": "Je n'ai pas trouvé dans les textes dont je dispose de quoi répondre de façon fiable à cette question.",
    "en": "I could not find anything in the available legal texts that answers this question reliably.",
}
EXTRACTIVE_INTRO = {
    "ar": "أهم النصوص ذات الصلة بسؤالك:",
    "fr": "Les dispositions les plus pertinentes pour votre question (texte original en arabe) :",
    "en": "The most relevant provisions for your question (original Arabic text):",
}
HISTORY_INTENT = re.compile(
    r"(قبل\s+(?:ال)?تنقيح|الصيغة\s+(?:الأصلية|القديمة|السابقة)|سابقا|كان\s+ينص|تاريخ(?:ي|ية)?\s+الفصل|"
    r"ancienne|historique|avant\s+(?:la\s+)?(?:réforme|modification)|originale|"
    r"previous(?:ly)?|original\s+(?:text|version|wording)|history|before\s+(?:the\s+)?amendment|used\s+to)",
    re.IGNORECASE)


@dataclass
class Answer:
    question: str
    language: str
    answer: str
    status: str                     # answered | abstained
    abstain_reason: str | None
    citations: list[dict]
    sources: list[dict]
    disclaimer: str
    verification: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    retrieval_mode: str = ""
    model: str = ""


def _source_view(c: dict, sid: int, rerank: float | None = None) -> dict:
    return {
        "id": f"S{sid}", "chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "doc_title": c["doc_title"],
        "doc_title_fr": c.get("doc_title_fr"), "article_label": c.get("article_label"),
        "article_label_display": c.get("article_label_display"), "language": c.get("language"),
        "work_id": c.get("work_id"),
        "chunk_type": c.get("chunk_type"), "context": c.get("context"), "text": c["text"],
        "page_start": c.get("page_start"), "page_end": c.get("page_end"), "source_url": c.get("source_url"),
        "legal_status": c.get("legal_status"), "authority": c.get("authority"),
        "amending_laws": c.get("amending_laws") or [], "repealed": c.get("repealed", False),
        "edition": c.get("edition"), "score": rerank,
    }


class RAGService:
    def __init__(self, settings: Settings, retriever: Retriever | None = None, llm=None):
        self.s = settings
        self.retriever = retriever or Retriever(settings)
        self._llm = llm
        self._cache: OrderedDict[tuple, Answer] = OrderedDict()

    @property
    def llm(self):
        if self._llm is None and self.s.llm_provider != "extractive":
            self._llm = get_llm(self.s)
        return self._llm

    def answer(self, question: str, mode: str | None = None, top_k: int | None = None,
               filters: dict | None = None) -> Answer:
        key = (question.strip(), mode, top_k, tuple(sorted((filters or {}).items())))
        if self.s.query_cache_size and key in self._cache:
            self._cache.move_to_end(key)
            cached = self._cache[key]
            return Answer(**{**cached.__dict__, "timings_ms": {"total": 0.0, "cache_hit": 1.0}})
        result = self._answer(question, mode, top_k, filters)
        if self.s.query_cache_size:
            self._cache[key] = result
            while len(self._cache) > self.s.query_cache_size:
                self._cache.popitem(last=False)
        return result

    def _answer(self, question: str, mode: str | None, top_k: int | None, filters: dict | None) -> Answer:
        t0 = time.perf_counter()
        wants_history = bool(HISTORY_INTENT.search(question)) or self.s.include_historical
        predicate = None if wants_history else (lambda c: c.get("legal_status") != "historical")
        ret = self.retriever.retrieve(question, mode=mode, top_k=top_k, filters=filters, predicate=predicate)
        lang = ret.query.language
        timings = dict(ret.timings_ms)
        hits = ret.hits
        chunks = [h.chunk for h in hits]
        sources = [_source_view(h.chunk, i, h.rerank_score) for i, h in enumerate(hits, 1)]

        def abstain(reason: str, extra: dict | None = None, usage: dict | None = None, model: str = "") -> Answer:
            timings["total"] = (time.perf_counter() - t0) * 1000
            return Answer(question, lang, ABSTAIN_MESSAGE[lang], "abstained", reason, [], sources,
                          DISCLAIMER[lang], extra or {}, timings, usage or {}, ret.mode, model)

        if not hits:
            return abstain("no_relevant_passages")
        if ret.reranked and not any(h.via_article_ref for h in hits):
            best = max(h.rerank_score or 0.0 for h in hits)
            if best < self.s.min_rerank_score:
                return abstain("low_retrieval_confidence", {"best_rerank_score": best})

        if self.s.llm_provider == "extractive":
            lines = [EXTRACTIVE_INTRO[lang]]
            for i, c in enumerate(chunks[:3], 1):
                lines.append(f"• {c['context'].split(' › ')[-1]}: {c['text'][:400]} [S{i}]")
            text = "\n".join(lines)
            ver = verify_answer(text, chunks, question)
            timings["total"] = (time.perf_counter() - t0) * 1000
            return Answer(question, lang, text, "answered", None, self._citations(ver, chunks, sources),
                          sources, DISCLAIMER[lang], ver.__dict__, timings, {}, ret.mode, "extractive")

        messages, used = build_messages(question, chunks, lang, self.s.max_context_chars)
        t = time.perf_counter()
        out = self.llm.complete(messages, self.s.llm_max_tokens, self.s.llm_temperature)
        timings["llm"] = (time.perf_counter() - t) * 1000
        usage = {"input_tokens": out.input_tokens, "output_tokens": out.output_tokens}
        raw = out.text.strip()
        text = normalize_citations(raw)

        if not text or ABSTAIN_TOKEN in text[:60]:
            return abstain("model_abstained", {"raw_answer": raw}, usage=usage, model=out.model)
        text = text.replace(ABSTAIN_TOKEN, "").strip()
        ver = verify_answer(text, used, question)
        if not ver.cited_ids:
            return abstain("answer_without_citations", {**ver.__dict__, "raw_answer": raw}, usage, out.model)
        if ver.invalid_ids:   # drop citations to non-existent sources
            text = CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= len(used) else "", text)
        timings["total"] = (time.perf_counter() - t0) * 1000
        usage["cost_usd"] = (
    api_cost(out.model, out.input_tokens, out.output_tokens)
    if self.s.llm_provider in ("anthropic", "openai")
    else compute_cost(timings["total"] / 1000)
)
        return Answer(question, lang, text, "answered", None, self._citations(ver, used, sources), sources,
                      DISCLAIMER[lang], ver.__dict__, timings, usage, ret.mode, out.model)

    @staticmethod
    def _citations(ver: Verification, used: list[dict], sources: list[dict]) -> list[dict]:
        return [sources[i - 1] for i in ver.cited_ids if i - 1 < len(sources)]
