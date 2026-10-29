"""Shared fixtures. Unit tests use a tiny synthetic corpus written in the style
of the Code (real article wording for a few articles) plus fake models, so
they run in seconds without downloading anything. Real-model behaviour is
measured by the evaluation harness in eval/, not here."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.core.config import Settings  # noqa: E402
from app.ingestion.chunker import chunk_article_units, finalize  # noqa: E402
from app.ingestion.loader import Page, SourceDoc  # noqa: E402
from app.ingestion.normalize import clean_text  # noqa: E402
from app.ingestion.structure import parse_units  # noqa: E402
from app.retrieval.bm25 import tokenize  # noqa: E402
from app.retrieval.retriever import Retriever  # noqa: E402
from app.retrieval.vector_store import VectorIndex  # noqa: E402

PAGE_5 = """مجلة الأحوال الشخصية
الكتاب الأول
الزواج
في المراكنة
الفصل الأول.- كل من الوعد بالزواج والمواعدة به لا يعتبر زواجا ولا يقضى
به.
الفصل 2.- (نقح بالقانون عدد 74 لسنة 1993 المؤرخ في 12 جويلية
1993).- لكل واحد من الخطيبين أن يسترد الهدايا التي يقدمها إلى الآخر، ما لم
يكن العدول من قبله أو وجد شرط خاص.
في الزواج
الفصل 3.- لا ينعقد الزواج إلا برضا الزوجين.
ويشترط لصحة الزواج إشهاد شاهدين من أهل الثقة وتسمية مهر للزوجة.
5"""
PAGE_6 = """الفصل 5 (نقح بالقانون عدد 32 لسنة 2007 المؤرخ في 14 ماي 2007).- يجب أن يكون كل من
الزوجين خلوا من الموانع الشرعية. وزيادة على ذلك فكل من لم يبلغ منهما ثمانية
عشر سنة كاملة لا يمكنه أن يبرم عقد زواج.
الفصل 6 (نقح بالقانون عدد 74 لسنة 1993 المؤرخ في 12 جويلية
1993).- زواج القاصر يتوقف على موافقة الولي والأم.
المهر
الفصل 12.- كل ما كان مباحا ومقوما بالمال تصلح تسميته مهرا وهو ملك للمرأة.
الفصل 13.- ليس للزوج أن يجبر المرأة على البناء إذا لم يدفع المهر، كما هو مبين
بال
الفصل 9.
وما بعده.
الفصل 4-- الحضانة حفظ الولد في مبيته والقيام بتربيته.
6"""
HIST_PAGE = """[7] كان الفصل 18 م.أ.ش في صياغته الأصلية ينص على أن: "تعدد الزوجات ممنوع."
والتزوج بأكثر من واحدة يستوجب عقابا بالسجن مدة عام."""


def make_doc(doc_id="csp", status="consolidated", extraction="ocr") -> SourceDoc:
    return SourceDoc(doc_id=doc_id, file="x.pdf", title="مجلة الأحوال الشخصية", language="ar",
                     extraction=extraction, legal_status=status, title_fr="Code du statut personnel",
                     source_url="https://example.invalid/csp.pdf")


@pytest.fixture
def pages():
    return [Page("csp", 5, clean_text(PAGE_5)), Page("csp", 6, clean_text(PAGE_6))]


@pytest.fixture
def units(pages):
    return parse_units("csp", pages)


@pytest.fixture
def chunks(units):
    src = make_doc()
    cs = chunk_article_units(src, units, 1200, 150)
    hist_src = make_doc("csp_hist", status="historical", extraction="text")
    hist_units = parse_units("csp_hist", [Page("csp_hist", 23, HIST_PAGE)], notes_pages=[23, 23],
                             trust_numbers=True)
    cs += chunk_article_units(hist_src, hist_units, 1200, 150)
    out, _ = finalize(cs)
    return out


class FakeEmbedder:
    """Bag-of-stems hashing embedder: deterministic, similar texts get similar vectors."""
    name = "fake"
    cache_size = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(256, dtype=np.float32)
        for tok in tokenize(text):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % 256] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    name = "fake-reranker"

    def score(self, query, passages):
        q = set(tokenize(query))
        return np.array([len(q & set(tokenize(p))) / (len(q) or 1) for p in passages], dtype=np.float32)


class FakeLLM:
    def __init__(self, reply="لا ينعقد الزواج إلا برضا الزوجين [S1]."):
        self.reply = reply
        self.calls = []

    def complete(self, messages, max_tokens, temperature):
        from app.generation.llm import LLMResult
        self.calls.append(messages)
        reply = self.reply(messages) if callable(self.reply) else self.reply
        return LLMResult(reply, 100, 20, 1.0, "fake-llm")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, min_rerank_score=0.05, top_k=3, candidate_k=10, rerank_k=10,
                    llm_provider="llamacpp")


@pytest.fixture
def retriever(settings, chunks):
    emb = FakeEmbedder()
    vecs = emb.embed_passages([c["embed_text"] for c in chunks])
    return Retriever(settings, chunks=chunks, vector_index=VectorIndex(vecs, [c["chunk_id"] for c in chunks], "fake"),
                     embedder=emb, reranker=FakeReranker())
