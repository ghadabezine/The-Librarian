import numpy as np
import pytest

from app.retrieval.bm25 import BM25Index, stem, tokenize
from app.retrieval.fusion import rrf
from app.retrieval.query import analyse


def test_rrf_rewards_agreement():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    order = [d for d, _ in fused]
    assert order[0] == 1 and set(order) == {1, 2, 3, 4}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_arabic_light_stemming_matches_prefixed_forms():
    assert stem("والزواج") == stem("الزواج") == stem("بالزواج")
    idx = BM25Index(["لا ينعقد الزواج إلا برضا الزوجين", "الحضانة حفظ الولد في مبيته",
                     "الهبة عقد بمقتضاه يملك شخص آخر مالا"])
    hits = idx.search("شروط انعقاد الزواج", 2)
    assert hits[0][0] == 0


def test_stopwords_removed():
    assert "في" not in tokenize("في الزواج") and "les" not in tokenize("les conditions")


def test_query_analysis_article_refs_and_language():
    info = analyse("ماذا ينص الفصل 5 من مجلة الأحوال الشخصية؟")
    assert info.language == "ar" and info.article_refs == ["5"]
    info = analyse("Que dit l'article 32 bis ?")
    assert info.language == "fr" and info.article_refs == ["32 مكرر"]
    assert analyse("Que dit l'article premier ?").article_refs == ["1"]


def test_cross_lingual_expansion_only_affects_keyword_query():
    info = analyse("What does Tunisian law say about child custody?")
    assert "حضانة" in info.keyword_query
    assert info.text == "What does Tunisian law say about child custody?"


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid", "hybrid_rerank"])
def test_all_modes_find_consent_article(retriever, mode):
    res = retriever.retrieve("هل ينعقد الزواج بدون رضا الزوجين؟", mode=mode, top_k=3)
    labels = [h.chunk["article_label"] for h in res.hits]
    assert "3" in labels, (mode, labels)
    assert res.mode == mode and "total" in res.timings_ms


def test_explicit_article_reference_is_pinned_first(retriever):
    res = retriever.retrieve("ماذا يقول الفصل 12 ؟", mode="hybrid_rerank", top_k=3)
    assert res.hits[0].chunk["article_label"] == "12" and res.hits[0].via_article_ref


def test_metadata_filter(retriever):
    res = retriever.retrieve("تعدد الزوجات", mode="bm25", top_k=5,
                             predicate=lambda c: c["legal_status"] != "historical")
    assert all(h.chunk["legal_status"] != "historical" for h in res.hits)
    res = retriever.retrieve("تعدد الزوجات", mode="bm25", top_k=5, filters={"doc_id": "csp_hist"})
    assert res.hits and all(h.chunk["doc_id"] == "csp_hist" for h in res.hits)


def test_vector_index_roundtrip(tmp_path):
    from app.retrieval.vector_store import VectorIndex
    v = np.eye(3, dtype=np.float32)
    VectorIndex(v, ["a", "b", "c"], "m").save(tmp_path)
    idx = VectorIndex.load(tmp_path)
    assert idx.search(np.array([0, 1, 0], dtype=np.float32), 1)[0][0] == 1
    assert idx.search(np.array([0, 1, 0], dtype=np.float32), 2, mask=np.array([True, False, True]))[0][0] in (0, 2)


def test_translations_collapsed(settings, chunks):
    from app.retrieval.retriever import Retriever
    from conftest import FakeEmbedder, FakeReranker
    from app.retrieval.vector_store import VectorIndex
    fr = dict(next(c for c in chunks if c["article_label"] == "3" and c["doc_id"] == "csp"),
              chunk_id="csp_fr:art3:1", doc_id="csp_fr", work_id="csp", language="fr",
              text="Article 3.- Le mariage n'est formé que par le consentement des deux époux.")
    fr["embed_text"] = fr["search_text"] = fr["text"]
    base = [dict(c, work_id=c["doc_id"]) for c in chunks]
    allc = base + [fr]
    emb = FakeEmbedder()
    vi = VectorIndex(emb.embed_passages([c["embed_text"] for c in allc]), [c["chunk_id"] for c in allc], "fake")
    for collapse in (True, False):
        settings.collapse_translations = collapse
        r = Retriever(settings, chunks=allc, vector_index=vi, embedder=emb, reranker=FakeReranker())
        res = r.retrieve("الزواج رضا الزوجين consentement des époux mariage", mode="hybrid", top_k=10)
        n = sum(1 for h in res.hits if h.chunk.get("work_id") == "csp" and h.chunk["article_label"] == "3")
        assert n == (1 if collapse else 2)
