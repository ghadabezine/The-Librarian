from app.ingestion.chunker import _split_long, chunk_article_units, chunk_fixed, finalize
from conftest import make_doc


def test_chunk_ids_unique_and_metadata_preserved(chunks):
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    c5 = next(c for c in chunks if c["article_label"] == "5" and c["doc_id"] == "csp")
    assert c5["page_start"] == 6 and c5["source_url"].startswith("https://")
    assert "الفصل 5" in c5["embed_text"] and c5["text"].startswith("الفصل 5")


def test_long_article_split_keeps_article_label(units):
    long = units[0]
    long.text = "\n".join(["فقرة طويلة عن الزواج والطلاق والحضانة " * 8] * 6)
    cs = chunk_article_units(make_doc(), [long], max_chars=400, overlap=80)
    assert len(cs) > 1
    assert all(c["article_label"] == long.article_label for c in cs)
    assert all(len(c["text"]) <= 480 for c in cs)
    assert cs[0]["parts"] == len(cs)


def test_split_overlap():
    text = "\n".join(f"جملة رقم {i} " * 10 for i in range(10))
    pieces = _split_long(text, 300, 60)
    assert len(pieces) > 2
    assert pieces[1].split("\n")[0] in pieces[0]   # overlap carried over


def test_fixed_strategy_records_covered_articles(units):
    cs = chunk_fixed(make_doc(), units, max_chars=300, overlap=50)
    assert cs and all(c["covered_articles"] is not None for c in cs)


def test_finalize_dedup_and_short_drop(chunks):
    dup = dict(chunks[0], chunk_id="csp:dup:1")
    short = dict(chunks[0], chunk_id="csp:short:1", text="قصير")
    out, report = finalize(chunks + [dup, short])
    assert "csp:dup:1" not in {c["chunk_id"] for c in out}
    assert report["dropped_short"] == ["csp:short:1"]
