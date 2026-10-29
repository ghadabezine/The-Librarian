from app.ingestion.loader import Page
from app.ingestion.structure import parse_units


def arts(units):
    return [u for u in units if u.unit_type == "article"]


def test_articles_detected_with_numbers(units):
    labels = [u.article_label for u in arts(units)]
    assert labels[:5] == ["1", "2", "3", "5", "6"]
    assert "12" in labels and "13" in labels


def test_book_and_subheading_metadata(units):
    a3 = next(u for u in arts(units) if u.article_label == "3")
    assert a3.book.startswith("الكتاب الأول")
    assert a3.heading == "في الزواج"
    a12 = next(u for u in arts(units) if u.article_label == "12")
    assert a12.heading == "المهر"


def test_amendment_extraction(units):
    a5 = next(u for u in arts(units) if u.article_label == "5")
    assert "القانون عدد 32 لسنة 2007" in a5.amending_laws
    assert not a5.repealed


def test_cross_reference_line_is_not_an_article(units):
    # "بال / الفصل 9." continues article 13; it must not open article 9.
    a13 = next(u for u in arts(units) if u.article_label == "13")
    assert "الفصل 9." in a13.text
    assert [u.article_label for u in arts(units)].count("9") == 0


def test_ocr_number_repair(units):
    # "الفصل 4--" after article 13 is article 14 with a dropped digit.
    a14 = next(u for u in arts(units) if u.article_label == "14")
    assert a14.number_inferred and "الحضانة" in a14.text


def test_page_numbers_dropped_and_pages_tracked(units):
    a3 = next(u for u in arts(units) if u.article_label == "3")
    assert a3.page_start == 5
    assert all(not u.text.strip().endswith("\n5") for u in units)


def test_repealed_article_group():
    page = Page("d", 4, "الفصول 3 و4 و5.- (أبطل العمل بهذه الفصول بمقتضى الفصل 5 من القانون عدد 40 لسنة 1957).")
    units = arts(parse_units("d", [page]))
    assert [u.article_number for u in units] == [3, 4, 5]
    assert all(u.repealed for u in units)


def test_historical_notes_link_to_article(chunks):
    note = next(c for c in chunks if c["chunk_type"] == "note")
    assert note["article_label"] == "18" and note["legal_status"] == "historical"
