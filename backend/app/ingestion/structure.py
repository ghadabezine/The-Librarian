"""Split a legal text into structural units (articles, notes, preambles).

Tunisian Arabic legal texts number their articles "الفصل N" (French
"Article N"); books and chapters are "الكتاب" / "الباب". OCR is imperfect,
so article numbers are checked against the running sequence and repaired
(flagged ``number_inferred``) when they are clearly misread, e.g. "الفصل 6"
between articles 145 and 147.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .loader import Page

ORDINAL_FIRST = {"الأول", "الاول", "الأولى", "الاولى"}
SUFFIXES = {"مكرر": "مكرر", "ثالثا": "ثالثا", "رابعا": "رابعا", "خامسا": "خامسا"}

ARTICLE_START = re.compile(
    r"^\s*(?:الفصل|ا\s?لفص[لا]|الفصيل)\s*(?P<num>الأول|الاول|\d{1,3})?\s*(?P<suf>مكرر|ثالثا|رابعا|خامسا)?"
    r"\s*(?P<rest>.*)?$"
)
ARTICLE_GROUP = re.compile(r"^\s*الفصول\s+(?P<nums>[\d\sو،,\-]+?)\s*[\.\-–—]")
HEADING = re.compile(r"^\s*(?P<kind>الكتاب|الباب|القسم|الجزء|العنوان)\b(?P<rest>.*)$")
ARABIC_MARKS = re.compile(r"[\u064B-\u0652\u0670]")
LEADING_NOISE = re.compile(r"^[\s\-–—_|~*'\"`,]+(?=[\u0600-\u06FF(\d])")
PAGE_NUMBER = re.compile(r"^\s*\d{1,3}\s*$")
AMENDMENT = re.compile(r"\((?:[^()]|\([^()]*\))*?(?:نقح|تقح|أضيف|اضيف|ألغي|الغي|أبطل|ابطل|نسخ|أتمم)(?:[^()]|\([^()]*\))*\)")
LAW_REF = re.compile(r"(القانون|بالقانون|المرسوم|بالمرسوم|الأمر|بالأمر)\s+عدد\s+(\d{1,4})\s+لسنة\s+(\d{4})")
REPEAL = re.compile(r"(?<![\u0600-\u06FF])(?:ألغي|الغي|ألغيت|أبطل|ابطل|نسخ)(?![\u0600-\u06FF])")
FOOTNOTE = re.compile(r"^\s*\((?:\*{1,3}|\d{1,2})\)\s*\S")
NOTE_START = re.compile(r"^\s*\[(\d{1,3})\]\s*(.*)$")
ARTICLE_MENTION = re.compile(r"الفصل\s*(\d{1,3})(?:\s*(مكرر))?")


@dataclass
class Unit:
    doc_id: str
    unit_type: str                  # article | preamble | note | heading_only
    text: str
    page_start: int
    page_end: int
    article_number: int | None = None
    article_suffix: str | None = None
    number_inferred: bool = False
    book: str | None = None
    section: str | None = None
    heading: str | None = None
    amendments: list[str] = field(default_factory=list)
    amending_laws: list[str] = field(default_factory=list)
    repealed: bool = False
    note_number: int | None = None
    mentions: list[str] = field(default_factory=list)
    footnotes: list[str] = field(default_factory=list)
    group: list[int] = field(default_factory=list)

    @property
    def article_label(self) -> str | None:
        if self.article_number is None:
            return None
        return f"{self.article_number}" + (f" {self.article_suffix}" if self.article_suffix else "")


def _lines(pages: list[Page]) -> list[tuple[int, str]]:
    out = []
    for page in pages:
        lines = [LEADING_NOISE.sub("", l).strip() for l in page.text.split("\n")]
        while lines and (not lines[-1] or PAGE_NUMBER.match(lines[-1])):
            lines.pop()
        out.extend((page.page, l) for l in lines if l)
    return out


def _parse_number(num: str | None) -> int | None:
    if not num:
        return None
    if num in ORDINAL_FIRST:
        return 1
    return int(num) if num.isdigit() else None


def _accept(line: str, m: re.Match, prev: int | None, trust_numbers: bool, allow_restart: bool,
            prev_line: str) -> tuple[bool, int | None, bool]:
    """Decide whether an "الفصل ..." line starts a new article.

    Returns (is_article, number, inferred). Lines that merely continue a
    cross-reference broken across lines ("... المنصوص عليها بال" / "الفصل 9.")
    must not open a new article.
    """
    parsed = _parse_number(m.group("num"))
    suffix = SUFFIXES.get(m.group("suf") or "")
    rest = (m.group("rest") or "").strip()
    # IORT marks article numbers with ".-"; OCR also renders it as '-', '--' or 'ب"'.
    strong = (rest.startswith(("(", ".-", "-", ".–", "–", 'ب"', '"')) or "-" in rest[:4])
    # Without a marker, an article start must follow the end of a sentence.
    after_sentence = (not prev_line or prev_line.endswith((".", ":", "؛", ")"))
                      or _is_short_heading(prev_line))
    # Text that continues a cross-reference ("الفصل 5 وما بعده", "الفصل 9 من القانون").
    reference = rest.startswith(("من ", "وما", "و ما", "والفصل", "و الفصل", "إلى", "الى", "بال", "أعلاه",
                                 "السابق", "المذكور", "المشار", "و ", "."))
    textual = bool(rest) and not reference
    sequential = prev is not None and parsed == prev + 1
    if parsed is None:
        if not strong:
            return False, None, False
        number, inferred = _repair(None, suffix, prev)
        return True, number, inferred
    if trust_numbers:   # clean text layer: headers are "الفصل 12" alone on a line
        return (not rest or strong), parsed, False
    plausible = prev is None or (parsed == 1 and allow_restart) or prev < parsed <= prev + 6 or (
        suffix and parsed == prev)
    repairable = prev is not None and str(prev + 1).endswith(str(parsed))
    if strong and plausible:
        return True, parsed, False
    if strong:
        number, inferred = _repair(parsed, suffix, prev)
        return True, number, inferred
    if sequential and (textual or rest.startswith("و ")) and not rest.startswith(("وما", "و ما")):
        return True, parsed, False   # exactly the next number: OCR turned ".-" into a letter
    if (plausible or repairable) and after_sentence and (textual or not rest):
        number, inferred = (parsed, False) if plausible else _repair(parsed, suffix, prev)
        return True, number, inferred
    return False, None, False


def _repair(parsed: int | None, suffix: str | None, prev: int | None) -> tuple[int | None, bool]:
    """Return (number, inferred) given the previously accepted article number."""
    if prev is None:
        return parsed, parsed is None
    if suffix and parsed == prev:
        return parsed, False
    expected = prev + 1
    if parsed is not None and prev < parsed <= prev + 6:
        return parsed, False          # normal progression (allowing for gaps)
    if parsed is not None and str(expected).endswith(str(parsed)):
        return expected, True         # leading digits dropped by OCR ("6" for 146)
    if parsed is None or parsed <= prev or parsed > prev + 6:
        return expected, True
    return parsed, False


def _amendment_info(text: str) -> tuple[list[str], list[str], bool]:
    head = text[:400]
    amendments = [m.group(0) for m in AMENDMENT.finditer(head)]
    laws = [f"{m.group(1).lstrip('ب')} عدد {m.group(2)} لسنة {m.group(3)}"
            for a in amendments for m in LAW_REF.finditer(a)]
    repealed = any(REPEAL.search(a) for a in amendments)
    return amendments, list(dict.fromkeys(laws)), repealed


def _strong_article(line: str) -> bool:
    m = ARTICLE_START.match(ARABIC_MARKS.sub("", line))
    if not m:
        return False
    rest = (m.group("rest") or "").strip()
    return not rest or rest.startswith(("(", ".-", "-", ".–", "–", 'ب"', '"')) or "-" in rest[:4]


def _is_short_heading(line: str) -> bool:
    words = line.split()
    return (0 < len(words) <= 7 and len(line) <= 45 and not line.endswith((".", "،", ":", "؛"))
            and not ARTICLE_START.match(line))


def parse_units(doc_id: str, pages: list[Page], notes_pages: list[int] | None = None,
                trust_numbers: bool = False, allow_restart: bool = False) -> list[Unit]:
    note_pages = set(range(notes_pages[0], notes_pages[1] + 1)) if notes_pages else set()
    body = [p for p in pages if p.page not in note_pages]
    notes = [p for p in pages if p.page in note_pages]
    units = _parse_articles(doc_id, _lines(body), trust_numbers, allow_restart)
    units += _parse_notes(doc_id, _lines(notes))
    for u in units:
        u.mentions = list(dict.fromkeys(
            m.group(1) + (" مكرر" if m.group(2) else "") for m in ARTICLE_MENTION.finditer(u.text)))
    return units


def _parse_articles(doc_id: str, lines: list[tuple[int, str]], trust_numbers: bool,
                    allow_restart: bool) -> list[Unit]:
    units: list[Unit] = []
    book = section = heading = None
    pending_heading: list[str] = []
    cur: Unit | None = None
    buf: list[str] = []
    prev_num: int | None = None
    i = 0

    def flush():
        nonlocal cur, buf
        if cur is not None:
            cur.text = "\n".join(buf).strip()
            if cur.footnotes:   # publisher's footnotes, kept apart from the article wording
                cur.text += "\n" + "\n".join("[هامش] " + f for f in cur.footnotes)
            if cur.unit_type == "article":
                cur.amendments, cur.amending_laws, cur.repealed = _amendment_info(cur.text)
            if cur.text:
                units.append(cur)
                for n in cur.group[1:]:   # "الفصول 3 و4 و5" -> one unit per article
                    units.append(replace(cur, article_number=n, group=[]))
        cur, buf = None, []

    while i < len(lines):
        page, line = lines[i]
        m_head = HEADING.match(line)
        if m_head and len(line) <= 60 and not ARTICLE_START.match(line) and "شارع" not in line:
            # Book / chapter heading; its title is usually on the next short line.
            title = line
            if i + 1 < len(lines) and _is_short_heading(lines[i + 1][1]) and not HEADING.match(lines[i + 1][1]):
                title += " : " + lines[i + 1][1]
                i += 1
            flush()
            if m_head.group("kind") == "الكتاب":
                book, section, heading = title, None, None
            else:
                section, heading = title, None
            i += 1
            continue

        m_art = ARTICLE_START.match(ARABIC_MARKS.sub("", line))
        m_group = ARTICLE_GROUP.match(line)
        accepted = False
        if m_art:
            prev_line = lines[i - 1][1] if i > 0 else ""
            accepted, number, inferred = _accept(line, m_art, prev_num, trust_numbers, allow_restart, prev_line)
        if accepted:
            flush()
            suffix = SUFFIXES.get(m_art.group("suf") or "")
            prev_num = number
            if pending_heading:
                heading = " ".join(pending_heading)
                pending_heading = []
            cur = Unit(doc_id, "article", "", page, page, number, suffix, inferred, book, section, heading)
            buf = [line]
            i += 1
            continue
        if m_group:
            flush()
            nums = [int(n) for n in re.findall(r"\d+", m_group.group("nums"))]
            cur = Unit(doc_id, "article", "", page, page, nums[0] if nums else None, None, False,
                       book, section, heading, group=nums)
            buf = [line]
            prev_num = max(nums) if nums else prev_num
            i += 1
            continue

        if FOOTNOTE.match(line) and cur is not None:
            cur.footnotes.append(line)
            i += 1
            continue

        # Short line after a finished sentence and directly followed by an
        # article: a sub-heading (e.g. "المهر").
        nxt = lines[i + 1][1] if i + 1 < len(lines) else ""
        nxt2 = lines[i + 2][1] if i + 2 < len(lines) else ""
        prev_line = lines[i - 1][1] if i > 0 else ""
        prev_done = not prev_line or prev_line.endswith((".", ":", "؛", ")")) or _is_short_heading(prev_line)
        if prev_done and _is_short_heading(line) and (
                _strong_article(nxt) or (_is_short_heading(nxt) and _strong_article(nxt2))):
            if cur is not None and cur.unit_type == "article":
                flush()
            pending_heading.append(line)
            i += 1
            continue

        if cur is None:
            cur = Unit(doc_id, "preamble", "", page, page, None, None, False, book, section, heading)
            buf = []
        cur.page_end = page
        buf.append(line)
        i += 1
    flush()
    return units


def _parse_notes(doc_id: str, lines: list[tuple[int, str]]) -> list[Unit]:
    units: list[Unit] = []
    cur: Unit | None = None
    buf: list[str] = []
    for page, line in lines:
        m = NOTE_START.match(line) or re.match(r"^\s*(\d{1,2})\s+(اضيف|أضيف.*)$", line)
        if m:
            if cur:
                cur.text = "\n".join(buf).strip()
                units.append(cur)
            cur = Unit(doc_id, "note", "", page, page, note_number=int(m.group(1)))
            buf = [line]
        elif cur:
            cur.page_end = page
            buf.append(line)
    if cur:
        cur.text = "\n".join(buf).strip()
        units.append(cur)
    for u in units:
        first = ARTICLE_MENTION.search(u.text)
        if first:   # the article a historical note is about
            u.article_number = int(first.group(1))
            u.article_suffix = "مكرر" if first.group(2) else None
    return units


# --------------------------------------------------------------------------- French
# The official French edition has a clean text layer. Article headers look like
# "Article premier.-", "Article 66 bis (Ajouté par Art. 2 de la loi n°2006-10 ...).-";
# footnotes sit below a "__________" rule; every page carries the watermark text.

FR_SUFFIXES = {"bis": "مكرر", "ter": "ثالثا", "quater": "رابعا", "quinquies": "خامسا"}
FR_SUFFIX_NAMES = {v: k for k, v in FR_SUFFIXES.items()}
FR_ARTICLE_START = re.compile(
    r"^\s*Article\s+(?P<num>premier|Premier|PREMIER|1er|\d{1,3})(?:\s*(?P<suf>bis|ter|quater|quinquies)\b)?"
    r"\s*(?P<rest>.*)$")
FR_ARTICLE_GROUP = re.compile(r"^\s*Articles\s+(?P<nums>\d{1,3}(?:\s*[-,et]+\s*\d{1,3})+)\s*\.?-")
FR_HEADING = re.compile(r"^\s*(?P<kind>LIVRE|CHAPITRE|Chapitre|TITRE|Titre|SECTION|Section)\b(?P<rest>.*)$")
FR_WATERMARK = re.compile(r"^\s*Imprimerie Officielle de la R[ée]publique Tunisienne\s*$")
FR_RULE = re.compile(r"^\s*_{5,}\s*$")
FR_LAW_REF = re.compile(r"(loi|décret-loi|décret|arrêté)\s+n\s*°\s*(\d{2,4})\s*-\s*(\d{1,4})", re.IGNORECASE)
FR_MENTION = re.compile(r"\barticles?\s+(\d{1,3})(?:\s+(bis|ter|quater))?", re.IGNORECASE)


def fr_display_label(label: str | None) -> str | None:
    if not label:
        return label
    num, _, suf = label.partition(" ")
    return num + (f" {FR_SUFFIX_NAMES.get(suf, suf)}" if suf else "")


def _fr_lines(pages: list[Page]) -> list[tuple[int, str, bool]]:
    """(page, line, is_footnote); drops watermark and page-number lines."""
    out = []
    for page in pages:
        footnote = False
        lines = [l.strip() for l in page.text.split("\n")]
        k = 0
        while k < len(lines):
            line = lines[k]
            if not line or FR_WATERMARK.match(line) or PAGE_NUMBER.match(line):
                k += 1
                continue
            if FR_RULE.match(line):
                footnote = True
                k += 1
                continue
            if line == "Article" and k + 1 < len(lines):   # header broken after the word
                line = "Article " + lines[k + 1].strip()
                k += 1
            out.append((page.page, line, footnote))
            k += 1
    return out


def _fr_amendment_info(head: str) -> tuple[list[str], list[str], bool]:
    m = re.match(r"^\s*Articles?\s+[^(]*?(\((?:[^()]|\([^()]*\))*\))", head)
    amendments = [m.group(1)] if m else []
    laws = [f"{a.group(1).lower()} n°{a.group(2)}-{a.group(3)}" for x in amendments for a in FR_LAW_REF.finditer(x)]
    repealed = any(re.search(r"abrog", x, re.I) and not re.search(r"remplac", x, re.I) for x in amendments)
    return amendments, list(dict.fromkeys(laws)), repealed


def parse_units_fr(doc_id: str, pages: list[Page], allow_restart: bool = False) -> list[Unit]:
    lines = _fr_lines(pages)
    units: list[Unit] = []
    book = section = heading = None
    pending: list[str] = []
    cur: Unit | None = None
    buf: list[str] = []

    def flush():
        nonlocal cur, buf
        if cur is not None:
            cur.text = "\n".join(buf).strip()
            if cur.footnotes:
                cur.text += "\n" + "\n".join("[Note] " + f for f in cur.footnotes)
            if cur.unit_type == "article":
                cur.amendments, cur.amending_laws, cur.repealed = _fr_amendment_info(cur.text[:500])
            if cur.text:
                units.append(cur)
                for n in cur.group[1:]:
                    units.append(replace(cur, article_number=n, group=[]))
        cur, buf = None, []

    for i, (page, line, is_note) in enumerate(lines):
        if is_note:
            if cur is not None:
                if FOOTNOTE.match(line) or not cur.footnotes:
                    cur.footnotes.append(line)
                else:
                    cur.footnotes[-1] += " " + line
            continue
        m_head = FR_HEADING.match(line)
        if m_head and len(line) <= 70 and not line.endswith("."):
            flush()
            nxt = lines[i + 1][1] if i + 1 < len(lines) else ""
            title = line
            if nxt and nxt.isupper() and len(nxt) <= 70 and not FR_ARTICLE_START.match(nxt):
                title += " : " + nxt
                pending = []
            if m_head.group("kind").upper() == "LIVRE":
                book, section, heading = title, None, None
            else:
                section, heading = title, None
            continue
        if title_consumed(lines, i):
            continue
        m_group = FR_ARTICLE_GROUP.match(line)
        if m_group:
            flush()
            nums = [int(n) for n in re.findall(r"\d+", m_group.group("nums"))]
            cur = Unit(doc_id, "article", "", page, page, nums[0], None, False, book, section, heading, group=nums)
            buf = [line]
            continue
        m_art = FR_ARTICLE_START.match(line)
        rest = (m_art.group("rest") or "").strip() if m_art else ""
        if m_art and rest.startswith((".-", "(", "-", ".–", "–", ". -", ". (")):
            flush()
            raw = m_art.group("num")
            number = 1 if raw.lower() in ("premier", "1er") else int(raw)
            suffix = FR_SUFFIXES.get((m_art.group("suf") or "").lower())
            if pending:
                heading = " ".join(pending)
                pending = []
            cur = Unit(doc_id, "article", "", page, page, number, suffix, False, book, section, heading)
            buf = [line]
            continue
        # Short line directly before an article: a sub-heading ("De la dot").
        nxt = lines[i + 1][1] if i + 1 < len(lines) else ""
        prev = lines[i - 1][1] if i > 0 else ""
        if (len(line) <= 60 and len(line.split()) <= 8 and not line.endswith((".", ",", ";", ":"))
                and FR_ARTICLE_START.match(nxt) and (not prev or prev.endswith((".", ":", ")")) or prev.isupper())):
            if cur is not None and cur.unit_type == "article":
                flush()
            pending.append(line)
            continue
        if cur is None:
            cur = Unit(doc_id, "preamble", "", page, page, None, None, False, book, section, heading)
            buf = []
        cur.page_end = page
        buf.append(line)
    flush()
    for u in units:
        u.mentions = list(dict.fromkeys(
            m.group(1) + (f" {FR_SUFFIXES[m.group(2).lower()]}" if m.group(2) else "")
            for m in FR_MENTION.finditer(u.text)))
    return units


def title_consumed(lines, i) -> bool:
    """True for the uppercase title line that was already joined to a heading."""
    if i == 0:
        return False
    prev = lines[i - 1][1]
    line = lines[i][1]
    return bool(FR_HEADING.match(prev)) and line.isupper() and len(line) <= 70 and not FR_ARTICLE_START.match(line)
