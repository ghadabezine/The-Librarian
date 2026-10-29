"""Query analysis: language, explicit article references, cross-lingual expansion.

Safeguard: the expansion terms below are only appended to the *keyword* (BM25)
query. The user's question is passed unchanged to the dense retriever, the
reranker and the LLM, so expansion can widen recall but cannot change the
legal meaning of what is being asked.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingestion.normalize import detect_language, search_form

# French/English legal terms -> Arabic terms used in the Code.
GLOSSARY: dict[str, str] = {
    "mariage": "زواج", "marriage": "زواج", "marry": "زواج", "married": "زواج", "epoux": "زوج زوجة",
    "spouse": "زوج زوجة", "spouses": "الزوجين", "husband": "الزوج", "wife": "الزوجة", "mari": "الزوج",
    "epouse": "الزوجة", "fiancailles": "خطبة الخطيبين", "engagement": "خطبة الخطيبين", "fiance": "الخطيبين",
    "dot": "مهر", "dowry": "مهر", "consentement": "رضا", "consent": "رضا", "temoin": "شاهدين",
    "witness": "شاهدين", "age": "سن سنة", "mineur": "قاصر", "minor": "قاصر", "tuteur": "ولي",
    "guardian": "ولي", "polygamie": "تعدد الزوجات", "polygamy": "تعدد الزوجات",
    "divorce": "طلاق", "divorcer": "طلاق", "repudiation": "طلاق", "conciliation": "صلح",
    "reconciliation": "صلح", "indemnite": "تعويض جراية", "compensation": "تعويض جراية",
    "garde": "حضانة", "custody": "حضانة", "hadhana": "حضانة", "visite": "زيارة", "visitation": "زيارة",
    "pension": "نفقة", "alimentaire": "نفقة", "alimony": "نفقة", "maintenance": "نفقة", "nafaqa": "نفقة",
    "viduite": "عدة", "idda": "عدة", "waiting": "عدة", "filiation": "نسب", "paternity": "نسب",
    "paternite": "نسب", "adoption": "تبني", "adopt": "تبني", "adopte": "تبني", "adopted": "تبني",
    "succession": "إرث ميراث تركة", "heritage": "إرث ميراث تركة", "inheritance": "إرث ميراث تركة",
    "heir": "وارث الورثة", "heirs": "الورثة", "heritier": "وارث الورثة", "inherit": "يرث",
    "testament": "وصية", "will": "وصية", "legs": "وصية", "bequest": "وصية", "donation": "هبة",
    "gift": "هبة الهدايا", "cadeaux": "الهدايا", "tutelle": "ولاية", "guardianship": "ولاية",
    "curatelle": "حجر", "interdiction": "حجر", "majorite": "رشد", "majority": "رشد",
    "absent": "مفقود", "missing": "مفقود", "disparu": "مفقود", "enfant": "الطفل المحضون الأبناء",
    "child": "الطفل المحضون الأبناء", "children": "الأبناء الأولاد", "enfants": "الأبناء الأولاد",
    "mere": "الأم", "mother": "الأم", "pere": "الأب", "father": "الأب", "nullite": "الزواج الفاسد",
    "void": "الزواج الفاسد", "invalid": "الزواج الفاسد", "empechement": "موانع", "impediment": "موانع",
    "obligation": "واجبات", "obligations": "واجبات", "duties": "واجبات", "droits": "حقوق", "rights": "حقوق",
    "logement": "مسكن", "housing": "مسكن", "domicile": "مسكن", "fonds": "صندوق ضمان النفقة",
    "fund": "صندوق ضمان النفقة", "trouve": "اللقيط", "foundling": "اللقيط", "assassin": "القتل",
    "murder": "القتل", "kill": "القتل", "share": "فرض نصيب", "part": "فرض نصيب", "quote": "فرض",
}

ARTICLE_REF = re.compile(
    r"(?:الفصل|للفصل|بالفصل|فصل|article|art\.?)\s*(\d{1,3})(?:\s*(مكرر|bis|ثالثا|ter))?",
    re.IGNORECASE,
)
ARTICLE_FIRST = re.compile(r"(?:الفصل الأول|الفصل الاول|article (?:premier|1er|first))", re.IGNORECASE)


@dataclass
class QueryInfo:
    text: str
    language: str
    article_refs: list[str] = field(default_factory=list)
    keyword_query: str = ""


def analyse(question: str) -> QueryInfo:
    q = question.strip()
    lang = detect_language(q)
    refs = []
    if ARTICLE_FIRST.search(q):
        refs.append("1")
    for num, suffix in ARTICLE_REF.findall(q):
        label = num + (" مكرر" if suffix and suffix.lower() in ("مكرر", "bis") else "")
        if label not in refs:
            refs.append(label)
    expansion = []
    if lang != "ar":
        for token in search_form(q).split():
            for key in (token, token.rstrip("s")):
                if key in GLOSSARY:
                    expansion.append(GLOSSARY[key])
                    break
    keyword_query = q + (" " + " ".join(expansion) if expansion else "")
    return QueryInfo(q, lang, refs, keyword_query)
