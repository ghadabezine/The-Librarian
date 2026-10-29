"""Post-generation checks that tie the answer back to the retrieved passages."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingestion.normalize import clean_text

CITATION = re.compile(r"\[S(\d{1,2})\]")
ARTICLE_IN_ANSWER = re.compile(
    r"(?:الفصل|للفصل|بالفصل|الفصلين|article|art\.)\s*(\d{1,3})(?:\s*(مكرر|bis))?", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w\[])(\d{1,4})(?![\w\]])")
SENTENCE_SPLIT = re.compile(r"(?<=[\.\!\?؟])\s+|\n+")


CITATION_GROUP = re.compile(r"[\[\(（【]\s*((?:S\s*\d{1,2}\s*(?:[,،؛;و&]|and|et)?\s*)+)[\]\)）】]")


def normalize_citations(text: str) -> str:
    """Rewrite citation variants ("[S1, S2]", "(S3)", "[S1 و S2]") as "[S1][S2]"."""
    def repl(m: re.Match) -> str:
        return "".join(f"[S{n}]" for n in re.findall(r"S\s*(\d{1,2})", m.group(1)))
    return CITATION_GROUP.sub(repl, text)


@dataclass
class Verification:
    cited_ids: list[int] = field(default_factory=list)
    invalid_ids: list[int] = field(default_factory=list)
    unverified_articles: list[str] = field(default_factory=list)
    unsupported_numbers: list[str] = field(default_factory=list)
    uncited_sentences: list[str] = field(default_factory=list)
    sentences: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.cited_ids) and not self.invalid_ids and not self.unverified_articles


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if len(s.strip()) > 3]


def verify_answer(answer: str, used_chunks: list[dict], question: str = "") -> Verification:
    v = Verification()
    ids = [int(m) for m in CITATION.findall(answer)]
    v.cited_ids = sorted({i for i in ids if 1 <= i <= len(used_chunks)})
    v.invalid_ids = sorted({i for i in ids if not 1 <= i <= len(used_chunks)})

    allowed_articles = set()
    for c in used_chunks:
        if c.get("article_label"):
            allowed_articles.add(c["article_label"])
        allowed_articles.update(c.get("mentions") or [])
    for num, suf in ARTICLE_IN_ANSWER.findall(answer):
        label = num + (" مكرر" if suf else "")
        if label not in allowed_articles and num not in allowed_articles:
            v.unverified_articles.append(label)

    # Every number in the answer (ages, durations, law numbers) must appear in a
    # cited source or in the question.
    cited_text = " ".join(clean_text(used_chunks[i - 1]["text"]) for i in v.cited_ids) + " " + question
    cited_text += " " + " ".join(used_chunks[i - 1].get("article_label") or "" for i in v.cited_ids)
    for n in NUMBER.findall(CITATION.sub("", answer)):
        if n not in cited_text and n not in v.unsupported_numbers:
            v.unsupported_numbers.append(n)

    sentences = split_sentences(answer)
    v.sentences = len(sentences)
    v.uncited_sentences = [s for s in sentences if not CITATION.search(s)]
    return v
