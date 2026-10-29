"""Text normalisation for Arabic and French legal text.

Two levels are kept apart on purpose:

* ``clean_text``   – conservative clean-up of extraction artefacts. The result is
  what we store and show to users, so it must not change the wording.
* ``search_form``  – aggressive folding used only for matching (BM25, dedup,
  citation checks): diacritics, tatweel, alef/ya/ta-marbuta variants, case and
  accents are folded so that spelling variants match.
"""
from __future__ import annotations

import re
import unicodedata

ARABIC_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
ARABIC_SCRIPT = re.compile(r"[؀-ۿ]")
# Zero-width and bidi control characters that PDF extraction leaves behind.
INVISIBLES = re.compile(r"[​-‏‪-‮⁦-⁩﻿­]")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = INVISIBLES.sub("", text)
    text = text.replace(TATWEEL, "")
    text = text.translate(ARABIC_INDIC_DIGITS)
    text = text.replace("\t", " ")
    # Re-join French words hyphenated across line breaks ("ma-\nriage").
    text = re.sub(r"([a-zà-ÿ])-\n([a-zà-ÿ])", r"\1\2", text)
    text = re.sub(r"[  ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Some PDF text layers store the lam-alef ligature in the wrong order, so
# "إلا" comes out as "إال" and "الأصلية" as "األصلية". Only unambiguous
# patterns are repaired; a word-internal "ال" can be legitimate, so it is left.
_LAM_ALEF_FIXES = [
    (re.compile(r"(?<![\u0600-\u06FF])ال(?![\u0600-\u06FF])"), "لا"),
    (re.compile(r"إال(?![\u0600-\u06FF])"), "إلا"),
    (re.compile(r"األ"), "الأ"), (re.compile(r"اإل"), "الإ"), (re.compile(r"اآل"), "الآ"),
]


def fix_lam_alef(text: str) -> str:
    for pattern, repl in _LAM_ALEF_FIXES:
        text = pattern.sub(repl, text)
    return text


def fold_arabic(text: str) -> str:
    text = ARABIC_DIACRITICS.sub("", text).replace(TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return text


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def search_form(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(ARABIC_INDIC_DIGITS)
    text = fold_arabic(text)
    text = strip_accents(text.lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_language(text: str) -> str:
    """Very small script/lexicon based detector: 'ar', 'fr' or 'en'."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    arabic = sum(1 for c in letters if ARABIC_SCRIPT.match(c))
    if arabic / len(letters) > 0.3:
        return "ar"
    words = set(re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower()))
    fr_markers = {"le", "la", "les", "des", "du", "de", "est", "quelles", "quels", "quel", "quelle",
                  "comment", "conditions", "droit", "droits", "époux", "épouse", "mariage", "divorce",
                  "enfant", "enfants", "garde", "loi", "selon", "pour", "une", "un", "et", "en",
                  "qu'est-ce", "sont", "peut", "dans", "sur", "avec", "pension", "succession", "héritage",
                  "que", "dit", "qui", "quoi", "il", "elle", "ce", "cette", "au", "aux", "doit", "faut",
                  "prévoit", "délai", "mère", "père", "père", "femme", "mari", "l'article", "d'un", "d'une"}
    en_markers = {"the", "what", "is", "are", "does", "how", "who", "law", "of", "and", "to", "can",
                  "marriage", "divorce", "child", "custody", "inheritance", "which", "under", "about"}
    fr = len(words & fr_markers) + sum(1 for c in text.lower() if c in "àâçéèêëîïôûùœ")
    en = len(words & en_markers)
    return "fr" if fr > en else "en"
