"""BM25 keyword index with Arabic/French aware tokenisation."""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

from ..ingestion.normalize import search_form

AR_STOP = set(search_form(
    "في من على الى إلى عن ان أن إن او أو ما لا لم لن هو هي هم التي الذي الذين اللذين هذا هذه ذلك تلك "
    "كل بعد قبل مع عند بين كان كانت يكون تكون قد ثم اذا إذا الا إلا غير به بها له لها لهم منه منها فيه "
    "فيها عليه عليها ولا وان وإن وما ومن او هل ماذا كيف متى لماذا اي أي ما هو ما هي حسب القانون التونسي تونس"
).split())
FR_STOP = set(search_form(
    "le la les un une des du de d l et ou en au aux a à est sont que qui quoi quel quelle quels quelles "
    "pour par sur dans avec ce cette ces se sa son ses leur leurs il elle ils elles on ne pas plus "
    "comment selon loi tunisie tunisien tunisienne droit"
).split())
EN_STOP = set(
    "the a an of to in on for by with is are was be what which who how does do can under about "
    "and or law tunisia tunisian legal".split()
)
STOP = AR_STOP | FR_STOP | EN_STOP

AR_PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال", "و")
AR_SUFFIXES = ("هما", "كما", "ات", "ون", "ين", "ان", "ها", "هم", "هن", "ه", "ي")
ARABIC = re.compile(r"[؀-ۿ]")


def stem(token: str) -> str:
    if ARABIC.search(token):
        for p in AR_PREFIXES:
            if token.startswith(p) and len(token) - len(p) >= 3:
                token = token[len(p):]
                break
        for s in AR_SUFFIXES:
            if token.endswith(s) and len(token) - len(s) >= 3:
                token = token[: -len(s)]
                break
        return token
    if len(token) > 4 and token.endswith("s"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in search_form(text).split() if t not in STOP and (len(t) > 1 or t.isdigit())]


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        self.tokens = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokens, k1=k1, b=b)

    def search(self, query: str, k: int, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        q = tokenize(query)
        if not q:
            return []
        scores = self.bm25.get_scores(q)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]
