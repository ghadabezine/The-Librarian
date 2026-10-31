// Finds which sentences of a source support the answer, by word overlap
// between the source sentence and the answer sentences that cite it.
// Lightweight and language-agnostic: works for Arabic and French; when the
// answer and the source are in different languages nothing is highlighted.

const STOP = new Set([
  // Arabic
  "الذي", "التي", "الذين", "على", "الى", "إلى", "عن", "في", "من", "مع", "هذا", "هذه", "ذلك", "تلك", "كل", "او", "أو",
  "الفصل", "فصل", "لا", "ما", "ان", "أن", "إن", "كان", "قد", "لم", "لدى", "بين", "عند", "بعد", "قبل", "غير", "حيث", "وفي", "التالية",
  // French
  "les", "des", "une", "dans", "par", "pour", "sur", "avec", "est", "sont", "qui", "que", "aux", "ces", "son", "ses",
  "article", "leur", "leurs", "pas", "peut", "être", "cette", "tout", "lorsque",
  // English
  "the", "and", "for", "with", "that", "this", "are", "was", "from", "can", "not", "has", "have",
]);

function norm(text) {
  return (text || "")
    .toLowerCase()
    .normalize("NFD").replace(/[̀-ͯ]/g, "")        // French accents
    .replace(/[ً-ْـ]/g, "")                    // Arabic diacritics, tatweel
    .replace(/[أإآ]/g, "ا").replace(/ى/g, "ي").replace(/ة/g, "ه")
    .replace(/\[s\d{1,2}\]/g, " ")
    .replace(/[^\p{L}\p{N}]+/gu, " ");
}

function tokens(text) {
  return norm(text).split(" ").filter((w) => w.length >= 3 && !STOP.has(w))
    .map((w) => (w.length > 4 && (w.startsWith("ال") || w.startsWith("وال")) ? w.replace(/^و?ال/, "") : w));
}

/** Answer sentences that cite `id`, as one bag of words. */
export function claimWords(answer, id) {
  const sentences = (answer || "").split(/(?<=[.!?؟\n])\s*/);
  const citing = [];
  sentences.forEach((s, i) => {
    if (!s.includes(`[${id}]`)) return;
    citing.push(s);
    // A marker placed after the full stop ("... المحكمة. [S1]") ends up alone:
    // it belongs to the sentence before it.
    if (tokens(s).length < 2 && i > 0) citing.push(sentences[i - 1]);
  });
  return new Set(tokens(citing.join(" ")));
}

/** Split a source text into segments that join back to the original text. */
export function segments(text) {
  const out = [];
  const re = /[^.؛;!?؟\n]+[.؛;!?؟]?\s*|\n/g;
  let m;
  while ((m = re.exec(text || "")) !== null) out.push(m[0]);
  return out;
}

/** Which segments share enough words with the claim to count as evidence. */
export function evidenceMarks(segs, claim) {
  return segs.map((seg) => {
    const ws = tokens(seg);
    if (!ws.length || !claim.size) return false;
    const hit = ws.filter((w) => claim.has(w)).length;
    return hit >= 4 || (hit >= 2 && hit / ws.length >= 0.34);
  });
}
