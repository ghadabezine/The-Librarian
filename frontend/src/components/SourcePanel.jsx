import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api.js";
import { claimWords, evidenceMarks, segments } from "../evidence.js";

function pdfLink(src) {
  if (!src.source_url) return null;
  return src.page_start ? `${src.source_url}#page=${src.page_start}` : src.source_url;
}

// One fetch per article, shared by every card that asks for it.
const versionCache = new Map();
function loadVersions(workId, label) {
  const key = `${workId}|${label}`;
  if (!versionCache.has(key)) {
    versionCache.set(key, api.article(workId, label).then((r) => r.versions).catch((e) => {
      versionCache.delete(key);
      throw e;
    }));
  }
  return versionCache.get(key);
}

function LegalText({ source, marks, clamp, active }) {
  const fr = source.language === "fr";
  const segs = useMemo(() => segments(source.text), [source.text]);
  return (
    <blockquote dir={fr ? "ltr" : "rtl"} lang={fr ? "fr" : "ar"}
      className={`${fr ? "leading-7 whitespace-pre-wrap" : "legal-text"} mt-2 border-s-4 ps-3 text-[14px]
        ${active ? "border-tn" : "border-sand-dark"} ${clamp ? "line-clamp-5" : ""}`}>
      {marks
        ? segs.map((seg, i) => marks[i]
          ? <mark key={i} className="rounded bg-amber-100 px-0.5 text-ink">{seg}</mark>
          : <span key={i}>{seg}</span>)
        : source.text}
    </blockquote>
  );
}

function VersionLabel({ source }) {
  const fr = source.language === "fr";
  return (
    <p dir={fr ? "ltr" : "rtl"} lang={fr ? "fr" : "ar"} className="mt-3 text-[11px] font-semibold uppercase tracking-wide text-ink-soft/70">
      {fr ? "Version française officielle (IORT 2024)" : "النص العربي الرسمي"}
    </p>
  );
}

function SourceCard({ t, s, answer, cited, isActive, onSelect, cardRef }) {
  const [view, setView] = useState("orig"); // orig | other | both
  const [versions, setVersions] = useState(null);
  const [loadState, setLoadState] = useState("idle"); // idle | loading | error

  const historical = s.legal_status === "historical";
  const link = pdfLink(s);
  const fr = s.language === "fr";
  const canTranslate = s.chunk_type === "article" && s.work_id && s.article_label && !historical;
  const otherLang = fr ? "ar" : "fr";

  // Highlight only on the open card, and only in the cited text itself.
  const marks = useMemo(() => {
    if (!isActive || !cited) return null;
    const m = evidenceMarks(segments(s.text), claimWords(answer, s.id));
    return m.some(Boolean) ? m : null;
  }, [isActive, cited, s.text, s.id, answer]);

  const choose = (v) => {
    setView(v);
    if (v === "orig" || versions || loadState === "loading") return;
    setLoadState("loading");
    loadVersions(s.work_id, s.article_label)
      .then((vs) => { setVersions(vs); setLoadState("idle"); })
      .catch(() => setLoadState("error"));
  };
  const other = versions?.find((v) => v.language === otherLang);
  const viewKey = view === "orig" ? s.language || "ar" : view === "other" ? otherLang : "both";

  return (
    <article ref={cardRef} onClick={() => onSelect(s.id)}
      className={`cursor-pointer rounded-xl border bg-white p-3 text-sm shadow-sm transition
        ${isActive ? "border-ink ring-2 ring-ink/20" : "border-sand-dark hover:border-ink/40"}`}>
      <header className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className="rounded bg-ink px-1.5 py-0.5 font-semibold text-sand">{s.id}</span>
        <span className={`rounded px-1.5 py-0.5 ${cited ? "bg-ok/10 text-ok" : "bg-sand text-ink-soft"}`}>
          {cited ? t.cited : t.retrieved}
        </span>
        <span className={`rounded px-1.5 py-0.5 ${historical ? "bg-warn/10 text-warn" : "bg-sand text-ink-soft"}`}>
          {historical ? t.historical : t.current}
        </span>
        {s.repealed && <span className="rounded bg-tn/10 px-1.5 py-0.5 text-tn">ملغى / abrogé</span>}
      </header>
      <h3 dir={fr ? "ltr" : "rtl"} lang={fr ? "fr" : "ar"} className={`${fr ? "" : "font-ar"} font-semibold leading-7`}>
        {s.article_label && s.chunk_type === "article"
          ? `${fr ? "Article" : "الفصل"} ${s.article_label_display || s.article_label} — ` : ""}{s.doc_title}
      </h3>
      {!fr && s.doc_title_fr && <p lang="fr" dir="ltr" className="text-end text-xs text-ink-soft">{s.doc_title_fr}</p>}
      <p className="mt-1 text-xs text-ink-soft">
        {s.page_start ? `${t.page} ${s.page_start}${s.page_end && s.page_end !== s.page_start ? `–${s.page_end}` : ""}` : ""}
        {s.amending_laws?.length ? ` · ${t.amendedBy}: ${s.amending_laws.join("، ")}` : ""}
      </p>

      {isActive && canTranslate && (
        <div role="group" aria-label="language" onClick={(e) => e.stopPropagation()}
          className="mt-2 inline-flex rounded-lg border border-sand-dark bg-sand p-0.5 text-xs">
          {["ar", "fr", "both"].map((k) => (
            <button key={k} type="button"
              onClick={() => choose(k === "both" ? "both" : k === (s.language || "ar") ? "orig" : "other")}
              aria-pressed={viewKey === k}
              className={`rounded-md px-2.5 py-1 transition ${viewKey === k ? "bg-white font-semibold text-ink shadow-sm" : "text-ink-soft hover:text-ink"}`}>
              {t.showIn[k]}
            </button>
          ))}
        </div>
      )}

      {(view === "orig" || view === "both") && (
        <>
          {view === "both" && <VersionLabel source={s} />}
          <LegalText source={s} marks={marks} clamp={!isActive} active={isActive} />
        </>
      )}
      {view !== "orig" && (
        loadState === "loading" ? <p className="mt-2 text-xs text-ink-soft">{t.loadingText}</p>
          : loadState === "error" ? <p className="mt-2 text-xs text-tn">{t.errServer}</p>
          : versions && !other ? <p className="mt-2 text-xs text-ink-soft">{t.noTranslation}</p>
          : other && (
            <>
              {view === "both" && <VersionLabel source={other} />}
              <LegalText source={other} clamp={false} active={isActive} />
            </>
          )
      )}

      {marks && view !== "other" && (
        <p className="mt-2 flex items-center gap-1.5 text-[11px] text-ink-soft">
          <span className="inline-block h-2.5 w-2.5 rounded-sm bg-amber-100 ring-1 ring-amber-200" />{t.evidence}
        </p>
      )}
      {link && (
        <a href={link} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}
          className="mt-2 inline-block text-xs text-ink underline underline-offset-2">{t.openPdf} ↗</a>
      )}
    </article>
  );
}

export default function SourcePanel({ t, response, activeId, onSelect }) {
  const refs = useRef({});
  useEffect(() => {
    if (activeId && refs.current[activeId]) {
      refs.current[activeId].scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [activeId, response]);

  if (!response) {
    return <p className="p-4 text-sm text-ink-soft">{t.sourcesHint}</p>;
  }
  const cited = new Set(response.citations.map((c) => c.id));
  return (
    <div className="space-y-3 p-3">
      {response.sources.map((s) => (
        <SourceCard key={s.chunk_id} t={t} s={s} answer={response.answer} cited={cited.has(s.id)}
          isActive={s.id === activeId} onSelect={onSelect} cardRef={(el) => (refs.current[s.id] = el)} />
      ))}
    </div>
  );
}
