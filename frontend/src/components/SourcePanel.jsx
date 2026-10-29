import { useEffect, useRef } from "react";

function pdfLink(src) {
  if (!src.source_url) return null;
  return src.page_start ? `${src.source_url}#page=${src.page_start}` : src.source_url;
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
      {response.sources.map((s) => {
        const isActive = s.id === activeId;
        const historical = s.legal_status === "historical";
        const link = pdfLink(s);
        const fr = s.language === "fr";
        return (
          <article key={s.chunk_id} ref={(el) => (refs.current[s.id] = el)} onClick={() => onSelect(s.id)}
            className={`cursor-pointer rounded-xl border bg-white p-3 text-sm shadow-sm transition
              ${isActive ? "border-ink ring-2 ring-ink/20" : "border-sand-dark hover:border-ink/40"}`}>
            <header className="mb-2 flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded bg-ink px-1.5 py-0.5 font-semibold text-sand">{s.id}</span>
              <span className={`rounded px-1.5 py-0.5 ${cited.has(s.id) ? "bg-ok/10 text-ok" : "bg-sand text-ink-soft"}`}>
                {cited.has(s.id) ? t.cited : t.retrieved}
              </span>
              <span className={`rounded px-1.5 py-0.5 ${historical ? "bg-warn/10 text-warn" : "bg-sand text-ink-soft"}`}>
                {historical ? t.historical : t.current}
              </span>
              {s.repealed && <span className="rounded bg-tn/10 px-1.5 py-0.5 text-tn">ملغى / abrogé</span>}
            </header>
            <h3 dir={fr ? "ltr" : "rtl"} className={`${fr ? "" : "font-ar"} font-semibold leading-7`}>
              {s.article_label && s.chunk_type === "article"
                ? `${fr ? "Article" : "الفصل"} ${s.article_label_display || s.article_label} — ` : ""}{s.doc_title}
            </h3>
            {!fr && s.doc_title_fr && <p className="text-xs text-ink-soft">{s.doc_title_fr}</p>}
            <p className="text-[11px] text-ink-soft/70">{fr ? "Version française officielle (IORT 2024)" : "النص العربي الرسمي"}</p>
            <p className="mt-1 text-xs text-ink-soft">
              {s.page_start ? `${t.page} ${s.page_start}${s.page_end && s.page_end !== s.page_start ? `–${s.page_end}` : ""}` : ""}
              {s.amending_laws?.length ? ` · ${t.amendedBy}: ${s.amending_laws.join("، ")}` : ""}
            </p>
            <blockquote dir={fr ? "ltr" : "rtl"} lang={fr ? "fr" : "ar"}
              className={`${fr ? "leading-7 whitespace-pre-wrap" : "legal-text"} mt-2 border-s-4 ps-3 text-[14px] ${isActive ? "border-tn" : "border-sand-dark"} ${isActive ? "" : "line-clamp-5"}`}>
              {s.text}
            </blockquote>
            {link && (
              <a href={link} target="_blank" rel="noreferrer" onClick={(e) => e.stopPropagation()}
                className="mt-2 inline-block text-xs text-ink underline underline-offset-2">{t.openPdf} ↗</a>
            )}
          </article>
        );
      })}
    </div>
  );
}
