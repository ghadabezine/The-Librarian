import { useState } from "react";

// Renders an answer, turning [S1]-style markers into buttons that select the
// source. Hovering or focusing a marker previews the cited article.
// Each line gets dir="auto" so the browser picks its direction from its first
// strong letter: an English or French line next to quoted Arabic text stays
// left-to-right, and Arabic lines stay right-to-left.

function articleTitle(s) {
  const fr = s.language === "fr";
  const art = s.article_label && s.chunk_type === "article"
    ? `${fr ? "Article" : "الفصل"} ${s.article_label_display || s.article_label} — ` : "";
  return art + s.doc_title;
}

// Shown only while hovered or focused, with fixed positioning, so it never
// stretches the chat's scroll area or gets clipped by it.
function CitePreview({ t, source, anchor }) {
  const fr = source.language === "fr";
  const W = Math.min(288, window.innerWidth - 16);
  const rtl = document.documentElement.dir === "rtl";
  let left = rtl ? anchor.right - W : anchor.left;
  left = Math.max(8, Math.min(left, window.innerWidth - W - 8));
  const below = anchor.bottom + 230 < window.innerHeight;
  const style = below ? { left, top: anchor.bottom + 6, width: W } : { left, bottom: window.innerHeight - anchor.top + 6, width: W };
  return (
    <span role="tooltip" style={style}
      className="pointer-events-none fixed z-50 block rounded-xl border border-sand-dark bg-white p-3 text-start text-ink shadow-lg">
      <span dir={fr ? "ltr" : "rtl"} lang={fr ? "fr" : "ar"} className={`block text-xs font-semibold ${fr ? "" : "font-ar"}`}>
        {articleTitle(source)}
      </span>
      <span dir={fr ? "ltr" : "rtl"} lang={fr ? "fr" : "ar"}
        className={`mt-1 line-clamp-4 text-[13px] ${fr ? "leading-6" : "legal-text leading-7"}`}>
        {source.text}
      </span>
      <span className="mt-1 block text-[11px] text-ink-soft/70">{t.openSource}</span>
    </span>
  );
}

function Cite({ t, id, source, active, onCite }) {
  const [anchor, setAnchor] = useState(null);
  const show = (e) => setAnchor(e.currentTarget.getBoundingClientRect());
  const hide = () => setAnchor(null);
  return (
    <>
      <button
        type="button"
        onClick={() => { hide(); onCite(id); }}
        onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}
        className={`mx-0.5 inline-flex items-center rounded-md px-1.5 py-0 text-xs font-semibold align-middle transition
          ${active ? "bg-ink text-sand" : "bg-sand-dark text-ink hover:bg-ink hover:text-sand"}`}
        aria-label={`source ${id}`}
      >
        {id}
      </button>
      {anchor && source && <CitePreview t={t} source={source} anchor={anchor} />}
    </>
  );
}

export default function AnswerText({ t, text, sources = [], onCite, activeId }) {
  const byId = Object.fromEntries(sources.map((s) => [s.id, s]));
  return (
    <div className="leading-8 text-[15px]">
      {text.split("\n").map((line, li) => (
        <div key={li} dir="auto" className="min-h-[1em] whitespace-pre-wrap">
          {line.split(/(\[S\d{1,2}\])/g).map((p, i) => {
            const m = p.match(/^\[S(\d{1,2})\]$/);
            if (!m) return <span key={i}>{p}</span>;
            const id = `S${m[1]}`;
            return <Cite key={i} t={t} id={id} source={byId[id]} active={activeId === id} onCite={onCite} />;
          })}
        </div>
      ))}
    </div>
  );
}
