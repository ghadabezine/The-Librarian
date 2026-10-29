import { textDir } from "../i18n.js";

// Renders an answer, turning [S1]-style markers into buttons that select the source.
export default function AnswerText({ text, onCite, activeId }) {
  const parts = text.split(/(\[S\d{1,2}\])/g);
  return (
    <div dir={textDir(text)} className="whitespace-pre-wrap leading-8 text-[15px]">
      {parts.map((p, i) => {
        const m = p.match(/^\[S(\d{1,2})\]$/);
        if (!m) return <span key={i}>{p}</span>;
        const id = `S${m[1]}`;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onCite(id)}
            className={`mx-0.5 inline-flex items-center rounded-md px-1.5 py-0 text-xs font-semibold align-middle transition
              ${activeId === id ? "bg-ink text-sand" : "bg-sand-dark text-ink hover:bg-ink hover:text-sand"}`}
            aria-label={`source ${id}`}
          >
            {id}
          </button>
        );
      })}
    </div>
  );
}
