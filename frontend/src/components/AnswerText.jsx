// Renders an answer, turning [S1]-style markers into buttons that select the source.
// Each line gets dir="auto" so the browser picks its direction from its first
// strong letter: an English or French line next to quoted Arabic text stays
// left-to-right, and Arabic lines stay right-to-left.
export default function AnswerText({ text, onCite, activeId }) {
  return (
    <div className="leading-8 text-[15px]">
      {text.split("\n").map((line, li) => (
        <div key={li} dir="auto" className="min-h-[1em] whitespace-pre-wrap">
          {line.split(/(\[S\d{1,2}\])/g).map((p, i) => {
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
      ))}
    </div>
  );
}
