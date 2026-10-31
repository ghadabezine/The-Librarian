import { textDir } from "../i18n.js";

// Empty-screen map of what the corpus covers: one card per topic, each with
// starter questions in the interface language.
export default function TopicCards({ t, lang, topics, onAsk }) {
  return (
    <div className="mx-auto max-w-3xl pt-4">
      <div className="mb-5 text-center">
        <h2 className="text-xl font-semibold text-ink">{t.topicsTitle}</h2>
        <p className="mt-1 text-sm text-ink-soft">{t.topicsHint}</p>
      </div>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {topics.map((tp) => (
          <div key={tp.id}
            className="group rounded-2xl border border-sand-dark bg-white p-4 shadow-sm transition
              hover:-translate-y-0.5 hover:border-ink/30 hover:shadow-md">
            <div className="mb-3 flex items-center gap-3">
              <span aria-hidden="true"
                className="flex h-10 w-10 items-center justify-center rounded-xl bg-sand text-xl transition group-hover:bg-sand-dark">
                {tp.icon}
              </span>
              <h3 className="font-semibold">{tp.title[lang]}</h3>
            </div>
            <ul className="space-y-1.5">
              {tp.q[lang].map((q) => (
                <li key={q}>
                  <button type="button" dir={textDir(q)} onClick={() => onAsk(q)}
                    className="w-full rounded-lg px-2 py-1 text-start text-sm text-ink-soft transition
                      hover:bg-sand hover:text-ink">
                    <span className="text-tn" aria-hidden="true">›</span> {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}
