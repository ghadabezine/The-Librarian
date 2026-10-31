import AnswerText from "./AnswerText.jsx";
import Feedback from "./Feedback.jsx";
import { textDir } from "../i18n.js";

export default function ChatMessage({ t, msg, onCite, activeSource }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div dir={textDir(msg.content)} className="max-w-[85%] rounded-2xl rounded-br-sm bg-ink px-4 py-2 text-sand shadow-sm">
          {msg.content}
        </div>
      </div>
    );
  }
  if (msg.error) {
    return (
      <div role="alert" className="rounded-xl border border-tn/40 bg-tn/5 px-4 py-3 text-sm text-tn">{msg.error}</div>
    );
  }
  const r = msg.response;
  const abstained = r.status === "abstained";
  // Extractive mode quotes the top passages without an LLM, so nothing was
  // generated or checked: say so instead of "answer grounded in sources".
  const extracted = !abstained && r.model === "extractive";
  const warnings = r.warnings || [];
  return (
    <div className={`rounded-2xl border bg-white px-4 py-3 shadow-sm ${abstained ? "border-warn/40" : "border-sand-dark"}`}>
      <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
        <span className={`rounded-full px-2 py-0.5 font-medium ${abstained ? "bg-warn/10 text-warn"
          : extracted ? "bg-sand-dark text-ink-soft" : "bg-ok/10 text-ok"}`}>
          {abstained ? `⚠ ${t.unsupported}` : extracted ? `≡ ${t.extracted}` : `✓ ${t.answered}`}
        </span>
        <span className="text-ink-soft/70">{r.model} · {Math.round(r.timings_ms.total)} ms</span>
      </div>
      <AnswerText t={t} text={r.answer} sources={r.sources} onCite={(id) => onCite(msg.id, id)}
        activeId={activeSource?.msgId === msg.id ? activeSource.sourceId : null} />
      {warnings.length > 0 && (
        <ul className="mt-2 space-y-1 text-xs text-warn">
          {warnings.some((w) => w.startsWith("cites_historical")) && <li>• {t.warnHistorical}</li>}
          {warnings.some((w) => w.startsWith("numbers_not_found")) && <li>• {t.warnNumbers}</li>}
          {warnings.some((w) => w.startsWith("unverified_article")) && <li>• {t.warnArticles}</li>}
        </ul>
      )}
      <p dir={textDir(r.disclaimer)} className="mt-3 text-[11px] text-ink-soft/70">{r.disclaimer}</p>
      <Feedback t={t} question={msg.question} response={r} />
    </div>
  );
}
