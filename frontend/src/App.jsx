import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api.js";
import { TOPICS, UI, textDir } from "./i18n.js";
import ChatMessage from "./components/ChatMessage.jsx";
import SourcePanel from "./components/SourcePanel.jsx";
import CorpusModal from "./components/CorpusModal.jsx";
import TopicCards from "./components/TopicCards.jsx";

let nextId = 1;

export default function App() {
  const [lang, setLang] = useState(() => {
    try { return localStorage.getItem("ui-lang") || "ar"; } catch { return "ar"; }
  });
  const t = UI[lang];
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [active, setActive] = useState(null); // {msgId, sourceId}
  const [showCorpus, setShowCorpus] = useState(false);
  const [showSourcesMobile, setShowSourcesMobile] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    document.documentElement.lang = lang;
    document.documentElement.dir = t.dir;
    try { localStorage.setItem("ui-lang", lang); } catch { /* storage unavailable */ }
  }, [lang, t.dir]);

  useEffect(() => {
    if (messages.length || loading) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // The source panel shows the selected answer, or the latest one.
  const panelMsg = useMemo(() => {
    const answers = messages.filter((m) => m.role === "assistant" && m.response);
    if (active) return answers.find((m) => m.id === active.msgId) || answers.at(-1);
    return answers.at(-1);
  }, [messages, active]);

  const ask = async (question) => {
    const q = question.trim();
    if (!q || loading) return;
    const history = messages
      .filter((m) => m.role === "user" || m.response)
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.role === "user" ? m.content : m.response.answer }));
    setMessages((ms) => [...ms, { id: nextId++, role: "user", content: q }]);
    setInput("");
    setLoading(true);
    try {
      const response = await api.chat(q, history);
      const id = nextId++;
      setMessages((ms) => [...ms, { id, role: "assistant", question: q, response }]);
      const first = response.citations[0]?.id || response.sources[0]?.id;
      setActive(first ? { msgId: id, sourceId: first } : null);
    } catch (e) {
      const error = e.message === "network" ? t.errNetwork : `${t.errServer}: ${e.message}`;
      setMessages((ms) => [...ms, { id: nextId++, role: "assistant", error }]);
    } finally {
      setLoading(false);
    }
  };

  const onCite = (msgId, sourceId) => {
    setActive({ msgId, sourceId });
    setShowSourcesMobile(true);
  };

  return (
    <div className="flex h-full flex-col">
      <header className="border-b border-sand-dark bg-ink text-sand">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3">
          <img src="/favicon.svg" alt="" className="h-9 w-9 rounded-lg" />
          <div className="min-w-0 flex-1">
            <h1 className="text-lg font-semibold leading-tight">{t.title}</h1>
            <p className="truncate text-xs text-sand/80">{t.subtitle}</p>
          </div>
          <button type="button" onClick={() => setShowCorpus(true)}
            className="hidden rounded-lg border border-sand/30 px-3 py-1 text-xs hover:bg-sand/10 sm:block">{t.corpus}</button>
          <button type="button" onClick={() => { setMessages([]); setActive(null); }}
            className="rounded-lg border border-sand/30 px-3 py-1 text-xs hover:bg-sand/10">{t.newChat}</button>
          <select value={lang} onChange={(e) => setLang(e.target.value)} aria-label="language"
            className="rounded-lg bg-ink-soft px-2 py-1 text-xs text-sand">
            <option value="ar">العربية</option>
            <option value="fr">Français</option>
            <option value="en">English</option>
          </select>
        </div>
      </header>

      <main className="mx-auto flex min-h-0 w-full max-w-7xl flex-1 flex-col lg:flex-row">
        <section className="flex min-h-0 flex-1 flex-col">
          <div className="flex-1 space-y-4 overflow-y-auto px-4 py-5">
            {messages.length === 0 && <TopicCards t={t} lang={lang} topics={TOPICS} onAsk={ask} />}
            {messages.map((m) => (
              <ChatMessage key={m.id} t={t} msg={m} onCite={onCite} activeSource={active} />
            ))}
            {loading && (
              <div className="flex items-center gap-2 text-sm text-ink-soft" aria-live="polite">
                <span className="h-2 w-2 animate-pulse rounded-full bg-tn" />
                {t.thinking}
              </div>
            )}
            <div ref={endRef} />
          </div>
          <form onSubmit={(e) => { e.preventDefault(); ask(input); }}
            className="border-t border-sand-dark bg-sand px-4 py-3">
            <div className="flex gap-2">
              <textarea value={input} onChange={(e) => setInput(e.target.value)} rows={2} maxLength={1000}
                dir={textDir(input) === "rtl" || (!input && t.dir === "rtl") ? "rtl" : "ltr"}
                onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input); } }}
                placeholder={t.placeholder} aria-label={t.placeholder}
                className="flex-1 resize-none rounded-xl border border-sand-dark bg-white px-3 py-2 focus:border-ink focus:outline-none" />
              <button type="submit" disabled={loading || !input.trim()}
                className="rounded-xl bg-tn px-5 font-medium text-white disabled:opacity-40">{t.send}</button>
            </div>
            <p className="mt-2 text-center text-[11px] text-ink-soft/70">{t.footer}</p>
          </form>
        </section>

        <aside className={`${showSourcesMobile ? "fixed inset-0 z-10 bg-sand" : "hidden"}
            min-h-0 border-sand-dark lg:static lg:block lg:w-[420px] lg:border-s`}>
          <div className="flex items-center justify-between border-b border-sand-dark px-4 py-2">
            <h2 className="font-semibold">{t.sources}</h2>
            <button type="button" onClick={() => setShowSourcesMobile(false)} className="text-sm lg:hidden">{t.close}</button>
          </div>
          <div className="h-[calc(100%-41px)] overflow-y-auto">
            <SourcePanel t={t} response={panelMsg?.response}
              activeId={active && panelMsg && active.msgId === panelMsg.id ? active.sourceId : null}
              onSelect={(sid) => panelMsg && setActive({ msgId: panelMsg.id, sourceId: sid })} />
          </div>
        </aside>
      </main>
      {showCorpus && <CorpusModal t={t} onClose={() => setShowCorpus(false)} />}
    </div>
  );
}
