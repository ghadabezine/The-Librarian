import { useEffect, useState } from "react";
import { api } from "../api.js";

export default function CorpusModal({ t, onClose }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    api.documents().then(setData).catch((e) => setError(e.message));
  }, []);
  return (
    <div className="fixed inset-0 z-20 flex items-center justify-center bg-ink/40 p-4" onClick={onClose}>
      <div role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}
        className="max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-2xl bg-sand p-5 shadow-xl">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-lg font-semibold">{t.corpusTitle}</h2>
          <button type="button" onClick={onClose} className="rounded px-2 py-1 hover:bg-sand-dark">{t.close}</button>
        </div>
        {error && <p className="text-tn">{error}</p>}
        {!data && !error && <p>…</p>}
        {data && (
          <>
            <p className="mb-3 text-sm text-ink-soft">
              {data.total_pages} {t.pages} · {data.total_chunks} chunks · {data.embedding_model}
            </p>
            <ul className="space-y-2">
              {data.documents.map((d) => (
                <li key={d.doc_id} className="rounded-lg border border-sand-dark bg-white p-3 text-sm">
                  <p dir="rtl" className="font-ar font-semibold">{d.title}</p>
                  {d.title_fr && <p className="text-xs text-ink-soft">{d.title_fr}</p>}
                  <p className="mt-1 text-xs text-ink-soft">
                    {d.pages} {t.pages} · {d.articles} {t.articles} ·{" "}
                    <span className={d.legal_status === "historical" ? "text-warn" : "text-ok"}>
                      {d.legal_status === "historical" ? t.historical : t.current}
                    </span>
                    {d.source_url && <> · <a className="underline" href={d.source_url} target="_blank" rel="noreferrer">PDF ↗</a></>}
                  </p>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
