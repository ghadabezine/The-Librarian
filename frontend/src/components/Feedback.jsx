import { useState } from "react";
import { api } from "../api.js";

export default function Feedback({ t, question, response }) {
  const [rating, setRating] = useState(null);
  const [issue, setIssue] = useState("");
  const [comment, setComment] = useState("");
  const [state, setState] = useState("idle"); // idle | sent | error

  const send = async (r, extra = {}) => {
    try {
      await api.feedback({
        question,
        answer: response.answer,
        rating: r,
        cited_chunk_ids: response.citations.map((c) => c.chunk_id),
        ...extra,
      });
      setState("sent");
    } catch {
      setState("error");
    }
  };

  if (state === "sent") return <p className="mt-2 text-xs text-ok">{t.thanks}</p>;

  return (
    <div className="mt-3 border-t border-sand-dark pt-2 text-xs text-ink-soft">
      <div className="flex items-center gap-2">
        <span>{t.helpful}</span>
        <button type="button" aria-label="helpful" onClick={() => { setRating("up"); send("up"); }}
          className="rounded px-2 py-0.5 hover:bg-sand-dark">👍</button>
        <button type="button" aria-label="not helpful" onClick={() => setRating("down")}
          className={`rounded px-2 py-0.5 hover:bg-sand-dark ${rating === "down" ? "bg-sand-dark" : ""}`}>👎</button>
      </div>
      {rating === "down" && (
        <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-center">
          <select value={issue} onChange={(e) => setIssue(e.target.value)}
            className="rounded border border-sand-dark bg-white px-2 py-1" aria-label={t.issue}>
            <option value="">{t.issue}</option>
            {Object.entries(t.issues).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
          <input value={comment} onChange={(e) => setComment(e.target.value)} placeholder={t.comment}
            maxLength={2000} className="flex-1 rounded border border-sand-dark bg-white px-2 py-1" />
          <button type="button" onClick={() => send("down", { issue: issue || null, comment: comment || null })}
            className="rounded bg-ink px-3 py-1 text-sand">{t.submit}</button>
        </div>
      )}
      {state === "error" && <p className="mt-1 text-tn">{t.errServer}</p>}
    </div>
  );
}
