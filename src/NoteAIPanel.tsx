import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "./ui";
import type { Note } from "./NotesList";

interface NoteSummary {
  summary: string;
  tasks: string[];
  decisions: string[];
  key_points: string[];
  message: string;
}

interface NotesAskResponse {
  answer: string;
  latency_ms: number;
}

interface Props {
  note: Note;
}

export function NoteAIPanel({ note }: Props) {
  const [open, setOpen] = useState(false);
  const [summarizing, setSummarizing] = useState(false);
  const [summary, setSummary] = useState<NoteSummary | null>(null);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const summarize = useCallback(async () => {
    setError(null);
    setSummarizing(true);
    try {
      const result = await invoke<NoteSummary>("summarize_note", { title: note.title, body: note.body });
      setSummary(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setSummarizing(false);
    }
  }, [note.title, note.body]);

  const ask = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed) return;
    setError(null);
    setAsking(true);
    setAnswer(null);
    try {
      const result = await invoke<NotesAskResponse>("ask_about_notes", {
        question: trimmed,
        notes: [{ title: note.title, body: note.body }],
      });
      setAnswer(result.answer);
    } catch (e) {
      setError(String(e));
    } finally {
      setAsking(false);
    }
  }, [question, note.title, note.body]);

  if (!open) {
    return (
      <button className="notes-ai-toggle" onClick={() => setOpen(true)}>
        ✦ Ask AI about this note
      </button>
    );
  }

  return (
    <div className="notes-ai-panel">
      <div className="notes-ai-panel-head">
        <span className="setup-section-label">AI</span>
        <button className="link-button" onClick={() => setOpen(false)}>
          Close
        </button>
      </div>

      {error && <p className="error">{error}</p>}

      <div className="notes-ai-actions">
        <Button variant="secondary" size="sm" onClick={summarize} disabled={summarizing || !note.body.trim()}>
          {summarizing ? "Summarizing…" : "Summarize"}
        </Button>
      </div>

      {summary && (
        <div className="notes-ai-summary">
          {summary.message && <p className="hint small">{summary.message}</p>}
          {summary.summary && <p className="notes-ai-summary-text">{summary.summary}</p>}
          {summary.key_points.length > 0 && (
            <div className="notes-ai-summary-group">
              <span className="setup-focus-label">Key points</span>
              <ul className="question-list">
                {summary.key_points.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>
          )}
          {summary.tasks.length > 0 && (
            <div className="notes-ai-summary-group">
              <span className="setup-focus-label">Tasks</span>
              <ul className="question-list">
                {summary.tasks.map((t, i) => (
                  <li key={i}>{t}</li>
                ))}
              </ul>
            </div>
          )}
          {summary.decisions.length > 0 && (
            <div className="notes-ai-summary-group">
              <span className="setup-focus-label">Decisions</span>
              <ul className="question-list">
                {summary.decisions.map((d, i) => (
                  <li key={i}>{d}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="notes-ai-ask">
        <textarea
          className="setup-textarea"
          placeholder="Ask a question about this note…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={2}
        />
        <Button variant="primary" size="sm" onClick={ask} disabled={asking || !question.trim()}>
          {asking ? "Asking…" : "Ask"}
        </Button>
      </div>

      {answer && <p className="notes-ai-answer">{answer}</p>}
    </div>
  );
}
