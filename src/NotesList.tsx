import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "./ui";

export interface Note {
  id: string;
  title: string;
  body: string;
  project: string | null;
  tags: string[];
  created_at_ms: number;
  updated_at_ms: number;
  linked_note_ids: string[];
}

function formatWhen(ms: number) {
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function preview(body: string): string {
  const trimmed = body.trim();
  if (!trimmed) return "Empty note";
  return trimmed.length > 90 ? `${trimmed.slice(0, 90)}…` : trimmed;
}

interface Props {
  refreshKey: number;
  selectedId: string | null;
  onSelect: (note: Note) => void;
  onCreate: () => void;
}

const SEARCH_DEBOUNCE_MS = 250;

export function NotesList({ refreshKey, selectedId, onSelect, onCreate }: Props) {
  const [notes, setNotes] = useState<Note[]>([]);
  const [query, setQuery] = useState("");

  const refresh = useCallback(async (q: string) => {
    try {
      const list = await invoke<Note[]>("search_notes", { query: q });
      setNotes(list);
    } catch {
      setNotes([]);
    }
  }, []);

  useEffect(() => {
    const timer = setTimeout(() => refresh(query), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [query, refresh, refreshKey]);

  return (
    <div className="notes-list">
      <div className="notes-list-head">
        <input
          className="search-input"
          type="text"
          placeholder="Search notes"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <Button variant="primary" size="sm" onClick={onCreate}>
          + New
        </Button>
      </div>

      <div className="notes-list-items">
        {notes.length === 0 ? (
          <p className="history-empty">
            {query.trim() ? "No notes match your search." : "No notes yet — create your first one."}
          </p>
        ) : (
          notes.map((note) => (
            <button
              key={note.id}
              className={["notes-list-item", note.id === selectedId ? "active" : ""].filter(Boolean).join(" ")}
              onClick={() => onSelect(note)}
            >
              <div className="notes-list-item-top">
                <span className="notes-list-item-title">{note.title || "Untitled note"}</span>
                <span className="notes-list-item-when">{formatWhen(note.updated_at_ms)}</span>
              </div>
              {note.project && <span className="document-type-tag">{note.project}</span>}
              <p className="notes-list-item-preview">{preview(note.body)}</p>
            </button>
          ))
        )}
      </div>
    </div>
  );
}
