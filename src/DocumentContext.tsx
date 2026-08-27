import { useCallback, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { readFile } from "@tauri-apps/plugin-fs";
import type { DocumentType } from "./types";
import { Button } from "./ui";

export interface DocumentContextSection {
  key: string;
  documentType: DocumentType;
  label: string;
  description?: string;
}

export type DocumentContextState = Record<
  string,
  { fileName: string | null; fileBytes: number[] | null; text: string }
>;

const EMPTY_SECTION_STATE = { fileName: null, fileBytes: null, text: "" };

export function emptyDocumentContextState(sections: DocumentContextSection[]): DocumentContextState {
  return Object.fromEntries(sections.map((s) => [s.key, { ...EMPTY_SECTION_STATE }]));
}

/// Fires `upload_document` for every section that has a file or pasted text.
/// Non-blocking by design (callers don't await the RAG pipeline finishing); errors are
/// logged, not surfaced, since a failed upload just means that document
/// won't be searchable — the session still starts.
export function uploadDocumentContext(sections: DocumentContextSection[], state: DocumentContextState) {
  for (const section of sections) {
    const value = state[section.key];
    if (!value) continue;
    if (value.fileBytes && value.fileName) {
      invoke("upload_document", {
        filename: value.fileName,
        bytes: value.fileBytes,
        documentType: section.documentType,
      }).catch((e) => console.error(`Failed to upload ${section.label}:`, e));
    } else if (value.text.trim()) {
      const bytes = Array.from(new TextEncoder().encode(value.text.trim()));
      invoke("upload_document", {
        filename: `${section.key}.txt`,
        bytes,
        documentType: section.documentType,
      }).catch((e) => console.error(`Failed to upload ${section.label}:`, e));
    }
  }
}

interface Props {
  sections: DocumentContextSection[];
  state: DocumentContextState;
  onChange: (state: DocumentContextState) => void;
}

export function DocumentContext({ sections, state, onChange }: Props) {
  const [error, setError] = useState<string | null>(null);

  const pickFile = useCallback(
    async (key: string) => {
      setError(null);
      try {
        const selected = await open({
          multiple: false,
          filters: [{ name: "Documents", extensions: ["pdf", "docx", "txt", "md", "markdown"] }],
        });
        if (!selected || Array.isArray(selected)) return;
        const bytes = await readFile(selected);
        const fileName = selected.split(/[\\/]/).pop() ?? "document";
        onChange({
          ...state,
          [key]: { fileName, fileBytes: Array.from(bytes), text: "" },
        });
      } catch (e) {
        setError(String(e));
      }
    },
    [state, onChange],
  );

  const clearFile = useCallback(
    (key: string) => {
      onChange({ ...state, [key]: { ...EMPTY_SECTION_STATE } });
    },
    [state, onChange],
  );

  const setText = useCallback(
    (key: string, text: string) => {
      onChange({ ...state, [key]: { fileName: null, fileBytes: null, text } });
    },
    [state, onChange],
  );

  return (
    <div className="setup-sections">
      {error && <p className="error">{error}</p>}
      {sections.map((section) => {
        const value = state[section.key] ?? EMPTY_SECTION_STATE;
        return (
          <div className="setup-section" key={section.key}>
            <div className="setup-section-head">
              <span className="setup-section-label">{section.label}</span>
              <span className="setup-section-optional">Optional</span>
            </div>
            {section.description && <p className="setup-subtitle">{section.description}</p>}

            <div className="setup-section-body">
              {value.fileName ? (
                <div className="setup-file-chip">
                  <span className="setup-file-name">{value.fileName}</span>
                  <button className="link-button" onClick={() => clearFile(section.key)}>
                    Remove
                  </button>
                </div>
              ) : (
                <>
                  <Button variant="secondary" size="sm" onClick={() => pickFile(section.key)}>
                    Upload a file
                  </Button>
                  <textarea
                    className="setup-textarea"
                    value={value.text}
                    onChange={(e) => setText(section.key, e.target.value)}
                    placeholder="Or paste text here"
                  />
                </>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
