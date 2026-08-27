import type { DocumentContextSection } from "./DocumentContext";

/// Context Veronica can draw on when answering — one "About you" document
/// (still tagged RESUME server-side, since `veronica::fetch_document_full_text`
/// reads that type unconditionally as full text — see veronica.rs) plus a
/// general documents catch-all covered by RAG retrieval.
export const VERONICA_CONTEXT_SECTIONS: DocumentContextSection[] = [
  { key: "about", documentType: "RESUME", label: "About you" },
  { key: "documents", documentType: "OTHER", label: "Documents" },
];
