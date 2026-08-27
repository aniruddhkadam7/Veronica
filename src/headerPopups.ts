import type { DocumentContextSection } from "./DocumentContext";

export const INTERVIEW_CONTEXT_SECTIONS: DocumentContextSection[] = [
  { key: "resume", documentType: "RESUME", label: "Resume / CV" },
  { key: "job_description", documentType: "JOB_DESCRIPTION", label: "Job Description" },
  { key: "context", documentType: "INTERVIEW_PREPARATION", label: "Upload documents" },
];

export const MEETING_CONTEXT_SECTIONS: DocumentContextSection[] = [
  { key: "agenda", documentType: "PROJECT", label: "Agenda" },
  { key: "reference", documentType: "COMPANY", label: "Reference material" },
  { key: "documents", documentType: "OTHER", label: "Meeting documents" },
];
