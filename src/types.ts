export type AudioSource = "SYSTEM_AUDIO" | "MICROPHONE";

export type RecordingState = "IDLE" | "STARTING" | "RECORDING" | "PAUSED" | "STOPPING" | "STOPPED";

export interface AudioLevelEvent {
  source: AudioSource;
  rms_level: number;
}

export interface TranscriptSegment {
  id: string;
  timestamp: number;
  source: AudioSource;
  partial_text: string | null;
  final_text: string | null;
  start_time: number | null;
  end_time: number | null;
}

export interface InterviewSession {
  id: string;
  started_at_ms: number;
  ended_at_ms: number | null;
  segments: TranscriptSegment[];
  paused_ms_total: number;
}

export interface RecordingStateEvent {
  state: RecordingState;
}

export type BackendStatus = "UNKNOWN" | "CONNECTING" | "CONNECTED" | "OFFLINE";

export type AnalysisPhase = "IDLE" | "ANALYZING" | "COMPLETED" | "FAILED";

export interface RetrievedSourceRef {
  filename: string;
  document_type: string;
  score: number;
  text: string;
}

export interface QuestionAnalysis {
  question_id: string;
  question: string;
  candidate_answer: string;
  assessment: string;
  score: number;
  strengths: string[];
  issues: string[];
  improved_answer: string;
  retrieved_sources: RetrievedSourceRef[];
  failed: boolean;
  error_message: string | null;
}

export interface OverallInterviewAnalysis {
  session_id: string;
  status: "completed" | "failed" | "partial";
  overall_score: number;
  technical_score: number;
  communication_score: number;
  practical_experience_score: number;
  confidence_score: number;
  summary: string;
  strengths: string[];
  weaknesses: string[];
  recommendations: string[];
  questions: QuestionAnalysis[];
  disclaimer: string;
  message: string;
}

export interface AnalysisProgressEvent {
  stage: string;
  detail: string;
  question_analysis: QuestionAnalysis | null;
  result: OverallInterviewAnalysis | null;
}

export type DocumentType =
  | "RESUME"
  | "JOB_DESCRIPTION"
  | "PROJECT"
  | "COMPANY"
  | "INTERVIEW_PREPARATION"
  | "TECHNICAL_NOTES"
  | "OTHER";

export type DocumentStatus =
  | "UPLOADING"
  | "EXTRACTING"
  | "CLEANING"
  | "CHUNKING"
  | "EMBEDDING"
  | "INDEXING"
  | "READY"
  | "ERROR";

export interface DocumentMetadata {
  document_id: string;
  filename: string;
  document_type: string;
  file_size: number;
  content_hash: string;
  created_at: number;
  updated_at: number;
  status: string;
  chunk_count: number;
  error_message: string | null;
}

export interface KnowledgeBaseStatus {
  document_count: number;
  chunk_count: number;
  status: string;
}

export interface SearchResultItem {
  chunk_id: string;
  score: number;
  text: string;
  metadata: {
    filename: string;
    document_type: string;
  };
}

export interface SearchResponse {
  results: SearchResultItem[];
  latency_ms: number;
}
