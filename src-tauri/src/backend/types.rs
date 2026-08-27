//! Wire types for the `POST /api/v1/interviews/analyze` (and
//! `/analyze/stream`) request/response, matching
//! `apps/backend/app/schemas/interview.py` and
//! `apps/backend/app/schemas/analysis.py` field-for-field.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WireAudioSource {
    SystemAudio,
    Microphone,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireTranscriptSegment {
    /// MM:SS or HH:MM:SS, matching the backend's validated timestamp format.
    pub timestamp: String,
    pub source: WireAudioSource,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireTranscript {
    pub duration_seconds: u64,
    pub segments: Vec<WireTranscriptSegment>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct WireCandidateContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume: Option<String>,
    #[serde(default)]
    pub projects: Vec<String>,
}

/// The minimum retrieved text needed for one question — never the source
/// document, never an embedding vector. Matches
/// `app/schemas/interview.py::RetrievedChunk`.
#[derive(Debug, Clone, Serialize)]
pub struct WireRetrievedChunk {
    pub text: String,
    pub source_filename: String,
    pub document_type: String,
    pub score: f64,
}

/// One interviewer question + candidate answer + the RAG context retrieved
/// specifically for that question. Matches
/// `app/schemas/interview.py::QuestionAnswer`.
#[derive(Debug, Clone, Serialize)]
pub struct WireQuestionAnswer {
    pub question_id: String,
    pub question: String,
    pub candidate_answer: String,
    pub timestamp: String,
    pub retrieved_context: Vec<WireRetrievedChunk>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InterviewAnalysisRequest {
    pub session_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub company: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_context: Option<WireCandidateContext>,
    pub transcript: WireTranscript,
    pub question_answers: Vec<WireQuestionAnswer>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RetrievedSourceRef {
    pub filename: String,
    pub document_type: String,
    pub score: f64,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QuestionAnalysis {
    pub question_id: String,
    pub question: String,
    pub candidate_answer: String,
    pub assessment: String,
    pub score: i32,
    pub strengths: Vec<String>,
    pub issues: Vec<String>,
    pub improved_answer: String,
    pub retrieved_sources: Vec<RetrievedSourceRef>,
    pub failed: bool,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OverallInterviewAnalysis {
    pub session_id: String,
    pub status: String,
    pub overall_score: i32,
    pub technical_score: i32,
    pub communication_score: i32,
    pub practical_experience_score: i32,
    pub confidence_score: i32,
    pub summary: String,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
    pub recommendations: Vec<String>,
    pub questions: Vec<QuestionAnalysis>,
    pub disclaimer: String,
    pub message: String,
}

/// One `data:` payload from the SSE stream — see
/// `apps/backend/app/services/analysis_service.py::AnalysisProgressEvent`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnalysisProgressEvent {
    pub stage: String,
    pub detail: String,
    #[serde(default)]
    pub question_analysis: Option<QuestionAnalysis>,
    #[serde(default)]
    pub result: Option<OverallInterviewAnalysis>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Wire types for the lightweight `POST /api/v1/ask` (and `/ask/stream`)
/// endpoint used by Interview Mode — matches `apps/backend/app/schemas/ask.py`.
/// Deliberately much smaller than the full analysis request: one question,
/// its already-locally-retrieved RAG context, and optional free-text
/// candidate context (no transcript, no question history, no rubric).
#[derive(Debug, Clone, Serialize)]
pub struct AskRetrievedChunk {
    pub text: String,
    pub source_filename: String,
    pub document_type: String,
    pub score: f64,
}

/// One completed exchange earlier in the current Interview Mode session.
/// Replayed to the model so follow-ups ("why did you choose that?") resolve
/// against what was already discussed.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationTurn {
    pub question: String,
    pub answer: String,
}

/// Every context field here is optional and purely additive — Veronica
/// answers the question with or without any of them. `retrieved_context`
/// being empty is a completely normal case, not a degraded one.
#[derive(Debug, Clone, Serialize)]
pub struct AskRequest {
    pub question: String,
    /// Prior turns of the current conversation, oldest first, excluding the
    /// question being asked now. Empty for the first question.
    pub conversation_history: Vec<ConversationTurn>,
    pub retrieved_context: Vec<AskRetrievedChunk>,
    /// Text from any documents the user has attached (see
    /// `DocumentContext.tsx`) — generic supporting material, not tied to any
    /// particular kind of document.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_context: Option<String>,
    /// "brief" | "default" | "detailed" — validated backend-side.
    pub answer_length: String,
    /// "natural" | "technical" | "concise" — validated backend-side.
    pub response_style: String,
    /// "natural" | "conversational" | "formal" — validated backend-side.
    pub humanization: String,
    /// "openai" | "anthropic" | "gemini" — the header dropdown's chosen model provider.
    /// `None` keeps the server-configured LLM_PROVIDER default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub llm_provider: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AskResponse {
    pub answer: String,
    pub latency_ms: f64,
}

/// Wire types for `POST /api/v1/setup/analyze` — matches
/// `apps/backend/app/schemas/setup_analysis.py`. Runs on the New Interview
/// setup page, before any interview session exists.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SetupAnalysisRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_description_text: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct SetupAnalysisResponse {
    #[serde(default)]
    pub job_title: Option<String>,
    #[serde(default)]
    pub company: Option<String>,
    #[serde(default)]
    pub seniority: Option<String>,
    #[serde(default)]
    pub employment_type: Option<String>,
    #[serde(default)]
    pub key_responsibilities: Vec<String>,
    #[serde(default)]
    pub required_skills: Vec<String>,
    #[serde(default)]
    pub technologies: Vec<String>,
    #[serde(default)]
    pub focus_areas: Vec<String>,
    #[serde(default)]
    pub candidate_highlights: Vec<String>,
}

/// Wire types for Notes Mode — matches `apps/backend/app/schemas/notes.py`.
#[derive(Debug, Clone, Serialize)]
pub struct NotesSummaryRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub body: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct NoteSummary {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub tasks: Vec<String>,
    #[serde(default)]
    pub decisions: Vec<String>,
    #[serde(default)]
    pub key_points: Vec<String>,
    #[serde(default)]
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NoteContext {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub body: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NotesAskRequest {
    pub question: String,
    pub notes: Vec<NoteContext>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NotesAskResponse {
    pub answer: String,
    pub latency_ms: f64,
}
