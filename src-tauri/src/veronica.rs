//! Veronica: the one assistant behind the one overlay. `ask_veronica` is the
//! single entry point — it answers a question OR, when the model's entire
//! response is an `ACTION: <NAME> | <target>` directive (see
//! `personal::prompts::veronica`'s ACTION-TAKING section), runs that action
//! through `crate::actions` (registry safety check -> fastest-method
//! router) and returns its result instead. Both cases stream through the
//! same `veronica:answer-delta`/`veronica:answer-complete` events, so the
//! overlay renders them identically — the user can't tell, from the UI,
//! whether their message was answered or acted on.

use tauri::{AppHandle, Emitter, State};

use crate::actions;
use crate::backend::{AskRequest, AskRetrievedChunk, ConversationTurn};
use crate::rag::{RagClient, RetrievalPlanner};
use crate::state::AppState;

/// Answer-shaping options chosen in the overlay's settings panel. Everything
/// is optional: `Default` reproduces the plain "natural, default length"
/// behavior, so a caller that supplies nothing still gets a good answer.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AskOptions {
    #[serde(default = "default_answer_length")]
    pub answer_length: String,
    #[serde(default = "default_response_style")]
    pub response_style: String,
    #[serde(default = "default_humanization")]
    pub humanization: String,
    /// "openai" | "anthropic" | "gemini" — the header dropdown's chosen model provider.
    /// `None` keeps the server-configured default.
    #[serde(default)]
    pub llm_provider: Option<String>,
}

fn default_answer_length() -> String {
    "default".to_string()
}

fn default_response_style() -> String {
    "natural".to_string()
}

fn default_humanization() -> String {
    "natural".to_string()
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            answer_length: default_answer_length(),
            response_style: default_response_style(),
            humanization: default_humanization(),
            llm_provider: None,
        }
    }
}

/// One prior exchange in this Veronica conversation, as sent by the overlay.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PriorTurn {
    pub question: String,
    pub answer: String,
}

/// How many prior turns to forward. The overlay keeps the whole conversation
/// on screen, but only the recent tail is worth paying for in the prompt:
/// every turn adds tokens to time-to-first-token, which is the number the
/// user feels. Six turns comfortably covers "why did you choose that?"
/// style follow-ups, which almost always refer to the last exchange or two.
const MAX_HISTORY_TURNS: usize = 6;

/// Normalizes the overlay's conversation history into what the backend
/// wants: the most recent complete turns, still oldest-first. Incomplete
/// turns are dropped rather than sent as empty strings — the backend schema
/// requires non-empty text on both sides, so a turn whose answer failed or
/// is still streaming would be rejected and take the whole request down
/// with it.
fn trim_history(turns: Vec<PriorTurn>) -> Vec<ConversationTurn> {
    let mut history: Vec<ConversationTurn> = turns
        .into_iter()
        .filter(|t| !t.question.trim().is_empty() && !t.answer.trim().is_empty())
        .map(|t| ConversationTurn { question: t.question.trim().to_string(), answer: t.answer.trim().to_string() })
        .collect();
    if history.len() > MAX_HISTORY_TURNS {
        history.drain(..history.len() - MAX_HISTORY_TURNS);
    }
    history
}

/// Fetches the full extracted text of the most recently uploaded document of
/// `document_type`, bypassing RAG chunk search entirely. Short documents
/// (a resume, notes) are cheap enough to just read in full — unlike the
/// general "Upload documents" catch-all, there is no size reason to chunk
/// and similarity-search them, and doing so only adds a race (the document
/// may not have finished indexing yet) and a chance of missing/skipping
/// content that full-text inclusion can't have. Returns `None` on any
/// failure (RAG unavailable, no matching document, still extracting) — this
/// must never fail the ask itself, only mean the answer proceeds without it.
async fn fetch_document_full_text(document_type: &str) -> Option<String> {
    let client = RagClient::new();
    let documents = client.list_documents(None).await.ok()?;
    let latest = documents
        .into_iter()
        .filter(|d| d.document_type == document_type && d.status == "READY")
        .max_by(|a, b| a.updated_at.total_cmp(&b.updated_at))?;
    client.get_document_text(&latest.document_id).await.ok().flatten()
}

/// Whether searching the user's own attached documents could plausibly
/// improve this answer.
///
/// Biased towards retrieving: a false positive costs one fast local search
/// whose empty/irrelevant result is harmless, while a false negative loses
/// real personalization. Only questions that are unambiguously about a
/// concept — a definitional opener with no second-person reference anywhere
/// — skip it.
fn retrieval_could_help(question: &str) -> bool {
    let lowered = question.to_lowercase();

    const PERSONAL_MARKERS: [&str; 12] = [
        "your ",
        "you ",
        "you'",
        "yourself",
        "have you",
        "did you",
        "tell me about a time",
        "walk me through",
        "worked on",
        "experience with",
        "your experience",
        "a project where",
    ];
    if PERSONAL_MARKERS.iter().any(|m| lowered.contains(m)) {
        return true;
    }

    const CONCEPTUAL_OPENERS: [&str; 12] = [
        "what is",
        "what are",
        "what's",
        "explain",
        "define",
        "how does",
        "how do",
        "how would",
        "why is",
        "why do",
        "difference between",
        "when should",
    ];
    let opener = lowered.trim_start_matches(|c: char| !c.is_alphanumeric());
    if CONCEPTUAL_OPENERS.iter().any(|o| opener.starts_with(o)) {
        return false;
    }

    true
}

/// One question, one answer:
///
///     question -> (retrieval, only when it could help) -> ONE LLM call -> stream
///
/// The uploaded resume/notes document, if any, is fetched as full text (see
/// `fetch_document_full_text`) and sent unconditionally on every question —
/// not gated behind `retrieval_could_help`, since reading a short document
/// costs nothing worth gating. `retrieved_context` (RAG) still covers the
/// general "Upload documents" catch-all category.
///
/// Streams back as `veronica:answer-delta` events, finishing with
/// `veronica:answer-complete`. If the model's entire response turns out to
/// be an `ACTION: <NAME> | <target>` directive (see
/// `personal::prompts::veronica`), that line is never shown to the user —
/// `run_action` replaces it with the actual result of running the action
/// (or a refusal) before the answer-complete event fires. A normal answer
/// never matches that shape and reaches the user exactly as streamed.
#[tauri::command]
pub async fn ask_veronica(
    app: AppHandle,
    _state: State<'_, AppState>,
    question: String,
    options: Option<AskOptions>,
    history: Option<Vec<PriorTurn>>,
) -> Result<String, String> {
    use crate::hardware::telemetry::{finish, FirstTokenTracker, PipelineStage, Stopwatch};

    let question_to_answer = Stopwatch::start();

    let trimmed = question.trim();
    if trimmed.is_empty() {
        return Err("no question text to send".into());
    }
    let options = options.unwrap_or_default();
    let history = trim_history(history.unwrap_or_default());

    let resume_fetch = fetch_document_full_text("RESUME");

    let retrieved = if retrieval_could_help(trimmed) {
        let cfg = crate::hardware::effective_config_checked(&app);
        let planner = RetrievalPlanner::new()
            .with_config(cfg.rag_top_k, cfg.rag_similarity_threshold, cfg.rag_max_context_chars)
            .with_timeout(std::time::Duration::from_millis(cfg.rag_retrieval_timeout_ms));
        let retrieval_timer = Stopwatch::start();
        let results = planner.plan_for_question(trimmed).await;
        finish(retrieval_timer, PipelineStage::RagRetrieval, &crate::hardware::perf_context(&app));
        results
    } else {
        log::debug!("Veronica: skipping retrieval for conceptual question");
        Vec::new()
    };

    let candidate_context = resume_fetch.await;

    let request = AskRequest {
        question: trimmed.to_string(),
        conversation_history: history,
        retrieved_context: retrieved
            .into_iter()
            .filter(|r| r.metadata.document_type != "RESUME")
            .map(|r| AskRetrievedChunk {
                text: r.text,
                source_filename: r.metadata.filename,
                document_type: r.metadata.document_type,
                score: r.score,
            })
            .collect(),
        candidate_context,
        answer_length: options.answer_length,
        response_style: options.response_style,
        humanization: options.humanization,
        llm_provider: options.llm_provider,
    };

    let app_for_events = app.clone();
    let llm_timer = Stopwatch::start();
    let first_token = FirstTokenTracker::new();
    let first_token_recorder = first_token.recorder();
    let on_delta = move |delta: &str| {
        first_token_recorder.mark();
        let _ = app_for_events.emit("veronica:answer-delta", delta);
    };
    let raw_answer = crate::personal::DirectLlmClient::new(request.llm_provider.as_deref())?
        .ask_stream(&request, on_delta)
        .await?;

    let ctx = crate::hardware::perf_context(&app);
    if let Some(ms) = first_token.elapsed_ms() {
        crate::hardware::telemetry::log_stage_ms(PipelineStage::LlmFirstToken, ms, &ctx);
    }
    finish(llm_timer, PipelineStage::LlmTotal, &ctx);

    // The model's entire response is checked against the ACTION line shape
    // (not just its first line) — a normal answer never matches this, so
    // this is a fast no-op for every ordinary question. Only when it
    // matches do we run the action and swap in its result; the raw
    // "ACTION: ..." line is never shown to the user.
    let answer = match actions::parse_action_line(&raw_answer) {
        Some(intent) => actions::execute(intent).await,
        None => raw_answer,
    };

    finish(question_to_answer, PipelineStage::QuestionToAnswer, &crate::hardware::perf_context(&app));
    let _ = app.emit("veronica:answer-complete", &answer);
    Ok(answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(q: &str, a: &str) -> PriorTurn {
        PriorTurn { question: q.to_string(), answer: a.to_string() }
    }

    #[test]
    fn history_keeps_the_most_recent_turns_oldest_first() {
        let turns: Vec<PriorTurn> = (0..MAX_HISTORY_TURNS + 3)
            .map(|i| turn(&format!("q{i}"), &format!("a{i}")))
            .collect();
        let trimmed = trim_history(turns);

        assert_eq!(trimmed.len(), MAX_HISTORY_TURNS);
        assert_eq!(trimmed.first().unwrap().question, "q3");
        assert_eq!(trimmed.last().unwrap().question, format!("q{}", MAX_HISTORY_TURNS + 2));
    }

    #[test]
    fn history_drops_incomplete_turns() {
        let trimmed = trim_history(vec![
            turn("answered", "yes"),
            turn("still streaming", ""),
            turn("", "orphan answer"),
            turn("  ", "   "),
        ]);
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0].question, "answered");
    }

    #[test]
    fn empty_history_is_fine() {
        assert!(trim_history(Vec::new()).is_empty());
    }

    #[test]
    fn conceptual_questions_skip_retrieval() {
        assert!(!retrieval_could_help("What is RAG?"));
        assert!(!retrieval_could_help("How does garbage collection work?"));
    }

    #[test]
    fn personal_questions_retrieve() {
        assert!(retrieval_could_help("Tell me about your experience with Python."));
        assert!(retrieval_could_help("What is your experience with Kubernetes?"));
    }

    #[test]
    fn ask_options_default_to_natural_default_length() {
        let options = AskOptions::default();
        assert_eq!(options.answer_length, "default");
        assert_eq!(options.response_style, "natural");
    }
}
