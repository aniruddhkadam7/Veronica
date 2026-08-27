//! Tauri commands for Interview Mode: overlay window lifecycle + the
//! ASK AI flow (optional local RAG -> backend -> LLM -> streamed answer).
//!
//! Retrieval's top_k/similarity_threshold/max_context_chars/timeout are all
//! hardware-tier-driven (`hardware::PerformanceManager::effective_config`)
//! rather than hardcoded here — see `hardware::manager` for the tier table
//! and docs/performance-tuning.md for the benchmark evidence behind it.

use tauri::{AppHandle, State};

use crate::rag::RagClient;
use crate::state::AppState;

use super::window::{self, OverlayCaptureStatus};

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
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub job_description: Option<String>,
    #[serde(default = "default_english_level")]
    pub english_level: String,
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

fn default_english_level() -> String {
    "simple".to_string()
}

fn default_humanization() -> String {
    "natural".to_string()
}

impl Default for AskOptions {
    fn default() -> Self {
        Self {
            answer_length: default_answer_length(),
            response_style: default_response_style(),
            role: None,
            job_description: None,
            english_level: default_english_level(),
            humanization: default_humanization(),
            llm_provider: None,
        }
    }
}

// Window creation/show/hide on Windows must happen on the same OS thread that
// owns the window message loop (the main thread). Tauri dispatches
// non-async `#[tauri::command]`s onto its blocking thread pool, NOT the main
// thread, so calling WebviewWindowBuilder::build()/show()/hide() directly
// from here deadlocks: the worker thread blocks waiting for the main-thread
// window/message APIs, while the main thread is itself waiting on IPC. Route
// the actual window work through `run_on_main_thread` and use a channel to
// bring the result back to the (async) command so the IPC call still
// completes normally instead of hanging forever.
#[tauri::command]
pub async fn show_interview_overlay(app: AppHandle) -> Result<OverlayCaptureStatus, String> {
    run_on_main(&app, window::show_overlay_window).await
}

#[tauri::command]
pub async fn hide_interview_overlay(app: AppHandle) -> Result<(), String> {
    run_on_main(&app, window::close_overlay_window).await
}

#[tauri::command]
pub async fn toggle_interview_overlay(app: AppHandle) -> Result<OverlayCaptureStatus, String> {
    run_on_main(&app, window::toggle_overlay_window).await
}

/// Applied immediately when the user flips "Always on top" in the overlay's
/// Settings panel — see `overlaySettings.ts`/`OverlaySettingsPanel.tsx`.
#[tauri::command]
pub async fn set_overlay_always_on_top(app: AppHandle, enabled: bool) -> Result<(), String> {
    run_on_main(&app, move |app| window::set_overlay_always_on_top(app, enabled)).await
}

/// Applied when the user changes "Overlay size" in Settings. `fraction` is
/// the side length as a fraction of the primary monitor's shorter dimension
/// (small=0.45, medium=0.6, large=0.75 — chosen client-side).
#[tauri::command]
pub async fn resize_interview_overlay(app: AppHandle, fraction: f64) -> Result<(), String> {
    run_on_main(&app, move |app| window::resize_overlay(app, fraction)).await
}

async fn run_on_main<T, F>(app: &AppHandle, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce(&AppHandle) -> Result<T, String> + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    let app_for_main = app.clone();
    app.run_on_main_thread(move || {
        let result = f(&app_for_main);
        let _ = tx.send(result);
    })
    .map_err(|e| format!("failed to schedule work on main thread: {e}"))?;

    tauri::async_runtime::spawn_blocking(move || {
        rx.recv()
            .map_err(|e| format!("main-thread task did not respond: {e}"))?
    })
    .await
    .map_err(|e| format!("main-thread task panicked: {e}"))?
}

/// Fetches the full extracted text of the most recently uploaded document of
/// `document_type` (RESUME or JOB_DESCRIPTION), bypassing RAG chunk search
/// entirely. CV and job description are short enough to just read in full —
/// unlike the "Upload documents" catch-all, there is no size reason to chunk
/// and similarity-search them, and doing so only adds a race (the document
/// may not have finished indexing yet) and a chance of missing/skipping
/// content that full-text inclusion can't have. Returns `None` on any
/// failure (RAG unavailable, no matching document, still extracting) —
/// exactly like `retrieval_could_help`'s failures, this must never fail the
/// ask itself, only mean the answer proceeds without that document.
pub(crate) async fn fetch_document_full_text(document_type: &str) -> Option<String> {
    let client = RagClient::new();
    let documents = client.list_documents(None).await.ok()?;
    let latest = documents
        .into_iter()
        .filter(|d| d.document_type == document_type && d.status == "READY")
        .max_by(|a, b| a.updated_at.total_cmp(&b.updated_at))?;
    client.get_document_text(&latest.document_id).await.ok().flatten()
}

// The ASK AI flow itself (retrieval -> AskRequest -> DirectLlmClient::ask_stream
// -> stream) now lives in `crate::veronica::ask_veronica`, which reuses
// `AskOptions`/`fetch_document_full_text`/`retrieval_could_help` from this
// module — see veronica.rs. `ask_interview_question` and its private
// `PriorTurn`/`trim_history`/`MAX_HISTORY_TURNS` (Interview-only duplicates
// of what's now `veronica::PriorTurn` etc.) were removed rather than kept as
// unused dead code once the overlay stopped calling them directly.

/// Whether searching the candidate's own documents could plausibly improve
/// this answer.
///
/// Biased towards retrieving: a false positive costs one fast local search
/// whose empty/irrelevant result is harmless, while a false negative loses
/// real personalization. Only questions that are unambiguously about a
/// concept — a definitional opener with no second-person reference anywhere —
/// skip it.
pub(crate) fn retrieval_could_help(question: &str) -> bool {
    let lowered = question.to_lowercase();

    // Any reference to the candidate makes this potentially personal,
    // whatever else the question looks like ("What is your experience with
    // Kubernetes?" opens definitionally but is entirely about them).
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

    // Purely definitional openers with no personal reference: general
    // knowledge answers these completely, and the CV cannot contribute.
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

/// Result of `start_backend_session`, distinguishing "the backend explicitly
/// said no" (`rejection` set — Start should stop) from every other outcome
/// (not signed in, credential store unavailable, network unreachable,
/// backend down — `rejection` is `None` and local recording should proceed
/// exactly as it always has). This feature has NO business blocking local
/// STT for any reason other than an actual entitlement decision — collapsing
/// every failure into one error type is what previously let an unrelated
/// infrastructure hiccup (e.g. Windows Credential Manager being briefly
/// unavailable) silently prevent recording from ever starting.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendSessionResult {
    /// Present only for a genuine backend rejection (no remaining minutes,
    /// concurrent session limit) — a user-presentable message the frontend
    /// should surface and use to stop Start.
    pub rejection: Option<String>,
}

impl BackendSessionResult {
    fn ok() -> Self {
        Self { rejection: None }
    }

    fn rejected(message: String) -> Self {
        Self { rejection: Some(message) }
    }
}

/// This is a personal build: there is no SaaS backend, no Supabase sign-in,
/// and no entitlement/session-authority system to check in with. These two
/// commands are kept as permanent no-ops (rather than removed) so the
/// frontend call sites around Start/Stop Interview — which call them
/// unconditionally — need no changes.
#[tauri::command]
pub async fn start_backend_session(
    _state: State<'_, AppState>,
    _stt_mode: String,
) -> Result<BackendSessionResult, String> {
    Ok(BackendSessionResult::ok())
}

#[tauri::command]
pub async fn end_backend_session(_state: State<'_, AppState>) -> Result<Option<()>, String> {
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conceptual_questions_skip_retrieval() {
        assert!(!retrieval_could_help("What is RAG?"));
        assert!(!retrieval_could_help("What is Kubernetes?"));
        assert!(!retrieval_could_help("Explain how TCP handles congestion."));
        assert!(!retrieval_could_help("What's the difference between a process and a thread?"));
        assert!(!retrieval_could_help("How does garbage collection work?"));
    }

    #[test]
    fn questions_about_the_candidate_retrieve() {
        assert!(retrieval_could_help("Tell me about your experience with Python."));
        assert!(retrieval_could_help("Have you worked with RAG?"));
        assert!(retrieval_could_help("Walk me through a project you're proud of."));
        assert!(retrieval_could_help("Tell me about a time you disagreed with a teammate."));
    }

    #[test]
    fn a_personal_reference_beats_a_definitional_opener() {
        // Opens like a concept question but is entirely about them — the
        // personal check must win, or "What is your experience with X?" would
        // lose its personalization.
        assert!(retrieval_could_help("What is your experience with Kubernetes?"));
        assert!(retrieval_could_help("How would you describe your testing approach?"));
    }

    #[test]
    fn unclassifiable_questions_default_to_retrieving() {
        // Biased towards retrieval: an irrelevant result is dropped by the
        // similarity threshold anyway, a missed one loses personalization.
        assert!(retrieval_could_help("Tell me about yourself."));
        assert!(retrieval_could_help("Kubernetes."));
    }

    fn turn(q: &str, a: &str) -> PriorTurn {
        PriorTurn {
            question: q.to_string(),
            answer: a.to_string(),
        }
    }

    #[test]
    fn history_keeps_the_most_recent_turns_oldest_first() {
        let turns: Vec<PriorTurn> = (0..MAX_HISTORY_TURNS + 3)
            .map(|i| turn(&format!("q{i}"), &format!("a{i}")))
            .collect();
        let trimmed = trim_history(turns);

        assert_eq!(trimmed.len(), MAX_HISTORY_TURNS);
        // The oldest turns are the ones dropped, and order is preserved.
        assert_eq!(trimmed.first().unwrap().question, "q3");
        assert_eq!(trimmed.last().unwrap().question, format!("q{}", MAX_HISTORY_TURNS + 2));
    }

    #[test]
    fn history_drops_incomplete_turns() {
        // A turn still streaming, or one whose request failed, has no answer.
        // Sending it would fail the backend's min_length validation and take
        // the whole follow-up down with it.
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
    fn history_is_trimmed_of_whitespace() {
        let trimmed = trim_history(vec![turn("  why?  ", "  because.  ")]);
        assert_eq!(trimmed[0].question, "why?");
        assert_eq!(trimmed[0].answer, "because.");
    }

    #[test]
    fn empty_history_is_fine() {
        assert!(trim_history(Vec::new()).is_empty());
    }

    #[test]
    fn ask_options_default_to_natural_default_length() {
        let options = AskOptions::default();
        assert_eq!(options.answer_length, "default");
        assert_eq!(options.response_style, "natural");
        assert!(options.role.is_none());
    }

    #[test]
    fn ask_options_deserialize_from_partial_frontend_payload() {
        let options: AskOptions = serde_json::from_str(r#"{"answerLength":"brief"}"#).unwrap();
        assert_eq!(options.answer_length, "brief");
        assert_eq!(options.response_style, "natural");
    }
}
