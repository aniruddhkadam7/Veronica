//! Veronica's action-taking system: a spoken request (already recognized as
//! an action by the overlay's wake-phrase check — see InterviewOverlay.tsx's
//! `tryVeronicaAction`) is classified into a fixed intent vocabulary by the
//! existing LLM, checked against a hardcoded safety table, and — only for
//! the safe set — executed via the fastest available native mechanism.
//!
//! Pipeline: Query Understanding (personal::prompts::intent) -> Intent ->
//! Safety Check (registry::risk_level) -> Fastest-Method Router
//! (router::execute) -> Execute (native).
//!
//! The LLM never executes anything itself — it only ever returns one of six
//! fixed intent names (personal/prompts/intent.rs's schema has no slot for
//! an arbitrary command), and this module's `Intent` enum has no variant
//! that could represent a destructive action (delete, format, registry/
//! security change, credential access, shutdown, arbitrary shell execution,
//! bulk destructive ops, or a consequential external send) — so there is no
//! code path, not even a guarded one, that could run any of those from a
//! voice command. Anything the classifier can't confidently map to the safe
//! six comes back as `Intent::Unknown`, which is refused before the router
//! is ever reached.

mod native;
mod registry;
mod router;

use tauri::AppHandle;

use crate::personal::prompts::intent::{self, ParsedIntent};
use crate::personal::DirectLlmClient;

pub use registry::RiskLevel;

/// The ONLY vocabulary the router/executor ever see. Mirrors
/// `personal::prompts::intent::ParsedIntent` one-to-one — kept as a
/// separate type (rather than reusing `ParsedIntent` directly in the
/// router) so the prompt-parsing module and the execution module don't need
/// to depend on each other's internals, but the shape is deliberately
/// identical: adding a new action means adding a variant to BOTH, which
/// keeps `registry::risk_level` and `intent::parse_intent` from silently
/// drifting apart on what's representable at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intent {
    OpenApp(String),
    OpenFile(String),
    OpenFolder(String),
    OpenUrl(String),
    QuerySystemInfo(String),
    Unknown,
}

impl From<ParsedIntent> for Intent {
    fn from(parsed: ParsedIntent) -> Self {
        match parsed {
            ParsedIntent::OpenApp(t) => Intent::OpenApp(t),
            ParsedIntent::OpenFile(t) => Intent::OpenFile(t),
            ParsedIntent::OpenFolder(t) => Intent::OpenFolder(t),
            ParsedIntent::OpenUrl(t) => Intent::OpenUrl(t),
            ParsedIntent::QuerySystemInfo(t) => Intent::QuerySystemInfo(t),
            ParsedIntent::Unknown => Intent::Unknown,
        }
    }
}

/// The one command the overlay calls for a recognized "Veronica, ..." voice
/// action. `utterance` is the text AFTER the wake phrase has already been
/// stripped client-side. Never fails outright for "didn't understand" or
/// "not allowed" cases — those come back as `Ok(refusal message)`, the same
/// way a normal Ask AI answer would, so the overlay can render them as a
/// plain turn. `Err` is reserved for real failures (no API key configured,
/// provider request failed) — the overlay's existing error handling already
/// knows how to surface those.
#[tauri::command]
pub async fn run_veronica_action(_app: AppHandle, utterance: String) -> Result<String, String> {
    let trimmed = utterance.trim();
    if trimmed.is_empty() {
        return Ok("I didn't catch an action in that.".to_string());
    }

    let (system_prompt, user_prompt) = intent::build_intent_prompt(trimmed);
    let raw = DirectLlmClient::new(None)?.classify(&system_prompt, &user_prompt).await?;
    let intent: Intent = intent::parse_intent(&raw).into();

    if matches!(intent, Intent::Unknown) {
        return Ok("I didn't recognize an action in that.".to_string());
    }

    match registry::risk_level(&intent) {
        RiskLevel::Blocked => Ok(registry::refusal_message(&intent)),
        RiskLevel::Safe => router::execute(&intent).await,
    }
}
