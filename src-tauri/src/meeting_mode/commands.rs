//! Tauri commands for Meeting Mode: overlay window lifecycle, the live
//! "ask" flow, live tracking of key points/decisions/action items, and
//! ending a meeting (summary + archive to history).
//!
//! Audio capture/transcript itself reuses the exact same commands Interview
//! Mode uses (`start_system_audio_capture`, `stop_audio_capture`,
//! `transcript:update` events) — SYSTEM_AUDIO is the other participants,
//! MICROPHONE is the user. Nothing meeting-specific needed duplicating there.

use tauri::{AppHandle, State};

use crate::backend::{MeetingSummaryRequest, MeetingTurnIn};
use crate::overlay_window;
use crate::state::{AppState, TrackedItem};

use super::MEETING_OVERLAY_LABEL;

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MeetingAskOptions {
    #[serde(default = "default_answer_length")]
    pub answer_length: String,
    #[serde(default = "default_response_style")]
    pub response_style: String,
    #[serde(default = "default_humanization")]
    pub humanization: String,
    #[serde(default)]
    pub meeting_title: Option<String>,
    #[serde(default)]
    pub participants: Option<String>,
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

impl Default for MeetingAskOptions {
    fn default() -> Self {
        Self {
            answer_length: default_answer_length(),
            response_style: default_response_style(),
            humanization: default_humanization(),
            meeting_title: None,
            participants: None,
            llm_provider: None,
        }
    }
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

#[tauri::command]
pub async fn show_meeting_overlay(app: AppHandle) -> Result<overlay_window::OverlayCaptureStatus, String> {
    run_on_main(&app, |app| {
        overlay_window::show_overlay_window(app, MEETING_OVERLAY_LABEL, "Smallbird Meeting — Overlay")
    })
    .await
}

#[tauri::command]
pub async fn hide_meeting_overlay(app: AppHandle) -> Result<(), String> {
    log::info!("hide_meeting_overlay: called");
    let result = run_on_main(&app, |app| overlay_window::close_overlay_window(app, MEETING_OVERLAY_LABEL)).await;
    log::info!("hide_meeting_overlay: run_on_main returned {result:?}");
    result
}

#[tauri::command]
pub async fn toggle_meeting_overlay(app: AppHandle) -> Result<overlay_window::OverlayCaptureStatus, String> {
    run_on_main(&app, |app| {
        overlay_window::toggle_overlay_window(app, MEETING_OVERLAY_LABEL, "Smallbird Meeting — Overlay")
    })
    .await
}

/// Applied immediately when the user flips "Always on top" in the overlay's
/// Settings panel — mirrors `interview_mode::commands::set_overlay_always_on_top`.
#[tauri::command]
pub async fn set_meeting_overlay_always_on_top(app: AppHandle, enabled: bool) -> Result<(), String> {
    run_on_main(&app, move |app| {
        overlay_window::set_overlay_always_on_top(app, MEETING_OVERLAY_LABEL, enabled)
    })
    .await
}

/// Applied when the user changes "Overlay size" in Settings. `fraction` is
/// the side length as a fraction of the primary monitor's shorter dimension
/// (small=0.45, medium=0.6, large=0.75 — chosen client-side).
#[tauri::command]
pub async fn resize_meeting_overlay(app: AppHandle, fraction: f64) -> Result<(), String> {
    run_on_main(&app, move |app| {
        overlay_window::resize_overlay(app, MEETING_OVERLAY_LABEL, fraction)
    })
    .await
}

// The live "quick answer" flow itself (retrieval -> MeetingAskRequest ->
// DirectLlmClient::meeting_ask_stream -> stream) now lives in
// `crate::veronica::ask_veronica`, which reuses `MeetingAskOptions` from this
// module — see veronica.rs. `ask_meeting_question` and its private
// `PriorTurn`/`trim_history` (Meeting-only duplicates of what's now
// `veronica::PriorTurn` etc.) were removed rather than kept as unused dead
// code once the overlay stopped calling them directly.

/// Tracked-item kind for `track_meeting_item` — matches the three lists the
/// overlay shows live during the meeting.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MeetingItemKind {
    KeyPoint,
    Decision,
    ActionItem,
}

#[tauri::command]
pub fn track_meeting_item(state: State<'_, AppState>, kind: MeetingItemKind, text: String) -> Result<(), String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Err("cannot track an empty item".into());
    }
    let mut session = state.meeting_session.lock().map_err(|e| e.to_string())?;
    let item = TrackedItem { text: trimmed.to_string() };
    match kind {
        MeetingItemKind::KeyPoint => session.key_points.push(item),
        MeetingItemKind::Decision => session.decisions.push(item),
        MeetingItemKind::ActionItem => session.action_items.push(item),
    }
    Ok(())
}

#[tauri::command]
pub fn clear_meeting_session(state: State<'_, AppState>) -> Result<(), String> {
    let mut session = state.meeting_session.lock().map_err(|e| e.to_string())?;
    *session = Default::default();
    Ok(())
}

#[derive(Debug, serde::Deserialize)]
pub struct MeetingTurnPayload {
    pub speaker: String,
    pub text: String,
}

/// Builds a structured end-of-meeting summary from the transcript + tracked
/// items. The caller (frontend) archives the result to Meeting history via
/// `archive_meeting`, which is what actually needs `started_at_ms`.
#[tauri::command]
pub async fn end_meeting(
    state: State<'_, AppState>,
    turns: Vec<MeetingTurnPayload>,
    meeting_title: Option<String>,
    participants: Option<String>,
) -> Result<crate::backend::MeetingSummary, String> {
    log::info!("end_meeting: called, {} turns", turns.len());
    let (key_points, decisions, action_items) = {
        let session = state.meeting_session.lock().map_err(|e| e.to_string())?;
        (
            session.key_points.iter().map(|i| i.text.clone()).collect::<Vec<_>>(),
            session.decisions.iter().map(|i| i.text.clone()).collect::<Vec<_>>(),
            session.action_items.iter().map(|i| i.text.clone()).collect::<Vec<_>>(),
        )
    };

    let request = MeetingSummaryRequest {
        turns: turns
            .iter()
            .map(|t| MeetingTurnIn { speaker: t.speaker.clone(), text: t.text.clone() })
            .collect(),
        key_points: key_points.clone(),
        decisions: decisions.clone(),
        action_items: action_items.clone(),
        meeting_title: meeting_title.clone(),
        participants: participants.clone(),
    };

    log::info!("end_meeting: summarizing via configured AI provider");
    let summary = crate::personal::DirectLlmClient::new(None)?.meeting_summarize(&request).await?;
    log::info!("end_meeting: summarize returned");

    Ok(summary)
}
