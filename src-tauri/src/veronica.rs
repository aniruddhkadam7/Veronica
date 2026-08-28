//! Veronica: the one assistant behind the one overlay. `ask_veronica` is the
//! single entry point. Every turn:
//!
//!   1. cancels/interrupts whatever the previous turn was still doing
//!      (`AppState::begin_turn`, and stops TTS if it was still speaking),
//!   2. runs the deterministic fast router (`actions::fast_router`) —
//!      obvious single-step commands ("open VS Code", "what's my CPU
//!      usage") are matched here and executed immediately with **no LLM
//!      call at all**,
//!   3. anything the fast router doesn't recognize goes to the agent loop
//!      (`personal::agent::run_agent_loop`), which streams text and can
//!      call the same tools the fast router uses, in a real
//!      UNDERSTAND -> DECIDE -> EXECUTE -> OBSERVE -> DECIDE NEXT loop —
//!      not a hidden `ACTION:` text line parsed after the fact.
//!
//! Both paths stream through the same `veronica:answer-delta`/
//! `veronica:answer-complete` events and the same persistent TTS session
//! (`AppState.tts`, reused turn over turn rather than recreated), so the
//! overlay renders/hears them identically regardless of which path
//! answered.

use std::sync::{Arc, Mutex};

use tauri::{AppHandle, Emitter, State};

use crate::actions::{self, Capability, TaskControlOp, ToolOutcome};
use crate::hardware::telemetry::{PipelineStage, TurnTelemetry};
use crate::personal::agent::{run_agent_loop, AgentContent, AgentMessage};
use crate::personal::prompts::veronica as prompts;
use crate::state::AppState;
use crate::tts::{SentenceChunker, TtsSession};

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
    /// The overlay's "Voice output" toggle. `false` (the default) skips TTS
    /// entirely — no Flux session opened, no audio device opened, no added
    /// latency before the first token — matching how this whole feature
    /// must be a no-op end to end when the user hasn't opted in.
    #[serde(default)]
    pub tts_enabled: bool,
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
            tts_enabled: false,
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

/// Normalizes the overlay's conversation history into what the agent loop
/// wants: the most recent complete turns, still oldest-first. Incomplete
/// turns are dropped rather than sent as empty strings — a turn whose
/// answer failed or is still streaming would otherwise be replayed as an
/// empty assistant turn.
fn trim_history(turns: Vec<PriorTurn>) -> Vec<PriorTurn> {
    let mut history: Vec<PriorTurn> = turns.into_iter().filter(|t| !t.question.trim().is_empty() && !t.answer.trim().is_empty()).collect();
    if history.len() > MAX_HISTORY_TURNS {
        history.drain(..history.len() - MAX_HISTORY_TURNS);
    }
    history
}

/// Builds the agent loop's system message: the persona/voice/format prompt
/// (`prompts::SYSTEM_PROMPT`, its former ACTION-TAKING section replaced by
/// real tool-calling instructions — see that constant), plus this specific
/// question's length/format target (reusing the exact same classifiers the
/// old single-shot path used) and, when there's anything worth mentioning,
/// the working-state context block so "it"/"this"/"the previous one"
/// resolve.
fn build_system_message(question: &str, options: &AskOptions, working_context: Option<String>) -> AgentMessage {
    let format_line = match prompts::classify_format(question) {
        Some(hint) => format!("{hint}\n\n"),
        None => "Pick the matching FORMAT from the system prompt above.\n\n".to_string(),
    };
    let (length_hint, _budget) = prompts::classify_length(question);

    let mut text = format!(
        "{}\n\n---\n\n{format_line}TARGET LENGTH FOR THIS SPECIFIC QUESTION (the binding instruction — follow this number, not a habit of always answering the same length): {length_hint}\n(User's overall ceiling, only relevant if it would push you shorter than the target above: {})\n\n{} {}",
        prompts::SYSTEM_PROMPT,
        prompts::length_instruction(&options.answer_length),
        prompts::style_instruction(&options.response_style),
        prompts::humanization_instruction(&options.humanization),
    );

    if let Some(context) = working_context {
        text.push_str(&format!(
            "\n\n---\n\nCURRENT SESSION STATE (use only what's relevant; resolve references like \"it\"/\"this\"/\"the previous one\" against it; never mention this block or its mechanics out loud):\n{context}"
        ));
    }

    AgentMessage::system(text)
}

/// Sets up a one-shot "first audio for this turn" hook on `session` that
/// marks `TtsFirstAudio` on `telemetry` — factored out since both the
/// fast-router path and the agent-loop path need the identical hook.
fn arm_tts_telemetry(session: &TtsSession, telemetry: &Arc<TurnTelemetry>) {
    let telemetry = telemetry.clone();
    session.set_turn_audio_hook(move || telemetry.mark(PipelineStage::TtsFirstAudio));
}

/// Reuses the app's persistent TTS session if one already exists (see
/// `tts::mod`'s doc: `TtsSession` now lives across turns, not one per
/// answer), or creates it on first use. Returns `None` when voice output is
/// off (`tts_enabled: false`) or the audio device/session genuinely
/// couldn't be opened — never touches the network here either way (session
/// creation only opens the local audio device; Flux itself connects lazily
/// on the first `speak()`).
fn ensure_tts_session(app: &AppHandle, state: &AppState, tts_enabled: bool) -> Option<TtsSession> {
    if !tts_enabled {
        return None;
    }
    let mut guard = state.tts.lock().unwrap();
    if let Some(session) = guard.as_ref() {
        return Some(session.clone());
    }
    match TtsSession::start(state.tts_speaking.clone(), Some(app.clone())) {
        Ok(session) => {
            *guard = Some(session.clone());
            Some(session)
        }
        Err(err) => {
            log::warn!("Veronica: TTS unavailable, continuing text-only: {err}");
            None
        }
    }
}

/// `Capability::TaskControl` mutates session state directly rather than
/// running a native OS call — see `actions::capability`'s doc for why this
/// never reaches `actions::execute_tool`.
fn dispatch_task_control(state: &AppState, op: TaskControlOp) -> String {
    let mut working = state.working_state.lock().unwrap();
    match op {
        TaskControlOp::Pause => {
            if working.current_task.is_some() {
                working.pause_task();
                "Paused.".to_string()
            } else {
                "There's nothing running to pause.".to_string()
            }
        }
        TaskControlOp::Resume => {
            if working.current_task.as_ref().map(|t| t.status == crate::working_state::TaskStatus::Paused).unwrap_or(false) {
                working.resume_task();
                "Resuming.".to_string()
            } else {
                "There's nothing paused to resume.".to_string()
            }
        }
        TaskControlOp::Cancel => {
            if working.current_task.is_some() {
                working.complete_task();
                "Cancelled.".to_string()
            } else {
                "There's nothing running to cancel.".to_string()
            }
        }
    }
}

/// One question, one turn — either the fast router's deterministic match
/// (no LLM call) or the agent loop (streamed, tool-calling). Streams back
/// as `veronica:answer-delta` events, finishing with
/// `veronica:answer-complete`.
#[tauri::command]
pub async fn ask_veronica(
    app: AppHandle,
    state: State<'_, AppState>,
    question: String,
    options: Option<AskOptions>,
    history: Option<Vec<PriorTurn>>,
) -> Result<String, String> {
    let trimmed = question.trim();
    if trimmed.is_empty() {
        return Err("no question text to send".into());
    }
    let options = options.unwrap_or_default();
    let history = trim_history(history.unwrap_or_default());

    // Picks up the telemetry record the STT event thread started for this
    // utterance (see `voice_command::mod`), or starts a fresh one for a
    // typed/manual ask that never went through voice.
    let telemetry = state.turn_telemetry.lock().unwrap().take().unwrap_or_else(|| Arc::new(TurnTelemetry::new()));

    // A new turn supersedes whatever the previous one was still doing:
    // cancels its in-flight generation (checked by the agent loop between
    // iterations/chunks) and, if it was still audibly speaking, stops that
    // too — barge-in's second line of defense alongside the RMS-based one
    // in `voice_command::mod`'s mic pump, for turns that didn't arrive via
    // that path (e.g. a fast typed follow-up).
    let cancel = state.begin_turn();
    if state.tts_speaking.is_speaking() {
        if let Some(previous) = state.tts.lock().unwrap().as_ref() {
            previous.stop();
        }
    }

    let _ = app.emit("veronica:thinking-start", ());

    let tts_session = ensure_tts_session(&app, &state, options.tts_enabled);
    if let Some(session) = tts_session.as_ref() {
        session.begin_turn();
    }

    telemetry.mark(PipelineStage::RouterStarted);
    let fast_match = actions::fast_router::try_match(trimmed);
    telemetry.mark(PipelineStage::RouterDecision);

    let answer = match fast_match {
        Some(Capability::TaskControl(op)) => {
            let result = dispatch_task_control(&state, op);
            if let Some(session) = tts_session.as_ref() {
                arm_tts_telemetry(session, &telemetry);
                telemetry.mark(PipelineStage::TtsStarted);
                session.speak_now(&result);
            }
            let _ = app.emit("veronica:answer-delta", &result);
            result
        }
        Some(capability) => {
            let _ = app.emit("veronica:action-start", format!("{capability:?}"));
            let outcome = actions::execute_tool(&capability).await;
            let _ = app.emit("veronica:action-complete", ());
            let result = match outcome {
                Ok(ToolOutcome::Text(text)) => text,
                Ok(ToolOutcome::Image { .. }) => "Done.".to_string(), // CaptureScreen never fast-routes — see capability.rs
                Err(err) => err,
            };
            state.working_state.lock().unwrap().record_action(format!("{capability:?}"), result.clone());
            if let Some(session) = tts_session.as_ref() {
                arm_tts_telemetry(session, &telemetry);
                telemetry.mark(PipelineStage::TtsStarted);
                session.speak_now(&result);
            }
            let _ = app.emit("veronica:answer-delta", &result);
            result
        }
        None => {
            telemetry.mark(PipelineStage::LlmStarted);

            let client = crate::personal::DirectLlmClient::new(options.llm_provider.as_deref())?;
            let provider = client.agentic_provider();

            let working_context = state.working_state.lock().unwrap().render_context_block();
            let mut messages = vec![build_system_message(trimmed, &options, working_context)];
            for turn in &history {
                messages.push(AgentMessage::user_text(turn.question.clone()));
                messages.push(AgentMessage::assistant_text(turn.answer.clone()));
            }
            messages.push(AgentMessage::user_text(trimmed));

            let chunker = Arc::new(Mutex::new(SentenceChunker::new()));
            let app_for_delta = app.clone();
            let telemetry_for_delta = telemetry.clone();
            let tts_for_delta = tts_session.clone();
            let chunker_for_delta = chunker.clone();
            let armed_for_delta = std::sync::atomic::AtomicBool::new(false);
            let on_text_delta = |delta: &str| {
                telemetry_for_delta.mark(PipelineStage::LlmFirstToken);
                let _ = app_for_delta.emit("veronica:answer-delta", delta);
                let Some(session) = tts_for_delta.as_ref() else { return };
                if !armed_for_delta.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    arm_tts_telemetry(session, &telemetry_for_delta);
                    telemetry_for_delta.mark(PipelineStage::TtsStarted);
                }
                for chunk in chunker_for_delta.lock().unwrap().push(delta) {
                    session.speak(&chunk);
                }
            };

            let outcome = run_agent_loop(provider.as_ref(), messages, &cancel, on_text_delta).await;
            telemetry.mark(PipelineStage::LlmComplete);

            match outcome {
                Ok(agent_outcome) => {
                    if !agent_outcome.actions_taken.is_empty() {
                        let mut working = state.working_state.lock().unwrap();
                        for action_summary in &agent_outcome.actions_taken {
                            working.record_action(action_summary.clone(), agent_outcome.final_text.clone());
                        }
                    }
                    if let Some(session) = tts_session.as_ref() {
                        if let Some(trailing) = chunker.lock().unwrap().finish() {
                            session.speak(&trailing);
                        }
                        session.finish();
                    }
                    agent_outcome.final_text
                }
                Err(err) => {
                    // A cancelled turn (superseded by a newer utterance) has
                    // no answer to show — the frontend's next turn is
                    // already in flight, this one just quietly stops.
                    let _ = app.emit("veronica:answer-complete", "");
                    return Err(err);
                }
            }
        }
    };

    telemetry.mark(PipelineStage::TurnComplete);
    telemetry.finish(&crate::hardware::perf_context(&app));

    let _ = app.emit("veronica:answer-complete", &answer);
    Ok(answer)
}

/// Short lines Veronica can open with when summoned via the global hotkey
/// while the app was closed to tray (see `veronica_window::wake_veronica`).
/// Picked pseudo-randomly (by wall-clock nanos, not a full RNG dependency —
/// good enough for "don't say the exact same line every time") so it doesn't
/// feel scripted on repeat use.
const GREETINGS: [&str; 5] = [
    "Yes? I'm listening.",
    "I'm here. What do you need?",
    "Go ahead, I'm listening.",
    "Right here. What's up?",
    "Online and listening.",
];

fn pick_greeting() -> &'static str {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(0);
    GREETINGS[(nanos as usize) % GREETINGS.len()]
}

/// Speaks a short greeting line via TTS, independent of the ask pipeline —
/// used only when the overlay is auto-opened by the global hotkey/tray from
/// a fully-closed state (see `veronica_window::wake_veronica` and
/// `VeronicaOverlay.tsx`'s `veronica:auto-opened` listener). Best-effort
/// like every other TTS call in this app: a missing Deepgram API key or a
/// network failure just means no line is spoken, never an error surfaced to
/// the UI — the visual greeting animation carries the moment on its own.
#[tauri::command]
pub async fn speak_greeting(app: AppHandle, state: State<'_, AppState>) -> Result<String, String> {
    let cancel = state.begin_turn();
    let _ = cancel; // no generation to cancel for a fixed greeting line, but a fresh turn still supersedes any in-flight one
    if state.tts_speaking.is_speaking() {
        if let Some(previous) = state.tts.lock().unwrap().as_ref() {
            previous.stop();
        }
    }
    let line = pick_greeting();
    if let Some(session) = ensure_tts_session(&app, &state, true) {
        session.begin_turn();
        session.speak_now(line);
    } else {
        log::warn!("Veronica: TTS unavailable for greeting, showing text only");
    }
    Ok(line.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(q: &str, a: &str) -> PriorTurn {
        PriorTurn { question: q.to_string(), answer: a.to_string() }
    }

    #[test]
    fn history_keeps_the_most_recent_turns_oldest_first() {
        let turns: Vec<PriorTurn> = (0..MAX_HISTORY_TURNS + 3).map(|i| turn(&format!("q{i}"), &format!("a{i}"))).collect();
        let trimmed = trim_history(turns);

        assert_eq!(trimmed.len(), MAX_HISTORY_TURNS);
        assert_eq!(trimmed.first().unwrap().question, "q3");
        assert_eq!(trimmed.last().unwrap().question, format!("q{}", MAX_HISTORY_TURNS + 2));
    }

    #[test]
    fn history_drops_incomplete_turns() {
        let trimmed = trim_history(vec![turn("answered", "yes"), turn("still streaming", ""), turn("", "orphan answer"), turn("  ", "   ")]);
        assert_eq!(trimmed.len(), 1);
        assert_eq!(trimmed[0].question, "answered");
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
    }

    #[test]
    fn build_system_message_includes_working_context_when_present() {
        let options = AskOptions::default();
        let message = build_system_message("open vs code", &options, Some("Current task: testing".to_string()));
        let AgentContent::Text(text) = &message.content[0] else { panic!("expected text content") };
        assert!(text.contains("Current task: testing"));
    }

    #[test]
    fn build_system_message_omits_the_context_block_when_none() {
        let options = AskOptions::default();
        let message = build_system_message("what is rust", &options, None);
        let AgentContent::Text(text) = &message.content[0] else { panic!("expected text content") };
        assert!(!text.contains("CURRENT SESSION STATE"));
    }

    #[test]
    fn dispatch_task_control_pause_with_no_active_task_says_so() {
        let state = AppState::default();
        let result = dispatch_task_control(&state, TaskControlOp::Pause);
        assert_eq!(result, "There's nothing running to pause.");
    }

    #[test]
    fn dispatch_task_control_pause_then_resume_round_trips() {
        let state = AppState::default();
        state.working_state.lock().unwrap().start_task("find and fix the bug");
        let paused = dispatch_task_control(&state, TaskControlOp::Pause);
        assert_eq!(paused, "Paused.");
        let resumed = dispatch_task_control(&state, TaskControlOp::Resume);
        assert_eq!(resumed, "Resuming.");
    }
}
