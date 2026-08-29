//! Veronica: the one assistant behind the one overlay. `ask_veronica` is the
//! single entry point for a real conversational turn — `try_interrupt` (see
//! its own doc) is the separate entry point for a dedicated interruption
//! command ("stop"/"wait"/"cancel"), called by the frontend BEFORE a turn
//! is ever created for an utterance, so a bare interruption never becomes
//! an `ask_veronica` call at all. Every real turn:
//!
//!   1. cancels/interrupts whatever the previous turn was still doing
//!      (`AppState::begin_turn`, and stops TTS if it was still speaking),
//!   2. runs the language/quality gate (`crate::language::detect`) — a
//!      confidently non-English utterance or a low-confidence/garbled one
//!      is answered with a short clarification and never reaches the
//!      router or an LLM at all,
//!   3. runs the deterministic fast router (`actions::fast_router`) —
//!      obvious single-step commands ("open VS Code", "what's my CPU
//!      usage") are matched here and executed immediately with **no LLM
//!      call at all**,
//!   4. anything the fast router doesn't recognize goes to the agent loop
//!      (`personal::agent::run_agent_loop`), which streams text and can
//!      call the same tools the fast router uses, in a real
//!      UNDERSTAND -> DECIDE -> EXECUTE -> OBSERVE -> DECIDE NEXT loop —
//!      not a hidden `ACTION:` text line parsed after the fact. Multi-step
//!      tool use also streams short, human-readable progress lines (see
//!      `personal::agent::tool_schema::progress_message`) into the same
//!      answer, never the raw tool name or its JSON arguments.
//!
//! Every path streams through the same `veronica:answer-delta`/
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

/// One prior exchange in this Veronica conversation — the shape
/// `run_agent_loop`'s message-building wants. Built internally from the
/// shared `ConversationStore` (see `trim_history`), not deserialized from
/// the frontend: neither window sends its own history anymore, since the
/// backend's own store is the single, authoritative source both draw from.
#[derive(Debug, Clone)]
pub struct PriorTurn {
    pub question: String,
    pub answer: String,
}

// ---------------------------------------------------------------------
// Event payloads — every one carries `turn_id` (camelCase on the wire:
// `turnId`) so the frontend's turn manager can correlate an incoming event
// back to the exact conversation turn it belongs to by IDENTITY, never by
// "whichever turn happens to be last in the list right now." That
// positional-correlation pattern (`prev[prev.length - 1]`) was the actual
// root cause of the live-observed bug where one turn's real answer landed
// on a DIFFERENT turn's message bubble while the first stayed stuck showing
// "Thinking…" forever — seen the moment more than one turn could exist
// with one still unresolved, for any reason (a rapid follow-up, a VAD
// utterance split, a race), not just one specific trigger. Turn-id-keyed
// payloads make that whole class of bug structurally impossible on the
// frontend, regardless of how/why turns end up overlapping.
// ---------------------------------------------------------------------

/// Bare `{turnId}` payload — used for every event that only needs to say
/// "this happened, for this turn" with no additional data
/// (`veronica:thinking-start`, `veronica:action-complete`).
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnIdPayload {
    turn_id: String,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct AnswerDeltaPayload {
    turn_id: String,
    delta: String,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct AnswerCompletePayload {
    turn_id: String,
    answer: String,
    /// `true` when this turn ended because a newer turn superseded it
    /// (barge-in, or a fast follow-up) rather than because it actually
    /// produced (or failed to produce) an answer — lets the frontend render
    /// an interrupted turn distinctly from a turn that legitimately
    /// completed with empty content.
    cancelled: bool,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct TurnErrorPayload {
    turn_id: String,
    message: String,
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ActionPayload {
    turn_id: String,
    action: String,
}

/// Optional for the frontend to display ("English") — see the
/// language-policy doc on `crate::language` and `run_turn`'s use of it.
/// `language` is the wire code ("en"/"unsupported").
#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct LanguageDetectedPayload {
    turn_id: String,
    language: String,
}

/// How many prior turns to forward. The overlay keeps the whole conversation
/// on screen, but only the recent tail is worth paying for in the prompt:
/// every turn adds tokens to time-to-first-token, which is the number the
/// user feels. Six turns comfortably covers "why did you choose that?"
/// style follow-ups, which almost always refer to the last exchange or two.
const MAX_HISTORY_TURNS: usize = 6;

/// Trims the shared conversation store's completed history down to the most
/// recent `MAX_HISTORY_TURNS`, oldest-first — everything already
/// non-empty/complete by construction (see `ConversationStore::
/// completed_history`), so this only bounds the count, unlike its
/// predecessor which also had to filter out incomplete caller-supplied
/// turns.
fn trim_history(turns: Vec<(String, String)>) -> Vec<PriorTurn> {
    let mut history: Vec<PriorTurn> = turns.into_iter().map(|(question, answer)| PriorTurn { question, answer }).collect();
    if history.len() > MAX_HISTORY_TURNS {
        history.drain(..history.len() - MAX_HISTORY_TURNS);
    }
    history
}

/// Builds the agent loop's system message: the persona/voice/format prompt
/// (`prompts::SYSTEM_PROMPT`, its former ACTION-TAKING section replaced by
/// real tool-calling instructions — see that constant, which now also
/// carries the fixed English-only language-policy instruction), plus this
/// specific question's length/format target (reusing the exact same
/// classifiers the old single-shot path used), and, when there's anything
/// worth mentioning, the working-state context block so "it"/"this"/"the
/// previous one" resolve.
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

/// Stops Veronica mid-answer: cancels whatever turn is currently generating
/// (same `cancel_current_turn` barge-in uses, so an in-flight LLM stream
/// actually stops doing work, not just the audio) and stops the persistent
/// TTS session if it's speaking right now. Called by the header's Stop
/// button alongside `stop_mic_assistant`, so "Stop" ends listening AND
/// speaking together rather than leaving a reply playing out after the mic
/// has already gone silent.
#[tauri::command]
pub fn stop_speaking(state: State<'_, AppState>) -> Result<(), String> {
    state.cancel_current_turn();
    if let Some(session) = state.tts.lock().unwrap().as_ref() {
        session.stop();
    }
    Ok(())
}

/// Checks whether `text` is a dedicated interruption command ("stop",
/// "wait", "hold on", "cancel", ...) rather than a normal request — see
/// `crate::interrupt`'s doc. When it is, this ALSO performs the
/// interruption (same effect as `stop_speaking`, plus a distinct
/// `veronica:interrupted` event) and returns `true`, so the frontend's
/// transcript handler can check-and-act in one call rather than racing a
/// separate detect-then-stop round trip. When it isn't, this is a pure,
/// side-effect-free `false` — the caller proceeds to treat the utterance as
/// a normal turn.
///
/// Called by the overlay/widget BEFORE a `Turn` is ever created for the
/// utterance — an interruption must never appear as a "YOU: stop" message,
/// and must never produce a visible assistant reply (no "(interrupted)",
/// no "Paused."). See requirement 6: stop is a control signal, not
/// conversation content.
#[tauri::command]
pub fn try_interrupt(app: AppHandle, state: State<'_, AppState>, text: String) -> Result<bool, String> {
    if !crate::interrupt::is_interrupt(&text) {
        return Ok(false);
    }
    log::info!("[INTERRUPT] user said {text:?} — treating as a control command, not a turn");
    state.cancel_current_turn();
    if let Some(session) = state.tts.lock().unwrap().as_ref() {
        session.stop();
    }
    let _ = app.emit("veronica:interrupted", ());
    Ok(true)
}

/// Returns the full shared conversation so far, oldest first — called by
/// the overlay on mount/show instead of starting from an empty list, and by
/// the widget if it ever needs the same history. See `conversation`'s doc:
/// this is the ONE place either window reads the conversation from, so
/// "open the overlay and see everything said through the widget" and vice
/// versa are both just this same call.
#[tauri::command]
pub fn get_conversation_history(state: State<'_, AppState>) -> Result<Vec<crate::conversation::ConversationTurn>, String> {
    Ok(state.conversation.lock().unwrap().snapshot())
}

/// Clears the shared conversation — called only when the user explicitly
/// ends the session (App.tsx's Stop), never by opening or closing the
/// overlay/widget. See `conversation`'s doc.
#[tauri::command]
pub fn reset_conversation(state: State<'_, AppState>) -> Result<(), String> {
    state.conversation.lock().unwrap().reset();
    Ok(())
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
/// as `veronica:answer-delta` events.
///
/// Every exit path funnels through the `match result` block at the bottom —
/// this is the actual fix for the "stuck in Thinking forever" bug reported
/// live: the previous version called `DirectLlmClient::new(...)?` directly
/// in the middle of this function. On failure (no API key configured for
/// the selected provider, or any other `?`-propagated error) that returned
/// straight out of the command WITHOUT ever emitting `veronica:answer-complete`
/// or `veronica:error` — and the frontend's orb state
/// (`useVeronicaOrbState.ts`) only ever clears `thinking` on one of those
/// two events (`veronica:thinking-start` sets it, nothing else touches it),
/// so it stayed stuck showing "Thinking…" until the NEXT successful turn
/// happened to fire one of those events. `run_turn` below is free to use
/// `?` internally however it needs to — since it's a separate function, its
/// early returns just become the `Err(err)` case of `result`, which THIS
/// function's terminal-event handling always sees and always acts on, no
/// matter how deep inside `run_turn` the failure happened.
#[tauri::command]
pub async fn ask_veronica(
    app: AppHandle,
    state: State<'_, AppState>,
    question: String,
    options: Option<AskOptions>,
    turn_id: String,
) -> Result<String, String> {
    let trimmed = question.trim();
    if trimmed.is_empty() {
        return Err("no question text to send".into());
    }
    let options = options.unwrap_or_default();
    // Derived from the ONE shared conversation store (see conversation.rs),
    // NOT a caller-supplied parameter — a follow-up asked through either
    // window now correctly resolves against whatever was said through
    // EITHER window, not just this same one's own client-side history.
    let history = trim_history(state.conversation.lock().unwrap().completed_history());

    // `turn_id` is generated by the CALLER (the frontend's turn manager —
    // see VeronicaOverlay.tsx/useAutoAsk.ts) the instant it decides to send
    // this question, and is what every event below is keyed by. This is
    // deliberately NOT the same id as `TurnTelemetry`'s own internal one
    // (which can be created earlier, by the STT event thread, before the
    // frontend has even decided this utterance is a complete question) —
    // that one stays a purely backend perf-log correlation id; this one is
    // the conversation/UI-facing turn identity.
    log::info!("[TURN_CREATED] turn_id={turn_id}");
    log::info!("[USER_MESSAGE] turn_id={turn_id} text={trimmed:?}");

    // Recorded into the ONE shared conversation store immediately — before
    // any router/LLM work starts — so this turn is visible to whichever
    // window (widget or overlay) reads the shared history next, regardless
    // of which window's mic session actually produced it. See
    // `conversation`'s doc: this is what makes "spoken through the widget,
    // visible in the overlay" (and vice versa) true, since the two windows
    // share no other state.
    state.conversation.lock().unwrap().create_turn(&turn_id, trimmed);

    let telemetry = state.turn_telemetry.lock().unwrap().take().unwrap_or_else(|| Arc::new(TurnTelemetry::new()));

    // A new turn supersedes whatever the previous one was still doing:
    // cancels its in-flight generation (checked by the agent loop between
    // iterations/chunks, and per network chunk within one provider call —
    // see `AgenticProvider::stream_agentic`) and, if it was still audibly
    // speaking, stops that too — barge-in's second line of defense
    // alongside the RMS-based one in `voice_command::mod`'s mic pump, for
    // turns that didn't arrive via that path (e.g. a fast typed follow-up).
    let cancel = state.begin_turn();
    if state.tts_speaking.is_speaking() {
        log::info!("[INTERRUPT] turn_id={turn_id} stopping in-progress speech for a new turn");
        if let Some(previous) = state.tts.lock().unwrap().as_ref() {
            previous.stop();
        }
    }

    let _ = app.emit("veronica:thinking-start", TurnIdPayload { turn_id: turn_id.clone() });

    let tts_session = ensure_tts_session(&app, &state, options.tts_enabled);
    if let Some(session) = tts_session.as_ref() {
        session.begin_turn();
    }

    let result = run_turn(&app, &state, &turn_id, trimmed, &options, &history, tts_session.as_ref(), &telemetry, &cancel).await;

    // Defense in depth: even if every internal path is correct, a turn that
    // technically completed (`Ok`) but whose own cancel token was flipped
    // while it was finishing must still be discarded, not delivered — a
    // newer turn already owns the conversation by this point. See
    // requirement 9 (stale responses must never overwrite a newer turn).
    let result = if cancel.is_cancelled() && result.is_ok() {
        log::warn!("[TURN_ERROR] turn_id={turn_id} completed but was already superseded — discarding its answer");
        Err("cancelled".to_string())
    } else {
        result
    };

    telemetry.mark(PipelineStage::TurnComplete);
    telemetry.finish(&crate::hardware::perf_context(&app));

    let final_result = match result {
        Ok(answer) => {
            log::info!("[TURN_COMPLETE] turn_id={turn_id}");
            state.conversation.lock().unwrap().complete_turn(&turn_id, &answer, false);
            let _ = app.emit("veronica:answer-complete", AnswerCompletePayload { turn_id: turn_id.clone(), answer: answer.clone(), cancelled: false });
            Ok(answer)
        }
        Err(err) if err == "cancelled" => {
            // Superseded by a newer turn — expected, not a failure the user
            // needs to hear about. Just clear THINKING/SPEAKING and return
            // to listening quietly; the newer turn is already in flight.
            log::info!("[TURN_COMPLETE] turn_id={turn_id} (interrupted/superseded)");
            state.conversation.lock().unwrap().complete_turn(&turn_id, "", true);
            let _ = app.emit("veronica:answer-complete", AnswerCompletePayload { turn_id: turn_id.clone(), answer: String::new(), cancelled: true });
            Err(err)
        }
        Err(err) => {
            // Requirement: THINKING must never be a terminal state, even on
            // a genuine failure (missing/invalid API key, network error,
            // provider outage, malformed response). Speak a short, honest
            // recovery line (if voice is on) and always emit the events
            // that clear the orb back to idle/listening either way.
            log::error!("[TURN_ERROR] turn_id={turn_id} {err}");
            let _ = app.emit("veronica:error", TurnErrorPayload { turn_id: turn_id.clone(), message: format!("Sorry, I ran into a problem: {err}") });
            if let Some(session) = tts_session.as_ref() {
                log::info!("[TTS_START] turn_id={turn_id} speaking error-recovery line");
                session.speak_now("Sorry, I ran into a problem there. Please try again.");
                log::info!("[TTS_COMPLETE] turn_id={turn_id}");
            }
            log::info!("[TURN_COMPLETE] turn_id={turn_id} (error)");
            state.conversation.lock().unwrap().complete_turn(&turn_id, "", false);
            let _ = app.emit("veronica:answer-complete", AnswerCompletePayload { turn_id: turn_id.clone(), answer: String::new(), cancelled: false });
            Err(err)
        }
    };
    log::info!("[LISTENING] turn_id={turn_id}");
    final_result
}

/// The actual fast-router-or-agent-loop turn logic, pulled out of
/// `ask_veronica` so every early return here (via `?` or otherwise) still
/// passes through that function's guaranteed terminal-event handling — see
/// its doc comment.
#[allow(clippy::too_many_arguments)]
async fn run_turn(
    app: &AppHandle,
    state: &AppState,
    turn_id: &str,
    trimmed: &str,
    options: &AskOptions,
    history: &[PriorTurn],
    tts_session: Option<&TtsSession>,
    telemetry: &Arc<TurnTelemetry>,
    cancel: &crate::state::CancelToken,
) -> Result<String, String> {
    // Language/quality gate: runs on the raw transcript, before the fast
    // router AND before any LLM call — see `crate::language`'s doc for why
    // enforcement lives here rather than at the STT-provider level (Groq's
    // Whisper API has no option to refuse non-English speech, only a hint
    // that biases recognition).
    //
    // Two distinct non-terminal outcomes, deliberately not collapsed into
    // one: `Unsupported` is confident positive evidence of a real
    // non-English utterance (a language-policy matter), while
    // `LowConfidence` is "no evidence either way" — almost always garbled/
    // misheard audio, not a real foreign-language request. Requirement 8
    // is explicit that the two must never share a message: repeating "I
    // only support English" for ordinary bad-audio noise reads as broken
    // and unhelpful, and inventing an answer from a low-confidence
    // transcript would be hallucinating the user's actual request. Either
    // way, no fast-router dispatch and no agent-loop/LLM request happens —
    // there is nothing reliable enough yet to act on.
    let language_decision = crate::language::detect(trimmed);
    log::info!("[LANGUAGE] turn_id={turn_id} detected={}", language_decision.code());
    let _ = app.emit("veronica:language-detected", LanguageDetectedPayload { turn_id: turn_id.to_string(), language: language_decision.code().to_string() });

    let clarification: Option<&str> = match language_decision {
        crate::language::Decision::Supported(_) => None,
        crate::language::Decision::Unsupported => Some(crate::language::rejection_message()),
        crate::language::Decision::LowConfidence => Some(crate::language::clarification_message()),
    };
    if let Some(message) = clarification {
        let message = message.to_string();
        if let Some(session) = tts_session {
            arm_tts_telemetry(session, telemetry);
            telemetry.mark(PipelineStage::TtsStarted);
            log::info!("[TTS_START] turn_id={turn_id}");
            session.speak_now(&message);
            log::info!("[TTS_COMPLETE] turn_id={turn_id}");
        }
        let _ = app.emit("veronica:answer-delta", AnswerDeltaPayload { turn_id: turn_id.to_string(), delta: message.clone() });
        return Ok(message);
    }

    telemetry.mark(PipelineStage::RouterStarted);
    let fast_match = actions::fast_router::try_match(trimmed);
    telemetry.mark(PipelineStage::RouterDecision);
    log::info!(
        "[ROUTER_DECISION] turn_id={turn_id} matched={}",
        fast_match.as_ref().map(|c| format!("{c:?}")).unwrap_or_else(|| "none (agent loop)".to_string())
    );

    match fast_match {
        Some(Capability::TaskControl(op)) => {
            let result = dispatch_task_control(state, op);
            if let Some(session) = tts_session {
                arm_tts_telemetry(session, telemetry);
                telemetry.mark(PipelineStage::TtsStarted);
                log::info!("[TTS_START] turn_id={turn_id}");
                session.speak_now(&result);
                log::info!("[TTS_COMPLETE] turn_id={turn_id}");
            }
            let _ = app.emit("veronica:answer-delta", AnswerDeltaPayload { turn_id: turn_id.to_string(), delta: result.clone() });
            Ok(result)
        }
        Some(capability) => {
            let _ = app.emit("veronica:action-start", ActionPayload { turn_id: turn_id.to_string(), action: format!("{capability:?}") });
            let outcome = actions::execute_tool(&capability).await;
            let _ = app.emit("veronica:action-complete", TurnIdPayload { turn_id: turn_id.to_string() });
            let result = match outcome {
                Ok(ToolOutcome::Text(text)) => text,
                Ok(ToolOutcome::Image { .. }) => "Done.".to_string(), // CaptureScreen never fast-routes — see capability.rs
                Err(err) => err,
            };
            state.working_state.lock().unwrap().record_action(format!("{capability:?}"), result.clone());
            if let Some(session) = tts_session {
                arm_tts_telemetry(session, telemetry);
                telemetry.mark(PipelineStage::TtsStarted);
                log::info!("[TTS_START] turn_id={turn_id}");
                session.speak_now(&result);
                log::info!("[TTS_COMPLETE] turn_id={turn_id}");
            }
            let _ = app.emit("veronica:answer-delta", AnswerDeltaPayload { turn_id: turn_id.to_string(), delta: result.clone() });
            Ok(result)
        }
        None => {
            log::info!("[LLM_START] turn_id={turn_id}");
            telemetry.mark(PipelineStage::LlmStarted);

            // Was `DirectLlmClient::new(...)?` — the exact call whose
            // failure used to bypass `ask_veronica`'s terminal-event
            // handling and leave the orb stuck on "Thinking…" forever. Now
            // a plain `?` inside THIS function, which is fine: it just
            // becomes `run_turn`'s own `Err`, which the caller always
            // handles.
            let client = crate::personal::DirectLlmClient::new(options.llm_provider.as_deref())?;
            let provider = client.agentic_provider();

            let working_context = state.working_state.lock().unwrap().render_context_block();
            let mut messages = vec![build_system_message(trimmed, options, working_context)];
            for turn in history {
                messages.push(AgentMessage::user_text(turn.question.clone()));
                messages.push(AgentMessage::assistant_text(turn.answer.clone()));
            }
            messages.push(AgentMessage::user_text(trimmed));

            let chunker = Arc::new(Mutex::new(SentenceChunker::new()));
            let app_for_delta = app.clone();
            let telemetry_for_delta = telemetry.clone();
            let tts_for_delta = tts_session.cloned();
            let chunker_for_delta = chunker.clone();
            let turn_id_for_delta = turn_id.to_string();
            let logged_first_token = std::sync::atomic::AtomicBool::new(false);
            let armed_for_tts = std::sync::atomic::AtomicBool::new(false);
            let on_text_delta = |delta: &str| {
                telemetry_for_delta.mark(PipelineStage::LlmFirstToken);
                if !logged_first_token.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    log::info!("[LLM_FIRST_TOKEN] turn_id={turn_id_for_delta}");
                }
                state.conversation.lock().unwrap().append_delta(&turn_id_for_delta, delta);
                let _ = app_for_delta.emit("veronica:answer-delta", AnswerDeltaPayload { turn_id: turn_id_for_delta.clone(), delta: delta.to_string() });
                let Some(session) = tts_for_delta.as_ref() else { return };
                if !armed_for_tts.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    arm_tts_telemetry(session, &telemetry_for_delta);
                    telemetry_for_delta.mark(PipelineStage::TtsStarted);
                    log::info!("[TTS_START] turn_id={turn_id_for_delta}");
                }
                for chunk in chunker_for_delta.lock().unwrap().push(delta) {
                    session.speak(&chunk);
                }
            };

            // Human-readable progress lines for multi-step tool use
            // (requirement 11) — streamed into the SAME assistant turn as a
            // normal delta (so the overlay's "exactly one assistant turn per
            // response" invariant holds — see `applyTurnEvent` in
            // VeronicaOverlay.tsx) rather than as a separate message.
            //
            // Spoken via a plain `speak()` (with an explicit trailing pause
            // so Flux treats it as a complete sentence), deliberately NOT
            // `speak_now()` — that also calls `finish()`, which sends Flux's
            // `Flush` command, and `Flush` means "the answer is over" in
            // Flux's own turn protocol (see deepgram_flux.rs). This is a
            // MID-answer status line with more text still to come (the next
            // loop iteration's real answer, or another tool call), so
            // ending the Flux turn here would be a real protocol-level lie,
            // not just a phrasing choice — `finish()` is only called once,
            // by the normal end-of-answer path below.
            let app_for_progress = app.clone();
            let tts_for_progress = tts_session.cloned();
            let turn_id_for_progress = turn_id.to_string();
            let on_progress = |message: &str| {
                let delta = format!("\n\n{message}");
                state.conversation.lock().unwrap().append_delta(&turn_id_for_progress, &delta);
                let _ = app_for_progress.emit("veronica:answer-delta", AnswerDeltaPayload { turn_id: turn_id_for_progress.clone(), delta });
                if let Some(session) = tts_for_progress.as_ref() {
                    session.speak(&format!("{message} "));
                }
            };

            // `?` here too: a cancelled or genuinely failed agent loop
            // becomes this function's `Err`, handled uniformly by the
            // caller — no special-casing needed at this call site anymore.
            let agent_outcome = run_agent_loop(provider.as_ref(), messages, cancel, turn_id, on_text_delta, on_progress).await?;
            telemetry.mark(PipelineStage::LlmComplete);
            log::info!("[LLM_COMPLETE] turn_id={turn_id} chars={}", agent_outcome.final_text.len());

            if !agent_outcome.actions_taken.is_empty() {
                let mut working = state.working_state.lock().unwrap();
                for action_summary in &agent_outcome.actions_taken {
                    working.record_action(action_summary.clone(), agent_outcome.final_text.clone());
                }
            }
            if let Some(session) = tts_session {
                if let Some(trailing) = chunker.lock().unwrap().finish() {
                    session.speak(&trailing);
                }
                session.finish();
                log::info!("[TTS_COMPLETE] turn_id={turn_id}");
            }
            Ok(agent_outcome.final_text)
        }
    }
}

/// Short lines Veronica can open with when summoned via the global hotkey
/// while the app was closed to tray (see `veronica_window::wake_veronica`).
/// Picked pseudo-randomly (by wall-clock nanos, not a full RNG dependency —
/// good enough for "don't say the exact same line every time") so it doesn't
/// feel scripted on repeat use.
const GREETINGS: [&str; 5] = [
    "Yes, Sir?",
    "I'm listening.",
    "Go ahead.",
    "Right here. What's up?",
    "Online, and no, nothing's on fire.",
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

    fn turn(q: &str, a: &str) -> (String, String) {
        (q.to_string(), a.to_string())
    }

    #[test]
    fn history_keeps_the_most_recent_turns_oldest_first() {
        let turns: Vec<(String, String)> = (0..MAX_HISTORY_TURNS + 3).map(|i| turn(&format!("q{i}"), &format!("a{i}"))).collect();
        let trimmed = trim_history(turns);

        assert_eq!(trimmed.len(), MAX_HISTORY_TURNS);
        assert_eq!(trimmed.first().unwrap().question, "q3");
        assert_eq!(trimmed.last().unwrap().question, format!("q{}", MAX_HISTORY_TURNS + 2));
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
