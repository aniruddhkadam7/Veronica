use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::audio::{PauseSignal, StopSignal};
use crate::rag::RagServiceHandle;
use crate::transcript::{RecordingState, TranscriptManager};
use crate::tts::{TtsSession, TtsSpeakingSignal};

/// Cancellation flag for one in-flight voice turn (fast-router execution or
/// agent-loop generation) — mirrors `audio::StopSignal`/`audio::PauseSignal`'s
/// existing `Arc<AtomicBool>` handle pattern. A new turn starting (the next
/// utterance's `Final` transcript arriving, or a barge-in) cancels whatever
/// token is currently active in `AppState.current_turn`; the agent loop and
/// the RAG/LLM streaming call sites check `is_cancelled()` between chunks/
/// iterations so an interrupted turn actually stops doing work instead of
/// only being stopped from being heard (that gap — TTS silenced but the LLM
/// stream still running to completion in the background — was the old
/// barge-in behavior; see `voice_command::mod`'s pump thread).
#[derive(Clone, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Handles for an in-progress recording session, held so Tauri commands can stop it
/// later. Wrapped in `Mutex` because Tauri commands run on arbitrary threads from
/// the async runtime.
pub struct CaptureSession {
    pub stop_signal: Option<StopSignal>,
    pub pause_signal: Option<PauseSignal>,
    pub system_audio_thread: Option<JoinHandle<()>>,
    pub mic_thread: Option<JoinHandle<()>>,
    pub pipeline_thread: Option<JoinHandle<()>>,
    pub recording_state: RecordingState,
}

impl Default for CaptureSession {
    fn default() -> Self {
        Self {
            stop_signal: None,
            pause_signal: None,
            system_audio_thread: None,
            mic_thread: None,
            pipeline_thread: None,
            recording_state: RecordingState::Idle,
        }
    }
}

#[derive(Default)]
pub struct AppState {
    pub capture: Mutex<CaptureSession>,
    pub transcript: Mutex<TranscriptManager>,
    /// `None` if the RAG service's venv wasn't found at startup (see
    /// `rag::process::RagServiceHandle::spawn`) — document upload/search
    /// commands report a clear "unavailable" error in that case rather than
    /// panicking.
    pub rag_service: Mutex<Option<RagServiceHandle>>,
    /// Mic-only capture session for Notes' voice dictation — reuses
    /// `CaptureSession` (system-audio fields simply stay `None`) so dictation
    /// gets the same start/pause/resume/stop lifecycle without a second
    /// struct. See `notes_mode::commands`.
    pub notes_dictation: Mutex<CaptureSession>,
    /// The most recent answer's TTS session, handed off here once its LLM
    /// stream finishes so playback of the last sentence(s) can continue
    /// after the `ask_veronica` command returns. `None` when voice output is
    /// off or no answer has been asked yet. See `veronica::ask_veronica`.
    pub tts: Mutex<Option<TtsSession>>,
    /// Shared, app-lifetime "is Veronica's own voice playing right now"
    /// signal — one instance for the whole app (not scoped per `TtsSession`,
    /// which is per-answer), so `voice_command::mod`'s mic-assistant pump
    /// can check it before forwarding audio to STT regardless of which
    /// answer/session is currently speaking. See `tts::TtsSpeakingSignal`'s
    /// doc for why this exists: without it, Veronica's own TTS output,
    /// picked up acoustically by the mic, gets transcribed and answered as
    /// if the user said it.
    pub tts_speaking: TtsSpeakingSignal,
    /// The currently in-flight voice turn's cancellation token, if any. See
    /// `CancelToken`'s doc. `None` between turns.
    pub current_turn: Mutex<Option<CancelToken>>,
    /// Lightweight cross-turn working memory (current app/window/file/
    /// project/task, recent actions, last result) so "it"/"this"/"the
    /// previous one" resolve across turns — see `working_state::WorkingState`.
    pub working_state: Mutex<crate::working_state::WorkingState>,
    /// The current utterance's latency instrumentation, created by
    /// `voice_command::mod`'s STT event thread (which marks the
    /// mic/speech/STT-side stages as they happen) and picked up by
    /// `veronica::ask_veronica` (which marks the router/LLM/TTS-side stages
    /// and logs the finished record) once the frontend's `invoke()` call
    /// for this turn arrives. `None` for a typed/manual ask that never went
    /// through the voice pipeline — `ask_veronica` starts a fresh one in
    /// that case instead of waiting for a slot that will never be filled.
    pub turn_telemetry: Mutex<Option<std::sync::Arc<crate::hardware::telemetry::TurnTelemetry>>>,
}

impl AppState {
    /// Starts a new voice turn: cancels whatever turn was previously
    /// in-flight (if any — a barge-in or a fast-follow-up utterance means
    /// the previous turn's answer is stale) and installs+returns a fresh
    /// `CancelToken` for this one. Every call site that starts generating a
    /// response (fast router execution, agent-loop streaming, RAG lookups)
    /// should hold the returned token and check `is_cancelled()` between
    /// units of work.
    pub fn begin_turn(&self) -> CancelToken {
        let mut guard = self.current_turn.lock().unwrap();
        if let Some(previous) = guard.take() {
            previous.cancel();
        }
        let token = CancelToken::new();
        *guard = Some(token.clone());
        token
    }

    /// Cancels whatever turn is currently in flight, without starting a new
    /// one — used by barge-in (`voice_command::mod`'s mic pump), which only
    /// knows "the user started talking over Veronica" at this instant, not
    /// yet what they said; the next `ask_veronica` call (once that new
    /// utterance is transcribed) is what calls `begin_turn()` for the
    /// replacement turn.
    pub fn cancel_current_turn(&self) {
        if let Some(token) = self.current_turn.lock().unwrap().as_ref() {
            token.cancel();
        }
    }
}
