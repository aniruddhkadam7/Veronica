use std::sync::Mutex;
use std::thread::JoinHandle;

use crate::audio::{PauseSignal, StopSignal};
use crate::rag::RagServiceHandle;
use crate::transcript::{RecordingState, TranscriptManager};
use crate::tts::{TtsSession, TtsSpeakingSignal};

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
}
