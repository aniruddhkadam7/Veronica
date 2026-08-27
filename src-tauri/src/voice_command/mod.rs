//! Mic-input assistant mode for the Interview Overlay: toggled on/off by the
//! overlay's mic button. While on, the user's own voice is transcribed
//! through the same STT sidecar and `TranscriptManager`/`transcript:update`
//! event every other capture source uses (see `commands::start_system_audio_capture`
//! and `transcript::TranscriptManager`) — tagged `AudioSource::Microphone` —
//! so the overlay's existing transcript listener can treat it exactly like
//! interviewer speech: it lands in the question box and, if Auto AI is on,
//! gets sent automatically once the user stops talking. This turns the
//! overlay into a talk-to-it assistant rather than only a system-audio
//! interview aid.
//!
//! Deliberately separate capture session from `commands::start_system_audio_capture`
//! (its own `StopSignal`/thread, not `AppState.capture`) so the mic can be
//! toggled independently of whether interviewer-audio capture is running —
//! `TranscriptManager` already keys in-progress segments per `AudioSource`
//! (see transcript/mod.rs), so both sources safely feed the same transcript
//! concurrently.
//!
//! Also hosts `launch_app`: the one system-level action available from the
//! overlay's Voice Commands settings (spawning an allowlisted program the
//! user configured ahead of time — see OverlaySettingsPanel.tsx). Kept here
//! since it shares this module's "voice-triggered action" scope, though it
//! runs independent of whether mic mode is on.

use std::sync::Mutex;
use std::thread::JoinHandle;

use tauri::{Emitter, Manager, State};

use crate::audio::{AudioSource, MicrophoneCapture, StopSignal};
use crate::state::AppState;

#[derive(Default)]
struct MicSessionHandles {
    stop_signal: Option<StopSignal>,
    mic_thread: Option<JoinHandle<()>>,
}

#[derive(Default)]
pub struct MicAssistantSession(Mutex<MicSessionHandles>);

#[tauri::command]
pub fn start_mic_assistant(
    app: tauri::AppHandle,
    session: State<'_, MicAssistantSession>,
) -> Result<(), String> {
    let mut guard = session.0.lock().map_err(|e| e.to_string())?;
    if guard.stop_signal.is_some() {
        return Err("mic assistant already running".into());
    }

    let (audio_tx, audio_rx) = crossbeam_channel::unbounded();
    let (stt_tx, stt_rx) = std::sync::mpsc::channel();
    let stop = StopSignal::new();

    let mic_thread: JoinHandle<()> = MicrophoneCapture::start(audio_tx, stop.clone())?;
    let stt_num_threads = crate::hardware::effective_config_checked(&app).stt_num_threads;
    let mut sidecar = crate::stt::SttSidecar::spawn(
        AudioSource::Microphone,
        stt_tx,
        Some(stt_num_threads),
        Some(&app),
    )?;

    // Same shape as start_capture_inner's events-forwarder thread: apply
    // each STT event to the shared TranscriptManager and emit the resulting
    // segment as "transcript:update" — the overlay's existing listener
    // already knows how to turn a segment into question-box text and (with
    // Auto AI on) an automatic ask.
    let app_for_events = app.clone();
    std::thread::Builder::new()
        .name("mic-assistant-events".into())
        .spawn(move || {
            for event in stt_rx.iter() {
                let state = app_for_events.state::<AppState>();
                let segment = state
                    .transcript
                    .lock()
                    .ok()
                    .and_then(|mut manager| manager.apply_event(event));
                if let Some(segment) = segment {
                    let _ = app_for_events.emit("transcript:update", &segment);
                }
            }
        })
        .map_err(|e| e.to_string())?;

    std::thread::Builder::new()
        .name("mic-assistant-pump".into())
        .spawn(move || {
            for chunk in audio_rx.iter() {
                if let Err(err) = sidecar.send_samples(&chunk.samples) {
                    log::warn!("mic assistant: failed to send audio to STT sidecar: {err}");
                }
            }
            let _ = sidecar.flush();
            sidecar.shutdown();
        })
        .map_err(|e| e.to_string())?;

    guard.stop_signal = Some(stop);
    guard.mic_thread = Some(mic_thread);

    Ok(())
}

#[tauri::command]
pub fn stop_mic_assistant(session: State<'_, MicAssistantSession>) -> Result<(), String> {
    let (stop_signal, mic_thread) = {
        let mut guard = session.0.lock().map_err(|e| e.to_string())?;
        (guard.stop_signal.take(), guard.mic_thread.take())
    };
    let Some(stop) = stop_signal else {
        return Err("mic assistant is not running".into());
    };
    stop.stop();
    if let Some(handle) = mic_thread {
        let _ = handle.join();
    }
    Ok(())
}

/// One allowlisted app entry, defined by the user in Settings — `path` is
/// whatever they typed there (an exe name resolvable on PATH, or a full
/// path), never text derived from speech.
#[tauri::command]
pub fn launch_app(path: String) -> Result<(), String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no path configured for this app".into());
    }
    std::process::Command::new(trimmed)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("failed to launch '{trimmed}': {e}"))
}
