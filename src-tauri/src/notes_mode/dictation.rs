//! Notes' voice dictation: mic-only capture (no system audio, no speaker
//! separation) through the same STT sidecar every other mode uses. Finalized
//! text accumulates in a buffer the frontend polls/reads on stop and appends
//! to the active note — this deliberately does not go through
//! `TranscriptManager`/`AppState.transcript`, which is Interview Mode's
//! recording state and must stay untouched and uncontended.

use std::sync::Mutex;
use std::thread::JoinHandle;

use tauri::{Emitter, Manager, State};

use crate::audio::{AudioSource, MicrophoneCapture, StopSignal};
use crate::state::AppState;
use crate::stt::SttSidecar;

#[derive(Default)]
struct DictationHandles {
    stop_signal: Option<StopSignal>,
    mic_thread: Option<JoinHandle<()>>,
    events_thread: Option<JoinHandle<()>>,
}

/// Buffer of finalized dictation text, shared between the events-forwarder
/// thread and the `stop_note_dictation` command. Kept separate from
/// `AppState` since it's plain accumulated text, not session state.
#[derive(Default)]
pub struct DictationBuffer(pub Mutex<String>);

#[derive(Default)]
pub struct DictationSession(Mutex<DictationHandles>);

#[tauri::command]
pub fn start_note_dictation(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    session: State<'_, DictationSession>,
    buffer: State<'_, DictationBuffer>,
) -> Result<(), String> {
    let mut handles = session.0.lock().map_err(|e| e.to_string())?;
    if handles.stop_signal.is_some() {
        return Err("dictation already running".into());
    }

    {
        let mut buf = buffer.0.lock().map_err(|e| e.to_string())?;
        buf.clear();
    }
    {
        let mut capture = state.notes_dictation.lock().map_err(|e| e.to_string())?;
        capture.recording_state = crate::transcript::RecordingState::Recording;
    }

    let (audio_tx, audio_rx) = crossbeam_channel::unbounded();
    let (stt_tx, stt_rx) = std::sync::mpsc::channel();
    let stop = StopSignal::new();

    let session_start = crate::hardware::telemetry::Stopwatch::start();
    let mic_thread = MicrophoneCapture::start(audio_tx, stop.clone())?;
    // `_checked`: same reasoning as start_system_audio_capture — dictation
    // session start is a natural memory-pressure checkpoint.
    let stt_num_threads = crate::hardware::effective_config_checked(&app).stt_num_threads;
    let mut sidecar = SttSidecar::spawn(AudioSource::Microphone, stt_tx, Some(stt_num_threads), Some(&app))?;
    crate::hardware::telemetry::finish(
        session_start,
        crate::hardware::telemetry::PipelineStage::SttSessionStart,
        &crate::hardware::perf_context(&app),
    );
    // STT/RAG scheduling coordination (Phase B): see
    // hardware::stt_rag_coordination's module doc — no-op on
    // Performance/HighPerformance tier or in MaximumPerformance mode.
    crate::hardware::stt_rag_coordination::on_stt_session_started(&app);

    // The events thread re-fetches the managed `DictationBuffer` from the
    // `AppHandle` (rather than capturing `buffer` directly) since Tauri
    // `State` borrows are not `'static` and can't be moved into a thread.
    let app_for_events = app.clone();
    let events_thread = std::thread::Builder::new()
        .name("notes-dictation-events".into())
        .spawn(move || {
            for event in stt_rx.iter() {
                if event.kind == crate::stt::SttEventKind::Final {
                    let state = app_for_events.state::<crate::notes_mode::dictation::DictationBuffer>();
                    if let Ok(mut buf) = state.0.lock() {
                        if !buf.is_empty() {
                            buf.push(' ');
                        }
                        buf.push_str(event.text.trim());
                    }
                    let _ = app_for_events.emit("notes:dictation-final", &event.text);
                } else {
                    let _ = app_for_events.emit("notes:dictation-partial", &event.text);
                }
            }
        })
        .map_err(|e| e.to_string())?;

    // Drain audio in a dedicated thread so writing to the sidecar's stdin
    // never blocks on stdout being read, matching the main recording
    // pipeline's threading (see commands::start_system_audio_capture).
    std::thread::Builder::new()
        .name("notes-dictation-pump".into())
        .spawn(move || {
            for chunk in audio_rx.iter() {
                if let Err(err) = sidecar.send_samples(&chunk.samples) {
                    log::warn!("notes dictation: failed to send audio to STT sidecar: {err}");
                }
            }
            let _ = sidecar.flush();
            sidecar.shutdown();
        })
        .map_err(|e| e.to_string())?;

    handles.stop_signal = Some(stop);
    handles.mic_thread = Some(mic_thread);
    handles.events_thread = Some(events_thread);

    Ok(())
}

/// Stops dictation and returns the accumulated finalized text so the
/// frontend can append it to the active note.
#[tauri::command]
pub fn stop_note_dictation(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    session: State<'_, DictationSession>,
    buffer: State<'_, DictationBuffer>,
) -> Result<String, String> {
    let (stop_signal, mic_thread) = {
        let mut handles = session.0.lock().map_err(|e| e.to_string())?;
        (handles.stop_signal.take(), handles.mic_thread.take())
    };

    if let Some(stop) = stop_signal {
        stop.stop();
    }
    if let Some(handle) = mic_thread {
        let _ = handle.join();
    }

    {
        let mut capture = state.notes_dictation.lock().map_err(|e| e.to_string())?;
        capture.recording_state = crate::transcript::RecordingState::Stopped;
    }

    // STT/RAG scheduling coordination (Phase B): releases this session's
    // throttle contribution. See hardware::stt_rag_coordination's module
    // doc — safe no-op if this session never activated it.
    crate::hardware::stt_rag_coordination::on_stt_session_ended(&app);

    let text = {
        let buf = buffer.0.lock().map_err(|e| e.to_string())?;
        buf.clone()
    };
    Ok(text)
}
