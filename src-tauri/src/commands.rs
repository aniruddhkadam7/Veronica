use std::io::Write;

use tauri::{Emitter, Manager, State};

use crate::audio::{
    AudioDeviceInfo, AudioDeviceManager, AudioSource, PauseSignal, StopSignal, SystemAudioCapture,
};
use crate::backend::{InterviewAnalysisRequest, SetupAnalysisRequest, SetupAnalysisResponse};
use crate::rag::{DocumentMetadata, KnowledgeBaseStatus, RagClient, SearchResponse};
use crate::state::AppState;
use crate::stt::SttSidecar;
use crate::transcript::{InterviewSession, RecordingState, TranscriptSegment};

// `pub(crate)`, not private: `voice_command::mod` also emits this same event
// shape from the mic-assistant pump (see its doc), so both capture paths
// (system-audio here, mic-assistant there) produce one consistent
// `audio:level` event the frontend can listen to regardless of source.
#[derive(Clone, serde::Serialize)]
pub(crate) struct AudioLevelEvent {
    pub(crate) source: AudioSource,
    pub(crate) rms_level: f32,
}

#[derive(Clone, serde::Serialize)]
struct RecordingStateEvent {
    state: RecordingState,
}

#[tauri::command]
pub fn list_output_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    AudioDeviceManager::list_output_devices()
}

#[tauri::command]
pub fn list_input_devices() -> Result<Vec<AudioDeviceInfo>, String> {
    AudioDeviceManager::list_input_devices()
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SelectedDevicesInfo {
    /// `None` means "use the system default" — see `state::SelectedDevices`'s doc.
    input_id: Option<String>,
    output_id: Option<String>,
}

#[tauri::command]
pub fn get_selected_devices(state: State<'_, AppState>) -> Result<SelectedDevicesInfo, String> {
    Ok(SelectedDevicesInfo {
        input_id: state.selected_devices.input(),
        output_id: state.selected_devices.output(),
    })
}

/// Sets the microphone Veronica listens through. `device_id` is a WASAPI
/// endpoint ID from `list_input_devices` (or `None`/omitted for "system
/// default"). Takes effect the next time the mic assistant/dictation session
/// starts — see `state::SelectedDevices`'s doc; it does not restart an
/// already-running session.
#[tauri::command]
pub fn set_input_device(state: State<'_, AppState>, device_id: Option<String>) -> Result<(), String> {
    state.selected_devices.set_input(device_id);
    Ok(())
}

/// Sets the speaker Veronica's voice plays through. `device_id` is the
/// output device's friendly name from `list_output_devices` (matched by name
/// against `cpal::Device::name()` — see `tts::player::open_output_stream`
/// for why output uses a name match while input uses a WASAPI endpoint ID),
/// or `None`/omitted for "system default". Takes effect the next time a TTS
/// session opens (session start, or the next answer if none is open yet).
#[tauri::command]
pub fn set_output_device(state: State<'_, AppState>, device_id: Option<String>) -> Result<(), String> {
    state.selected_devices.set_output(device_id);
    Ok(())
}

#[derive(Clone, serde::Serialize)]
pub struct TtsVoiceInfo {
    model: &'static str,
    label: &'static str,
}

/// The full Deepgram Flux voice catalog for the Settings -> Audio voice
/// picker — see `tts::deepgram_flux::VOICES`.
#[tauri::command]
pub fn list_tts_voices() -> Vec<TtsVoiceInfo> {
    crate::tts::deepgram_flux::VOICES
        .iter()
        .map(|v| TtsVoiceInfo { model: v.model, label: v.label })
        .collect()
}

#[tauri::command]
pub fn get_selected_voice(state: State<'_, AppState>) -> String {
    state.selected_voice.get()
}

/// Sets the Flux voice Veronica speaks with. `voice_model` must be one of
/// `list_tts_voices`' `model` values. Takes effect the next time a TTS
/// session opens (mirrors `set_output_device`) — an already-open Flux
/// session keeps using the voice it connected with.
#[tauri::command]
pub fn set_tts_voice(state: State<'_, AppState>, voice_model: String) -> Result<(), String> {
    if !crate::tts::deepgram_flux::is_known_voice(&voice_model) {
        return Err(format!("unknown Flux voice model: {voice_model}"));
    }
    state.selected_voice.set(voice_model);
    Ok(())
}

/// Line spoken by the Settings -> Audio voice picker's per-voice preview
/// button — short enough to finish quickly, and generic enough that it makes
/// sense read in any of Flux's 36 voices. Says whichever name that specific
/// voice would actually self-identify with (`assistant_name_for_voice`:
/// "Veronica" for a female voice, "Mark" for a male one), so the preview
/// matches what the assistant would really say once that voice is selected.
fn voice_preview_text(voice_model: &str) -> String {
    let name = crate::tts::deepgram_flux::assistant_name_for_voice(voice_model);
    format!("Hi, I'm {name}. This is how I sound.")
}

/// Speaks this voice's preview line (see `voice_preview_text`) with
/// `voice_model`, on a throwaway TTS session (its own audio device handle
/// and Flux connection, not `AppState.tts` — the app's persistent,
/// cross-turn session tied to whichever voice is actually selected) so
/// previewing a voice never disturbs an in-progress real answer or leaves
/// the persistent session pinned to a voice the user only auditioned but
/// didn't pick. Runs on a background thread and returns immediately:
/// playback finishes on its own, there is nothing for the caller to await.
#[tauri::command]
pub fn preview_tts_voice(app: tauri::AppHandle, voice_model: String) -> Result<(), String> {
    if !crate::tts::deepgram_flux::is_known_voice(&voice_model) {
        return Err(format!("unknown Flux voice model: {voice_model}"));
    }
    let preview_text = voice_preview_text(&voice_model);
    std::thread::Builder::new()
        .name("tts-voice-preview".into())
        .spawn(move || {
            let speaking = crate::tts::TtsSpeakingSignal::default();
            match crate::tts::TtsSession::start_with_voice(speaking, Some(app), voice_model) {
                Ok(session) => {
                    session.begin_turn();
                    session.speak_now(&preview_text);
                }
                Err(err) => log::warn!("Voice preview: failed to start TTS session: {err}"),
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// How long a Start request will wait for a previous session's teardown
/// (`RecordingState::Stopping`) to finish before giving up, when it lands in
/// that narrow window rather than seeing a clean `Idle`/`Stopped`. Bounded
/// and polled once per interval — never an unbounded/forever retry — so a
/// genuinely stuck teardown still surfaces a clear error instead of hanging
/// the caller.
const STOPPING_WAIT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
const STOPPING_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

/// Starts the full local pipeline: WASAPI system-audio capture -> STT
/// sidecar -> TranscriptManager -> `transcript:update` / `audio:level` events to
/// the frontend. No network call happens here; the backend is only contacted later
/// when the user explicitly clicks "Analyze Interview" (see docs/architecture.md).
///
/// Deterministic lifecycle: `Idle`/`Stopped` -> `Starting` -> (STT sidecar
/// confirmed ready -> audio capture confirmed initialized) -> `Recording`,
/// or -> `Idle` with a real error on any failure along the way, with
/// whatever was already spawned cleaned up before returning. The `Starting`
/// state itself is what makes a second, concurrent Start request harmless
/// (it is rejected immediately with the same "capture already running"
/// every caller already treats as expected — see e.g.
/// `InterviewOverlay.tsx`) instead of racing a second WASAPI/STT startup
/// against the first. A request that instead lands mid-`Stopping` (a
/// previous session's teardown still in flight) waits briefly for it to
/// finish rather than starting a new session while the old one's
/// device/process handles may still be live. See
/// `docs/performance-tuning.md`'s STT-start-reliability section for the
/// full root-cause writeup this replaces, and its real-i3-latency-diagnosis
/// section for why the sidecar is started before audio capture rather than
/// after.
#[tauri::command]
pub fn start_system_audio_capture(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
) -> Result<(), String> {
    log::info!("start_system_audio_capture: invoked");

    let mut waited = std::time::Duration::ZERO;
    loop {
        let mut session = state.capture.lock().map_err(|e| e.to_string())?;
        match session.recording_state {
            RecordingState::Idle | RecordingState::Stopped => {
                session.recording_state = RecordingState::Starting;
                break;
            }
            RecordingState::Starting | RecordingState::Recording | RecordingState::Paused => {
                // The exact string every frontend call site already treats
                // as a harmless "someone else already started it" outcome —
                // see e.g. InterviewOverlay.tsx's mount effect, which races
                // this exact call against InterviewSetup's.
                return Err("capture already running".into());
            }
            RecordingState::Stopping => {
                if waited >= STOPPING_WAIT_TIMEOUT {
                    return Err(
                        "a previous session is still shutting down — try again in a moment".into(),
                    );
                }
                drop(session);
                std::thread::sleep(STOPPING_POLL_INTERVAL);
                waited += STOPPING_POLL_INTERVAL;
                continue;
            }
        }
    }
    let _ = app.emit("recording:state", RecordingStateEvent { state: RecordingState::Starting });

    let result = start_capture_inner(&app, &state);

    match &result {
        Ok(()) => {
            let mut session = state.capture.lock().map_err(|e| e.to_string())?;
            session.recording_state = RecordingState::Recording;
            drop(session);
            let _ = app.emit("recording:state", RecordingStateEvent { state: RecordingState::Recording });
            log::info!("start_system_audio_capture: recording started");
        }
        Err(err) => {
            let mut session = state.capture.lock().map_err(|e| e.to_string())?;
            session.recording_state = RecordingState::Idle;
            session.stop_signal = None;
            session.pause_signal = None;
            session.system_audio_thread = None;
            session.mic_thread = None;
            session.pipeline_thread = None;
            drop(session);
            log::error!("start_system_audio_capture: failed to start: {err}");
            let _ = app.emit("recording:state", RecordingStateEvent { state: RecordingState::Idle });
        }
    }

    result
}

/// The actual startup sequence, factored out of `start_system_audio_capture`
/// so the state-machine bookkeeping (guard, rollback, event emission) around
/// it stays in one place regardless of which step fails. Everything spawned
/// here that later steps depend on is cleaned up on its own failure path
/// before returning `Err` — no partially-started session is ever left
/// behind for the caller to roll back blindly.
fn start_capture_inner(app: &tauri::AppHandle, state: &State<'_, AppState>) -> Result<(), String> {
    {
        let mut transcript = state.transcript.lock().map_err(|e| e.to_string())?;
        transcript.clear();
    }

    let (stt_tx, stt_rx) = std::sync::mpsc::channel();
    let stop = StopSignal::new();
    let pause = PauseSignal::new();

    let session_start = crate::hardware::telemetry::Stopwatch::start();

    // STT/RAG scheduling coordination (Phase B): signal RAG indexing to
    // yield BEFORE spawning the sidecar, not after — the sidecar's own
    // model-load is itself the most CPU/memory-constrained moment of the
    // whole startup sequence on Entry/Standard-tier hardware, and it used
    // to run fully unthrottled while RAG's background indexing/embedding
    // competed with it for the same scarce resources. No-op on
    // Performance/HighPerformance tier or when Adaptive doesn't call for
    // throttling — see hardware::stt_rag_coordination's module doc.
    crate::hardware::stt_rag_coordination::on_stt_session_started(app);

    // `_checked`: this is a natural checkpoint (recording session start) —
    // feeds a fresh RAM reading into the sustained-pressure tracker, which
    // may clamp `stt_num_threads` down if the machine has been under
    // memory pressure across the last couple of checkpoints (see
    // hardware::pressure). Any resulting downgrade/restoration is logged
    // from within effective_config_checked itself.
    let stt_num_threads = crate::hardware::effective_config_checked(app).stt_num_threads;

    // The STT sidecar is spawned BEFORE audio capture starts — deliberately
    // reversed from the original order. `SttSidecar::spawn` blocks until the
    // model has finished loading and the sidecar signals ready (see
    // stt/sidecar.rs). Starting audio capture first meant WASAPI spent that
    // entire model-load window pushing chunks into an unbounded channel with
    // nothing yet consuming them — on slow hardware (e.g. an i3/8GB machine,
    // where model load can take several seconds instead of the ~100ms seen
    // on faster dev hardware) that produced a real backlog: silence for a
    // few seconds, then the backlogged audio decoded in a burst the instant
    // the sidecar came up, before settling into real-time. Starting the
    // sidecar first means no audio channel — and so no backlog — exists
    // until STT is already able to keep up with it. See
    // `docs/performance-tuning.md`'s real-i3-latency-diagnosis section.
    let sidecar = match SttSidecar::spawn(AudioSource::SystemAudio, stt_tx, Some(stt_num_threads), Some(app)) {
        Ok(sidecar) => sidecar,
        Err(err) => {
            // Audio capture hasn't started yet at this point — nothing to
            // stop/join here, only the RAG throttle signal to undo before
            // surfacing the error.
            crate::hardware::stt_rag_coordination::on_stt_session_ended(app);
            return Err(err);
        }
    };
    // Each log site reads a fresh `perf_context(app)` rather than one
    // captured up front — `effective_config_checked` above is what actually
    // feeds this session's RAM reading into the pressure tracker, so a
    // context captured before that checkpoint would log the previous
    // session's (possibly stale) pressure state / thread count instead of
    // the one actually in effect for this session.
    crate::hardware::telemetry::log_stage_ms(
        crate::hardware::telemetry::PipelineStage::SttReady,
        session_start.elapsed().as_millis(),
        &crate::hardware::perf_context(app),
    );

    // Audio capture starts only now that the sidecar is confirmed ready to
    // consume it — no window remains during which chunks can queue up
    // unread.
    let (audio_tx, audio_rx) = crossbeam_channel::unbounded();
    // TEMP DIAGNOSTIC: a sender clone purely for polling backlog depth
    // (`.len()`) — crossbeam senders/receivers are safe to clone for a
    // second producer, but never clone the *receiver* for this purpose, as
    // that would steal chunks meant for the real pipeline. Expected to stay
    // ~0 now that capture only starts once STT is ready.
    let audio_tx_for_diag = audio_tx.clone();

    let audio_thread = match SystemAudioCapture::start(audio_tx, stop.clone()) {
        Ok(handle) => handle,
        Err(err) => {
            // The sidecar is already spawned (and ready) at this point —
            // shut it down cleanly and undo the throttle signal before
            // surfacing the error, so a failed Start never leaves an
            // orphaned STT process running.
            sidecar.shutdown();
            crate::hardware::stt_rag_coordination::on_stt_session_ended(app);
            return Err(err);
        }
    };
    crate::hardware::telemetry::log_stage_ms(
        crate::hardware::telemetry::PipelineStage::AudioCaptureReady,
        session_start.elapsed().as_millis(),
        &crate::hardware::perf_context(app),
    );
    // By this point the whole startup sequence — STT model load AND audio
    // capture init — is confirmed complete.
    crate::hardware::telemetry::finish(
        session_start,
        crate::hardware::telemetry::PipelineStage::SttSessionStart,
        &crate::hardware::perf_context(app),
    );

    let app_for_pipeline = app.clone();
    let pause_for_pipeline = pause.clone();
    let tts_speaking_for_pipeline = state.tts_speaking.clone();

    // ===== TEMP DIAGNOSTIC INSTRUMENTATION — STT latency investigation =====
    // Gated behind STT_DIAG=1 (OFF by default): read-only logging, changes no
    // STT/RAG/VAD/endpointing behavior. For removal after the investigation
    // — see docs/performance-tuning.md's real-i3-latency-diagnosis section.
    let diag_enabled = std::env::var("STT_DIAG").map(|v| v == "1").unwrap_or(false);
    let diag_start = std::time::Instant::now();
    if diag_enabled {
        let diag_stop = stop.clone();
        let diag_app = app.clone();
        std::thread::Builder::new()
            .name("diag-resource-sampler".into())
            .spawn(move || {
                let mut sys = sysinfo::System::new_all();
                while !diag_stop.is_stopped() {
                    std::thread::sleep(std::time::Duration::from_millis(1000));
                    sys.refresh_cpu_usage();
                    let cpu = sys.global_cpu_usage();
                    let avail = crate::hardware::available_ram_mb();
                    let cfg = crate::hardware::effective_config_checked(&diag_app);
                    let pressure = crate::hardware::perf_context(&diag_app).pressure;
                    log::info!(
                        "DIAG t={}ms cpu={:.1}% avail_ram_mb={} stt_threads={} pressure={:?}",
                        diag_start.elapsed().as_millis(),
                        cpu,
                        avail,
                        cfg.stt_num_threads,
                        pressure,
                    );
                }
            })
            .ok();
    }
    // ===== END TEMP DIAGNOSTIC (resource sampler) =====

    let pipeline_thread = std::thread::Builder::new()
        .name("audio-stt-pipeline".into())
        .spawn(move || {
            // Forward audio chunks to the sidecar and levels to the UI, on this
            // thread, while a second thread below drains STT events. Two separate
            // threads are required so writing audio to the sidecar's stdin never
            // blocks waiting for its stdout to be read (see docs/progress.md Step 4).
            let stt_events_app = app_for_pipeline.clone();
            let events_thread = std::thread::Builder::new()
                .name("stt-events-forwarder".into())
                .spawn(move || {
                    // First-partial/first-final latency, measured from this
                    // forwarder thread's own start (which begins essentially
                    // at session start) to the first occurrence of each
                    // event kind — the number the user actually experiences
                    // (audio flowing -> text appearing), not a cross-process
                    // clock reconciliation with the Python sidecar's own
                    // monotonic timestamps.
                    let stt_clock = crate::hardware::telemetry::Stopwatch::start();
                    let mut first_partial_logged = false;
                    let mut first_final_logged = false;

                    for event in stt_rx.iter() {
                        // ===== TEMP DIAGNOSTIC: every partial/final, not just
                        // the first — text length only, never content, per
                        // telemetry.rs's existing no-content-logging rule.
                        if diag_enabled {
                            log::info!(
                                "DIAG t={}ms stt_event kind={:?} text_len={}",
                                diag_start.elapsed().as_millis(),
                                event.kind,
                                event.text.len(),
                            );
                        }
                        // ===== END TEMP DIAGNOSTIC =====

                        if !first_partial_logged && event.kind == crate::stt::SttEventKind::Partial {
                            first_partial_logged = true;
                            crate::hardware::telemetry::log_stage_ms(
                                crate::hardware::telemetry::PipelineStage::SttFirstPartial,
                                stt_clock.elapsed().as_millis(),
                                &crate::hardware::perf_context(&stt_events_app),
                            );
                        }
                        if !first_final_logged && event.kind == crate::stt::SttEventKind::Final {
                            first_final_logged = true;
                            crate::hardware::telemetry::log_stage_ms(
                                crate::hardware::telemetry::PipelineStage::SttFinal,
                                stt_clock.elapsed().as_millis(),
                                &crate::hardware::perf_context(&stt_events_app),
                            );
                        }

                        let state = stt_events_app.state::<AppState>();
                        let segment = state
                            .transcript
                            .lock()
                            .ok()
                            .and_then(|mut manager| manager.apply_event(event));
                        if let Some(segment) = segment {
                            let _ = stt_events_app.emit("transcript:update", &segment);
                        }
                    }
                })
                .ok();

            // ===== TEMP DIAGNOSTIC: RMS-threshold speech onset/offset
            // detector, purely observational (does NOT feed VAD/endpointing
            // — those stay untouched in Python), plus outbound channel
            // backlog at each transition, to see whether audio is queuing up
            // behind STT rather than being delivered promptly.
            let diag_audio_tx = audio_tx_for_diag.clone();
            let mut diag_speaking = false;
            // ===== END TEMP DIAGNOSTIC (setup) =====

            // The loop itself lives in `audio::pipeline` so that the headless
            // `pipeline_test` binary drives the same code rather than a copy.
            crate::audio::run_stt_pipeline(audio_rx, sidecar, pause_for_pipeline, Some(tts_speaking_for_pipeline), move |chunk| {
                let _ = app_for_pipeline.emit(
                    "audio:level",
                    AudioLevelEvent {
                        source: chunk.source,
                        rms_level: chunk.rms_level,
                    },
                );
                // ===== TEMP DIAGNOSTIC =====
                if diag_enabled {
                    const SPEECH_RMS_THRESHOLD: f32 = 0.02;
                    let is_speech = chunk.rms_level > SPEECH_RMS_THRESHOLD;
                    if is_speech && !diag_speaking {
                        diag_speaking = true;
                        log::info!(
                            "DIAG t={}ms speech_onset rms={:.4} backlog={}",
                            diag_start.elapsed().as_millis(),
                            chunk.rms_level,
                            diag_audio_tx.len(),
                        );
                    } else if !is_speech && diag_speaking {
                        diag_speaking = false;
                        log::info!(
                            "DIAG t={}ms speech_offset rms={:.4} backlog={}",
                            diag_start.elapsed().as_millis(),
                            chunk.rms_level,
                            diag_audio_tx.len(),
                        );
                    }
                }
                // ===== END TEMP DIAGNOSTIC =====
            });

            if let Some(handle) = events_thread {
                let _ = handle.join();
            }
        })
        .map_err(|e| e.to_string())?;

    let mut session = state.capture.lock().map_err(|e| e.to_string())?;
    session.stop_signal = Some(stop);
    session.pause_signal = Some(pause);
    session.system_audio_thread = Some(audio_thread);
    session.pipeline_thread = Some(pipeline_thread);
    Ok(())
}

/// Pauses recording: audio capture keeps running internally (so the OS device
/// buffer never overflows) but no audio reaches the STT sidecar or the UI meter,
/// and no transcript segment changes. The sidecar process is left running so
/// resuming is instant and the transcript-so-far is untouched.
#[tauri::command]
pub fn pause_recording(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("pause_recording: invoked");
    let mut session = state.capture.lock().map_err(|e| e.to_string())?;
    let Some(pause) = session.pause_signal.as_ref() else {
        return Err("no active recording to pause".into());
    };
    if session.recording_state != RecordingState::Recording {
        return Err(format!(
            "cannot pause from state {:?}",
            session.recording_state
        ));
    }
    pause.set_paused(true);
    session.recording_state = RecordingState::Paused;
    drop(session);

    {
        let mut transcript = state.transcript.lock().map_err(|e| e.to_string())?;
        transcript.mark_paused();
    }

    let _ = app.emit(
        "recording:state",
        RecordingStateEvent {
            state: RecordingState::Paused,
        },
    );
    log::info!("pause_recording: done");
    Ok(())
}

#[tauri::command]
pub fn resume_recording(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("resume_recording: invoked");
    let mut session = state.capture.lock().map_err(|e| e.to_string())?;
    let Some(pause) = session.pause_signal.as_ref() else {
        return Err("no active recording to resume".into());
    };
    if session.recording_state != RecordingState::Paused {
        return Err(format!(
            "cannot resume from state {:?}",
            session.recording_state
        ));
    }
    pause.set_paused(false);
    session.recording_state = RecordingState::Recording;
    drop(session);

    {
        let mut transcript = state.transcript.lock().map_err(|e| e.to_string())?;
        transcript.mark_resumed();
    }

    let _ = app.emit(
        "recording:state",
        RecordingStateEvent {
            state: RecordingState::Recording,
        },
    );
    log::info!("resume_recording: done");
    Ok(())
}

/// Stops recording: halts WASAPI capture, flushes and finalizes the STT sidecar
/// (so any in-progress utterance is finalized rather than dropped), and marks the
/// session as completed. Does not contact the backend — that only happens if/when
/// the user explicitly clicks "Analyze Interview".
#[tauri::command]
pub fn stop_audio_capture(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    log::info!("stop_audio_capture: invoked");
    // Take the handles out and drop the lock *before* joining threads below —
    // joining can block for as long as the STT sidecar takes to flush and exit,
    // and holding `state.capture` for that whole span would deadlock every other
    // recording command (get_recording_state, pause/resume, a fresh start) that
    // also needs this same mutex while Stop is still in flight. The
    // `Stopping` state set here (rather than jumping straight to `Stopped`)
    // is what a concurrent Start request sees and waits on instead of
    // racing a new WASAPI/STT session against this teardown's still-live
    // device/process handles — see `start_system_audio_capture`.
    let (pause_signal, stop_signal, system_audio_thread, mic_thread, pipeline_thread) = {
        let mut session = state.capture.lock().map_err(|e| e.to_string())?;
        session.recording_state = RecordingState::Stopping;
        (
            session.pause_signal.take(),
            session.stop_signal.take(),
            session.system_audio_thread.take(),
            session.mic_thread.take(),
            session.pipeline_thread.take(),
        )
    };
    let _ = app.emit("recording:state", RecordingStateEvent { state: RecordingState::Stopping });

    if let Some(pause) = pause_signal {
        // Make sure a paused pipeline thread doesn't sit forever skipping chunks;
        // stopping the capture thread below closes its channel either way, but
        // un-pausing first ensures any final in-flight chunk still reaches STT.
        pause.set_paused(false);
    }
    if let Some(stop) = stop_signal {
        stop.stop();
    }
    if let Some(handle) = system_audio_thread {
        let _ = handle.join();
    }
    if let Some(handle) = mic_thread {
        let _ = handle.join();
    }
    if let Some(handle) = pipeline_thread {
        let _ = handle.join();
    }

    {
        let mut transcript = state.transcript.lock().map_err(|e| e.to_string())?;
        transcript.mark_stopped();
    }

    {
        let mut session = state.capture.lock().map_err(|e| e.to_string())?;
        session.recording_state = RecordingState::Stopped;
    }

    // STT/RAG scheduling coordination (Phase B): releases the throttle this
    // session may have activated. Safe no-op if it never did (different
    // tier/mode, or this call races a start that hasn't incremented yet —
    // see on_stt_session_ended's own guard against underflow).
    crate::hardware::stt_rag_coordination::on_stt_session_ended(&app);

    let _ = app.emit(
        "recording:state",
        RecordingStateEvent {
            state: RecordingState::Stopped,
        },
    );
    log::info!("stop_audio_capture: done");
    Ok(())
}

#[tauri::command]
pub fn get_current_session(state: State<'_, AppState>) -> Result<InterviewSession, String> {
    let transcript = state.transcript.lock().map_err(|e| e.to_string())?;
    Ok(transcript.session().clone())
}

#[tauri::command]
pub fn get_recording_state(state: State<'_, AppState>) -> Result<RecordingState, String> {
    let session = state.capture.lock().map_err(|e| e.to_string())?;
    Ok(session.recording_state)
}

/// Resets everything to start a brand new interview: any previous recording must
/// already be stopped (recording_state must not be Recording/Paused).
#[tauri::command]
pub fn start_new_interview(state: State<'_, AppState>) -> Result<(), String> {
    let mut session = state.capture.lock().map_err(|e| e.to_string())?;
    if matches!(
        session.recording_state,
        RecordingState::Recording | RecordingState::Paused | RecordingState::Starting | RecordingState::Stopping
    ) {
        return Err("stop the current recording before starting a new interview".into());
    }
    session.recording_state = RecordingState::Idle;
    session.pause_signal = None;
    session.stop_signal = None;
    drop(session);

    let mut transcript = state.transcript.lock().map_err(|e| e.to_string())?;
    transcript.clear();
    Ok(())
}

#[tauri::command]
pub fn clear_transcript(state: State<'_, AppState>) -> Result<(), String> {
    let mut transcript = state.transcript.lock().map_err(|e| e.to_string())?;
    transcript.clear();
    Ok(())
}

fn format_hh_mm_ss(total_ms: u64) -> String {
    let total_secs = total_ms / 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn iso8601(ms: u64) -> String {
    // Minimal dependency-free ISO-8601-ish UTC timestamp (date is not computed;
    // we keep this local-storage-only export simple and avoid pulling in a full
    // date/time crate just for file headers/JSON export metadata).
    let secs = ms / 1000;
    let millis = ms % 1000;
    format!("{secs}.{millis:03}Z")
}

/// Opens a native "Save As" dialog and writes the current transcript as plain
/// text. Runs entirely locally — no network call.
#[tauri::command]
pub fn export_transcript_txt(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<Option<String>, String> {
    let session = {
        let transcript = state.transcript.lock().map_err(|e| e.to_string())?;
        transcript.session().clone()
    };

    let mut out = String::new();
    out.push_str("INTERVIEW TRANSCRIPT\n");
    out.push_str(&format!("Session ID: {}\n", session.id));
    out.push_str(&format!("Started: {}\n", iso8601(session.started_at_ms)));
    if let Some(ended) = session.ended_at_ms {
        out.push_str(&format!("Ended: {}\n", iso8601(ended)));
    }
    out.push_str(&format!("Duration: {}\n", format_hh_mm_ss(session.elapsed_ms())));
    out.push_str(&format!("Words: {}\n\n", session.word_count()));

    for segment in &session.segments {
        let Some(text) = segment.final_text.as_deref() else {
            continue;
        };
        let relative_ms = segment
            .start_time
            .map(|s| (s * 1000.0) as u64)
            .unwrap_or(0);
        out.push_str(&format!("[{}]\n", format_hh_mm_ss(relative_ms)));
        out.push_str(text);
        out.push_str("\n\n");
    }

    let default_name = format!("interview-transcript-{}.txt", session.id);
    let Some(path) = pick_save_path(&app, &default_name, "Text", &["txt"])? else {
        return Ok(None);
    };
    write_file(&path, out.as_bytes())?;
    Ok(Some(path))
}

/// Opens a native "Save As" dialog and writes the current transcript as
/// structured JSON. Runs entirely locally — no network call.
#[tauri::command]
pub fn export_transcript_json(app: tauri::AppHandle, state: State<'_, AppState>) -> Result<Option<String>, String> {
    let session = {
        let transcript = state.transcript.lock().map_err(|e| e.to_string())?;
        transcript.session().clone()
    };

    #[derive(serde::Serialize)]
    struct ExportSegment {
        timestamp: String,
        source: AudioSource,
        text: String,
    }

    #[derive(serde::Serialize)]
    struct ExportPayload {
        #[serde(rename = "sessionId")]
        session_id: String,
        #[serde(rename = "startedAt")]
        started_at: String,
        #[serde(rename = "endedAt")]
        ended_at: Option<String>,
        #[serde(rename = "durationMs")]
        duration_ms: u64,
        #[serde(rename = "wordCount")]
        word_count: usize,
        segments: Vec<ExportSegment>,
    }

    let payload = ExportPayload {
        session_id: session.id.clone(),
        started_at: iso8601(session.started_at_ms),
        ended_at: session.ended_at_ms.map(iso8601),
        duration_ms: session.elapsed_ms(),
        word_count: session.word_count(),
        segments: session
            .segments
            .iter()
            .filter_map(|s| {
                s.final_text.as_ref().map(|text| ExportSegment {
                    timestamp: iso8601(s.timestamp),
                    source: s.source,
                    text: text.clone(),
                })
            })
            .collect(),
    };

    let json = serde_json::to_string_pretty(&payload).map_err(|e| e.to_string())?;

    let default_name = format!("interview-transcript-{}.json", session.id);
    let Some(path) = pick_save_path(&app, &default_name, "JSON", &["json"])? else {
        return Ok(None);
    };
    write_file(&path, json.as_bytes())?;
    Ok(Some(path))
}

fn pick_save_path(
    app: &tauri::AppHandle,
    default_name: &str,
    filter_name: &str,
    extensions: &[&str],
) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;

    let path = app
        .dialog()
        .file()
        .set_file_name(default_name)
        .add_filter(filter_name, extensions)
        .blocking_save_file();

    Ok(path.map(|p| p.to_string()))
}

fn write_file(path: &str, contents: &[u8]) -> Result<(), String> {
    let mut file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    file.write_all(contents).map_err(|e| e.to_string())?;
    Ok(())
}

/// Personal build: there is no SaaS backend to be connected to. Kept as a
/// permanent `false` no-op rather than removed, in case any future UI code
/// polls it for a connectivity indicator.
#[tauri::command]
pub async fn check_backend_connection() -> Result<bool, String> {
    Ok(false)
}

/// Runs the full Step 10 analysis pipeline: extract question/answer pairs
/// from the finalized transcript (deterministic, local, no LLM — see
/// `crate::analyzer`), run a local RAG search for each question (local, see
/// `crate::rag::RetrievalPlanner`), then stream the resulting
/// question+context set to the backend for LLM analysis. Only reachable from
/// the "Analyze Interview" button on the post-stop summary screen — never
/// called automatically when recording stops. Sends text only (transcript-
/// derived question/answer text plus retrieved chunk text); no audio, no
/// original documents, no embeddings, no vector database.
///
/// Streams progress as `analysis:progress` Tauri events (one per SSE event
/// from the backend) so the frontend can show incremental status without the
/// command needing to return until the whole analysis finishes.
#[tauri::command]
pub async fn analyze_interview(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    role: Option<String>,
    company: Option<String>,
    job_description: Option<String>,
) -> Result<crate::backend::OverallInterviewAnalysis, String> {
    let session = {
        let transcript = state.transcript.lock().map_err(|e| e.to_string())?;
        transcript.session().clone()
    };

    if session.segments.is_empty() {
        return Err("cannot analyze an empty transcript".into());
    }

    let qa_pairs = crate::analyzer::extract_question_answers(&session);

    // Tier-driven top_k/similarity_threshold/max_context_chars, but no
    // `.with_timeout()` — this offline analysis path intentionally keeps the
    // client's generous default HTTP budget rather than the tight
    // live-answer timeout, since there's no user waiting on an immediate
    // answer here. `_checked` once here (not per qa_pair in the loop below)
    // — analysis-start is the checkpoint, not each of its many retrievals.
    let cfg = crate::hardware::effective_config_checked(&app);
    let planner = crate::rag::RetrievalPlanner::new().with_config(
        cfg.rag_top_k,
        cfg.rag_similarity_threshold,
        cfg.rag_max_context_chars,
    );
    let mut wire_question_answers = Vec::with_capacity(qa_pairs.len());
    for pair in &qa_pairs {
        let results = planner.plan_for_question(&pair.question).await;
        wire_question_answers.push(crate::backend::WireQuestionAnswer {
            question_id: pair.question_id.clone(),
            question: pair.question.clone(),
            candidate_answer: pair.candidate_answer.clone(),
            timestamp: pair.timestamp.clone(),
            retrieved_context: results
                .into_iter()
                .map(|r| crate::backend::WireRetrievedChunk {
                    text: r.text,
                    source_filename: r.metadata.filename,
                    document_type: r.metadata.document_type,
                    score: r.score,
                })
                .collect(),
        });
    }

    let request = build_analysis_request(&session, role, company, job_description, wire_question_answers);

    let app_for_events = app.clone();
    let on_event = move |event: &crate::backend::AnalysisProgressEvent| {
        let _ = app_for_events.emit("analysis:progress", event);
    };
    crate::personal::DirectLlmClient::new(None)?.analyze_stream(&request, on_event).await
}

fn build_analysis_request(
    session: &InterviewSession,
    role: Option<String>,
    company: Option<String>,
    job_description: Option<String>,
    question_answers: Vec<crate::backend::WireQuestionAnswer>,
) -> InterviewAnalysisRequest {
    let segments = session
        .segments
        .iter()
        .filter_map(|segment| wire_segment(session, segment))
        .collect();

    InterviewAnalysisRequest {
        session_id: session.id.clone(),
        role,
        company,
        job_description,
        candidate_context: None,
        transcript: crate::backend::WireTranscript {
            duration_seconds: session.elapsed_ms() / 1000,
            segments,
        },
        question_answers,
    }
}

fn wire_segment(
    session: &InterviewSession,
    segment: &TranscriptSegment,
) -> Option<crate::backend::WireTranscriptSegment> {
    let text = segment.final_text.clone()?;
    let relative_ms = segment.timestamp.saturating_sub(session.started_at_ms);
    Some(crate::backend::WireTranscriptSegment {
        timestamp: format_relative_timestamp(relative_ms),
        source: match segment.source {
            AudioSource::SystemAudio => crate::backend::WireAudioSource::SystemAudio,
            AudioSource::Microphone => crate::backend::WireAudioSource::Microphone,
        },
        text,
    })
}

fn format_relative_timestamp(ms: u64) -> String {
    let total_secs = ms / 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

fn rag_unavailable_err() -> String {
    "RAG service is not available (it may still be starting up, or the \
     sidecars/rag-lite binary was not found — see README.md's \"Getting the \
     model and sidecar binaries\" section)"
        .to_string()
}

/// Every RAG command below checks `state.rag_service` is `Some` before making
/// any HTTP call — this is what makes "RAG unavailable" a clean user-facing
/// error rather than a raw connection-refused message, and it's also what lets
/// the rest of the app (recording, transcript, export, analyze) keep working
/// normally even if the RAG service's environment was never set up.
fn ensure_rag_available(state: &State<'_, AppState>) -> Result<(), String> {
    let slot = state.rag_service.lock().map_err(|e| e.to_string())?;
    if slot.is_none() {
        return Err(rag_unavailable_err());
    }
    Ok(())
}

#[tauri::command]
pub async fn check_rag_connection(state: State<'_, AppState>) -> Result<bool, String> {
    if ensure_rag_available(&state).is_err() {
        return Ok(false);
    }
    let client = RagClient::new();
    Ok(client.health_check().await.is_ok())
}

/// New Interview setup page's auto-analysis: summarizes pasted/uploaded CV
/// and job-description text into a short "Interview Focus" for the setup
/// page to display, using your own configured AI provider directly — no
/// RAG/embedding involved, so unlike the commands below it does not require
/// the RAG service to be available.
#[tauri::command]
pub async fn analyze_setup_context(
    resume_text: Option<String>,
    job_description_text: Option<String>,
) -> Result<SetupAnalysisResponse, String> {
    let request = SetupAnalysisRequest { resume_text, job_description_text };
    crate::personal::DirectLlmClient::new(None)?.analyze_setup(&request).await
}

#[tauri::command]
pub async fn upload_document(
    state: State<'_, AppState>,
    filename: String,
    bytes: Vec<u8>,
    document_type: String,
    agent_id: Option<String>,
) -> Result<DocumentMetadata, String> {
    ensure_rag_available(&state)?;
    let client = RagClient::new();
    client
        .upload_document(filename, bytes, &document_type, agent_id.as_deref())
        .await
}

/// Polled by the New Interview setup page after uploading a PDF/DOCX resume
/// or job description, so its auto-analysis summary can use the file's real
/// extracted text instead of being limited to pasted/plain-text input.
/// Returns `None` while the RAG service is still extracting/chunking/
/// embedding the document — the frontend treats that as "try again shortly."
#[tauri::command]
pub async fn get_document_text(
    state: State<'_, AppState>,
    document_id: String,
) -> Result<Option<String>, String> {
    ensure_rag_available(&state)?;
    let client = RagClient::new();
    client.get_document_text(&document_id).await
}

#[tauri::command]
pub async fn list_documents(
    state: State<'_, AppState>,
    agent_id: Option<String>,
) -> Result<Vec<DocumentMetadata>, String> {
    ensure_rag_available(&state)?;
    let client = RagClient::new();
    client.list_documents(agent_id.as_deref()).await
}

#[tauri::command]
pub async fn delete_document(state: State<'_, AppState>, document_id: String) -> Result<(), String> {
    ensure_rag_available(&state)?;
    let client = RagClient::new();
    client.delete_document(&document_id).await
}

#[tauri::command]
pub async fn clear_knowledge_base(state: State<'_, AppState>, agent_id: Option<String>) -> Result<(), String> {
    ensure_rag_available(&state)?;
    let client = RagClient::new();
    client.clear_knowledge_base(agent_id.as_deref()).await
}

#[tauri::command]
pub async fn knowledge_base_status(
    state: State<'_, AppState>,
    agent_id: Option<String>,
) -> Result<KnowledgeBaseStatus, String> {
    ensure_rag_available(&state)?;
    let client = RagClient::new();
    client.knowledge_base_status(agent_id.as_deref()).await
}

#[tauri::command]
pub async fn search_knowledge_base(
    state: State<'_, AppState>,
    query: String,
    top_k: u32,
    agent_id: Option<String>,
) -> Result<SearchResponse, String> {
    ensure_rag_available(&state)?;
    let client = RagClient::new();
    client.search(&query, top_k, agent_id.as_deref()).await
}
