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

use std::sync::Mutex;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use tauri::{Emitter, Manager, State};

use crate::audio::{compute_rms, AudioSource, MicrophoneCapture, StopSignal};
use crate::state::AppState;

/// RMS level (0.0-1.0, see `audio::compute_rms`) a mic chunk must clear to
/// count as "the user is talking over Veronica" while TTS is muting the mic.
/// Deliberately well above normal-speech RMS (typically ~0.02-0.1 close to a
/// mic): there is no acoustic echo cancellation in this app, so Veronica's
/// own voice, played through speakers and picked up by the same mic she's
/// listening on, is on the same channel as the user's real voice — this
/// threshold plus `BARGE_IN_SUSTAIN` are the only guard against her own TTS
/// falsely triggering a barge-in. Tunable via `VERONICA_BARGE_IN_RMS` for
/// setups with louder speaker bleed (a laptop's built-in speakers/mic) or a
/// headset (no bleed at all, could safely go lower).
fn barge_in_rms_threshold() -> f32 {
    std::env::var("VERONICA_BARGE_IN_RMS")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .unwrap_or(0.12)
}

/// How long above-threshold audio must be sustained, continuously, before
/// it's treated as real user speech rather than a brief loud transient in
/// Veronica's own TTS bleed (a plosive, a loud word). Chosen short enough
/// that barge-in still feels immediate to the user, long enough that a
/// single loud TTS chunk can't trip it — TTS speech has much more varied
/// short-term energy than someone deliberately interrupting to talk.
const BARGE_IN_SUSTAIN: Duration = Duration::from_millis(180);

/// How long to keep the mic muted after `TtsSpeakingSignal` reports Veronica
/// has genuinely stopped (both the Flux session closed AND the player's sink
/// finished draining — see `TtsSpeakingSignal`'s doc), before trusting mic
/// input as real user speech again.
///
/// Exists because un-muting exactly on that instant was measured to still
/// pick up Veronica's own trailing audio — acoustic/electronic tail from a
/// Bluetooth output device — as a short phrase (typically the last word or
/// two of her own answer, e.g. "...thank you for asking, sir" tailing off
/// into what Whisper transcribes as a bare "Thank you.") and answer it as a
/// new real question, which can then itself produce a short reply that
/// echoes again — a live-observed, confirmed-in-logs runaway loop, not a
/// hypothetical. `is_speaking()` alone (the mute condition below) cannot see
/// this: it goes false the instant playback is DONE, not once the room has
/// actually gone quiet.
///
/// Kept deliberately short — real-word latency (how soon after Veronica
/// finishes can the user's own next utterance start being heard) matters
/// more here than defense-in-depth, since a too-long value would itself
/// clip the start of a fast follow-up. 150ms was picked as a starting point
/// per live testing (not a guess left untuned): short enough to be
/// imperceptible as a delay before listening resumes, long enough in
/// practice to clear the trailing echo that produced the repeated "Thank
/// you." loop. Tunable via `VERONICA_TTS_SETTLE_MS` without a rebuild if a
/// particular output device needs more (or less) margin.
fn tts_settle_duration() -> Duration {
    std::env::var("VERONICA_TTS_SETTLE_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(150))
}

/// Root-cause fix for a false/stale-transcript bug distinct from the
/// TTS-tail settle window above: when muting the mic for TTS (below), the
/// code used to call `sidecar.flush()` unconditionally "so a half-decoded
/// utterance doesn't sit in the decoder." But the local engine has no real
/// speech/non-speech judgment by default (its VAD pre-gate is a pass-
/// through — see `stt::sidecar::SttSidecar::flush`'s doc) — ANY audio at
/// all (room tone, a keyboard click, mic self-noise) can leave it with an
/// in-progress "utterance" the instant Veronica starts talking, with
/// nothing to do with what the user actually said. An unconditional flush
/// force-finalized that fragment and shipped it to Groq, which could
/// hallucinate a short, plausible phrase (e.g. "Thank you.") from audio the
/// user never spoke — a false user turn Veronica would then reply to.
///
/// The fix: only flush if at least this much audio is already buffered
/// (see `SttSidecar::pending_audio_duration`). A stray chunk or two of
/// ambient noise (well under 200ms) never reaches this — flush becomes a
/// no-op for it, and the fragment simply stays buffered, gated off from
/// Groq the same way any other muted-for-TTS audio is, until either it
/// grows into something substantial or gets cleared by the NEXT real
/// flush/finalization. A genuine short utterance the user was mid-saying
/// when Veronica started talking (e.g. one word, ~300ms+) still clears this
/// bar and gets finalized/transcribed promptly, so legitimate speech isn't
/// silently dropped — this is a minimum-plausible-utterance floor, not a
/// delay: it adds no wait time to anything that already exceeds it.
fn min_flush_worthy_audio() -> Duration {
    std::env::var("VERONICA_MIN_FLUSH_AUDIO_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(350))
}

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

    // `MicrophoneCapture::start` only spawns its capture thread and returns
    // immediately (no readiness handshake back from WASAPI init) — unlike
    // the system-audio path, there is no separate "capture ready" instant to
    // measure here, so this session only has one meaningful startup number:
    // time until the STT sidecar (which DOES block until it signals ready)
    // is usable.
    let session_start = Instant::now();
    let selected_input = app.state::<AppState>().selected_devices.input();
    let mic_thread: JoinHandle<()> = MicrophoneCapture::start(audio_tx, stop.clone(), selected_input)?;
    let stt_num_threads = crate::hardware::effective_config_checked(&app).stt_num_threads;
    let mut sidecar = crate::stt::SttSidecar::spawn(
        AudioSource::Microphone,
        stt_tx,
        Some(stt_num_threads),
        Some(&app),
    )?;
    {
        use crate::hardware::telemetry::{log_stage_ms, PipelineStage};
        let ctx = crate::hardware::perf_context(&app);
        let startup_ms = session_start.elapsed().as_millis();
        log_stage_ms(PipelineStage::SttReady, startup_ms, &ctx);
        log_stage_ms(PipelineStage::SttSessionStart, startup_ms, &ctx);
    }

    // Same shape as start_capture_inner's events-forwarder thread: apply
    // each STT event to the shared TranscriptManager and emit the resulting
    // segment as "transcript:update" — the overlay's existing listener
    // already knows how to turn a segment into question-box text and (with
    // Auto AI on) an automatic ask.
    let app_for_events = app.clone();
    let app_for_telemetry = app.clone();
    std::thread::Builder::new()
        .name("mic-assistant-events".into())
        .spawn(move || {
            // Marks the moment the previous utterance ended (or session
            // start) so the FIRST partial after that point can be timed as
            // "silence end -> first transcript text" — the number the user
            // actually feels as "did it hear me start talking". Only the
            // first partial after each reset is logged (matches
            // `PipelineStage::SttFirstPartial`'s doc: speech start -> first
            // partial), not every partial in the utterance.
            let mut utterance_boundary = Instant::now();
            let mut first_partial_seen = false;
            // The current utterance's latency record — created on its first
            // `Partial` (see below) and handed off to `AppState.turn_telemetry`
            // for `veronica::ask_veronica` to pick up and continue once the
            // frontend's `invoke()` call for this utterance arrives. Real
            // `mic_detected`/`speech_started`/`speech_ended` timestamps, not
            // approximations: this thread genuinely observes each of those
            // moments as they happen. `stt_started`/`stt_first_result` are
            // deliberately NOT marked here — by the time this thread sees a
            // `Final` event at all, `stt::sidecar`'s reader thread has
            // already called Groq and gotten a response (that's what
            // triggers the event being sent), so there is no meaningfully
            // earlier "STT dispatched" instant observable from this vantage
            // point without instrumenting `stt::sidecar` itself, which this
            // pass leaves untouched (see that module's isolated-interface
            // doc). Marking them here anyway at the same instant as
            // `stt_final` would just report a fake zero, so they're left
            // unmarked — `TurnTelemetry::finish` reports "n/a" for any delta
            // that needs them, rather than a fabricated number.
            let mut current_telemetry: Option<std::sync::Arc<crate::hardware::telemetry::TurnTelemetry>> = None;
            for event in stt_rx.iter() {
                use crate::hardware::telemetry::{log_stage_ms, PipelineStage, TurnTelemetry};
                match event.kind {
                    crate::stt::SttEventKind::Partial => {
                        if !first_partial_seen {
                            first_partial_seen = true;
                            let ctx = crate::hardware::perf_context(&app_for_telemetry);
                            log_stage_ms(
                                PipelineStage::SttFirstPartial,
                                utterance_boundary.elapsed().as_millis(),
                                &ctx,
                            );
                            let telemetry = std::sync::Arc::new(TurnTelemetry::new());
                            telemetry.mark(PipelineStage::MicDetected);
                            telemetry.mark(PipelineStage::SpeechStarted);
                            let state = app_for_telemetry.state::<AppState>();
                            *state.turn_telemetry.lock().unwrap() = Some(telemetry.clone());
                            current_telemetry = Some(telemetry);
                        }
                    }
                    crate::stt::SttEventKind::Final => {
                        if let Some(telemetry) = current_telemetry.take() {
                            telemetry.mark(PipelineStage::SpeechEnded);
                            telemetry.mark(PipelineStage::SttFinal);
                        }
                        // start_time/end_time are process-relative monotonic
                        // seconds from the sidecar's own clock (speech-end to
                        // Groq-response isn't directly timed here since Groq
                        // runs on its own detached thread — see
                        // stt::sidecar's reader thread doc — but the local
                        // VAD's speech duration is a useful companion number
                        // to the values logged around ask_veronica).
                        if let (Some(start), Some(end)) = (event.start_time, event.end_time) {
                            let ctx = crate::hardware::perf_context(&app_for_telemetry);
                            let speech_ms = ((end - start).max(0.0) * 1000.0) as u128;
                            log_stage_ms(PipelineStage::SttFinal, speech_ms, &ctx);
                        }
                        utterance_boundary = Instant::now();
                        first_partial_seen = false;
                    }
                }
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

    let tts_speaking = app.state::<AppState>().tts_speaking.clone();
    let mic_muted = app.state::<AppState>().mic_muted.clone();
    let app_for_level = app.clone();
    let app_for_barge_in = app.clone();
    let barge_in_rms = barge_in_rms_threshold();
    let settle_duration = tts_settle_duration();
    std::thread::Builder::new()
        .name("mic-assistant-pump".into())
        .spawn(move || {
            // While Veronica's own TTS is speaking (checked live per chunk,
            // not just at loop start — playback can start/stop mid-loop),
            // the chunk is still drained from the channel (so the capture
            // thread's send never blocks) but withheld from STT — otherwise
            // her own voice, picked up acoustically by the mic, gets
            // transcribed and answered as if the user said it. Mirrors
            // audio::pipeline::run_stt_pipeline's PauseSignal handling.
            //
            // Barge-in: while muted, chunks are still scanned (not sent to
            // STT — no AEC exists here, so that stream is contaminated with
            // Veronica's own voice) for sustained above-threshold energy. If
            // the user is genuinely talking over her, this is what notices
            // and cuts her off — see `barge_in_rms_threshold`/`BARGE_IN_SUSTAIN`
            // for why a single loud chunk isn't enough on its own.
            let mut was_muted = false;
            let mut loud_since: Option<Instant> = None;
            // Whether the user's own mute toggle (the header's mute button,
            // separate from Stop) was active on the previous chunk — tracked
            // so re-enabling it flushes the decoder exactly once, same as
            // `was_muted` does for the TTS-speaking case below.
            let mut was_user_muted = false;
            // Set the instant `is_speaking` is first observed false after
            // being true — see `tts_settle_duration`'s doc for why the mic
            // stays muted a little past that instant rather than un-muting
            // on it directly. `None` means either TTS has never spoken yet
            // this session, or the settle window already elapsed and normal
            // listening has resumed.
            let mut settling_since: Option<Instant> = None;
            for chunk in audio_rx.iter() {
                // Emitted unconditionally (even while muted for TTS) so the
                // orb widgets' "listening" animation reflects real mic input
                // the same way commands.rs's system-audio path already does
                // via run_stt_pipeline's on_level callback — this is the
                // mic-assistant path's equivalent, previously missing
                // entirely (chunk.rms_level was computed by MicrophoneCapture
                // but never read here).
                let _ = app_for_level.emit(
                    "audio:level",
                    crate::commands::AudioLevelEvent { source: chunk.source, rms_level: chunk.rms_level },
                );

                // Deliberate user mute takes priority over everything else,
                // including barge-in: unlike the TTS-speaking mute below
                // (which exists only to stop Veronica hearing herself, so a
                // loud enough voice should always be able to interrupt it),
                // the user explicitly asked not to be listened to, so no
                // amount of speech energy should override that.
                if mic_muted.is_muted() {
                    if !was_user_muted {
                        if let Err(err) = sidecar.flush() {
                            log::warn!("mic assistant: failed to flush STT sidecar before user mute: {err}");
                        }
                        was_user_muted = true;
                        loud_since = None;
                        log::info!("[TTS_MUTE_STATE] source=mic muted=true reason=user_muted");
                    }
                    continue;
                }
                if was_user_muted {
                    was_user_muted = false;
                    log::info!("[TTS_MUTE_STATE] source=mic muted=false reason=user_unmuted");
                }

                let is_speaking = tts_speaking.is_speaking();
                if is_speaking {
                    // Actively speaking again (e.g. a fast follow-up answer
                    // right after a brief pause) — cancel any settle window
                    // from a previous stop rather than let a stale timer
                    // un-mute mid-answer once it happens to elapse.
                    settling_since = None;
                    if !was_muted {
                        // Entering mute: finalize whatever was already
                        // heard so a half-decoded utterance doesn't sit in
                        // the decoder and bleed into audio heard after
                        // Veronica stops talking — but ONLY if there's
                        // plausibly a real utterance to finalize. See
                        // `min_flush_worthy_audio`'s doc: an unconditional
                        // flush here was the actual root cause of a false
                        // "Thank you."-style user turn appearing from
                        // ordinary ambient audio, independent of TTS
                        // acoustic bleed and independent of the settle
                        // window below (which guards un-muting, not this).
                        let pending = sidecar.pending_audio_duration();
                        if pending >= min_flush_worthy_audio() {
                            if let Err(err) = sidecar.flush() {
                                log::warn!("mic assistant: failed to flush STT sidecar before muting for TTS: {err}");
                            }
                        } else if pending > Duration::ZERO {
                            // Too short to plausibly be real speech — discard
                            // rather than flush, so it's neither sent to Groq
                            // NOR left sitting in the decoder to contaminate
                            // whatever the user says once unmuted again.
                            log::info!("[STT_FLUSH_SKIPPED] reason=tts_speaking pending_ms={}", pending.as_millis());
                            if let Err(err) = sidecar.discard_pending() {
                                log::warn!("mic assistant: failed to discard pending STT audio before muting for TTS: {err}");
                            }
                        }
                        was_muted = true;
                        loud_since = None;
                        log::info!("[TTS_MUTE_STATE] source=mic muted=true reason=tts_speaking");
                        // Real "Veronica is speaking" signal for the orb
                        // widgets, piggybacked on this loop's existing
                        // per-chunk read of `tts_speaking` (already polled at
                        // mic-chunk cadence for the mute logic above) rather
                        // than adding a new poller or threading an AppHandle
                        // into tts::player (see that module's doc on why its
                        // Sink/OutputStream must stay confined to one thread
                        // with no Tauri coupling).
                        let _ = app_for_level.emit("tts:speaking-changed", true);
                    }

                    let rms = compute_rms(&chunk.samples);
                    if rms >= barge_in_rms {
                        let started = loud_since.get_or_insert_with(Instant::now);
                        if started.elapsed() >= BARGE_IN_SUSTAIN {
                            log::info!("mic assistant: barge-in detected (rms={rms:.3}), stopping TTS");
                            let barge_in_state = app_for_barge_in.state::<AppState>();
                            if let Some(session) = barge_in_state.tts.lock().unwrap().as_ref() {
                                session.stop();
                            }
                            // Also cancels any in-flight agent-loop
                            // generation for the turn Veronica was still
                            // speaking — without this, only the AUDIO
                            // stopped; the LLM stream (and any tool calls it
                            // was still making) previously kept running to
                            // completion in the background, wasting the
                            // rest of that turn's latency/cost for an answer
                            // nobody will hear.
                            barge_in_state.cancel_current_turn();
                            loud_since = None;
                            // Send this triggering chunk to STT immediately
                            // too (rather than waiting for the next loop
                            // iteration to naturally un-mute): the user was
                            // already mid-word when sustained energy crossed
                            // the threshold, so this chunk is real speech,
                            // not TTS bleed.
                            was_muted = false;
                            log::info!("[TTS_MUTE_STATE] source=mic muted=false reason=barge_in");
                            let _ = app_for_level.emit("tts:speaking-changed", false);
                            if let Err(err) = sidecar.send_samples(&chunk.samples) {
                                log::warn!("mic assistant: failed to send audio to STT sidecar: {err}");
                            }
                        }
                    } else {
                        loud_since = None;
                    }
                    continue;
                }
                if was_muted {
                    // TTS has genuinely stopped (is_speaking is false), but
                    // stay muted a little longer to let Veronica's own
                    // acoustic/electronic tail decay — see
                    // `tts_settle_duration`'s doc for the exact failure this
                    // prevents (her own trailing words transcribed and
                    // answered as a new user turn). No barge-in scanning
                    // during this window either: unlike the actively-speaking
                    // case above, there is no live TTS to interrupt anymore,
                    // so a loud chunk here is either the tail itself or the
                    // user's real next utterance — either way, waiting the
                    // (short) remainder of the window before listening again
                    // is correct, not a barge-in decision.
                    let is_first_settle_chunk = settling_since.is_none();
                    let settle_start = settling_since.get_or_insert_with(Instant::now);
                    if is_first_settle_chunk {
                        log::info!("[TTS_MUTE_STATE] source=mic muted=true reason=settling settle_ms={}", settle_duration.as_millis());
                    }
                    if settle_start.elapsed() < settle_duration {
                        continue;
                    }
                    was_muted = false;
                    settling_since = None;
                    loud_since = None;
                    log::info!("[TTS_MUTE_STATE] source=mic muted=false reason=settle_elapsed");
                    let _ = app_for_level.emit("tts:speaking-changed", false);
                }
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

/// Toggles the explicit user mute (header's mute button) on or off. Safe to
/// call whether or not the mic assistant is currently running — it just
/// sets the shared flag the pump loop reads, so a mute set before Start
/// takes effect the moment the session begins, and it clears itself when
/// `stop_mic_assistant` isn't called and the same session is later resumed.
#[tauri::command]
pub fn set_mic_muted(state: State<'_, AppState>, muted: bool) -> Result<(), String> {
    state.mic_muted.set_muted(muted);
    Ok(())
}

#[tauri::command]
pub fn get_mic_muted(state: State<'_, AppState>) -> Result<bool, String> {
    Ok(state.mic_muted.is_muted())
}

#[tauri::command]
pub fn stop_mic_assistant(
    state: State<'_, AppState>,
    session: State<'_, MicAssistantSession>,
) -> Result<(), String> {
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
    // Reset for the next session — otherwise a muted session ended via Stop
    // would leave the next Start silently pre-muted with no visible cause.
    state.mic_muted.set_muted(false);
    Ok(())
}
