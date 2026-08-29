use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use tauri::{path::BaseDirectory, AppHandle, Manager};

use crate::audio::{AudioSource, TARGET_SAMPLE_RATE};
use crate::process_util::hidden_command;

use super::events::{SidecarLine, SttEvent, SttEventKind};
use super::groq;

/// How long `SttSidecar::spawn` will wait for the local VAD engine to signal
/// `{"type":"ready"}` (or an error, or exit) before giving up. Overridable
/// via `STT_READY_TIMEOUT_MS` — the default is generous relative to normal
/// model-load time (typically a couple of seconds) specifically to tolerate
/// a slower load under CPU/memory contention on constrained hardware,
/// without leaving a genuinely hung process waiting forever.
fn ready_timeout() -> Duration {
    std::env::var("STT_READY_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(20_000))
}

fn stt_package_dir() -> std::path::PathBuf {
    // In dev, CARGO_MANIFEST_DIR is src-tauri; this personal repo's flat
    // layout has no packages/stt monorepo tree — the dev-tree VAD-engine
    // source (streaming_asr_sidecar/), its venv (streaming_asr_sidecar/.venv),
    // the prebuilt frozen sidecar (sidecars/stt-sidecar/), and the model
    // (models/stt/) all live directly at the repo root, one level up.
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn script_path() -> std::path::PathBuf {
    stt_package_dir().join("streaming_asr_sidecar").join("sidecar.py")
}

/// The local VAD engine's own virtualenv interpreter, at
/// `streaming_asr_sidecar/.venv` (repo root, sibling to `models/` and
/// `sidecars/` in this flat personal repo — there is no `packages/stt`
/// monorepo tree here). sherpa-onnx and its ONNX Runtime dependency are
/// heavy enough to deserve isolation from any other Python tooling on the
/// dev machine.
///
/// Only meaningful in dev: a plain venv is not relocatable (its `python.exe`
/// hardcodes the absolute path of the base Python install it was created
/// from — see `pyvenv.cfg`), so it is never bundled into a release build.
/// Installed builds instead run the PyInstaller-frozen executable resolved
/// by `frozen_sidecar_path` below, which has no such dependency.
fn stt_venv_python() -> Option<std::path::PathBuf> {
    let candidate = stt_package_dir()
        .join("streaming_asr_sidecar")
        .join(".venv")
        .join("Scripts")
        .join("python.exe");
    candidate.exists().then_some(candidate)
}

/// The PyInstaller-frozen sidecar bundled as a resource for release builds
/// (see `bundle.resources` in `tauri.conf.json`, built via
/// `packages/stt/scripts/freeze_sidecar.py`) — a single self-contained
/// executable with its own Python runtime and sherpa-onnx/numpy embedded, so
/// end users need nothing installed to run it. `None` when there's no
/// `AppHandle` (the headless `bin/pipeline_test*.rs` binaries) or the
/// resource simply isn't there.
///
/// `tauri dev` has no bundled-resource directory at all (Tauri's resource
/// resolver only exists for a built app), so this also checks the
/// repo-relative dev build output directly, so `tauri dev` gets the same
/// fast startup as a release build whenever the frozen sidecar has been
/// built at least once — falling back to the venv path only if it genuinely
/// hasn't.
fn frozen_sidecar_path(app: Option<&AppHandle>) -> Option<std::path::PathBuf> {
    if let Some(app) = app {
        if let Ok(candidate) = app.path().resolve("stt-sidecar/stt-sidecar.exe", BaseDirectory::Resource) {
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    let dev_candidate = stt_package_dir()
        .join("sidecars")
        .join("stt-sidecar")
        .join("stt-sidecar.exe");
    dev_candidate.exists().then_some(dev_candidate)
}

/// The STT model directory bundled as a resource (see `bundle.resources` in
/// `tauri.conf.json`). `None` falls back to `sidecar.py`'s own
/// `STT_MODEL_DIR`-or-relative-default resolution, which is what dev/test
/// runs (including the headless `bin/pipeline_test*.rs` binaries) rely on.
fn resource_model_dir(app: Option<&AppHandle>) -> Option<std::path::PathBuf> {
    let app = app?;
    let candidate = app.path().resolve("stt-model", BaseDirectory::Resource).ok()?;
    candidate.is_dir().then_some(candidate)
}

/// A running local VAD/endpointing engine process for one audio source
/// (system audio or microphone), paired with Groq Cloud transcription.
/// Owns the child process and the threads that pump audio in / read
/// detected-utterance events out.
pub struct SttSidecar {
    child: Child,
    stdin: Option<ChildStdin>,
    reader_thread: Option<JoinHandle<()>>,
    /// The current utterance's raw samples, appended to by `send_samples`
    /// and drained by the reader thread on each detected utterance boundary
    /// to send to Groq. This is the "send only VAD-detected speech
    /// segments" cut point: the local engine's own endpoint detection,
    /// running on this same audio, decides where the buffer gets cut, not a
    /// fixed duration or every chunk individually.
    groq_utterance_buffer: Arc<Mutex<Vec<f32>>>,
}

impl SttSidecar {
    /// Spawns the local sherpa-onnx process that does voice-activity and
    /// utterance-endpoint detection, and wires its detected utterances to
    /// Groq Cloud's Whisper API for transcription (see `groq.rs`). The local
    /// engine's own transcribed text is never used — it exists here purely
    /// to decide *when* an utterance is complete; Groq is the only source of
    /// transcript text.
    ///
    /// `events_tx` receives every partial/final event, forwarded from a
    /// dedicated reader thread so that writing audio to stdin never blocks
    /// on draining stdout (a naive write-then-read pattern deadlocks on
    /// Windows once the pipe buffer fills).
    ///
    /// `num_threads`: an explicit thread count (from
    /// `hardware::PerformanceManager::effective_config()`) takes priority
    /// over the `STT_NUM_THREADS` environment variable when `Some` — this is
    /// the tier-driven path production call sites use. `None` preserves the
    /// pre-existing env-var-pass-through behavior unchanged, which
    /// `bin/pipeline_test.rs` (a headless test binary with no
    /// `PerformanceManager`/`AppHandle` available) relies on. Two sidecars
    /// (system audio + microphone) can be spawned concurrently — passing the
    /// value explicitly here, rather than mutating `std::env` before each
    /// spawn, avoids a process-global race between them.
    ///
    /// Blocks (up to `ready_timeout()`) until the local engine signals
    /// `{"type":"ready"}`, reports `{"type":"error",...}`, or exits without
    /// either — so the caller never reports "recording started" while the
    /// engine is still loading or has already crashed.
    ///
    /// `app`: threaded through so a release build can find and run the
    /// PyInstaller-frozen sidecar bundled as a resource (see
    /// `frozen_sidecar_path`) instead of requiring `streaming_asr_sidecar/.venv`
    /// — which only ever exists on a dev machine, never on an end user's PC —
    /// to be present. `None` for the headless `bin/pipeline_test*.rs`
    /// binaries, which have no `AppHandle` and always exercise the
    /// dev-tree venv/script path below instead.
    pub fn spawn(
        source: AudioSource,
        events_tx: Sender<SttEvent>,
        num_threads: Option<u32>,
        app: Option<&AppHandle>,
    ) -> Result<Self, String> {
        let source_arg = match source {
            AudioSource::SystemAudio => "SYSTEM_AUDIO",
            AudioSource::Microphone => "MICROPHONE",
        };

        // Two ways to end up with something runnable, tried in order: the
        // dev-tree venv interpreter running the source script directly (the
        // primary path whenever a dev venv is present), or the frozen,
        // fully self-contained sidecar bundled into a release build (no
        // Python needed on the target machine at all) as a fallback for
        // machines with no such venv.
        let mut used_frozen_sidecar_without_bundle = false;
        let (executable, args) = if let Some(venv) = stt_venv_python() {
            let script = script_path();
            if !script.exists() {
                return Err(format!("STT sidecar script not found at {}", script.display()));
            }
            (
                venv.to_string_lossy().to_string(),
                vec![script.to_string_lossy().to_string(), source_arg.to_string()],
            )
        } else if let Some(frozen) = frozen_sidecar_path(app) {
            // `resource_model_dir(app)` below tells us whether this frozen
            // exe actually came from a real Tauri resource bundle (release
            // build) or from `frozen_sidecar_path`'s dev-mode fallback
            // directly — only the latter needs `STT_MODEL_DIR` set
            // explicitly, since the frozen exe's own relative-default model
            // resolution doesn't work outside a PyInstaller bundle's real
            // install location.
            used_frozen_sidecar_without_bundle = resource_model_dir(app).is_none();
            (frozen.to_string_lossy().to_string(), vec![source_arg.to_string()])
        } else {
            return Err(
                "STT venv not found at streaming_asr_sidecar/.venv — run: \
                 py -3 -m venv streaming_asr_sidecar/.venv && \
                 streaming_asr_sidecar/.venv/Scripts/python.exe -m pip install sherpa-onnx numpy"
                    .to_string(),
            );
        };

        // Trailing silence before the current utterance is finalized —
        // this IS the endpoint-detection tuning, independent of who
        // transcribes the resulting audio. Settable via the launch
        // environment so it can be tuned without a rebuild.
        let end_silence_ms = std::env::var("STT_END_SILENCE_MS").ok();

        // The local engine's own VAD pre-gate was silently discarding a
        // large share of real microphone chunks as "non-speech" before they
        // ever reached the endpoint detector (observed 48-64% of chunks
        // skipped on live mic input vs. ~3-6% on clean pre-recorded test
        // audio) — the direct cause of utterances finalizing on garbled/
        // incomplete audio. Defaulting it off here trades a little extra
        // CPU for not dropping real speech; still overridable via
        // STT_VAD_GATE_ENABLED for anyone who wants the gate back (e.g. a
        // genuinely noisy room).
        let vad_gate_enabled = std::env::var("STT_VAD_GATE_ENABLED").unwrap_or_else(|_| "false".to_string());

        let mut command = hidden_command(&executable);
        command.args(&args);
        if let Some(ms) = end_silence_ms {
            command.env("STT_END_SILENCE_MS", ms);
        }
        command.env("STT_VAD_GATE_ENABLED", vad_gate_enabled);
        match num_threads {
            Some(threads) => {
                command.env("STT_NUM_THREADS", threads.to_string());
            }
            None => {
                if let Ok(threads) = std::env::var("STT_NUM_THREADS") {
                    command.env("STT_NUM_THREADS", threads);
                }
            }
        }
        // A bundled model resource (release builds) takes priority over
        // whatever STT_MODEL_DIR the parent process happens to have set —
        // the dev/test fallback below is only meaningful when there's no
        // bundle to find one in.
        //
        // The frozen exe's own default model path (sidecar.py's
        // `_resolve_model_dir`) is resolved relative to `__file__`, which
        // under PyInstaller points into the frozen bundle's temp extraction
        // directory, not this repo — so when the dev-mode fallback above
        // picked the frozen exe with no Tauri resource bundle to resolve
        // `resource_model_dir` from, it still needs `STT_MODEL_DIR` pointed
        // at the repo's `models/stt/` directory explicitly, exactly as a
        // real release build's resource bundle would.
        if let Some(dir) = resource_model_dir(app) {
            command.env("STT_MODEL_DIR", dir);
        } else if let Ok(dir) = std::env::var("STT_MODEL_DIR") {
            command.env("STT_MODEL_DIR", dir);
        } else if used_frozen_sidecar_without_bundle {
            let dev_model_dir = stt_package_dir().join("models").join("stt").join("nemo-fastconformer-80ms-int8");
            if dev_model_dir.is_dir() {
                command.env("STT_MODEL_DIR", dev_model_dir);
            }
        }

        let groq_utterance_buffer = Arc::new(Mutex::new(Vec::new()));

        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn STT sidecar ({executable}): {e}"))?;

        let stdin = child.stdin.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| "sidecar stdout not captured".to_string())?;
        let stderr = child.stderr.take();

        if let Some(stderr) = stderr {
            std::thread::Builder::new()
                .name("stt-sidecar-stderr".into())
                .spawn(move || {
                    let reader = BufReader::new(stderr);
                    for line in reader.lines().map_while(Result::ok) {
                        log::warn!("[stt sidecar stderr] {line}");
                    }
                })
                .ok();
        }

        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        // Cloned (not moved-by-reference) so it can live in the reader
        // thread's `'static` closure below — `app` itself is only a borrow
        // for the duration of this `spawn()` call, needed above purely for
        // path resolution. `AppHandle::clone()` is cheap (an Arc internally).
        // `None` for the headless test binaries (see this fn's doc), so
        // those keep emitting nothing, exactly as before this change.
        let app_for_errors = app.cloned();

        let reader_buffer = groq_utterance_buffer.clone();
        let reader_thread = std::thread::Builder::new()
            .name("stt-sidecar-reader".into())
            .spawn(move || {
                // Set at most once, on whichever comes first: a Ready line,
                // an Error line, or stdout closing without either. Every
                // later line is forwarded normally regardless — this only
                // gates the one-time readiness signal `spawn()` is waiting
                // on, never the ongoing partial/final event stream.
                let mut ready_signaled = false;
                let reader = BufReader::new(stdout);
                for line in reader.lines().map_while(Result::ok) {
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<SidecarLine>(&line) {
                        Ok(SidecarLine::Ready) => {
                            log::info!("STT VAD engine ready ({source_arg}); transcription via Groq Cloud (whisper-large-v3-turbo)");
                            if !ready_signaled {
                                ready_signaled = true;
                                let _ = ready_tx.send(Ok(()));
                            }
                        }
                        Ok(SidecarLine::Partial { text: _, source: _ }) => {
                            // The local engine's own partial text is not
                            // forwarded: Groq has no meaningful concept of a
                            // "partial" for an utterance still in progress
                            // (it only ever transcribes a complete buffered
                            // segment on `Final`), and forwarding text that
                            // will never match what the user eventually sees
                            // would be misleading rather than merely absent.
                        }
                        Ok(SidecarLine::Final {
                            text: _local_text,
                            source,
                            start_time,
                            end_time,
                        }) => {
                            // The local engine has decided this utterance is
                            // complete — that decision (VAD/endpointing) is
                            // the only thing it's kept running for. Take the
                            // buffered raw audio for this utterance and ask
                            // Groq on a SEPARATE thread, not inline here:
                            // this loop is the only thing draining the local
                            // engine's stdout, which keeps emitting partials
                            // for whatever the user says next while a Groq
                            // request is in flight. Blocking this loop on
                            // that request (Groq's own timeout is up to
                            // 15s+) stalls stdout drainage; once the local
                            // engine's stdout pipe buffer fills from
                            // undrained partial-JSON lines, its writes block
                            // and the whole child process — including its
                            // stdin — stops responding, which was observed
                            // to kill the sidecar outright, not just delay
                            // one utterance's transcript. Groq is the only
                            // source of transcript text, so an error becomes
                            // an empty final rather than a silently-wrong
                            // local transcript; per-utterance Groq calls have
                            // no ordering requirement between each other, so
                            // spawning a new thread per utterance (mic
                            // utterances are seconds apart, never a tight
                            // loop) is simpler than a bounded worker pool.
                            let samples = reader_buffer.lock().map(|mut b| std::mem::take(&mut *b)).unwrap_or_default();
                            let events_tx = events_tx.clone();
                            let app_for_groq_error = app_for_errors.clone();
                            std::thread::spawn(move || {
                                let text = match groq::transcribe(&samples, TARGET_SAMPLE_RATE) {
                                    Ok(groq_text) => groq_text,
                                    Err(err) => {
                                        log::error!("Groq transcription failed for this utterance: {err}");
                                        // Real error signal for the orb
                                        // widgets — this failure previously
                                        // had no path to the frontend at all
                                        // (not even via invoke().catch(),
                                        // since it happens on a background
                                        // thread with no command in flight):
                                        // the utterance was just silently
                                        // dropped as an empty final below.
                                        if let Some(app) = app_for_groq_error.as_ref() {
                                            use tauri::Emitter;
                                            let _ = app.emit("veronica:error", format!("Speech transcription failed: {err}"));
                                        }
                                        String::new()
                                    }
                                };
                                if text.is_empty() {
                                    return;
                                }
                                let _ = events_tx.send(SttEvent {
                                    kind: SttEventKind::Final,
                                    text,
                                    source,
                                    start_time: Some(start_time),
                                    end_time: Some(end_time),
                                });
                            });
                        }
                        Ok(SidecarLine::Error { message }) => {
                            log::error!("STT sidecar error: {message}");
                            if !ready_signaled {
                                ready_signaled = true;
                                let _ = ready_tx.send(Err(message));
                            } else if let Some(app) = app_for_errors.as_ref() {
                                // Post-startup error: a pre-ready one is
                                // already surfaced via spawn()'s own Err
                                // return above, so only emit here for
                                // failures happening after the session was
                                // already reported healthy — those had no
                                // path to the frontend before this.
                                use tauri::Emitter;
                                let _ = app.emit("veronica:error", format!("Speech recognition error: {message}"));
                            }
                        }
                        Err(err) => {
                            log::warn!("unparseable STT sidecar line: {line} ({err})");
                        }
                    }
                }
                // stdout closed (process exited) without ever signaling
                // ready — e.g. it crashed during model load before managing
                // to emit even an Error line. Surface that as a failure too,
                // rather than leaving `spawn()` waiting until its timeout
                // for a process that has already exited.
                if !ready_signaled {
                    let _ = ready_tx.send(Err("STT sidecar exited before signaling ready".to_string()));
                }
            })
            .map_err(|e| e.to_string())?;

        match ready_rx.recv_timeout(ready_timeout()) {
            Ok(Ok(())) => {}
            Ok(Err(message)) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader_thread.join();
                return Err(format!("STT sidecar failed to start: {message}"));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = reader_thread.join();
                return Err("STT sidecar did not become ready in time".to_string());
            }
        }

        Ok(Self {
            child,
            stdin,
            reader_thread: Some(reader_thread),
            groq_utterance_buffer,
        })
    }

    /// Encodes f32 samples (range -1.0..1.0) as PCM16 and writes them to the
    /// local engine's stdin using the length-prefixed frame protocol, so it
    /// can keep detecting utterance boundaries. Also appends the raw samples
    /// to the current utterance's buffer (drained by the reader thread on
    /// each detected boundary and sent to Groq) — this is the "send only
    /// VAD-detected speech segments" cut point: the local engine's endpoint
    /// detection, running on this same audio, decides where that buffer
    /// gets cut and sent, not a fixed duration or every chunk individually.
    pub fn send_samples(&mut self, samples: &[f32]) -> Result<(), String> {
        if let Ok(mut buf) = self.groq_utterance_buffer.lock() {
            buf.extend_from_slice(samples);
        }
        let Some(stdin) = self.stdin.as_mut() else {
            return Err("sidecar stdin already closed".into());
        };
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let clamped = s.clamp(-1.0, 1.0);
            let sample_i16 = (clamped * i16::MAX as f32) as i16;
            bytes.extend_from_slice(&sample_i16.to_le_bytes());
        }
        write_frame(stdin, &bytes)
    }

    /// How much audio (as a duration, at `TARGET_SAMPLE_RATE`) is currently
    /// sitting in the not-yet-finalized utterance buffer — i.e. exactly what
    /// `flush()` would force-finalize and ship to Groq if called right now.
    /// Callers use this to decide whether a flush is worth triggering at
    /// all: see `flush`'s doc for why an unconditional flush is unsafe.
    pub fn pending_audio_duration(&self) -> Duration {
        let sample_count = self.groq_utterance_buffer.lock().map(|buf| buf.len()).unwrap_or(0);
        Duration::from_secs_f64(sample_count as f64 / TARGET_SAMPLE_RATE as f64)
    }

    /// Sends the zero-length flush marker so the local engine finalizes any
    /// in-progress utterance immediately (used when the user pauses/stops
    /// recording, or the mic is about to be muted for TTS) — that
    /// finalization is what triggers sending the buffered audio to Groq for
    /// this last utterance.
    ///
    /// The local engine has no real speech/non-speech judgment of its own
    /// by default (`STT_VAD_GATE_ENABLED` is off — see `sidecar.py`'s
    /// `PassThroughGate`), so ANY audio at all — room tone, a keyboard
    /// click, mic self-noise — can leave it with an in-progress "utterance"
    /// at the moment this is called. An unconditional flush would force-
    /// finalize that fragment and send it to Groq, which can hallucinate a
    /// short plausible phrase (e.g. "Thank you.") from marginal audio the
    /// user never actually spoke. Callers MUST check `pending_audio_duration`
    /// first and skip the flush when it's below a reasoned minimum — see
    /// `voice_command::mod`'s mute-entry call site, which is exactly this
    /// case (muting for TTS is not "the user paused mid-sentence," so a tiny
    /// fragment there is far more likely to be noise than real speech).
    pub fn flush(&mut self) -> Result<(), String> {
        let Some(stdin) = self.stdin.as_mut() else {
            return Err("sidecar stdin already closed".into());
        };
        write_frame(stdin, &[])
    }

    /// Throws away whatever fragment of audio is currently in-progress in
    /// the local decoder WITHOUT transcribing it — the counterpart to
    /// `flush()` for exactly the case that method's doc warns about: a
    /// too-short-to-be-real fragment (see `pending_audio_duration`) that
    /// should never reach Groq at all, and must not be left sitting in the
    /// decoder either (a stale fragment left in place would otherwise get
    /// silently prepended to whatever real speech the user says once the
    /// mic is un-muted again, corrupting that later, genuine utterance).
    /// A one-byte frame on the wire — real audio frames are always an even
    /// number of bytes (PCM16LE), so this is unambiguous against the
    /// existing zero-byte flush marker and real audio, with no wire-format
    /// version bump needed. See `sidecar.py`'s `DISCARD_MARKER` handler.
    pub fn discard_pending(&mut self) -> Result<(), String> {
        if let Ok(mut buf) = self.groq_utterance_buffer.lock() {
            buf.clear();
        }
        let Some(stdin) = self.stdin.as_mut() else {
            return Err("sidecar stdin already closed".into());
        };
        write_frame(stdin, &[0u8])
    }

    /// Closes stdin (signals end-of-stream) and waits for the process to exit.
    pub fn shutdown(mut self) {
        self.stdin.take();
        let _ = self.child.wait();
        if let Some(handle) = self.reader_thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for SttSidecar {
    /// Safety net for paths that don't call `shutdown()` explicitly (a panic
    /// unwinding through this struct, or the struct simply being dropped early):
    /// without this, the child Python process would keep running as an orphan,
    /// since Windows does not kill child processes when their parent exits.
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn write_frame(stdin: &mut ChildStdin, payload: &[u8]) -> Result<(), String> {
    let len = payload.len() as u32;
    stdin.write_all(&len.to_le_bytes()).map_err(|e| e.to_string())?;
    if !payload.is_empty() {
        stdin.write_all(payload).map_err(|e| e.to_string())?;
    }
    stdin.flush().map_err(|e| e.to_string())?;
    Ok(())
}
