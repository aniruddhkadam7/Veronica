use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use tauri::{path::BaseDirectory, AppHandle, Manager};

use crate::audio::AudioSource;
use crate::process_util::hidden_command;

use super::events::{SidecarLine, SttEvent, SttEventKind};

/// How long `SttSidecar::spawn` will wait for the sidecar to signal
/// `{"type":"ready"}` (or an error, or exit) before giving up. Overridable
/// via `STT_READY_TIMEOUT_MS`, same pattern as the sidecar's other tuning
/// knobs (`STT_NUM_THREADS`, `STT_END_SILENCE_MS`, ...) — the default is
/// generous relative to normal model-load time (typically a couple of
/// seconds) specifically to tolerate a slower load under CPU/memory
/// contention on constrained hardware, without leaving a genuinely hung
/// process waiting forever.
fn ready_timeout() -> Duration {
    std::env::var("STT_READY_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(20_000))
}

/// Locates a working Python 3 interpreter. Tries `py -3` first (the standard Windows
/// launcher, which resolves correctly even when bare `python`/`python3` are shadowed
/// by the Microsoft Store app-execution-alias stub — see docs/architecture.md), then
/// falls back to `python`.
fn find_python() -> Option<(String, Vec<String>)> {
    if hidden_command("py")
        .args(["-3", "--version"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Some(("py".to_string(), vec!["-3".to_string()]));
    }
    if hidden_command("python")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        return Some(("python".to_string(), vec![]));
    }
    None
}

fn stt_package_dir() -> std::path::PathBuf {
    // In dev, CARGO_MANIFEST_DIR is src-tauri; this personal repo's flat
    // layout has no packages/stt monorepo tree — only the prebuilt sidecar
    // (sidecars/stt-sidecar/) and model (models/stt/) at the repo root, one
    // level up. There is no dev-tree venv/script path in this repo at all
    // (script_path()/stt_venv_python() below will simply never resolve to
    // anything that exists), so this function only matters for locating the
    // prebuilt sidecar/model in dev mode.
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Which local engine the sidecar should run.
///
/// `StreamingAsr` (sherpa-onnx / NeMo FastConformer 80ms) is the production
/// engine as of the benchmark in `docs/stt-benchmark.md`. PocketSphinx is kept
/// reachable via `STT_ENGINE=pocketsphinx` so the two can be compared on the
/// same machine without a rebuild, and so there is a fallback if the streaming
/// venv is ever missing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SttEngineKind {
    StreamingAsr,
    PocketSphinx,
}

impl SttEngineKind {
    fn from_env() -> Self {
        match std::env::var("STT_ENGINE").unwrap_or_default().trim().to_ascii_lowercase().as_str() {
            "pocketsphinx" | "sphinx" => Self::PocketSphinx,
            _ => Self::StreamingAsr,
        }
    }

    fn script_path(self) -> std::path::PathBuf {
        let base = stt_package_dir();
        match self {
            Self::StreamingAsr => base.join("streaming_asr_sidecar").join("sidecar.py"),
            Self::PocketSphinx => base.join("pocketsphinx_sidecar").join("sidecar.py"),
        }
    }
}

/// The STT sidecar's own virtualenv interpreter. sherpa-onnx and its ONNX
/// Runtime dependency are heavy enough to deserve isolation, mirroring how
/// `packages/rag` and `apps/backend` each own a venv rather than sharing one.
///
/// Only meaningful in dev: a plain venv is not relocatable (its `python.exe`
/// hardcodes the absolute path of the base Python install it was created
/// from — see `pyvenv.cfg`), so it is never bundled into a release build.
/// Installed builds instead run the PyInstaller-frozen executable resolved
/// by `frozen_sidecar_path` below, which has no such dependency.
fn stt_venv_python() -> Option<std::path::PathBuf> {
    let candidate = stt_package_dir()
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
/// resolver only exists for a built app), so this used to unconditionally
/// fall back to the venv path below in dev — meaning every dev-mode STT
/// start paid for a cold Python interpreter + sherpa-onnx import + model load
/// from disk (several seconds), even on a machine that had already run
/// `freeze_sidecar.py` and had the exact same frozen exe sitting right there
/// unused. This now also checks that repo-relative dev build output
/// directly, so `tauri dev` gets the same fast startup as a release build
/// whenever the frozen sidecar has been built at least once — falling back
/// to the venv path only if it genuinely hasn't.
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

/// A running PocketSphinx sidecar process for one audio source (system audio or
/// microphone). Owns the child process and the threads that pump audio in / read
/// transcript events out.
pub struct SttSidecar {
    child: Child,
    stdin: Option<ChildStdin>,
    reader_thread: Option<JoinHandle<()>>,
}

impl SttSidecar {
    /// Spawns the sidecar process. `events_tx` receives every partial/final/error
    /// event the sidecar produces, forwarded from a dedicated reader thread so that
    /// writing audio to stdin never blocks on draining stdout (a naive
    /// write-then-read pattern deadlocks on Windows once the pipe buffer fills —
    /// see docs/progress.md Step 4 for how this was diagnosed).
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
    /// Blocks (up to `ready_timeout()`) until the sidecar signals
    /// `{"type":"ready"}`, reports `{"type":"error",...}`, or exits without
    /// either. Earlier versions of this function returned `Ok` the instant
    /// the child process was spawned — before the model had loaded, and
    /// sometimes before Python had even finished importing its
    /// dependencies. On constrained hardware (slow CPU, competing for RAM
    /// with the RAG service's own startup) that load can be slow or can
    /// fail outright (e.g. an allocation failure), and neither was ever
    /// visible to the caller: the command would report success and the UI
    /// would show "Recording" with a sidecar that was still loading, or had
    /// already crashed, and would never produce a single transcript event.
    /// See `docs/performance-tuning.md`'s STT-start-reliability section.
    ///
    /// `app`: threaded through so a release build can find and run the
    /// PyInstaller-frozen sidecar bundled as a resource (see
    /// `frozen_sidecar_path`) instead of requiring `packages/stt/.venv` —
    /// which only ever exists on a dev machine, never on an end user's PC —
    /// to be present. `None` for the headless `bin/pipeline_test*.rs`
    /// binaries, which have no `AppHandle` and always exercise the
    /// dev-tree venv/script path below instead.
    pub fn spawn(
        source: AudioSource,
        events_tx: Sender<SttEvent>,
        num_threads: Option<u32>,
        app: Option<&AppHandle>,
    ) -> Result<Self, String> {
        let engine = SttEngineKind::from_env();

        let source_arg = match source {
            AudioSource::SystemAudio => "SYSTEM_AUDIO",
            AudioSource::Microphone => "MICROPHONE",
        };

        // Three ways to end up with something runnable, tried in order: the
        // frozen, fully self-contained sidecar bundled into a release build
        // (no Python needed on the target machine at all); the dev-tree
        // venv interpreter running the source script directly; or, for the
        // PocketSphinx comparison engine only (never frozen — it's not the
        // production engine), whatever system Python is on PATH.
        let mut used_frozen_sidecar_without_bundle = false;
        let (executable, args) = if let (SttEngineKind::StreamingAsr, Some(frozen)) =
            (engine, frozen_sidecar_path(app))
        {
            // `resource_model_dir(app)` below tells us whether this frozen
            // exe actually came from a real Tauri resource bundle (release
            // build) or from `frozen_sidecar_path`'s dev-mode fallback onto
            // `packages/stt/dist/` directly — only the latter needs
            // `STT_MODEL_DIR` set explicitly, since the frozen exe's own
            // relative-default model resolution doesn't work outside a
            // PyInstaller bundle's real install location.
            used_frozen_sidecar_without_bundle = resource_model_dir(app).is_none();
            (frozen.to_string_lossy().to_string(), vec![source_arg.to_string()])
        } else {
            let (python, mut base_args) = match (engine, stt_venv_python()) {
                (SttEngineKind::StreamingAsr, Some(venv)) => {
                    (venv.to_string_lossy().to_string(), Vec::new())
                }
                (SttEngineKind::StreamingAsr, None) => {
                    return Err(
                        "STT venv not found at packages/stt/.venv — run: \
                         py -3 -m venv packages/stt/.venv && \
                         packages/stt/.venv/Scripts/python.exe -m pip install sherpa-onnx numpy"
                            .to_string(),
                    )
                }
                (SttEngineKind::PocketSphinx, _) => find_python().ok_or_else(|| {
                    "no Python 3 interpreter found (tried `py -3` and `python`)".to_string()
                })?,
            };

            let script = engine.script_path();
            if !script.exists() {
                return Err(format!("STT sidecar script not found at {}", script.display()));
            }

            base_args.push(script.to_string_lossy().to_string());
            base_args.push(source_arg.to_string());
            (python, base_args)
        };

        // Trailing silence before the current utterance is finalized. Settable
        // via the launch environment so it can be tuned without a rebuild. The
        // two engines want different defaults, so each sidecar picks its own
        // when this is unset rather than having one value imposed here.
        let end_silence_ms = std::env::var("STT_END_SILENCE_MS").ok();

        let mut command = hidden_command(&executable);
        command.args(&args);
        if let Some(ms) = end_silence_ms {
            command.env("STT_END_SILENCE_MS", ms);
        }
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
        // directory, not this repo — so when `tauri dev`'s dev-mode fallback
        // above picked the frozen exe from `packages/stt/dist/` (no
        // Tauri resource bundle to resolve `resource_model_dir` from), it
        // still needs `STT_MODEL_DIR` pointed at the repo's `models/stt/`
        // directory explicitly, exactly as a real release build's resource
        // bundle would.
        if let Some(dir) = resource_model_dir(app) {
            command.env("STT_MODEL_DIR", dir);
        } else if let Ok(dir) = std::env::var("STT_MODEL_DIR") {
            command.env("STT_MODEL_DIR", dir);
        } else if used_frozen_sidecar_without_bundle {
            let dev_model_dir = stt_package_dir()
                .join("models")
                .join("stt")
                .join("nemo-fastconformer-80ms-int8");
            if dev_model_dir.is_dir() {
                command.env("STT_MODEL_DIR", dev_model_dir);
            }
        }

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
                            log::info!("STT sidecar ready ({source_arg})");
                            if !ready_signaled {
                                ready_signaled = true;
                                let _ = ready_tx.send(Ok(()));
                            }
                        }
                        Ok(SidecarLine::Partial { text, source }) => {
                            let _ = events_tx.send(SttEvent {
                                kind: SttEventKind::Partial,
                                text,
                                source,
                                start_time: None,
                                end_time: None,
                            });
                        }
                        Ok(SidecarLine::Final {
                            text,
                            source,
                            start_time,
                            end_time,
                        }) => {
                            let _ = events_tx.send(SttEvent {
                                kind: SttEventKind::Final,
                                text,
                                source,
                                start_time: Some(start_time),
                                end_time: Some(end_time),
                            });
                        }
                        Ok(SidecarLine::Error { message }) => {
                            log::error!("STT sidecar error: {message}");
                            if !ready_signaled {
                                ready_signaled = true;
                                let _ = ready_tx.send(Err(message));
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
        })
    }

    /// Encodes f32 samples (range -1.0..1.0) as PCM16 and writes them to the
    /// sidecar's stdin using the length-prefixed frame protocol.
    pub fn send_samples(&mut self, samples: &[f32]) -> Result<(), String> {
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

    /// Sends the zero-length flush marker so the sidecar finalizes any in-progress
    /// utterance immediately (used when the user pauses/stops recording).
    pub fn flush(&mut self) -> Result<(), String> {
        let Some(stdin) = self.stdin.as_mut() else {
            return Err("sidecar stdin already closed".into());
        };
        write_frame(stdin, &[])
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
