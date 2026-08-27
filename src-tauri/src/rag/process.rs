use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::OnceLock;
use std::time::Duration;

use tauri::{path::BaseDirectory, AppHandle, Manager};

use crate::process_util::hidden_command;

const RAG_PORT: u16 = 8100;

/// Per-launch shared secret required on every RAG service request (except
/// /health) — see packages/rag/app/main.py's auth middleware. Generated once
/// per desktop app process and handed to the RAG child via env var at spawn
/// time, so no other local process can drive this service just by knowing
/// its (fixed, well-known) port.
static INTERNAL_TOKEN: OnceLock<String> = OnceLock::new();

fn internal_token() -> &'static str {
    INTERNAL_TOKEN.get_or_init(|| {
        use ring::rand::{SecureRandom, SystemRandom};
        let mut bytes = [0u8; 32];
        SystemRandom::new()
            .fill(&mut bytes)
            .expect("failed to generate RAG internal auth token");
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    })
}

fn rag_package_dir() -> PathBuf {
    // This personal repo's flat layout has no packages/rag monorepo tree —
    // only the prebuilt rag-lite resource (sidecars/rag-lite/) at the repo
    // root. `rag_venv_python()` below will simply never find a venv here
    // (there isn't one in this repo), which is expected: the frozen
    // `rag-lite` resource (see `frozen_rag_lite_path`) is the only supported
    // path in a release build, and is what `tauri dev` needs to resolve too.
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Locates the RAG service's own virtualenv interpreter. It has much heavier
/// dependencies (torch, sentence-transformers) than the STT sidecar, so it
/// gets its own venv (`packages/rag/.venv`) rather than sharing one — this
/// mirrors how `apps/backend` has its own venv too.
///
/// Dev/test only, same caveat as `stt::sidecar`'s `stt_venv_python`: a venv
/// is not relocatable (its `python.exe` hardcodes the absolute path of the
/// base Python install it was created from), so it's never what a release
/// build ships or runs — see `frozen_rag_lite_path` below.
fn rag_venv_python() -> Option<PathBuf> {
    let candidate = rag_package_dir().join(".venv").join("Scripts").join("python.exe");
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// The PyInstaller-frozen "rag-lite" service bundled as a resource for
/// release builds (see `bundle.resources` in `tauri.conf.json`, built via
/// `packages/rag/scripts/freeze_rag_lite.py`) — text extraction
/// (pypdf/python-docx) with no `torch`/`sentence-transformers`, so document
/// upload and the setup screen's CV/JD text analysis work with nothing
/// installed on the target machine. Semantic search over uploaded documents
/// during an interview is the one thing this build can't do — already a
/// best-effort/non-fatal path everywhere it's used (see
/// `RetrievalPlanner`), never a hard requirement. `None` when there's no
/// `AppHandle` or the resource isn't there (plain `cargo build`/`tauri dev`,
/// where the full venv above is used instead).
fn frozen_rag_lite_path(app: Option<&AppHandle>) -> Option<PathBuf> {
    let app = app?;
    let candidate = app.path().resolve("rag-lite/rag-lite.exe", BaseDirectory::Resource).ok()?;
    candidate.exists().then_some(candidate)
}

pub struct RagServiceHandle {
    child: Option<Child>,
}

impl RagServiceHandle {
    pub fn base_url() -> String {
        format!("http://127.0.0.1:{RAG_PORT}")
    }

    /// The shared secret required on every RAG service request — see
    /// `internal_token()`. Generated lazily so callers (e.g. `RagClient::new`)
    /// can fetch it even before `spawn` has run.
    pub fn auth_token() -> &'static str {
        internal_token()
    }

    /// Spawns the RAG service as a child process. Returns `Ok(None)` (not an
    /// error) if neither the frozen `rag-lite` resource nor the dev venv are
    /// available — the caller can still run the rest of the app without
    /// document upload/RAG features available, rather than failing the whole
    /// app to launch.
    ///
    /// `embed_config`: hardware-tier-driven (`RAG_EMBED_BATCH_SIZE`,
    /// `RAG_TORCH_THREADS`), read once at spawn time — Python's `Settings`
    /// class (packages/rag/app/core/config.py) reads these at process
    /// startup and caches them, so applying a new value requires restarting
    /// this process (see `restart`), the same way `packages/rag`'s own
    /// embedding model is loaded once and reused for the process lifetime.
    /// Ignored (but harmless) by the frozen `rag-lite` build, which never
    /// loads an embedding model to begin with.
    ///
    /// `app`: threaded through so a release build can find and run the
    /// frozen `rag-lite` resource — see `frozen_rag_lite_path`. `None` keeps
    /// today's dev-tree venv behavior unchanged.
    pub fn spawn(embed_config: EmbedProcessConfig, app: Option<&AppHandle>) -> Result<Option<Self>, String> {
        let mut command = if let Some(frozen) = frozen_rag_lite_path(app) {
            let mut command = hidden_command(&frozen);
            command.env("RAG_PORT", RAG_PORT.to_string());
            // Defense in depth: RAG_INTERNAL_TOKEN is always set below
            // regardless, so this fail-closed check (packages/rag/app/
            // main.py) never actually fires today — but a release build is
            // exactly the case where "silently start unauthenticated if the
            // token were ever missing" must be a hard error, not a warning.
            // Never set on the dev-tree venv path (the `else` branch below),
            // which should keep failing open/warning for local development.
            command.env("RAG_ENV", "production");
            command
        } else {
            let Some(python) = rag_venv_python() else {
                log::warn!(
                    "RAG service venv not found at packages/rag/.venv — document upload/RAG search will be unavailable"
                );
                return Ok(None);
            };
            let mut command = hidden_command(&python);
            command
                .args([
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    &RAG_PORT.to_string(),
                ])
                .current_dir(rag_package_dir());
            command
        };

        let mut child = command
            .env("RAG_EMBED_BATCH_SIZE", embed_config.embed_batch_size.to_string())
            .env("RAG_TORCH_THREADS", embed_config.torch_threads.to_string())
            .env("RAG_INTERNAL_TOKEN", internal_token())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("failed to spawn RAG service: {e}"))?;

        forward_child_output(child.stdout.take(), "stdout");
        forward_child_output(child.stderr.take(), "stderr");

        Ok(Some(Self { child: Some(child) }))
    }

    pub fn shutdown(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }

    /// Kills the current process and spawns a fresh one with new
    /// embedding-config env vars. Used when the user changes performance
    /// mode — RAG's Python settings are fixed at process startup (see
    /// `spawn`'s doc comment), so a live in-process reconfiguration isn't
    /// possible without a deeper change to how `packages/rag` caches
    /// settings. Callers should treat the brief gap between shutdown and
    /// the new process becoming healthy the same way initial startup is
    /// treated (poll `/health`), not as an error.
    pub fn restart(&mut self, embed_config: EmbedProcessConfig, app: Option<&AppHandle>) -> Result<(), String> {
        self.shutdown();
        let Some(mut spawned) = Self::spawn(embed_config, app)? else {
            return Err("RAG service venv not found — cannot restart".to_string());
        };
        // `.take()`, not a whole-struct move — `spawned` still runs its
        // `Drop` impl at the end of this scope, which is a no-op once its
        // `child` has been taken (mirrors `shutdown()`'s own `self.child.take()`).
        self.child = spawned.child.take();
        Ok(())
    }
}

/// The subset of `hardware::PerformanceConfig` relevant to spawning the RAG
/// child process — kept as its own small struct (rather than passing the
/// full `PerformanceConfig` here) so `rag::process` does not need to depend
/// on the `hardware` module's other STT/retrieval fields it has no use for.
#[derive(Debug, Clone, Copy)]
pub struct EmbedProcessConfig {
    pub embed_batch_size: u32,
    pub torch_threads: u32,
}

impl Drop for RagServiceHandle {
    /// Same reasoning as `SttSidecar`'s Drop impl: without this, an unclean
    /// Rust-side exit would leave the RAG service (and the memory-heavy
    /// embedding model it has loaded) running as an orphaned process.
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Forwards a child process's stdout/stderr into the `log` crate on a
/// dedicated thread, the same pattern used for the STT sidecar's stderr
/// (`stt::sidecar`) — keeps RAG service logs visible in the desktop app's own
/// log output without blocking anything on the pipe being drained.
fn forward_child_output<R>(pipe: Option<R>, label: &'static str)
where
    R: std::io::Read + Send + 'static,
{
    let Some(pipe) = pipe else { return };
    std::thread::Builder::new()
        .name(format!("rag-service-{label}"))
        .spawn(move || {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(pipe);
            for line in reader.lines().map_while(Result::ok) {
                log::info!("[rag service {label}] {line}");
            }
        })
        .ok();
}

/// Polls the RAG service's `/health` endpoint until it responds or the timeout
/// elapses. Called once after spawning, before the app reports the service as
/// available.
pub fn wait_until_healthy(timeout: Duration) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    let url = format!("{}/health", RagServiceHandle::base_url());
    while std::time::Instant::now() < deadline {
        if let Ok(response) = reqwest::blocking::get(&url) {
            if response.status().is_success() {
                return true;
            }
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    false
}

/// The embedding model can take a while to load on first request (see
/// docs/progress.md Step 9 — ~35s cold, cached after) plus uvicorn's own
/// startup, so this uses a generous timeout; a `false` return just means the
/// RAG service commands will report "unavailable" until it does come up.
pub fn wait_until_healthy_default() -> bool {
    wait_until_healthy(Duration::from_secs(60))
}
