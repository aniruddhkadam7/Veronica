//! Installs rustls's default crypto provider once per process.
//!
//! `reqwest` here is built with `default-features = false` (no built-in TLS
//! backend of its own), but `rustls` still ends up in the dependency tree
//! transitively (via other plugins), which means every `reqwest::Client`
//! actually performing an HTTPS request — including the direct calls to
//! OpenAI/Anthropic/Gemini in `personal::providers::*` — goes through
//! `rustls::ClientConfig` under the hood. rustls 0.23+ requires an explicit
//! crypto provider to be installed process-wide before any such client is
//! built — otherwise every `reqwest::Client::builder().build()` in the
//! process panics with "No rustls crypto provider is configured."
//!
//! `install_default()` is itself safe to call from multiple threads/tests —
//! it no-ops (returns `Err`, ignored here) if a provider is already
//! installed, so this is safe to call at the top of both `lib.rs::run()` and
//! any test module that constructs a `reqwest::Client`.
use std::sync::Once;

static INIT: Once = Once::new();

/// Cheap to call from every `reqwest::Client` construction site (production
/// and test) — the actual install only runs once per process via `Once`.
pub fn ensure_installed() {
    INIT.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}
