//! Process-wide pooled HTTP clients, built once and cloned (cheap — a
//! `reqwest::Client`/`reqwest::blocking::Client` is an `Arc` around its
//! connection pool internally) at every call site that previously built a
//! fresh client per request (Groq STT, RAG, and all three direct LLM
//! providers). Reusing one client keeps TCP/TLS connections (and, for
//! HTTP/2 endpoints, the multiplexed connection itself) alive across calls
//! instead of paying a fresh handshake on every single turn.
//!
//! A `OnceLock` static (rather than threading a client through `AppState`
//! into every call site — `RagClient`'s upload/list/delete/search methods,
//! three provider modules' `generate`/`stream`/`stream_agentic`, and the STT
//! sidecar's per-utterance transcription thread) keeps this change additive:
//! every existing function signature that only needs a client stays the
//! same shape, just swaps `reqwest::Client::new()` for `shared_async_client()`.

use std::sync::OnceLock;

static ASYNC_CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
static BLOCKING_CLIENT: OnceLock<reqwest::blocking::Client> = OnceLock::new();

/// The shared async client for RAG and the three direct LLM provider calls.
/// No default timeout here — every call site applies its own per-request
/// `.timeout(...)` (reqwest supports overriding the client default per
/// request), matching each site's existing, deliberately-chosen timeout
/// value rather than forcing one shared value on all of them.
pub fn shared_async_client() -> reqwest::Client {
    ASYNC_CLIENT
        .get_or_init(|| reqwest::Client::builder().build().expect("failed to build shared reqwest client"))
        .clone()
}

/// The shared blocking client for Groq STT (`stt::groq::transcribe`), called
/// from a plain `std::thread`, never from async context — see that module's
/// doc for why a blocking client is used at all.
pub fn shared_blocking_client() -> reqwest::blocking::Client {
    BLOCKING_CLIENT
        .get_or_init(|| reqwest::blocking::Client::builder().build().expect("failed to build shared blocking reqwest client"))
        .clone()
}
