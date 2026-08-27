//! Deepgram Cloud TTS (Aura-1) client for one chunk of text.
//!
//! This is the only TTS provider in the app — there is no local model, no
//! fallback. Deepgram's `/v1/speak` endpoint is called with
//! `encoding=linear16&container=none`: raw 16-bit signed little-endian PCM
//! frames stream back in the HTTP response body as they're synthesized, with
//! no WAV/container header to wait for or parse, so playback can start on
//! the very first chunk of bytes rather than waiting for the whole response.
//!
//! Blocking HTTP client, not async: the caller (`tts::player`) runs this on
//! a dedicated `std::thread` per sentence-chunk, mirroring
//! `stt::groq::transcribe`'s reasoning exactly — `reqwest`'s `blocking`
//! feature is already a declared dependency feature (see Cargo.toml), and
//! `ask_veronica`'s `on_delta` callback (the caller of `TtsSession::speak`)
//! is a plain synchronous closure invoked from inside an async Tauri
//! command's streaming loop, so blocking it on network I/O would stall that
//! whole stream, not just speech — a thread per chunk avoids that instead of
//! threading a tokio runtime through a sync callback.

use std::io::Read;
use std::time::Duration;

const SPEAK_URL: &str = "https://api.deepgram.com/v1/speak";

/// Aura-1 voice. Fixed, not user-configurable: the user picked this specific
/// voice, and switching it is a deliberate code change, not a runtime
/// setting (unlike the API key, which is a secret entered in Settings and
/// must never live in source).
const VOICE_MODEL: &str = "aura-asteria-en";

/// Deepgram's documented Aura-1 sample rate for linear16 output. Fixed
/// alongside `VOICE_MODEL` — the player (`tts::player`) is built assuming
/// this exact rate, so changing one without the other would produce audio
/// at the wrong pitch/speed.
pub const SAMPLE_RATE: u32 = 24_000;

/// Generous relative to Deepgram's documented sub-300ms time-to-first-byte
/// for Aura, but bounded — a hung request must not leave speech silently
/// stalled forever. Overridable, same pattern as `GROQ_TIMEOUT_MS`.
fn request_timeout() -> Duration {
    std::env::var("DEEPGRAM_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(15_000))
}

/// Distinguishes failure modes so the caller's log line says something
/// actionable. The `Display` impl is what ends up in logs — never includes
/// the API key.
#[derive(Debug)]
pub enum DeepgramError {
    MissingKey,
    Network(String),
    Timeout,
    RateLimited,
    Http { status: u16, body: String },
}

impl std::fmt::Display for DeepgramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingKey => write!(f, "no Deepgram API key configured (Settings -> API Keys)"),
            Self::Network(e) => write!(f, "network error reaching Deepgram: {e}"),
            Self::Timeout => write!(f, "Deepgram request timed out"),
            Self::RateLimited => write!(f, "Deepgram rate limit hit (429)"),
            Self::Http { status, body } => write!(f, "Deepgram returned {status}: {body}"),
        }
    }
}

/// Loads the key from the same Windows Credential Manager store as the LLM
/// provider keys (Settings -> API Keys in the main window), read fresh on
/// every call rather than cached — so saving/clearing the key in Settings
/// picks up on the next sentence spoken, matching `stt::groq::api_key`'s
/// reasoning exactly. Never logged.
fn api_key() -> Result<String, DeepgramError> {
    match crate::personal::api_key_store::load_key("deepgram") {
        Ok(Some(key)) if !key.trim().is_empty() => Ok(key),
        _ => Err(DeepgramError::MissingKey),
    }
}

/// Synthesizes `text` and calls `on_chunk` with each raw linear16 PCM chunk
/// (16-bit signed little-endian, mono, `SAMPLE_RATE` Hz) as it arrives over
/// the HTTP response body — this is what makes playback start on the first
/// chunk rather than waiting for the whole utterance. `on_chunk` runs on
/// this same thread; the caller is expected to hand chunks to a player
/// queue quickly rather than block here itself.
pub fn speak_streaming(text: &str, mut on_chunk: impl FnMut(&[u8])) -> Result<(), DeepgramError> {
    // Checked before api_key(): there is nothing to speak either way, so
    // this must not require a configured key to report "nothing to do" —
    // keeps behavior identical whether or not Deepgram is set up yet.
    if text.trim().is_empty() {
        return Ok(());
    }
    let key = api_key()?;

    let client = reqwest::blocking::Client::builder()
        .timeout(request_timeout())
        .build()
        .map_err(|e| DeepgramError::Network(e.to_string()))?;

    // Built manually rather than via a `.query(&[...])` builder call: the
    // parameter values here are all fixed ASCII literals (a voice model
    // name, an encoding name, "none", a sample rate), so there is no user
    // input to percent-encode and no need for a query-building dependency
    // just for this one request.
    let url = format!("{SPEAK_URL}?model={VOICE_MODEL}&encoding=linear16&container=none&sample_rate={SAMPLE_RATE}");

    let mut response = client
        .post(url)
        // Deepgram's REST API authenticates standard API keys with
        // `Authorization: Token <key>`, not `Bearer` — verified against the
        // real API (a Bearer-formatted request came back 401 INVALID_AUTH
        // even with a valid key). `Bearer` is reserved for short-lived JWTs,
        // which this app never uses.
        .header("Authorization", format!("Token {key}"))
        .json(&serde_json::json!({ "text": text }))
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                DeepgramError::Timeout
            } else {
                DeepgramError::Network(e.to_string())
            }
        })?;

    let status = response.status();
    if status.as_u16() == 429 {
        return Err(DeepgramError::RateLimited);
    }
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(DeepgramError::Http { status: status.as_u16(), body });
    }

    // Read in fixed-size chunks rather than the whole body at once — this is
    // what actually delivers "audio starts playing as soon as possible":
    // each `read()` call returns as soon as the OS socket has bytes, which
    // for a streaming response is well before the full utterance has been
    // synthesized. 4096 bytes = ~85ms of audio at 24kHz mono 16-bit, small
    // enough to keep latency low, large enough to not be dominated by
    // per-call syscall overhead.
    let mut buf = [0u8; 4096];
    loop {
        let n = response.read(&mut buf).map_err(|e| DeepgramError::Network(e.to_string()))?;
        if n == 0 {
            break;
        }
        on_chunk(&buf[..n]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // The API key now comes from Windows Credential Manager (see `api_key`),
    // not an env var this test suite can control in isolation — mirrors
    // `stt::groq`'s tests exactly. `#[ignore]`'d for the same reason.
    #[test]
    #[ignore = "depends on real Windows Credential Manager state (no stored 'deepgram' key) — not controllable from a unit test"]
    fn missing_key_is_reported_without_making_a_request() {
        let err = speak_streaming("hello", |_| {}).unwrap_err();
        assert!(matches!(err, DeepgramError::MissingKey));
    }

    #[test]
    fn empty_text_short_circuits_to_ok_with_no_chunks_regardless_of_key_state() {
        // No network call is made for empty/whitespace-only text, checked
        // before `api_key()` is even consulted — see `speak_streaming`'s
        // early return above — so this holds whether or not a real key
        // happens to be configured.
        let mut chunks = 0;
        let result = speak_streaming("   ", |_| chunks += 1);
        assert!(result.is_ok());
        assert_eq!(chunks, 0);
    }

    #[test]
    fn display_never_includes_a_key_value() {
        let err = DeepgramError::MissingKey;
        assert!(!err.to_string().to_lowercase().contains("token"));
    }
}
