//! Groq Cloud Whisper transcription for one already-endpointed speech
//! segment — the only source of transcript text in this app.
//!
//! This is deliberately not a second STT pipeline: the local sherpa-onnx
//! sidecar (`SttSidecar`) still does all capture-format handling, streaming,
//! and — critically — voice-activity/endpoint detection. This module only
//! transcribes each utterance it detects, called from `SttSidecar`'s
//! stdout-reader thread once an utterance boundary is found.
//!
//! Blocking HTTP client, not async: the caller is a plain `std::thread`
//! reading the sidecar's stdout line-by-line, not a tokio task, and
//! `reqwest`'s `blocking` feature is already a declared dependency feature
//! (see Cargo.toml) — spinning up a tokio runtime just for this one call
//! would be more moving parts for no benefit, since the caller's own loop
//! is already synchronous.

use std::io::Write as _;
use std::time::Duration;

const TRANSCRIPTIONS_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";
const MODEL: &str = "whisper-large-v3-turbo";

/// Generous relative to Groq's typical sub-second response time for a
/// short utterance, but bounded — a hung request must not stall the STT
/// event stream (and therefore the whole live transcript) indefinitely.
/// Overridable for anyone on an unusually slow connection, same pattern as
/// `STT_READY_TIMEOUT_MS`.
fn request_timeout() -> Duration {
    std::env::var("GROQ_TIMEOUT_MS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(Duration::from_millis(15_000))
}

/// Distinguishes failure modes so the caller's log line says something
/// actionable (missing key vs. a transient network/rate-limit blip vs. an
/// API-shape mismatch). The `Display` impl is what ends up in logs — never
/// includes the API key.
#[derive(Debug)]
pub enum GroqError {
    MissingKey,
    Network(String),
    Timeout,
    RateLimited,
    Http { status: u16, body: String },
    UnexpectedResponse(String),
}

impl std::fmt::Display for GroqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingKey => write!(f, "no Groq API key configured (Settings -> API Keys)"),
            Self::Network(e) => write!(f, "network error reaching Groq: {e}"),
            Self::Timeout => write!(f, "Groq request timed out"),
            Self::RateLimited => write!(f, "Groq rate limit hit (429)"),
            Self::Http { status, body } => write!(f, "Groq returned {status}: {body}"),
            Self::UnexpectedResponse(e) => write!(f, "Groq returned an unexpected response: {e}"),
        }
    }
}

/// Loads the key from the same Windows Credential Manager store as the LLM
/// provider keys (Settings -> API Keys in the main window) rather than an
/// env var — read fresh on every call rather than cached, so saving/
/// clearing the key in Settings and restarting just the mic/capture session
/// (not the whole app) picks up the change. Never logged: only
/// `GroqError::MissingKey`'s fixed message is ever surfaced, never the key's
/// value or even whether a malformed one was set.
fn api_key() -> Result<String, GroqError> {
    match crate::personal::api_key_store::load_key("groq") {
        Ok(Some(key)) if !key.trim().is_empty() => Ok(key),
        _ => Err(GroqError::MissingKey),
    }
}

/// Transcribes one complete utterance's worth of 16kHz mono f32 samples
/// (range -1.0..1.0, the same format `SttSidecar::send_samples` encodes for
/// the local sidecar) via Groq's Whisper endpoint. Encodes to a small WAV in
/// memory — no temp file — since Groq's multipart endpoint wants a named
/// file part, not a raw byte stream.
pub fn transcribe(samples: &[f32], sample_rate: u32) -> Result<String, GroqError> {
    // Checked before api_key(): there is nothing to transcribe either way,
    // so this must not require a configured key to report "no text" —
    // keeps behavior identical whether or not Groq is set up yet.
    if samples.is_empty() {
        return Ok(String::new());
    }
    let key = api_key()?;

    let wav_bytes = encode_wav_pcm16(samples, sample_rate);

    let client = reqwest::blocking::Client::builder()
        .timeout(request_timeout())
        .build()
        .map_err(|e| GroqError::Network(e.to_string()))?;

    let part = reqwest::blocking::multipart::Part::bytes(wav_bytes)
        .file_name("segment.wav")
        .mime_str("audio/wav")
        .map_err(|e| GroqError::Network(e.to_string()))?;
    let form = reqwest::blocking::multipart::Form::new()
        .text("model", MODEL)
        .text("response_format", "json")
        .part("file", part);

    let response = client
        .post(TRANSCRIPTIONS_URL)
        .bearer_auth(&key)
        .multipart(form)
        .send()
        .map_err(|e| {
            if e.is_timeout() {
                GroqError::Timeout
            } else {
                GroqError::Network(e.to_string())
            }
        })?;

    let status = response.status();
    if status.as_u16() == 429 {
        return Err(GroqError::RateLimited);
    }
    if !status.is_success() {
        let body = response.text().unwrap_or_default();
        return Err(GroqError::Http { status: status.as_u16(), body });
    }

    let json: serde_json::Value = response
        .json()
        .map_err(|e| GroqError::UnexpectedResponse(e.to_string()))?;
    Ok(json.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string())
}

/// Encodes f32 samples (-1.0..1.0) as a minimal mono 16-bit PCM WAV file in
/// memory. Mirrors the same clamp-and-scale `SttSidecar::send_samples` uses
/// for the local sidecar's PCM16LE frames, just wrapped in a WAV header
/// instead of this crate's own length-prefixed protocol, since Groq expects
/// a standard audio file format.
fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let sample_i16 = (clamped * i16::MAX as f32) as i16;
        pcm.extend_from_slice(&sample_i16.to_le_bytes());
    }

    let data_len = pcm.len() as u32;
    let byte_rate = sample_rate * 2; // mono, 16-bit
    let mut wav = Vec::with_capacity(44 + pcm.len());
    let _ = wav.write_all(b"RIFF");
    let _ = wav.write_all(&(36 + data_len).to_le_bytes());
    let _ = wav.write_all(b"WAVEfmt ");
    let _ = wav.write_all(&16u32.to_le_bytes()); // PCM fmt chunk size
    let _ = wav.write_all(&1u16.to_le_bytes()); // PCM
    let _ = wav.write_all(&1u16.to_le_bytes()); // mono
    let _ = wav.write_all(&sample_rate.to_le_bytes());
    let _ = wav.write_all(&byte_rate.to_le_bytes());
    let _ = wav.write_all(&2u16.to_le_bytes()); // block align
    let _ = wav.write_all(&16u16.to_le_bytes()); // bits per sample
    let _ = wav.write_all(b"data");
    let _ = wav.write_all(&data_len.to_le_bytes());
    let _ = wav.write_all(&pcm);
    wav
}

#[cfg(test)]
mod tests {
    use super::*;

    // The API key now comes from Windows Credential Manager (see `api_key`),
    // not an env var this test suite can control in isolation — whether
    // `transcribe` sees `MissingKey` depends on whether a real "groq" entry
    // happens to be stored on the machine running the test, which a unit
    // test must not assume either way. `#[ignore]`'d for the same reason
    // `api_key_store`'s own round-trip test is: requires a real, clean
    // Credential Manager state. Run manually with `cargo test -- --ignored`
    // on a real desktop session after clearing any stored "groq" key.
    #[test]
    #[ignore = "depends on real Windows Credential Manager state (no stored 'groq' key) — not controllable from a unit test"]
    fn missing_key_is_reported_without_making_a_request() {
        let err = transcribe(&[0.0, 0.1, -0.1], 16_000).unwrap_err();
        assert!(matches!(err, GroqError::MissingKey));
    }

    #[test]
    fn empty_samples_short_circuit_to_empty_text_regardless_of_key_state() {
        // No network call is made for empty input, checked before `api_key()`
        // is even consulted — see `transcribe`'s early return above — so this
        // holds whether or not a real key happens to be configured.
        let result = transcribe(&[], 16_000);
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn wav_header_reports_correct_data_length_and_format() {
        let samples = vec![0.0f32; 1600]; // 100ms at 16kHz
        let wav = encode_wav_pcm16(&samples, 16_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        let data_len = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_len, 1600 * 2); // 16-bit samples
        assert_eq!(wav.len(), 44 + 1600 * 2);
    }

    #[test]
    fn display_never_includes_a_key_value() {
        let err = GroqError::MissingKey;
        assert!(!err.to_string().to_lowercase().contains("gsk_"));
    }
}
