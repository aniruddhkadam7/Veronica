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

/// Below this RMS (0.0-1.0, see `audio::compute_rms`), an utterance is
/// treated as near-silence/room noise and never sent to Groq at all — the
/// cheapest possible guard against Whisper-family models' well-documented
/// tendency to *hallucinate* plausible-sounding text (often a short stock
/// phrase in an unexpected language) when given silence or noise instead of
/// real speech, rather than reporting "no speech". This only catches the
/// obvious case (near-total silence); see `is_likely_hallucination` below
/// for the case where the local VAD still triggered on something with real
/// energy (a click, a cough, faint background sound) that isn't speech.
const MIN_SPEECH_RMS: f32 = 0.006;

/// Standard Whisper hallucination heuristic (the same one whisper.cpp/
/// faster-whisper use): a segment Whisper itself is unsure contains speech
/// at all reports a high `no_speech_prob` and/or a very negative
/// `avg_logprob` (low confidence in its own token choices) — exactly the
/// signature of "there was no real speech here, but the model still had to
/// emit *something*." Thresholds are the commonly-used defaults for this
/// exact check, not tuned against this app's own data.
const NO_SPEECH_PROB_THRESHOLD: f64 = 0.6;
const AVG_LOGPROB_THRESHOLD: f64 = -1.0;

/// ISO-639-1 code passed as Whisper's `language` hint. English is now the
/// only supported language (see `language.rs`), so — unlike a multilingual
/// allowlist, which Whisper's API has no way to express — a single hard
/// language hint is exactly the right tool here: it biases recognition
/// toward English with no other-supported-language tradeoff to worry about.
const LANGUAGE_HINT: &str = "en";

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
    // Near-silence/room-noise guard — see `MIN_SPEECH_RMS`'s doc. Cheaper
    // than a network call, and the local VAD's own endpoint detection can
    // still trigger on non-speech energy (a click, a chair creak) even with
    // its speech pre-gate disabled (see `stt::sidecar`'s doc on
    // `STT_VAD_GATE_ENABLED`), so this is a real, needed second check, not
    // a redundant one.
    let rms = crate::audio::compute_rms(samples);
    if rms < MIN_SPEECH_RMS {
        // Distinguishes a silence-triggered empty final (this branch) from a
        // genuine "user said nothing transcribable" case (Groq called, came
        // back empty) in logs — requirement 10's confidence/completeness
        // observability. Never logs audio content, only the RMS number.
        log::info!("[STT_SILENCE_DISCARDED] rms={rms:.4}");
        return Ok(String::new());
    }
    let key = api_key()?;

    let wav_bytes = encode_wav_pcm16(samples, sample_rate);

    // Shared, process-wide client (see `http_client`) instead of a fresh one
    // per utterance — keeps the TCP/TLS connection to Groq warm across
    // consecutive utterances in the same session rather than paying a full
    // handshake every time. The per-call timeout still applies, via the
    // request builder rather than the client itself, so `GROQ_TIMEOUT_MS`
    // stays live-tunable exactly as before.
    let client = crate::http_client::shared_blocking_client();

    let part = reqwest::blocking::multipart::Part::bytes(wav_bytes)
        .file_name("segment.wav")
        .mime_str("audio/wav")
        .map_err(|e| GroqError::Network(e.to_string()))?;
    // "verbose_json" (not "json") specifically to get each segment's
    // `no_speech_prob`/`avg_logprob` back — see `is_likely_hallucination`.
    // The plain "json" format only ever returns `text`, with no way to tell
    // "real speech, confidently transcribed" apart from "no real speech,
    // but the model emitted a plausible-looking phrase anyway" (this app's
    // exact live-observed failure: a short foreign-language phrase
    // transcribed from silence/noise and then acted on as if the user had
    // said it).
    //
    // `language: "en"` biases Whisper toward English — a soft recognition
    // hint, not a hard filter (it can still transcribe other languages it
    // hears). See `language.rs` for the actual hard enforcement, which
    // happens after transcription on the returned text.
    let form = reqwest::blocking::multipart::Form::new()
        .text("model", MODEL)
        .text("response_format", "verbose_json")
        .text("language", LANGUAGE_HINT)
        .part("file", part);

    let response = client
        .post(TRANSCRIPTIONS_URL)
        .bearer_auth(&key)
        .timeout(request_timeout())
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
    let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("").trim().to_string();
    if text.is_empty() {
        return Ok(text);
    }
    if is_likely_hallucination(&json) {
        log::info!("[STT_HALLUCINATION_DISCARDED] text={text:?}");
        return Ok(String::new());
    }
    Ok(text)
}

/// Applies the standard Whisper hallucination heuristic (see
/// `NO_SPEECH_PROB_THRESHOLD`/`AVG_LOGPROB_THRESHOLD`'s doc) across every
/// segment in a `verbose_json` response. `false` (never filtered) if the
/// response has no `segments` array at all — an unexpected shape must fail
/// open to "trust the text", not silently start dropping every real
/// transcript.
fn is_likely_hallucination(response: &serde_json::Value) -> bool {
    let Some(segments) = response.get("segments").and_then(|s| s.as_array()) else {
        return false;
    };
    if segments.is_empty() {
        return false;
    }
    segments.iter().all(|segment| {
        let no_speech_prob = segment.get("no_speech_prob").and_then(|v| v.as_f64());
        let avg_logprob = segment.get("avg_logprob").and_then(|v| v.as_f64());
        no_speech_prob.is_some_and(|p| p >= NO_SPEECH_PROB_THRESHOLD) || avg_logprob.is_some_and(|p| p <= AVG_LOGPROB_THRESHOLD)
    })
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

    // -- is_likely_hallucination: the live-observed bug this guards against
    // (a short foreign-language phrase transcribed from silence/noise and
    // then acted on as if the user had said it) --

    #[test]
    fn high_no_speech_prob_is_flagged_as_hallucination() {
        let response = serde_json::json!({
            "text": "그럴까?",
            "segments": [{"no_speech_prob": 0.85, "avg_logprob": -0.3}],
        });
        assert!(is_likely_hallucination(&response));
    }

    #[test]
    fn very_negative_avg_logprob_is_flagged_as_hallucination() {
        let response = serde_json::json!({
            "text": "some garbled text",
            "segments": [{"no_speech_prob": 0.1, "avg_logprob": -1.5}],
        });
        assert!(is_likely_hallucination(&response));
    }

    #[test]
    fn confident_real_speech_is_not_flagged() {
        let response = serde_json::json!({
            "text": "Open VS Code.",
            "segments": [{"no_speech_prob": 0.02, "avg_logprob": -0.15}],
        });
        assert!(!is_likely_hallucination(&response));
    }

    #[test]
    fn one_confident_segment_among_several_is_enough_to_trust_the_transcript() {
        // A multi-segment utterance where only PART of it looks like real
        // speech must not be discarded wholesale.
        let response = serde_json::json!({
            "text": "Open VS Code.",
            "segments": [
                {"no_speech_prob": 0.9, "avg_logprob": -0.2},
                {"no_speech_prob": 0.02, "avg_logprob": -0.1},
            ],
        });
        assert!(!is_likely_hallucination(&response));
    }

    #[test]
    fn missing_segments_array_fails_open_never_filters() {
        let response = serde_json::json!({"text": "some text"});
        assert!(!is_likely_hallucination(&response));
    }

    #[test]
    fn empty_segments_array_fails_open() {
        let response = serde_json::json!({"text": "", "segments": []});
        assert!(!is_likely_hallucination(&response));
    }

    #[test]
    fn near_silence_never_reaches_the_network() {
        // RMS well below MIN_SPEECH_RMS — must short-circuit to empty text
        // exactly like the empty-samples case, regardless of key state.
        let quiet = vec![0.0001f32; 16_000];
        assert_eq!(transcribe(&quiet, 16_000).unwrap(), "");
    }
}
