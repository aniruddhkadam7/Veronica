//! Speech-to-text pipeline.
//!
//! `SttSidecar` wraps a local sherpa-onnx child process (audio in over
//! stdin, detected-utterance events out over stdout) purely for
//! voice-activity/endpoint detection — it decides where one utterance ends
//! and the next begins. Its own transcribed text is never used.
//!
//! Groq Cloud's Whisper API (`whisper-large-v3-turbo`, see `groq.rs`) is the
//! only source of transcript text: each time the local engine detects an
//! utterance boundary, the raw audio buffered for that utterance (not
//! continuous silence — only what the local engine judged to be speech) is
//! sent to Groq and its result becomes the `Final` event. If Groq fails for
//! any reason (missing/invalid `GROQ_API_KEY`, timeout, rate limit, network
//! error), that utterance is dropped rather than silently reported with
//! placeholder or stale text — see `sidecar.rs`'s reader thread.
//!
//! No component in this module (or anywhere else in the app) tries to detect
//! whether a piece of text is a "question" or decide when to call an LLM. It only
//! turns audio into text events.

mod events;
mod groq;
mod sidecar;

pub use events::{SttEvent, SttEventKind};
pub use sidecar::SttSidecar;
