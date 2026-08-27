//! Local speech-to-text pipeline.
//!
//! `SttEngine` wraps the CMU PocketSphinx sidecar (see
//! `packages/stt/pocketsphinx_sidecar/sidecar.py`) as a child process: audio in over
//! stdin, transcript events out over stdout. No audio or transcript data leaves this
//! machine here — this module has no network client.
//!
//! No component in this module (or anywhere else in the app) tries to detect
//! whether a piece of text is a "question" or decide when to call an LLM. It only
//! turns audio into partial/final text events.

mod events;
mod sidecar;

pub use events::{SttEvent, SttEventKind};
pub use sidecar::SttSidecar;
