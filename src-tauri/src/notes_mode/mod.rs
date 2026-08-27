//! Notes Mode: a normal in-app workspace (no overlay window, unlike
//! Interview/Meeting) for quick notes, voice-to-text dictation, AI summaries,
//! and "ask AI about my notes".

pub mod ai;
pub mod dictation;
pub mod store;

pub use dictation::{DictationBuffer, DictationSession};
pub use store::NotesStore;
