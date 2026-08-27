//! Meeting Mode: a frameless, always-on-top overlay window for live
//! meetings — transcript, live suggested answers, and tracked key points/
//! decisions/action items, ending in an AI meeting summary archived to
//! Meeting history.
//!
//! Built on the shared `overlay_window` module (parameterized by label)
//! rather than a bespoke window implementation.

pub mod commands;
pub mod history;

pub const MEETING_OVERLAY_LABEL: &str = "meeting-overlay";
