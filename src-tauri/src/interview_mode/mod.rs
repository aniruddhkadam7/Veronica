//! Interview Mode: a small, frameless, transparent, always-on-top overlay
//! window that shows the live transcript and lets the user press ENTER to get
//! an AI-generated answer for the currently-displayed question, without
//! leaving that one window (spec: no page navigation during an interview).
//!
//! This is additive — the existing main window (recording, prepare, full
//! analysis dashboard) is untouched and still reachable; Interview Mode is a
//! second window the user can open/close independently.

pub mod commands;
mod window;

pub use window::{
    close_overlay_window, is_capture_excluded, show_overlay_window, toggle_overlay_window,
    OverlayCaptureStatus, OVERLAY_WINDOW_LABEL,
};
