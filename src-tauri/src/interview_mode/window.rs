//! Interview Mode's overlay window — thin wrapper over the shared
//! `overlay_window` module (used by Meeting Mode too) with Interview
//! Mode's own window label/title baked in. Used to be its own
//! near-duplicate implementation of every function below; that's gone now
//! in favor of one shared implementation both modes call into, so a fix to
//! one (e.g. the overlay's on-screen positioning) can't quietly apply to
//! only one mode's overlay while the other keeps the old behavior.

use tauri::AppHandle;

use crate::overlay_window;

pub use overlay_window::OverlayCaptureStatus;

// Label kept as "interview-overlay" (not renamed) to minimize churn — this is
// now Veronica's one overlay window for both Interview and Meeting mode, not
// Interview-exclusive; only the displayed title changed.
pub const OVERLAY_WINDOW_LABEL: &str = "interview-overlay";
const OVERLAY_TITLE: &str = "Veronica";

pub fn show_overlay_window(app: &AppHandle) -> Result<OverlayCaptureStatus, String> {
    overlay_window::show_overlay_window(app, OVERLAY_WINDOW_LABEL, OVERLAY_TITLE)
}

pub fn close_overlay_window(app: &AppHandle) -> Result<(), String> {
    overlay_window::close_overlay_window(app, OVERLAY_WINDOW_LABEL)
}

pub fn toggle_overlay_window(app: &AppHandle) -> Result<OverlayCaptureStatus, String> {
    overlay_window::toggle_overlay_window(app, OVERLAY_WINDOW_LABEL, OVERLAY_TITLE)
}

pub fn set_overlay_always_on_top(app: &AppHandle, enabled: bool) -> Result<(), String> {
    overlay_window::set_overlay_always_on_top(app, OVERLAY_WINDOW_LABEL, enabled)
}

pub fn resize_overlay(app: &AppHandle, fraction: f64) -> Result<(), String> {
    overlay_window::resize_overlay(app, OVERLAY_WINDOW_LABEL, fraction)
}

pub fn is_capture_excluded(app: &AppHandle) -> bool {
    overlay_window::is_capture_excluded(app, OVERLAY_WINDOW_LABEL)
}
