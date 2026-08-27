//! Windows screen-capture exclusion for the Interview Mode overlay window.
//!
//! Uses the native `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)` Win32 API
//! (available since Windows 10 version 2004) to ask the OS compositor to omit
//! this window from screen captures, screen recordings, and most screen-share
//! pipelines (which typically capture via the same OS-level APIs this affects —
//! e.g. Desktop Duplication API / Windows.Graphics.Capture).
//!
//! IMPORTANT — this is not a guarantee. `SetWindowDisplayAffinity` affects
//! capture through the standard Windows compositor APIs. It does NOT protect
//! against:
//!   - A physical camera pointed at the screen.
//!   - Third-party/legacy capture methods that bypass DWM (rare in practice,
//!     but exist for the older `WDA_MONITOR` era of GPU/driver-level capture).
//!   - The overlay being visible in the *local* display at all — that's the
//!     point; only the *captured/shared* output excludes it.
//! This module never claims protection succeeded unless the Win32 call itself
//! reported success — see `enable_capture_exclusion`'s return value, which the
//! UI surfaces honestly as "Enabled" or "Unavailable" rather than assuming.

use tauri::WebviewWindow;
use windows::Win32::UI::WindowsAndMessaging::{
    SetWindowDisplayAffinity, WDA_EXCLUDEFROMCAPTURE, WDA_NONE,
};

/// Attempts to exclude `window` from screen captures/screen-share. Returns
/// `Ok(())` only if the underlying Win32 call actually succeeded; returns
/// `Err` with a description otherwise — callers must not assume protection is
/// active just because this function was called.
pub fn enable_capture_exclusion(window: &WebviewWindow) -> Result<(), String> {
    let hwnd = window.hwnd().map_err(|e| format!("failed to get window handle: {e}"))?;
    unsafe {
        SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
            .map_err(|e| format!("SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) failed: {e}"))
    }
}

/// Restores normal (capturable) display affinity for `window`.
pub fn disable_capture_exclusion(window: &WebviewWindow) -> Result<(), String> {
    let hwnd = window.hwnd().map_err(|e| format!("failed to get window handle: {e}"))?;
    unsafe {
        SetWindowDisplayAffinity(hwnd, WDA_NONE)
            .map_err(|e| format!("SetWindowDisplayAffinity(WDA_NONE) failed: {e}"))
    }
}
