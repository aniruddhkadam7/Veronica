//! Shared frameless/always-on-top overlay window mechanics, parameterized by
//! window label — used by `veronica_window`, which calls through to this
//! module with Veronica's label/title baked in. Kept parameterized rather
//! than hardcoded to one label since this module previously served two
//! separate overlay windows (Interview/Meeting Mode, since merged into one
//! Veronica overlay) and diverging duplicates of this logic had quietly
//! drifted apart before that merge.

use tauri::{
    window::Color, AppHandle, Emitter, LogicalPosition, LogicalSize, Manager, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder,
};

use crate::main_window::MAIN_WINDOW_LABEL;
use crate::windows_capture_protection;

const OVERLAY_FALLBACK_SIDE: f64 = 560.0;
const DEFAULT_SIZE_FRACTION: f64 = 0.75;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct OverlayCaptureStatus {
    pub excluded: bool,
}

/// Creates the overlay window (if it doesn't exist), centers it on the
/// primary monitor (see `center_overlay_on_screen`), applies capture
/// exclusion, shows it, then hides the main window so the overlay is the
/// only thing visible — hidden LAST, only once the overlay is actually
/// positioned/shown, so there's no visible gap where the app looks like it
/// closed. `title` is the OS window title; `label` distinguishes this
/// overlay from the other mode's overlay (both modes never have an overlay
/// open at the same time, but each needs its own window identity).
pub fn show_overlay_window(app: &AppHandle, label: &str, title: &str) -> Result<OverlayCaptureStatus, String> {
    let excluded = if let Some(existing) = app.get_webview_window(label) {
        // Reset the frontend's session-scoped React state before showing a
        // *reused* overlay window (the WebView2 process/DOM is never torn
        // down between close and reopen — only hidden) — otherwise a
        // meeting ended (or mid-ending) on the previous close would still
        // be showing its Summary screen (or stuck "Ending…") the instant
        // this window reappears for a brand-new meeting, before any of
        // this new session's events have even arrived. Best-effort: the
        // webview may not have a listener registered yet on the very first
        // reuse, which is fine — there's nothing stale to clear at that
        // point anyway.
        let _ = existing.emit("overlay:reset-session", ());
        center_overlay_on_screen(&existing, false);
        existing.show().map_err(|e| e.to_string())?;
        existing.set_focus().map_err(|e| e.to_string())?;
        // Screen-capture exclusion is currently disabled — the overlay
        // should stay visible in screen shares/recordings rather than being
        // hidden from them.
        let _ = windows_capture_protection::disable_capture_exclusion(&existing);
        false
    } else {
        let window = build_overlay_window(app, label, title)?;
        center_overlay_on_screen(&window, true);

        let _ = windows_capture_protection::disable_capture_exclusion(&window);
        log::info!("overlay '{label}': screen-capture exclusion disabled (visible in screen share)");

        window.show().map_err(|e| e.to_string())?;
        window.set_focus().map_err(|e| e.to_string())?;
        false
    };

    if let Some(main) = app.get_webview_window(MAIN_WINDOW_LABEL) {
        if let Err(e) = main.hide() {
            log::warn!("failed to hide main window when entering overlay '{label}': {e}");
        }
    }

    Ok(OverlayCaptureStatus { excluded })
}

/// Hides the overlay and brings the main window back, since it's hidden on
/// entry (see `show_overlay_window`).
///
/// Also emits `interview-mode:overlay-closed` to every window: the overlay
/// and main window are separate webviews with no shared React state, so the
/// main window's own session status has no other way to learn the overlay
/// closed unless it was the one that called this. Reusing the same event
/// name across both modes (rather than a Meeting-specific one) keeps the
/// main window's listener a single one covering every mode, since it only
/// ever has one active overlay/session at a time regardless of mode.
pub fn close_overlay_window(app: &AppHandle, label: &str) -> Result<(), String> {
    log::info!("close_overlay_window '{label}': on main thread, hiding");
    if let Some(window) = app.get_webview_window(label) {
        window.hide().map_err(|e| e.to_string())?;
    }
    log::info!("close_overlay_window '{label}': overlay hidden, showing main");
    if let Some(main) = app.get_webview_window(MAIN_WINDOW_LABEL) {
        main.show().map_err(|e| e.to_string())?;
        main.set_focus().map_err(|e| e.to_string())?;
    }
    log::info!("close_overlay_window '{label}': main shown, emitting event");
    let _ = app.emit("interview-mode:overlay-closed", ());
    log::info!("close_overlay_window '{label}': done");
    Ok(())
}

pub fn toggle_overlay_window(app: &AppHandle, label: &str, title: &str) -> Result<OverlayCaptureStatus, String> {
    if let Some(window) = app.get_webview_window(label) {
        let visible = window.is_visible().unwrap_or(false);
        if visible {
            close_overlay_window(app, label)?;
            return Ok(OverlayCaptureStatus { excluded: is_capture_excluded(app, label) });
        }
    }
    show_overlay_window(app, label, title)
}

pub fn is_capture_excluded(app: &AppHandle, label: &str) -> bool {
    app.get_webview_window(label).is_some()
}

pub fn set_overlay_always_on_top(app: &AppHandle, label: &str, enabled: bool) -> Result<(), String> {
    if let Some(window) = app.get_webview_window(label) {
        window.set_always_on_top(enabled).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn resize_overlay(app: &AppHandle, label: &str, fraction: f64) -> Result<(), String> {
    let Some(window) = app.get_webview_window(label) else {
        return Ok(());
    };
    if let Ok(Some(monitor)) = window.primary_monitor() {
        let scale = monitor.scale_factor();
        let monitor_size = monitor.size().to_logical::<f64>(scale);
        let side = (monitor_size.width.min(monitor_size.height) * fraction).max(320.0);
        let _ = window.set_size(LogicalSize::new(side, side));
    }
    center_overlay_on_screen(&window, false);
    Ok(())
}

fn build_overlay_window(app: &AppHandle, label: &str, title: &str) -> Result<WebviewWindow, String> {
    WebviewWindowBuilder::new(app, label, WebviewUrl::App("index.html".into()))
        .title(title)
        .inner_size(OVERLAY_FALLBACK_SIDE, OVERLAY_FALLBACK_SIDE)
        .min_inner_size(320.0, 320.0)
        .resizable(true)
        .decorations(false)
        .transparent(true)
        .background_color(Color(0, 0, 0, 0))
        .always_on_top(true)
        .skip_taskbar(true)
        .shadow(false)
        .visible(false)
        .focused(false)
        .build()
        .map_err(|e| format!("failed to create overlay window '{label}': {e}"))
}

/// How far above true vertical center the overlay sits, so it reads as
/// docked close to the app's toolbar rather than dead-center of the screen
/// — the main window itself always opens near the top of the screen (see
/// main_window.rs's `position_top_center`/`TOP_MARGIN`), so the overlay
/// should land close to it rather than drifting toward the middle of a
/// tall monitor.
const UPWARD_BIAS: f64 = 40.0;

/// Positions the overlay centered on the primary monitor — horizontally
/// exactly centered, vertically offset upward by `UPWARD_BIAS`. If
/// `apply_default_size` is set (only true right after `build_overlay_window`
/// creates a brand-new window), also sizes it to a square roughly
/// three-quarters of the primary monitor's shorter dimension first.
/// `resize_overlay` sets its own explicit size before calling this purely
/// for repositioning, so it always passes `false`.
///
/// Always computed from the monitor and the overlay's own *intended* size
/// (recomputed here with the same `DEFAULT_SIZE_FRACTION` math
/// `apply_default_size` uses, rather than read back via
/// `window.outer_size()`) — reading the size back immediately after
/// `set_size()` isn't guaranteed to reflect the new size yet on every
/// platform, and using a stale/default size there was the actual cause of
/// the overlay occasionally landing at an unrelated position on screen.
/// This intentionally doesn't depend on the main window's own position at
/// all — the main window is about to be hidden right after this runs
/// anyway (see `show_overlay_window`), so anchoring to a window that's
/// mid-hide was also part of the inconsistency.
fn center_overlay_on_screen(window: &WebviewWindow, apply_default_size: bool) {
    let Ok(Some(monitor)) = window.primary_monitor() else {
        return;
    };
    let scale = monitor.scale_factor();
    let monitor_size = monitor.size().to_logical::<f64>(scale);
    let monitor_pos = monitor.position().to_logical::<f64>(scale);

    let side = if apply_default_size {
        let side = (monitor_size.width.min(monitor_size.height) * DEFAULT_SIZE_FRACTION).max(360.0);
        let _ = window.set_size(LogicalSize::new(side, side));
        side
    } else {
        window
            .outer_size()
            .map(|s| s.to_logical::<f64>(scale).width)
            .unwrap_or_else(|_| (monitor_size.width.min(monitor_size.height) * DEFAULT_SIZE_FRACTION).max(360.0))
    };

    let x = monitor_pos.x + (monitor_size.width - side) / 2.0;
    let y = (monitor_pos.y + (monitor_size.height - side) / 2.0 - UPWARD_BIAS).max(monitor_pos.y);
    let _ = window.set_position(LogicalPosition::new(x, y));
}
