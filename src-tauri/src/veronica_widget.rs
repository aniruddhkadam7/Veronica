//! Veronica's "widget" — a small, always-on-top, chrome-less window docked
//! to the bottom-right corner of the screen, showing only the pulsing orb
//! (see `VeronicaWidget.tsx`). No conversation UI at all — this is the
//! "don't show me the chat, just something I can glance at / tap" entry
//! point requested alongside the full overlay and the global hotkey.
//!
//! Distinct from `veronica_window` (the full chat overlay): both can exist
//! independently, but showing one hides the other (see `show_widget`) so
//! there's never a redundant orb floating next to the full overlay.

use tauri::{AppHandle, LogicalPosition, LogicalSize, Manager, WebviewUrl, WebviewWindow, WebviewWindowBuilder};

use crate::veronica_window::OVERLAY_WINDOW_LABEL;
use crate::windows_capture_protection;

pub const WIDGET_WINDOW_LABEL: &str = "veronica-widget";
const WIDGET_TITLE: &str = "Veronica";

/// Default square side, used only until the user picks a different orb size
/// in Settings (see `resize_veronica_widget`) — sized generously around the
/// default 160px ParticlesOrb (see VeronicaWidget.tsx) so its particles'
/// outward expansion/ripple at high energy states (listening/speaking)
/// finishes well before reaching the window edge, which would otherwise clip
/// that motion against the window's hard rectangular boundary.
const WIDGET_SIDE: f64 = 220.0;

/// How much bigger the window is than the orb it hosts, in both directions —
/// same margin `WIDGET_SIDE`/160px default orb implies (220/160 = 1.375),
/// kept as a ratio so `resize_veronica_widget` can preserve the same
/// clipping-free clearance at any orb size the user picks.
const WIDGET_MARGIN_RATIO: f64 = WIDGET_SIDE / 160.0;

/// Gap from the right screen edge so the widget doesn't sit flush against
/// the corner.
const RIGHT_MARGIN: f64 = 24.0;

/// Gap from the bottom screen edge — larger than `RIGHT_MARGIN` so the
/// widget clears the Windows taskbar (and system tray icons) instead of
/// sitting right against it.
const BOTTOM_MARGIN: f64 = 90.0;

fn build_widget_window(app: &AppHandle) -> Result<WebviewWindow, String> {
    let window = WebviewWindowBuilder::new(app, WIDGET_WINDOW_LABEL, WebviewUrl::App("index.html".into()))
        .title(WIDGET_TITLE)
        .inner_size(WIDGET_SIDE, WIDGET_SIDE)
        .resizable(false)
        .decorations(false)
        .transparent(true)
        .background_color(tauri::window::Color(0, 0, 0, 0))
        .always_on_top(true)
        .skip_taskbar(true)
        .shadow(false)
        .visible(false)
        .focused(false)
        .build()
        .map_err(|e| format!("failed to create Veronica widget window: {e}"))?;

    // Without this, DWM draws its default accent-colored 1px border around
    // the window regardless of `decorations(false)` — on this window, fully
    // transparent apart from the round orb, that thin square outline reads
    // as an obvious box drawn around the circle. See
    // `main_window::suppress_window_border`'s doc for the full context (the
    // main window hits the same issue and fixes it the same way).
    crate::main_window::suppress_window_border(&window);

    Ok(window)
}

/// Docks the window to the bottom-right corner at the given square side.
/// Re-docking (rather than just resizing in place) on every size change
/// keeps the visible orb anchored to the same corner instead of growing
/// toward the center of the screen, which is what a naive `set_size` alone
/// would do (Tauri resizes from the window's top-left origin).
fn dock_bottom_right(window: &WebviewWindow, side: f64) {
    let Ok(Some(monitor)) = window.primary_monitor() else {
        return;
    };
    let scale = monitor.scale_factor();
    let monitor_size = monitor.size().to_logical::<f64>(scale);
    let monitor_pos = monitor.position().to_logical::<f64>(scale);

    let x = monitor_pos.x + monitor_size.width - side - RIGHT_MARGIN;
    let y = monitor_pos.y + monitor_size.height - side - BOTTOM_MARGIN;
    let _ = window.set_size(LogicalSize::new(side, side));
    let _ = window.set_position(LogicalPosition::new(x, y));
}

/// Shows the widget (creating it on first use), docked to the bottom-right
/// corner, and hides the full chat overlay if it happens to be open — the
/// two are alternate views of "Veronica is available", never shown at once.
/// Does NOT touch the main window's visibility, unlike the full overlay's
/// `show_overlay_window` — the widget is meant to float alongside whatever
/// else is on screen, not replace it.
pub fn show_widget(app: &AppHandle) -> Result<(), String> {
    if let Some(overlay) = app.get_webview_window(OVERLAY_WINDOW_LABEL) {
        let _ = overlay.hide();
    }

    if let Some(existing) = app.get_webview_window(WIDGET_WINDOW_LABEL) {
        dock_bottom_right(&existing, WIDGET_SIDE);
        existing.show().map_err(|e| e.to_string())?;
        // Screen-capture exclusion is currently disabled — the widget should
        // stay visible in screen shares/recordings. See show_overlay_window
        // for the matching decision on the full overlay.
        let _ = windows_capture_protection::disable_capture_exclusion(&existing);
    } else {
        let window = build_widget_window(app)?;
        dock_bottom_right(&window, WIDGET_SIDE);
        let _ = windows_capture_protection::disable_capture_exclusion(&window);
        window.show().map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn hide_widget(app: &AppHandle) -> Result<(), String> {
    if let Some(window) = app.get_webview_window(WIDGET_WINDOW_LABEL) {
        window.hide().map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn resize_widget_inner(app: &AppHandle, orb_size: f64) -> Result<(), String> {
    let Some(window) = app.get_webview_window(WIDGET_WINDOW_LABEL) else {
        // Not created yet (widget never shown this session) — Settings simply
        // saves the choice (see widgetSettings.ts) for the next time it opens,
        // nothing to resize right now.
        return Ok(());
    };
    let side = orb_size * WIDGET_MARGIN_RATIO;
    dock_bottom_right(&window, side);
    Ok(())
}

#[tauri::command]
pub async fn show_veronica_widget(app: AppHandle) -> Result<(), String> {
    crate::veronica_window::run_on_main(&app, |app| show_widget(app)).await
}

#[tauri::command]
pub async fn hide_veronica_widget(app: AppHandle) -> Result<(), String> {
    crate::veronica_window::run_on_main(&app, |app| hide_widget(app)).await
}

/// Applied when the user changes the orb size slider in Settings. Resizes
/// and re-docks the (already-open) widget window to keep the same
/// clipping-free margin around whatever size orb is now showing — a no-op if
/// the widget hasn't been opened yet this session.
#[tauri::command]
pub async fn resize_veronica_widget(app: AppHandle, orb_size: f64) -> Result<(), String> {
    crate::veronica_window::run_on_main(&app, move |app| resize_widget_inner(app, orb_size)).await
}
