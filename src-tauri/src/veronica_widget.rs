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

/// Fixed square side. Sized generously around the 56px orb (see
/// `.veronica-widget-orb` in overlay.css) so its glow rings — which expand
/// to ~2.2x their base size while fading out — finish shrinking to nothing
/// well before reaching the window edge. A tighter window clipped the ring's
/// soft-edged glow against the window's hard rectangular boundary, which
/// read as a faint square flickering in time with the pulse.
const WIDGET_SIDE: f64 = 160.0;

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

fn dock_bottom_right(window: &WebviewWindow) {
    let Ok(Some(monitor)) = window.primary_monitor() else {
        return;
    };
    let scale = monitor.scale_factor();
    let monitor_size = monitor.size().to_logical::<f64>(scale);
    let monitor_pos = monitor.position().to_logical::<f64>(scale);

    let x = monitor_pos.x + monitor_size.width - WIDGET_SIDE - RIGHT_MARGIN;
    let y = monitor_pos.y + monitor_size.height - WIDGET_SIDE - BOTTOM_MARGIN;
    let _ = window.set_size(LogicalSize::new(WIDGET_SIDE, WIDGET_SIDE));
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
        dock_bottom_right(&existing);
        existing.show().map_err(|e| e.to_string())?;
        let _ = windows_capture_protection::enable_capture_exclusion(&existing);
    } else {
        let window = build_widget_window(app)?;
        dock_bottom_right(&window);
        if windows_capture_protection::enable_capture_exclusion(&window).is_err() {
            log::warn!("Veronica widget: screen-capture exclusion FAILED/UNAVAILABLE");
        }
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

#[tauri::command]
pub async fn show_veronica_widget(app: AppHandle) -> Result<(), String> {
    crate::veronica_window::run_on_main(&app, |app| show_widget(app)).await
}

#[tauri::command]
pub async fn hide_veronica_widget(app: AppHandle) -> Result<(), String> {
    crate::veronica_window::run_on_main(&app, |app| hide_widget(app)).await
}
