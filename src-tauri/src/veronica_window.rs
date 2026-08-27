//! Veronica's overlay window — thin wrapper over the shared
//! `overlay_window` module with Veronica's window label/title baked in.
//! There is now only one overlay in the whole app, so this label is no
//! longer chosen to be neutral between two modes — it's just Veronica's
//! window.

use tauri::{AppHandle, Emitter, State};

use crate::overlay_window;
use crate::state::AppState;

pub use overlay_window::OverlayCaptureStatus;

pub const OVERLAY_WINDOW_LABEL: &str = "veronica-overlay";
const OVERLAY_TITLE: &str = "Veronica";

// `pub(crate)`, not private: `veronica_widget` also calls this directly (to
// expand the widget into the full overlay), and reuses `run_on_main` below
// to do so from its own `#[tauri::command]`s rather than duplicating the
// main-thread dispatch dance.
pub(crate) fn show_overlay_window(app: &AppHandle) -> Result<OverlayCaptureStatus, String> {
    overlay_window::show_overlay_window(app, OVERLAY_WINDOW_LABEL, OVERLAY_TITLE)
}

fn close_overlay_window(app: &AppHandle) -> Result<(), String> {
    overlay_window::close_overlay_window(app, OVERLAY_WINDOW_LABEL)
}

fn toggle_overlay_window(app: &AppHandle) -> Result<OverlayCaptureStatus, String> {
    overlay_window::toggle_overlay_window(app, OVERLAY_WINDOW_LABEL, OVERLAY_TITLE)
}

/// Called directly (not through `run_on_main`) from the global-shortcut
/// handler in `lib.rs`, which already runs on the main thread.
pub fn toggle_overlay_window_sync(app: &AppHandle) -> Result<OverlayCaptureStatus, String> {
    toggle_overlay_window(app)
}

/// Brings Veronica to the front from a fully-closed-to-tray state — called
/// from the global hotkey handler and the tray icon (click or "Show
/// Veronica" menu item). Unlike a plain toggle, this only ever opens: it's
/// what a user reaching for Veronica while every window is hidden expects,
/// not a 50/50 toggle that might close something that isn't even visible.
///
/// If the app really was fully hidden beforehand (both windows closed —
/// the "app is closed" case from the hotkey's perspective, even though the
/// process kept running in the tray), emits `veronica:auto-opened` so the
/// overlay knows to play its greeting animation/voice line, matching the
/// "wake up and say something" behavior a hotkey-summoned assistant should
/// have. A plain re-focus while the overlay was already open does not
/// re-trigger the greeting.
pub fn wake_veronica(app: &AppHandle) {
    let was_fully_hidden = crate::tray::app_was_fully_hidden(app);
    let app_for_main = app.clone();
    let result = app.run_on_main_thread(move || {
        if let Err(err) = show_overlay_window(&app_for_main) {
            log::warn!("failed to show Veronica overlay from hotkey/tray: {err}");
            return;
        }
        if was_fully_hidden {
            let _ = app_for_main.emit("veronica:auto-opened", ());
        }
    });
    if let Err(err) = result {
        log::warn!("failed to schedule Veronica wake-up on main thread: {err}");
    }
}

fn set_overlay_always_on_top_inner(app: &AppHandle, enabled: bool) -> Result<(), String> {
    overlay_window::set_overlay_always_on_top(app, OVERLAY_WINDOW_LABEL, enabled)
}

fn resize_overlay_inner(app: &AppHandle, fraction: f64) -> Result<(), String> {
    overlay_window::resize_overlay(app, OVERLAY_WINDOW_LABEL, fraction)
}

// Window creation/show/hide on Windows must happen on the same OS thread
// that owns the window message loop (the main thread). Tauri dispatches
// non-async `#[tauri::command]`s onto its blocking thread pool, NOT the
// main thread, so calling WebviewWindowBuilder::build()/show()/hide()
// directly from here deadlocks. Route the actual window work through
// `run_on_main_thread` and use a channel to bring the result back to the
// (async) command so the IPC call still completes normally.
pub(crate) async fn run_on_main<T, F>(app: &AppHandle, f: F) -> Result<T, String>
where
    T: Send + 'static,
    F: FnOnce(&AppHandle) -> Result<T, String> + Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();
    let app_for_main = app.clone();
    app.run_on_main_thread(move || {
        let result = f(&app_for_main);
        let _ = tx.send(result);
    })
    .map_err(|e| format!("failed to schedule work on main thread: {e}"))?;

    tauri::async_runtime::spawn_blocking(move || {
        rx.recv()
            .map_err(|e| format!("main-thread task did not respond: {e}"))?
    })
    .await
    .map_err(|e| format!("main-thread task panicked: {e}"))?
}

#[tauri::command]
pub async fn show_interview_overlay(app: AppHandle) -> Result<OverlayCaptureStatus, String> {
    run_on_main(&app, show_overlay_window).await
}

#[tauri::command]
pub async fn hide_interview_overlay(app: AppHandle) -> Result<(), String> {
    run_on_main(&app, close_overlay_window).await
}

#[tauri::command]
pub async fn toggle_interview_overlay(app: AppHandle) -> Result<OverlayCaptureStatus, String> {
    run_on_main(&app, toggle_overlay_window).await
}

/// Applied immediately when the user flips "Always on top" in the overlay's
/// Settings panel.
#[tauri::command]
pub async fn set_overlay_always_on_top(app: AppHandle, enabled: bool) -> Result<(), String> {
    run_on_main(&app, move |app| set_overlay_always_on_top_inner(app, enabled)).await
}

/// Applied when the user changes "Overlay size" in Settings. `fraction` is
/// the side length as a fraction of the primary monitor's shorter dimension
/// (small=0.45, medium=0.6, large=0.75 — chosen client-side).
#[tauri::command]
pub async fn resize_interview_overlay(app: AppHandle, fraction: f64) -> Result<(), String> {
    run_on_main(&app, move |app| resize_overlay_inner(app, fraction)).await
}

/// Result of `start_backend_session`, kept for compatibility with the
/// frontend's existing unconditional call. There is no SaaS backend / sign-
/// in / entitlement system in this personal build, so this always succeeds.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendSessionResult {
    pub rejection: Option<String>,
}

#[tauri::command]
pub async fn start_backend_session(
    _state: State<'_, AppState>,
    _stt_mode: String,
) -> Result<BackendSessionResult, String> {
    Ok(BackendSessionResult { rejection: None })
}

#[tauri::command]
pub async fn end_backend_session(_state: State<'_, AppState>) -> Result<Option<()>, String> {
    Ok(None)
}
