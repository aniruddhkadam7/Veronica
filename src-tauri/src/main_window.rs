//! Main window titlebar cosmetics, plus the compact header's anchored
//! dropdown/popover mechanics. The window is frameless (`decorations: false`
//! in tauri.conf.json) with its own custom title-bar strip (App.tsx's
//! `.title-bar`) driving Minimize/Maximize/Restore/Close — there is no
//! native OS titlebar.
//!
//! The window's real OS size is fixed at 760x720 from launch
//! (tauri.conf.json) and never resized for a popover open/close — an
//! earlier version grew/shrank the actual window height per popover
//! (`set_popover_content_height` used to call `set_size`), which reliably
//! read as the whole window visibly moving/jittering on every click no
//! matter how the resize itself was smoothed. 720px is tall enough for the
//! tallest popover (Settings, ~560px including its own chrome) plus the
//! 88px toolbar and gaps; App.css hides the unused space by keeping
//! `.app-shell-compact`'s own background transparent (see
//! `.compact-transparent`) whenever no popover is open, so the window's
//! true fixed bounds are never visually apparent — only the popover's own
//! opaque box appears to "open", exactly like a normal in-page dropdown,
//! while the OS window underneath never changes size or position.
//! `set_popover_content_height` is kept only as a no-op command so the
//! existing frontend call sites (see HeaderDropdown.tsx) don't need to
//! change.

use tauri::{AppHandle, LogicalPosition, Manager};
use windows::Win32::Foundation::COLORREF;
use windows::Win32::Graphics::Dwm::{
    DwmSetWindowAttribute, DWMWA_BORDER_COLOR, DWMWA_CAPTION_COLOR, DWMWA_TEXT_COLOR,
};

pub const MAIN_WINDOW_LABEL: &str = "main";

const COMPACT_WIDTH: f64 = 760.0;

/// The visible height of the main window's toolbar (32px custom title bar +
/// 56px header) — what's actually painted on screen, distinct from the
/// window's real fixed OS height (720px, tauri.conf.json). `pub` so other
/// windows that need to dock relative to what the user actually sees (the
/// Veronica overlay — see veronica_window.rs's dock_below_main_window)
/// use this instead of the real (much taller) `outer_size()`.
pub const COMPACT_HEIGHT: f64 = 88.0;

// Small gap from the very top of the screen so the window doesn't sit flush
// against the physical screen edge — reads as "top of the screen" without
// touching it.
const TOP_MARGIN: f64 = 24.0;

/// Positions the main window horizontally centered and near the top of the
/// primary monitor, then shows it. Called once at launch (see `lib.rs`'s
/// setup hook) so the compact toolbar always opens in the same predictable
/// spot rather than wherever the OS last placed it or however it cascades
/// new windows. The window starts `"visible": false` in tauri.conf.json
/// specifically so this can position it first — showing it already visible
/// at its eventual default OS placement and then immediately moving it
/// would be a visible jump/flash on every launch.
pub fn position_top_center(app: &AppHandle) {
    let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) else {
        return;
    };
    if let Ok(Some(monitor)) = window.primary_monitor() {
        let scale = monitor.scale_factor();
        let monitor_size = monitor.size().to_logical::<f64>(scale);
        let monitor_pos = monitor.position().to_logical::<f64>(scale);

        let x = monitor_pos.x + (monitor_size.width - COMPACT_WIDTH) / 2.0;
        let y = monitor_pos.y + TOP_MARGIN;

        let _ = window.set_position(LogicalPosition::new(x, y));
    }
    let _ = window.show();
}

/// No-op: the main window's real size is fixed (tauri.conf.json) and never
/// resized per popover — see this module's doc comment for why. Kept as a
/// command purely so HeaderDropdown.tsx's existing `invoke` call sites
/// don't need to change.
#[tauri::command]
pub fn set_popover_content_height(_app: AppHandle, _content_height: f64) -> Result<(), String> {
    Ok(())
}

/// Matches the native Windows title bar's caption/text color to the app
/// header's own background (`--surface: #ffffff` / `--text: #17181c` in
/// App.css), and suppresses DWM's default accent-colored window border —
/// with `decorations: false` there's no native caption left for the first
/// two attributes to paint, but the third still matters: Win11 draws a 1px
/// accent-colored border around every top-level window regardless of
/// `decorations`, which read as a stray bright (theme-accent, often
/// magenta) outline around our own CSS-drawn rounded card
/// (`.app-shell-compact` in App.css already paints its own neutral
/// border+shadow). `DWMWA_COLOR_NONE` turns that OS border off so only our
/// own CSS border is visible. Windows 11 only (all three attributes are a
/// no-op — not an error — on Windows 10, which predates this DWM API);
/// failures are logged, never fatal, since this is a cosmetic touch, not
/// functional.
pub fn apply_light_titlebar(app: &AppHandle) {
    let Some(window) = app.get_webview_window(MAIN_WINDOW_LABEL) else {
        return;
    };
    let Ok(hwnd) = window.hwnd() else {
        return;
    };

    // COLORREF is 0x00BBGGRR, not 0x00RRGGBB.
    const CAPTION_COLOR: u32 = 0x00FFFFFF; // #ffffff, R=G=B=0xff so byte order doesn't matter here
    const TEXT_COLOR: u32 = 0x001C1817; // #17181c
    const DWMWA_COLOR_NONE: u32 = 0xFFFFFFFE;

    unsafe {
        let caption = COLORREF(CAPTION_COLOR);
        if let Err(e) = DwmSetWindowAttribute(
            hwnd,
            DWMWA_CAPTION_COLOR,
            &caption as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<COLORREF>() as u32,
        ) {
            log::warn!("failed to set title bar caption color (expected on Windows 10): {e}");
        }
        let text = COLORREF(TEXT_COLOR);
        if let Err(e) = DwmSetWindowAttribute(
            hwnd,
            DWMWA_TEXT_COLOR,
            &text as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<COLORREF>() as u32,
        ) {
            log::warn!("failed to set title bar text color (expected on Windows 10): {e}");
        }
        let border = COLORREF(DWMWA_COLOR_NONE);
        if let Err(e) = DwmSetWindowAttribute(
            hwnd,
            DWMWA_BORDER_COLOR,
            &border as *const _ as *const std::ffi::c_void,
            std::mem::size_of::<COLORREF>() as u32,
        ) {
            log::warn!("failed to suppress window border color (expected on Windows 10): {e}");
        }
    }
}
