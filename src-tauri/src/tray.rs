//! System tray icon. Exists so the app can keep running in the background
//! after the main window is closed (see `lib.rs`'s `CloseRequested` handler)
//! — without a tray icon there would be no visible way to bring the app back
//! or fully quit it once every window is hidden, and the global hotkeys
//! (Ctrl+Shift+Space / Ctrl+Shift+V) would be running with no indication the
//! process is still alive.

use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Manager};

use crate::main_window::MAIN_WINDOW_LABEL;

const QUIT_ID: &str = "quit";
const SHOW_ID: &str = "show";

/// Builds and shows the tray icon with a "Show Veronica" / "Quit" menu.
/// Left-clicking the icon itself also shows the overlay (see the tray event
/// handler below), matching the hotkey's behavior — the menu items are there
/// for discoverability/mouse users, not the only way in.
pub fn setup(app: &AppHandle) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, SHOW_ID, "Show Veronica", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, QUIT_ID, "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &quit])?;

    TrayIconBuilder::new()
        .icon(app.default_window_icon().cloned().ok_or(tauri::Error::InvalidIcon(
            std::io::Error::new(std::io::ErrorKind::NotFound, "no default window icon configured"),
        ))?)
        .tooltip("Veronica")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            QUIT_ID => app.exit(0),
            SHOW_ID => crate::veronica_window::wake_veronica(app),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let tauri::tray::TrayIconEvent::Click {
                button: tauri::tray::MouseButton::Left,
                button_state: tauri::tray::MouseButtonState::Up,
                ..
            } = event
            {
                crate::veronica_window::wake_veronica(tray.app_handle());
            }
        })
        .build(app)?;

    Ok(())
}

/// Whether every user-facing window is currently hidden — i.e. the app is
/// running "closed to tray". Used by the global-shortcut handler to decide
/// whether opening the overlay counts as Veronica being freshly "woken up"
/// (and should therefore greet) versus just being toggled while already in
/// use.
pub fn app_was_fully_hidden(app: &AppHandle) -> bool {
    let main_hidden = app
        .get_webview_window(MAIN_WINDOW_LABEL)
        .map(|w| !w.is_visible().unwrap_or(false))
        .unwrap_or(true);
    let overlay_hidden = app
        .get_webview_window(crate::veronica_window::OVERLAY_WINDOW_LABEL)
        .map(|w| !w.is_visible().unwrap_or(false))
        .unwrap_or(true);
    main_hidden && overlay_hidden
}
