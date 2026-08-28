//! The Execute step: native OS calls only (tier 1 in the router's priority
//! list) — no MCP, no UI automation, since nothing in the current safe
//! action set needs them. Every function here follows the codebase's
//! existing convention for wrapping fallible OS calls
//! (windows_capture_protection.rs, process_util.rs): `Result<_, String>`,
//! `.map_err(|e| format!("<what failed>: {e}"))`.

use std::path::{Path, PathBuf};

use windows::core::PCWSTR;
use windows::Win32::System::Power::{GetSystemPowerStatus, SYSTEM_POWER_STATUS};
use windows::Win32::System::SystemInformation::GetLocalTime;
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE};
use winreg::RegKey;

use crate::process_util::hidden_command;

const APP_PATHS_KEY: &str = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths";

/// Looks up `name` in the registry's "App Paths" key (checked under both
/// HKCU and HKLM, matching how Windows itself resolves `Start-Process`/Run
/// dialog launches by bare name), falling back to a Start Menu shortcut
/// scan if the registry has no entry. Errors clearly rather than silently
/// no-op-ing when nothing matches, so the user gets "couldn't find X"
/// instead of Veronica just going quiet.
pub fn resolve_and_launch_app(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("no app name given".into());
    }

    let path = resolve_app_path(trimmed)
        .ok_or_else(|| format!("I couldn't find an app named \"{trimmed}\"."))?;

    hidden_command(&path)
        .spawn()
        .map(|_| format!("Opening {trimmed}."))
        .map_err(|e| format!("failed to launch \"{trimmed}\": {e}"))
}

fn resolve_app_path(name: &str) -> Option<PathBuf> {
    if let Some(path) = registry_app_path(name) {
        return Some(path);
    }
    start_menu_shortcut(name)
}

/// Registry lookup mirrors how `ShellExecute`/the Run dialog resolve a bare
/// app name: `App Paths\<name>.exe`'s default value is the full path.
/// Windows registers most installed desktop apps here regardless of
/// whether the user ever added them to Veronica's own Settings allowlist
/// (see voice_command::launch_app, which stays allowlist-only and
/// unaffected by this — this is a separate, broader resolution path only
/// used by the new action system).
fn registry_app_path(name: &str) -> Option<PathBuf> {
    let candidates = [format!("{name}.exe"), name.to_string()];
    for hive in [HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE] {
        let root = RegKey::predef(hive);
        let Ok(app_paths) = root.open_subkey(APP_PATHS_KEY) else { continue };
        for subkey_name in app_paths.enum_keys().flatten() {
            if candidates.iter().any(|c| subkey_name.eq_ignore_ascii_case(c)) {
                if let Ok(subkey) = app_paths.open_subkey(&subkey_name) {
                    if let Ok(default_value) = subkey.get_value::<String, _>("") {
                        let path = PathBuf::from(default_value);
                        if path.exists() {
                            return Some(path);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Falls back to a recursive filename scan of the Start Menu's Programs
/// folders (per-user and all-users) when the registry has no App Paths
/// entry — covers apps that only ship a Start Menu shortcut (.lnk) without
/// registering themselves in App Paths.
fn start_menu_shortcut(name: &str) -> Option<PathBuf> {
    let target_stem = name.to_lowercase();
    let mut roots = Vec::new();
    if let Ok(appdata) = std::env::var("APPDATA") {
        roots.push(PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs"));
    }
    if let Ok(programdata) = std::env::var("PROGRAMDATA") {
        roots.push(PathBuf::from(programdata).join(r"Microsoft\Windows\Start Menu\Programs"));
    }

    for root in roots {
        if let Some(found) = find_shortcut(&root, &target_stem, 0) {
            return Some(found);
        }
    }
    None
}

/// Small bounded recursive walk (Start Menu folders are shallow — a depth
/// cap avoids any pathological symlink loop turning this into an unbounded
/// scan) matching a `.lnk` file's stem against `target_stem`.
fn find_shortcut(dir: &Path, target_stem: &str, depth: u8) -> Option<PathBuf> {
    if depth > 6 {
        return None;
    }
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = find_shortcut(&path, target_stem, depth + 1) {
                return Some(found);
            }
            continue;
        }
        if path.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("lnk")) == Some(true) {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
            if stem == target_stem || stem.contains(target_stem) {
                return Some(path);
            }
        }
    }
    None
}

/// Opens a file, folder, or URL with whatever the OS has registered as its
/// default handler (Explorer, the default browser, the associated app) via
/// `ShellExecuteW`'s `"open"` verb — the native-API tier for this, not a
/// shelled-out `explorer.exe`/`start` call.
pub fn open_path_or_url(target: &str) -> Result<String, String> {
    let trimmed = target.trim();
    if trimmed.is_empty() {
        return Err("no target given".into());
    }

    let operation: Vec<u16> = "open\0".encode_utf16().collect();
    let file: Vec<u16> = format!("{trimmed}\0").encode_utf16().collect();

    let result = unsafe {
        ShellExecuteW(
            None,
            PCWSTR(operation.as_ptr()),
            PCWSTR(file.as_ptr()),
            PCWSTR::null(),
            PCWSTR::null(),
            SW_SHOWNORMAL,
        )
    };

    // Per the Win32 ShellExecute contract, a return value greater than 32
    // indicates success; anything else is an error code.
    if (result.0 as usize) > 32 {
        Ok(format!("Opening {trimmed}."))
    } else {
        Err(format!("couldn't open \"{trimmed}\" (error code {})", result.0 as usize))
    }
}

// ---------------------------------------------------------------------
// Fast-router native actions (window ops, CPU/memory, volume, clipboard,
// screen capture) — added for the low-latency re-architecture. Same
// conventions as the block above: `Result<_, String>`, native Win32/OS
// calls only, no shelling out.
// ---------------------------------------------------------------------

use windows::core::BOOL;
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{
    GetForegroundWindow, GetSystemMetrics, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible, PostMessageW,
    SetForegroundWindow, ShowWindow, EnumWindows, SM_CXSCREEN, SM_CYSCREEN, SW_MAXIMIZE, SW_MINIMIZE, SW_RESTORE, WM_CLOSE,
};

/// Finds the first visible top-level window whose title contains `needle`
/// (case-insensitive) — the same "substring match" spirit as
/// `resolve_app_path`'s Start Menu shortcut scan above, not exact-title
/// matching (voice input is never going to say a window's exact title).
fn find_window_by_title(needle: &str) -> Option<HWND> {
    struct SearchCtx {
        needle: String,
        found: Option<isize>,
    }

    unsafe extern "system" fn enum_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let ctx = &mut *(lparam.0 as *mut SearchCtx);
        if IsWindowVisible(hwnd).as_bool() {
            let len = GetWindowTextLengthW(hwnd);
            if len > 0 {
                let mut buf = vec![0u16; len as usize + 1];
                let read = GetWindowTextW(hwnd, &mut buf);
                if read > 0 {
                    let title = String::from_utf16_lossy(&buf[..read as usize]).to_lowercase();
                    if title.contains(&ctx.needle) {
                        ctx.found = Some(hwnd.0 as isize);
                        return BOOL(0);
                    }
                }
            }
        }
        BOOL(1)
    }

    let mut ctx = SearchCtx { needle: needle.to_lowercase(), found: None };
    unsafe {
        let _ = EnumWindows(Some(enum_proc), LPARAM(&mut ctx as *mut SearchCtx as isize));
    }
    ctx.found.map(|raw| HWND(raw as *mut _))
}

/// Resolves a window operation's target: the named window if given, else
/// whatever window currently has focus — "minimize this"/"close this" with
/// no explicit app name is a normal thing to say.
fn resolve_target_window(target: Option<&str>) -> Result<HWND, String> {
    match target {
        Some(name) => find_window_by_title(name).ok_or_else(|| format!("I couldn't find a window matching \"{name}\".")),
        None => {
            let hwnd = unsafe { GetForegroundWindow() };
            if hwnd.0.is_null() {
                Err("there's no focused window right now".to_string())
            } else {
                Ok(hwnd)
            }
        }
    }
}

/// Focuses an already-running window matching `name` if one exists,
/// otherwise launches it fresh via `resolve_and_launch_app`. This is what
/// `Capability::LaunchOrFocusApp` (the fast router's target for "open"/
/// "launch"/"start") actually runs — checking for an already-open window
/// first both avoids spawning a duplicate instance of apps that support only
/// one, and is the lower-latency path when the app is already running (no
/// process spawn/startup at all, just a foreground-window switch).
pub fn launch_or_focus_app(name: &str) -> Result<String, String> {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return Err("no app name given".into());
    }
    if let Some(hwnd) = find_window_by_title(trimmed) {
        unsafe {
            let _ = ShowWindow(hwnd, SW_RESTORE);
            let _ = SetForegroundWindow(hwnd);
        }
        return Ok(format!("{trimmed} is already open."));
    }
    resolve_and_launch_app(trimmed)
}

pub fn window_minimize(target: Option<&str>) -> Result<String, String> {
    let hwnd = resolve_target_window(target)?;
    unsafe {
        let _ = ShowWindow(hwnd, SW_MINIMIZE);
    }
    Ok("Minimized.".to_string())
}

pub fn window_maximize(target: Option<&str>) -> Result<String, String> {
    let hwnd = resolve_target_window(target)?;
    unsafe {
        let _ = ShowWindow(hwnd, SW_MAXIMIZE);
    }
    Ok("Maximized.".to_string())
}

pub fn window_close(target: Option<&str>) -> Result<String, String> {
    let hwnd = resolve_target_window(target)?;
    unsafe {
        // WM_CLOSE (a polite close request the app can intercept, e.g. to
        // prompt "save changes?") rather than force-terminating the process.
        let _ = PostMessageW(Some(hwnd), WM_CLOSE, WPARAM(0), LPARAM(0));
    }
    Ok("Closed.".to_string())
}

pub fn window_focus(target: Option<&str>) -> Result<String, String> {
    let hwnd = resolve_target_window(target)?;
    unsafe {
        let _ = ShowWindow(hwnd, SW_RESTORE);
        let _ = SetForegroundWindow(hwnd);
    }
    Ok("Done.".to_string())
}

/// CPU usage via `sysinfo` — same pattern already used by
/// `hardware::profile`'s hardware-detection pass (two `refresh_cpu_usage()`
/// calls separated by `sysinfo::MINIMUM_CPU_UPDATE_INTERVAL`; sysinfo's CPU
/// percentage is delta-based and reads as 0 without a real interval between
/// samples — not an artificial delay, a real OS measurement requirement).
pub fn query_cpu_usage() -> Result<String, String> {
    let mut sys = sysinfo::System::new();
    sys.refresh_cpu_usage();
    std::thread::sleep(sysinfo::MINIMUM_CPU_UPDATE_INTERVAL);
    sys.refresh_cpu_usage();
    let pct = sys.global_cpu_usage();
    Ok(format!("CPU usage is {pct:.0} percent."))
}

pub fn query_memory_usage() -> Result<String, String> {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let used = sys.used_memory();
    let total = sys.total_memory();
    let pct = if total > 0 { (used as f64 / total as f64) * 100.0 } else { 0.0 };
    let used_gb = used as f64 / 1_073_741_824.0;
    let total_gb = total as f64 / 1_073_741_824.0;
    Ok(format!("Memory usage is {pct:.0} percent — {used_gb:.1} of {total_gb:.1} gigabytes."))
}

pub fn screen_width_height() -> (i32, i32) {
    unsafe { (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN)) }
}

/// Answers a small fixed set of system-info questions directly via native
/// APIs — no shelling out to `wmic`/PowerShell for these.
pub fn query_system_info(kind: &str) -> Result<String, String> {
    match kind.trim().to_lowercase().as_str() {
        "time" => {
            let now = unsafe { GetLocalTime() };
            let hour_12 = match now.wHour % 12 {
                0 => 12,
                h => h,
            };
            let period = if now.wHour < 12 { "AM" } else { "PM" };
            Ok(format!("It's {hour_12}:{:02} {period}.", now.wMinute))
        }
        "battery" => {
            let mut status = SYSTEM_POWER_STATUS::default();
            unsafe { GetSystemPowerStatus(&mut status) }
                .map_err(|e| format!("failed to read battery status: {e}"))?;
            if status.BatteryLifePercent == 255 {
                Ok("This device doesn't report a battery level.".to_string())
            } else {
                let charging = status.ACLineStatus == 1;
                Ok(format!(
                    "Battery is at {}%{}.",
                    status.BatteryLifePercent,
                    if charging { " and charging" } else { "" }
                ))
            }
        }
        "volume" => Err("I can't check the system volume yet.".to_string()),
        other => Err(format!("I don't know how to check \"{other}\".")),
    }
}
