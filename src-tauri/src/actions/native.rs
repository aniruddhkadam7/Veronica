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
