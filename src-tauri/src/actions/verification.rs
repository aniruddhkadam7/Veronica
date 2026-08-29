//! Generic post-action verification: after a risky/consequential action
//! runs, re-check that it actually had the claimed effect and append a short
//! confirmation/warning note to the result — a closed lookup table keyed by
//! `Capability` shape (same discipline as `registry::risk_level_for_capability`),
//! not a hardcoded call at each individual tool's call site.
//!
//! Unlike the risk-tier match, an omitted entry here is NOT a compile error
//! — a missing verification degrades user-facing confidence, not safety
//! (the action already ran, gated correctly by `execute_tool`/`registry`
//! before this ever runs). New Sensitive/Destructive capabilities SHOULD get
//! an entry here, but the crate still builds without one.

use std::path::Path;
use std::time::Duration;

use tauri::{AppHandle, Runtime};

use super::capability::{Capability, FileOp, StorageOp};
use super::{execute_tool, ToolOutcome};

/// Runs `capability`, then — if a verification check exists for its shape —
/// re-checks the claimed effect and appends a short note to the result text.
/// The one call site both the fast router and the agent loop should use
/// instead of calling `execute_tool` directly. `app` is forwarded to
/// `execute_tool` unchanged (only `SchedulerOp`/`WatcherOp` actually need it).
pub async fn execute_and_verify<R: Runtime>(capability: &Capability, confirmed: bool, app: &AppHandle<R>) -> Result<ToolOutcome, String> {
    let outcome = execute_tool(capability, confirmed, app).await?;
    let ToolOutcome::Text(action_text) = &outcome else { return Ok(outcome) };

    let note = match capability {
        Capability::StorageOp(StorageOp::DeleteFile { path }) => Some(verify_deleted(path)),
        Capability::FileOp(FileOp::WriteFile { path, content, .. }) => Some(verify_content_written(path, content.len())),
        Capability::FileOp(FileOp::CreateFile { path, content }) => Some(verify_content_written(path, content.as_deref().unwrap_or("").len())),
        _ => None,
    };

    match note {
        Some(note) => Ok(ToolOutcome::Text(format!("{action_text}\n\n{note}"))),
        None => Ok(outcome),
    }
}

fn verify_deleted(path: &str) -> String {
    if Path::new(path).exists() {
        format!("Warning: \"{path}\" still appears to exist.")
    } else {
        "Confirmed: the file is gone.".to_string()
    }
}

fn verify_content_written(path: &str, expected_len: usize) -> String {
    match std::fs::metadata(path) {
        Ok(meta) if meta.len() as usize >= expected_len => format!("Confirmed: {} bytes written.", meta.len()),
        Ok(meta) => format!("Warning: expected at least {expected_len} bytes but found {}.", meta.len()),
        Err(_) => format!("Could not verify: \"{path}\" is no longer readable."),
    }
}

/// Re-queries `sysinfo` after a short delay to confirm a killed process is
/// actually gone — separate async entry point since `execute_and_verify`
/// above is synchronous-shaped apart from `execute_tool` itself; wired in
/// once `ProcessOp::Kill` lands (Phase 3).
pub async fn verify_process_gone(pid: u32) -> String {
    tokio::time::sleep(Duration::from_millis(300)).await;
    let mut sys = sysinfo::System::new();
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    if sys.process(sysinfo::Pid::from_u32(pid)).is_some() {
        "Warning: the process still appears to be running.".to_string()
    } else {
        "Confirmed: the process is no longer running.".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn mock_app_handle() -> tauri::AppHandle<tauri::test::MockRuntime> {
        let app = tauri::test::mock_app();
        tauri::Manager::manage(&app, crate::state::AppState::default());
        app.handle().clone()
    }

    #[test]
    fn execute_and_verify_appends_confirmation_after_a_successful_delete() {
        let dir = std::env::temp_dir().join(format!("veronica_test_verify_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("to_delete.txt");
        fs::write(&file, "x").unwrap();

        let app = mock_app_handle();
        let capability = Capability::StorageOp(StorageOp::DeleteFile { path: file.to_str().unwrap().to_string() });
        let outcome = tauri::async_runtime::block_on(execute_and_verify(&capability, true, &app)).unwrap();
        let ToolOutcome::Text(text) = outcome else { panic!("expected text outcome") };
        assert!(text.contains("Confirmed: the file is gone."));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn execute_and_verify_passes_through_capabilities_with_no_check() {
        let app = mock_app_handle();
        let capability = Capability::SystemInfo(crate::actions::capability::SystemInfoKind::Cpu);
        let outcome = tauri::async_runtime::block_on(execute_and_verify(&capability, false, &app)).unwrap();
        // No verification note appended — just the plain CPU-usage text.
        let ToolOutcome::Text(text) = outcome else { panic!("expected text outcome") };
        assert!(!text.contains("Confirmed:"));
    }
}
