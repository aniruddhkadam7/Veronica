//! Veronica's action-taking system. `Capability` (see `capability.rs`) is
//! the closed vocabulary — every request to do something on the computer
//! resolves to one of its variants, either deterministically
//! (`actions::fast_router`, no LLM involved) or via a real structured tool
//! call the agent loop makes (`personal::agent`) — and `execute_tool` is the
//! one place that actually runs one, checked against `registry`'s
//! three-tier safety classification first.
//!
//! Safety model: `RiskLevel::Safe` capabilities run immediately.
//! `Sensitive`/`Destructive` capabilities are withheld — `execute_tool`
//! returns `ToolOutcome::NeedsConfirmation` instead of running them — unless
//! the caller passes `confirmed: true`, which only happens after the user
//! has explicitly said yes (voice or the overlay's confirmation dialog; see
//! `veronica.rs`'s `pending_confirmation` state machine and
//! `confirmation::classify_reply`). The LLM never bypasses this: it can only
//! ever name one of the fixed tools in
//! `personal::agent::tool_schema::all_tools()`, and every one of those maps
//! onto a `Capability` classified here — there is no path from a model's
//! tool call straight to unconfirmed execution of anything above `Safe`.

mod capability;
mod clipboard;
pub mod context;
pub mod fast_router;
pub mod filesystem;
mod native;
mod network;
mod processes;
pub mod registry;
pub mod scheduler;
mod screen;
pub mod storage;
mod terminal;
pub mod verification;
mod volume;
pub mod watchers;

pub use capability::{
    Capability, ClipboardOp, FileOp, NetworkQuery, ProcessOp, ProcessQuery, SchedulerOp, StorageOp, StorageQuery, SystemInfoKind, TaskControlOp, TerminalOp, ToolOutcome, VolumeOp, WatcherOp,
    WindowOp, WindowQueryOp,
};
pub use registry::RiskLevel;
pub use scheduler::SchedulerRegistry;
pub use watchers::WatcherRegistry;

use tauri::{AppHandle, Manager, Runtime};

/// The one execution entry point for the `Capability` vocabulary — called
/// (indirectly, via `verification::execute_and_verify`) by both
/// `actions::fast_router` (deterministic matches) and `personal::agent`'s
/// tool loop (LLM-decided tool calls).
///
/// `confirmed` must be `true` only when the user has already explicitly
/// agreed to a specific pending `Sensitive`/`Destructive` capability (see
/// this module's doc) — passing `true` unconditionally would defeat the
/// entire confirmation gate.
///
/// `app` is needed only by `SchedulerOp`/`WatcherOp` (to reach
/// `AppState.scheduler`/`AppState.watchers` and to emit Tauri events from a
/// background thread) — every other capability ignores it. Threaded through
/// here rather than special-cased at the caller so `execute_tool` stays the
/// single execution chokepoint for every `Capability` variant, with no
/// carve-outs.
///
/// `Capability::TaskControl` is NOT handled here — it mutates
/// `AppState.working_state` (pause/resume/cancel the current task), which
/// this module has no access to; the caller (`veronica::ask_veronica`)
/// dispatches it directly. Passing one here is a caller bug, not a runtime
/// possibility reachable from user input — see the `unreachable!` below.
pub async fn execute_tool<R: Runtime>(capability: &Capability, confirmed: bool, app: &AppHandle<R>) -> Result<ToolOutcome, String> {
    let risk = registry::risk_level_for_capability(capability);
    if !confirmed && matches!(risk, RiskLevel::Sensitive | RiskLevel::Destructive) {
        return Ok(ToolOutcome::NeedsConfirmation { capability: capability.clone(), voice_prompt: registry::confirmation_prompt_for(capability), risk });
    }

    match capability {
        Capability::LaunchOrFocusApp(name) => native::launch_or_focus_app(name).map(ToolOutcome::text),
        Capability::LaunchAppWithArg { app, arg } => native::launch_app_with_arg(app, arg).map(ToolOutcome::text),
        Capability::OpenPath(target) => native::open_path_or_url(target).map(ToolOutcome::text),
        Capability::WindowOp { op, target } => {
            let target = target.as_deref();
            match op {
                WindowOp::Minimize => native::window_minimize(target),
                WindowOp::Maximize => native::window_maximize(target),
                WindowOp::Close => native::window_close(target),
                WindowOp::Focus => native::window_focus(target),
            }
            .map(ToolOutcome::text)
        }
        Capability::WindowQuery(query) => match query {
            WindowQueryOp::ListOpen => native::enumerate_visible_windows(),
            WindowQueryOp::GetActive => native::get_active_window(),
        }
        .map(ToolOutcome::text),
        Capability::SystemInfo(kind) => match kind {
            SystemInfoKind::Time => native::query_system_info("time"),
            SystemInfoKind::Battery => native::query_system_info("battery"),
            SystemInfoKind::Cpu => native::query_cpu_usage(),
            SystemInfoKind::Memory => native::query_memory_usage(),
            SystemInfoKind::Volume => volume::get_volume_percent(),
            SystemInfoKind::DiskSpace => native::query_system_info("diskspace"),
            SystemInfoKind::Uptime => native::query_system_info("uptime"),
        }
        .map(ToolOutcome::text),
        Capability::VolumeOp(op) => match op {
            VolumeOp::Up(amount) => volume::adjust_volume(amount.map(|a| a as i32).unwrap_or(10)),
            VolumeOp::Down(amount) => volume::adjust_volume(-(amount.map(|a| a as i32).unwrap_or(10))),
            VolumeOp::Mute => volume::set_mute(true),
            VolumeOp::Unmute => volume::set_mute(false),
            VolumeOp::SetPercent(pct) => volume::set_volume_percent(*pct),
        }
        .map(ToolOutcome::text),
        Capability::Clipboard(op) => match op {
            ClipboardOp::Read => clipboard::read_text(),
            ClipboardOp::Write(text) => clipboard::write_text(text),
        }
        .map(ToolOutcome::text),
        Capability::CaptureScreen => {
            screen::capture_primary_screen_png().map(|png_bytes| ToolOutcome::Image { media_type: "image/png", png_bytes })
        }
        Capability::SearchKnowledgeBase(query) => search_knowledge_base(query).await.map(ToolOutcome::text),
        Capability::FileOp(op) => match op {
            FileOp::CreateFolder { path } => filesystem::create_folder(path),
            FileOp::CreateFile { path, content } => filesystem::create_file(path, content.as_deref()),
            FileOp::WriteFile { path, content, append } => filesystem::write_file(path, content, *append),
            FileOp::ReadFile { path } => filesystem::read_file(path),
        }
        .map(ToolOutcome::text),
        Capability::StorageQuery(query) => match query {
            StorageQuery::ListFolder { path } => storage::list_folder(path),
            StorageQuery::SearchFiles { root, query, max_results } => {
                storage::search_files(root, query, *max_results, &crate::state::CancelToken::new())
            }
            StorageQuery::LargestFiles { root, top_n } => storage::largest_files(root, *top_n, &crate::state::CancelToken::new()),
            StorageQuery::DiskUsage { drive } => storage::disk_usage(drive.as_deref()),
        }
        .map(ToolOutcome::text),
        Capability::StorageOp(op) => match op {
            StorageOp::DeleteFile { path } => storage::delete_file(path),
            StorageOp::MoveOrRename { from, to } => storage::move_or_rename(from, to),
        }
        .map(ToolOutcome::text),
        Capability::ProcessQuery(query) => match query {
            ProcessQuery::List => processes::list_processes(),
            ProcessQuery::FindByName(name) => processes::find_by_name(name),
        }
        .map(ToolOutcome::text),
        Capability::ProcessOp(op) => match op {
            ProcessOp::Kill { pid, name } => processes::kill_process(*pid, name.as_deref()),
        }
        .map(ToolOutcome::text),
        Capability::NetworkQuery(query) => match query {
            NetworkQuery::Status => network::network_status(),
            NetworkQuery::PingHost { host } => network::ping_host(host),
            NetworkQuery::ListeningPorts => network::listening_ports(),
        }
        .map(ToolOutcome::text),
        Capability::TerminalOp(TerminalOp::RunCommand { command, working_dir }) => {
            terminal::run_command(command, working_dir.as_deref(), &crate::state::CancelToken::new()).await.map(ToolOutcome::text)
        }
        Capability::SchedulerOp(op) => {
            let state = app.state::<crate::state::AppState>();
            match op {
                SchedulerOp::ScheduleOnce { run_at_unix_ms, description, action } => scheduler::schedule_once(&state.scheduler, *run_at_unix_ms, description, (**action).clone()),
                SchedulerOp::CancelScheduled { id } => scheduler::cancel_scheduled(&state.scheduler, id),
                SchedulerOp::ListScheduled => scheduler::list_scheduled(&state.scheduler),
            }
            .map(ToolOutcome::text)
        }
        Capability::WatcherOp(op) => {
            let state = app.state::<crate::state::AppState>();
            match op {
                WatcherOp::WatchPath { path, description } => watchers::watch_path(&state.watchers, app.clone(), path, description),
                WatcherOp::StopWatch { id } => watchers::stop_watch(&state.watchers, id),
                WatcherOp::ListWatches => watchers::list_watches(&state.watchers),
            }
            .map(ToolOutcome::text)
        }
        Capability::TaskControl(_) => unreachable!("TaskControl is dispatched by the caller against AppState.working_state, never reaches execute_tool"),
    }
}

/// Backs `Capability::SearchKnowledgeBase` — wraps the existing
/// `RetrievalPlanner` (unchanged), now called only when the agent loop
/// itself decides to, instead of being pre-fetched unconditionally before
/// every question (see the audit's finding on `veronica::retrieval_could_help`).
async fn search_knowledge_base(query: &str) -> Result<String, String> {
    let planner = crate::rag::RetrievalPlanner::new();
    let results = planner.plan_for_question(query).await;
    if results.is_empty() {
        return Ok("No relevant documents found.".to_string());
    }
    let joined = results
        .into_iter()
        .map(|r| format!("From {} ({}): {}", r.metadata.filename, r.metadata.document_type, r.text))
        .collect::<Vec<_>>()
        .join("\n\n");
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::Manager;

    /// `tauri::test::mock_app()` gives a real `AppHandle` backed by a mock
    /// runtime — lets these tests exercise `execute_tool` end-to-end
    /// (including the `SchedulerOp`/`WatcherOp` arms, which genuinely need
    /// one) instead of only checking `registry::risk_level_for_capability`
    /// in isolation.
    fn mock_app_handle() -> tauri::AppHandle<tauri::test::MockRuntime> {
        let app = tauri::test::mock_app();
        app.manage(crate::state::AppState::default());
        app.handle().clone()
    }

    #[test]
    fn execute_tool_returns_needs_confirmation_for_destructive_when_unconfirmed() {
        let dir = std::env::temp_dir().join(format!("veronica_test_mod_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("x.txt");
        std::fs::write(&file, "x").unwrap();

        let app = mock_app_handle();
        let capability = Capability::StorageOp(StorageOp::DeleteFile { path: file.to_str().unwrap().to_string() });
        let outcome = tauri::async_runtime::block_on(execute_tool(&capability, false, &app)).unwrap();
        assert!(matches!(outcome, ToolOutcome::NeedsConfirmation { .. }));
        assert!(file.exists(), "the file must not actually be deleted while unconfirmed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn execute_tool_runs_when_confirmed_true_for_the_same_capability() {
        let dir = std::env::temp_dir().join(format!("veronica_test_mod_confirmed_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("x.txt");
        std::fs::write(&file, "x").unwrap();

        let app = mock_app_handle();
        let capability = Capability::StorageOp(StorageOp::DeleteFile { path: file.to_str().unwrap().to_string() });
        let outcome = tauri::async_runtime::block_on(execute_tool(&capability, true, &app)).unwrap();
        assert!(matches!(outcome, ToolOutcome::Text(_)));
        assert!(!file.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn safe_capabilities_never_need_confirmation() {
        let app = mock_app_handle();
        let capability = Capability::SystemInfo(SystemInfoKind::Cpu);
        let outcome = tauri::async_runtime::block_on(execute_tool(&capability, false, &app)).unwrap();
        assert!(matches!(outcome, ToolOutcome::Text(_)));
    }

    #[test]
    fn scheduler_op_list_scheduled_reaches_appstate_via_the_apphandle() {
        let app = mock_app_handle();
        let capability = Capability::SchedulerOp(SchedulerOp::ListScheduled);
        let outcome = tauri::async_runtime::block_on(execute_tool(&capability, false, &app)).unwrap();
        assert!(matches!(outcome, ToolOutcome::Text(text) if text.contains("Nothing is scheduled")));
    }

    #[test]
    fn scheduler_op_schedule_once_needs_confirmation_and_never_reaches_appstate_unconfirmed() {
        let app = mock_app_handle();
        let capability = Capability::SchedulerOp(SchedulerOp::ScheduleOnce {
            run_at_unix_ms: 0,
            description: "test".to_string(),
            action: Box::new(Capability::SystemInfo(SystemInfoKind::Time)),
        });
        let outcome = tauri::async_runtime::block_on(execute_tool(&capability, false, &app)).unwrap();
        assert!(matches!(outcome, ToolOutcome::NeedsConfirmation { .. }));
    }
}
