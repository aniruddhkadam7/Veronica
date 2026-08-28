//! Veronica's action-taking system. `Capability` (see `capability.rs`) is
//! the closed vocabulary — every request to do something on the computer
//! resolves to one of its variants, either deterministically
//! (`actions::fast_router`, no LLM involved) or via a real structured tool
//! call the agent loop makes (`personal::agent`) — and `execute_tool` is the
//! one place that actually runs one, checked against a hardcoded safety
//! table first.
//!
//! The LLM never executes anything itself — it only ever names one of the
//! fixed tools in `personal::agent::tool_schema::all_tools()` (there is no
//! tool, and no `Capability` variant, that could represent a destructive
//! action: delete, format, registry/security change, credential access,
//! shutdown, arbitrary shell execution, bulk destructive ops, or a
//! consequential external send) — so there is no code path, not even a
//! guarded one, that could run any of those from a request to Veronica.

mod capability;
mod clipboard;
pub mod fast_router;
mod native;
mod registry;
mod screen;
mod volume;

pub use capability::{Capability, ClipboardOp, SystemInfoKind, TaskControlOp, ToolOutcome, VolumeOp, WindowOp};
pub use registry::RiskLevel;

/// The one execution entry point for the `Capability` vocabulary —
/// called by both `actions::fast_router` (deterministic matches) and
/// `personal::agent`'s tool loop (LLM-decided tool calls). Every variant is
/// closed-vocabulary and `Safe` by construction (see `registry::risk_level`,
/// extended alongside this for the same variants) — nothing here can
/// represent a destructive action, matching this module's existing
/// guarantee for `Intent`.
///
/// `Capability::TaskControl` is NOT handled here — it mutates
/// `AppState.working_state` (pause/resume/cancel the current task), which
/// this module has no access to; the caller (`veronica::ask_veronica`)
/// dispatches it directly. Passing one here is a caller bug, not a runtime
/// possibility reachable from user input — see the `unreachable!` below.
pub async fn execute_tool(capability: &Capability) -> Result<ToolOutcome, String> {
    match registry::risk_level_for_capability(capability) {
        RiskLevel::Blocked => Err(registry::refusal_message_for_capability(capability)),
        RiskLevel::Safe => match capability {
            Capability::LaunchOrFocusApp(name) => native::launch_or_focus_app(name).map(ToolOutcome::text),
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
            Capability::SystemInfo(kind) => match kind {
                SystemInfoKind::Time => native::query_system_info("time"),
                SystemInfoKind::Battery => native::query_system_info("battery"),
                SystemInfoKind::Cpu => native::query_cpu_usage(),
                SystemInfoKind::Memory => native::query_memory_usage(),
                SystemInfoKind::Volume => volume::get_volume_percent(),
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
            Capability::TaskControl(_) => unreachable!("TaskControl is dispatched by the caller against AppState.working_state, never reaches execute_tool"),
        },
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

