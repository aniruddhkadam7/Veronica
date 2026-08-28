//! Lightweight, in-memory-only working state for one Veronica session —
//! what the agent loop needs to resolve "it"/"this"/"the previous one" and
//! to support pausing/resuming a multi-step task. Not a database, not
//! persisted across app restarts: it lives in `AppState.working_state`
//! (a `Mutex`, same storage pattern as `AppState.transcript`/`AppState.tts`)
//! for exactly as long as the app process does.

use std::collections::VecDeque;

/// How many recent action summaries to keep — enough for a few turns of
/// "the previous one"-style reference resolution without the prompt payload
/// this gets rendered into (see `render_context_block`) growing unbounded
/// over a long session.
const MAX_RECENT_ACTIONS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus {
    Active,
    Paused,
    Complete,
}

/// One multi-step goal the agent loop is (or was) working through — e.g.
/// "find the authentication problem and fix it" spanning several tool calls
/// across possibly several turns.
#[derive(Debug, Clone)]
pub struct TaskState {
    pub goal: String,
    pub steps_done: Vec<String>,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, Default)]
pub struct WorkingState {
    pub current_app: Option<String>,
    pub current_window: Option<String>,
    pub current_file: Option<String>,
    pub current_folder: Option<String>,
    pub current_project: Option<String>,
    pub current_task: Option<TaskState>,
    pub recent_actions: VecDeque<String>,
    pub last_result: Option<String>,
}

impl WorkingState {
    /// Records one action's outcome, trimming `recent_actions` to
    /// `MAX_RECENT_ACTIONS`. Called after every fast-router execution and
    /// every agent-loop tool call so "it"/"the previous one" always resolve
    /// to something real, not stale.
    pub fn record_action(&mut self, summary: impl Into<String>, result: impl Into<String>) {
        self.recent_actions.push_back(summary.into());
        while self.recent_actions.len() > MAX_RECENT_ACTIONS {
            self.recent_actions.pop_front();
        }
        self.last_result = Some(result.into());
    }

    /// Starts (replacing any previous) multi-step task, `Active` from the
    /// start.
    pub fn start_task(&mut self, goal: impl Into<String>) {
        self.current_task = Some(TaskState { goal: goal.into(), steps_done: Vec::new(), status: TaskStatus::Active });
    }

    /// "pause"/"stop that" — a `TaskControl` capability the fast router
    /// resolves with no LLM call. A no-op (not an error) if there is no
    /// active task, since the fast router has no reliable way to distinguish
    /// "pause the task" from a bare "pause" utterance with nothing running.
    pub fn pause_task(&mut self) {
        if let Some(task) = self.current_task.as_mut() {
            if task.status == TaskStatus::Active {
                task.status = TaskStatus::Paused;
            }
        }
    }

    /// "resume"/"continue" — reactivates a paused task so the next
    /// agent-loop call re-injects its goal/steps-done instead of starting
    /// over. No-op if there is no paused task.
    pub fn resume_task(&mut self) {
        if let Some(task) = self.current_task.as_mut() {
            if task.status == TaskStatus::Paused {
                task.status = TaskStatus::Active;
            }
        }
    }

    pub fn complete_task(&mut self) {
        if let Some(task) = self.current_task.as_mut() {
            task.status = TaskStatus::Complete;
        }
    }

    /// Renders a short, bounded block of the current state for injection
    /// into the agent loop's prompt — `None` when there is nothing worth
    /// mentioning (a fresh session), so an empty-but-present block never
    /// wastes prompt tokens on a normal first question.
    pub fn render_context_block(&self) -> Option<String> {
        let mut lines = Vec::new();
        if let Some(app) = &self.current_app {
            lines.push(format!("Current application: {app}"));
        }
        if let Some(window) = &self.current_window {
            lines.push(format!("Current window: {window}"));
        }
        if let Some(file) = &self.current_file {
            lines.push(format!("Current file: {file}"));
        }
        if let Some(folder) = &self.current_folder {
            lines.push(format!("Current folder: {folder}"));
        }
        if let Some(project) = &self.current_project {
            lines.push(format!("Current project: {project}"));
        }
        if let Some(task) = &self.current_task {
            lines.push(format!(
                "Current task ({:?}): {} — steps done so far: {}",
                task.status,
                task.goal,
                if task.steps_done.is_empty() { "none yet".to_string() } else { task.steps_done.join("; ") }
            ));
        }
        if !self.recent_actions.is_empty() {
            lines.push(format!("Recent actions this session: {}", self.recent_actions.iter().cloned().collect::<Vec<_>>().join("; ")));
        }
        if let Some(result) = &self.last_result {
            lines.push(format!("Result of the most recent action (what \"it\"/\"that\" refers to): {result}"));
        }

        if lines.is_empty() {
            None
        } else {
            Some(lines.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_state_renders_no_context_block() {
        assert_eq!(WorkingState::default().render_context_block(), None);
    }

    #[test]
    fn record_action_sets_last_result_and_appends_to_recent() {
        let mut ws = WorkingState::default();
        ws.record_action("opened VS Code", "Opening VS Code.");
        assert_eq!(ws.last_result.as_deref(), Some("Opening VS Code."));
        assert_eq!(ws.recent_actions.len(), 1);
    }

    #[test]
    fn recent_actions_trims_to_max() {
        let mut ws = WorkingState::default();
        for i in 0..(MAX_RECENT_ACTIONS + 5) {
            ws.record_action(format!("action {i}"), format!("result {i}"));
        }
        assert_eq!(ws.recent_actions.len(), MAX_RECENT_ACTIONS);
        assert_eq!(ws.recent_actions.back().unwrap(), &format!("action {}", MAX_RECENT_ACTIONS + 4));
    }

    #[test]
    fn pause_then_resume_round_trips_status() {
        let mut ws = WorkingState::default();
        ws.start_task("find and fix the bug");
        assert_eq!(ws.current_task.as_ref().unwrap().status, TaskStatus::Active);
        ws.pause_task();
        assert_eq!(ws.current_task.as_ref().unwrap().status, TaskStatus::Paused);
        ws.resume_task();
        assert_eq!(ws.current_task.as_ref().unwrap().status, TaskStatus::Active);
    }

    #[test]
    fn pause_with_no_active_task_is_a_harmless_no_op() {
        let mut ws = WorkingState::default();
        ws.pause_task(); // no task at all
        assert!(ws.current_task.is_none());
    }

    #[test]
    fn context_block_includes_task_and_recent_actions() {
        let mut ws = WorkingState::default();
        ws.start_task("find the authentication problem and fix it");
        ws.record_action("searched code", "found the issue in auth.rs");
        let block = ws.render_context_block().unwrap();
        assert!(block.contains("find the authentication problem and fix it"));
        assert!(block.contains("found the issue in auth.rs"));
    }
}
