//! Scheduler: fires a `Capability` at a future point in time. One dedicated
//! background `std::thread` (matching this crate's convention for
//! background work — `tokio` here only has the `"time"` feature, not `"rt"`)
//! started once at app setup (`spawn_scheduler_thread`, called from
//! `lib.rs`), polling an in-memory job list every second and firing due jobs
//! through `actions::execute_tool` directly.
//!
//! A scheduled job's inner `Capability` is re-validated `Safe`-only both at
//! schedule time (`validate_schedulable`) AND again right before firing —
//! there is no live turn to confirm a `Sensitive`/`Destructive` action
//! against when it fires unattended, so anything above `Safe` is rejected
//! outright rather than silently downgraded or auto-confirmed. This is a
//! deliberate, documented limitation, not an oversight.

use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager};

use super::capability::Capability;
use super::registry::{self, RiskLevel};

#[derive(Debug, Clone)]
pub struct ScheduledJob {
    pub id: String,
    pub run_at_unix_ms: i64,
    pub description: String,
    pub action: Capability,
}

/// Registry of pending scheduled jobs — lives in `AppState`, guarded the
/// same way every other cross-thread piece of session state is
/// (`Mutex`-wrapped, cheap to lock briefly).
#[derive(Default)]
pub struct SchedulerRegistry(pub Mutex<Vec<ScheduledJob>>);

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// Rejects scheduling anything above `Safe` — see the module doc. Called at
/// schedule time so the user gets immediate feedback ("I can only schedule
/// safe actions") rather than a job that silently never fires.
pub fn validate_schedulable(action: &Capability) -> Result<(), String> {
    match registry::risk_level_for_capability(action) {
        RiskLevel::Safe => Ok(()),
        _ => Err("I can only schedule actions that don't need confirmation — that one would need your approval, so I can't schedule it unattended.".to_string()),
    }
}

pub fn schedule_once(registry_state: &SchedulerRegistry, run_at_unix_ms: i64, description: &str, action: Capability) -> Result<String, String> {
    validate_schedulable(&action)?;
    let id = format!("job-{}-{}", now_unix_ms(), description.len());
    let job = ScheduledJob { id: id.clone(), run_at_unix_ms, description: description.to_string(), action };
    registry_state.0.lock().unwrap().push(job);
    Ok(format!("Scheduled: {description}."))
}

pub fn cancel_scheduled(registry_state: &SchedulerRegistry, id: &str) -> Result<String, String> {
    let mut jobs = registry_state.0.lock().unwrap();
    let before = jobs.len();
    jobs.retain(|j| j.id != id);
    if jobs.len() < before {
        Ok(format!("Cancelled scheduled job {id}."))
    } else {
        Err(format!("I couldn't find a scheduled job with id \"{id}\"."))
    }
}

pub fn list_scheduled(registry_state: &SchedulerRegistry) -> Result<String, String> {
    let jobs = registry_state.0.lock().unwrap();
    if jobs.is_empty() {
        return Ok("Nothing is scheduled.".to_string());
    }
    let listed = jobs.iter().map(|j| format!("{} (id {})", j.description, j.id)).collect::<Vec<_>>().join(", ");
    Ok(format!("Scheduled: {listed}."))
}

/// Starts the background polling thread — called once from `lib.rs`'s
/// setup. Holds an `AppHandle` (not a raw `AppState` reference) so it can
/// reach `AppState` from its own thread via Tauri's `app.state::<T>()`,
/// which works from any thread.
pub fn spawn_scheduler_thread(app: AppHandle) {
    std::thread::spawn(move || loop {
        std::thread::sleep(std::time::Duration::from_secs(1));
        let due: Vec<ScheduledJob> = {
            let state = app.state::<crate::state::AppState>();
            let mut jobs = state.scheduler.0.lock().unwrap();
            let now = now_unix_ms();
            let (due, remaining): (Vec<_>, Vec<_>) = jobs.drain(..).partition(|j| j.run_at_unix_ms <= now);
            *jobs = remaining;
            due
        };
        for job in due {
            // Re-validate at fire time — the registry's own classification
            // could not have changed since schedule time in this version,
            // but re-checking here is cheap and keeps the "never fire
            // anything above Safe unattended" guarantee independent of
            // whatever validated it at schedule time.
            if registry::risk_level_for_capability(&job.action) != RiskLevel::Safe {
                log::warn!("[SCHEDULER] job {} skipped — no longer Safe at fire time", job.id);
                continue;
            }
            let app_for_job = app.clone();
            let job_id = job.id.clone();
            let description = job.description.clone();
            tauri::async_runtime::spawn(async move {
                let outcome = super::execute_tool(&job.action, true, &app_for_job).await;
                let result_text = match outcome {
                    Ok(super::ToolOutcome::Text(text)) => text,
                    Ok(_) => "Done.".to_string(),
                    Err(err) => err,
                };
                log::info!("[SCHEDULER] fired job {job_id}: {description} -> {result_text}");
                let state = app_for_job.state::<crate::state::AppState>();
                state.working_state.lock().unwrap().record_action(format!("(scheduled) {description}"), result_text);
                let _ = app_for_job.emit("veronica:scheduled-fired", serde_json::json!({ "id": job_id, "description": description }));
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::capability::SystemInfoKind;

    #[test]
    fn schedule_once_rejects_a_sensitive_or_destructive_inner_capability() {
        let registry_state = SchedulerRegistry::default();
        let action = Capability::StorageOp(super::super::capability::StorageOp::DeleteFile { path: "x.txt".to_string() });
        assert!(schedule_once(&registry_state, now_unix_ms() + 1000, "delete x.txt", action).is_err());
    }

    #[test]
    fn schedule_once_accepts_a_safe_inner_capability() {
        let registry_state = SchedulerRegistry::default();
        let action = Capability::SystemInfo(SystemInfoKind::Time);
        assert!(schedule_once(&registry_state, now_unix_ms() + 1000, "check the time", action).is_ok());
    }

    #[test]
    fn list_scheduled_reflects_a_just_scheduled_job() {
        let registry_state = SchedulerRegistry::default();
        schedule_once(&registry_state, now_unix_ms() + 1000, "check the time", Capability::SystemInfo(SystemInfoKind::Time)).unwrap();
        let listed = list_scheduled(&registry_state).unwrap();
        assert!(listed.contains("check the time"));
    }

    #[test]
    fn schedule_then_cancel_then_list_shows_it_gone() {
        let registry_state = SchedulerRegistry::default();
        schedule_once(&registry_state, now_unix_ms() + 1000, "check the time", Capability::SystemInfo(SystemInfoKind::Time)).unwrap();
        let id = registry_state.0.lock().unwrap()[0].id.clone();
        cancel_scheduled(&registry_state, &id).unwrap();
        assert_eq!(list_scheduled(&registry_state).unwrap(), "Nothing is scheduled.");
    }

    #[test]
    fn cancel_scheduled_on_an_unknown_id_is_a_clear_error() {
        let registry_state = SchedulerRegistry::default();
        assert!(cancel_scheduled(&registry_state, "not-a-real-id").is_err());
    }
}
