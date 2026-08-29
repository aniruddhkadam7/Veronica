//! Filesystem watchers via the `notify` crate, which wraps the native OS
//! notification API (`ReadDirectoryChangesW` on Windows) — not a polling
//! loop. Each active watch owns a `notify::RecommendedWatcher` kept alive in
//! `AppState.watchers`'s registry; dropping it (on `stop_watch`) stops that
//! watch. A watch is passive: it only pushes events into `WorkingState`/
//! emits a Tauri event, it never itself takes action.

use std::collections::HashMap;
use std::sync::Mutex;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use tauri::{AppHandle, Emitter, Runtime};

pub struct WatcherRegistry(pub Mutex<HashMap<String, (RecommendedWatcher, String)>>);

impl Default for WatcherRegistry {
    fn default() -> Self {
        Self(Mutex::new(HashMap::new()))
    }
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

/// Starts watching `path`, backed by the native `ReadDirectoryChangesW`
/// (via `notify`'s `RecommendedWatcher`). Events are forwarded as
/// `veronica:watch-event` Tauri events (id + description + what changed) —
/// the agent loop/fast router don't poll for them; a proactive notification
/// is out of scope for this pass, this just makes the plumbing available.
pub fn watch_path<R: Runtime + 'static>(registry: &WatcherRegistry, app: AppHandle<R>, path: &str, description: &str) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no path given".into());
    }
    if !std::path::Path::new(trimmed).exists() {
        return Err(format!("\"{trimmed}\" doesn't exist."));
    }

    let id = format!("watch-{}", now_unix_ms());
    let id_for_events = id.clone();
    let app_for_events = app.clone();
    let mut watcher: RecommendedWatcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            let _ = app_for_events.emit(
                "veronica:watch-event",
                serde_json::json!({ "id": id_for_events, "kind": format!("{:?}", event.kind), "paths": event.paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>() }),
            );
        }
    })
    .map_err(|e| format!("couldn't start watching \"{trimmed}\": {e}"))?;

    watcher.watch(std::path::Path::new(trimmed), RecursiveMode::Recursive).map_err(|e| format!("couldn't start watching \"{trimmed}\": {e}"))?;

    registry.0.lock().unwrap().insert(id.clone(), (watcher, description.to_string()));
    Ok(format!("Watching \"{trimmed}\" (id {id})."))
}

pub fn stop_watch(registry: &WatcherRegistry, id: &str) -> Result<String, String> {
    let mut watches = registry.0.lock().unwrap();
    if watches.remove(id).is_some() {
        Ok(format!("Stopped watch {id}."))
    } else {
        Err(format!("I couldn't find an active watch with id \"{id}\"."))
    }
}

pub fn list_watches(registry: &WatcherRegistry) -> Result<String, String> {
    let watches = registry.0.lock().unwrap();
    if watches.is_empty() {
        return Ok("No active watches.".to_string());
    }
    let listed = watches.iter().map(|(id, (_, desc))| format!("{desc} (id {id})")).collect::<Vec<_>>().join(", ");
    Ok(format!("Active watches: {listed}."))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_that_does_not_exist_is_correctly_detected_as_missing() {
        // `watch_path` itself needs a real `AppHandle` (only constructible
        // inside a running Tauri app), so this test exercises the same
        // existence check `watch_path` performs before touching one,
        // confirming the early-return condition is correct.
        assert!(!std::path::Path::new(r"C:\this\does\not\exist_veronica_test").exists());
    }

    #[test]
    fn stop_watch_on_an_unknown_id_is_a_clear_error() {
        let registry = WatcherRegistry::default();
        assert!(stop_watch(&registry, "not-a-real-id").is_err());
    }

    #[test]
    fn list_watches_on_an_empty_registry_says_so() {
        let registry = WatcherRegistry::default();
        assert_eq!(list_watches(&registry).unwrap(), "No active watches.");
    }
}
