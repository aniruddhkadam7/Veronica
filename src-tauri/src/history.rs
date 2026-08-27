//! Interview history: a small on-disk archive of past Interview Mode
//! sessions, so the main window can list what happened in prior interviews.
//!
//! This is deliberately separate from `transcript::InterviewSession` (the raw
//! STT segments for the *current* recording) — history entries are the ASK AI
//! question/answer turns from the overlay, which is what a person actually
//! wants to review afterward, not the raw dictation feed.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HistoryTurn {
    pub question: String,
    pub answer: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterviewHistoryEntry {
    pub id: String,
    pub started_at_ms: u64,
    pub ended_at_ms: u64,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub company: Option<String>,
    pub turns: Vec<HistoryTurn>,
}

#[derive(Default)]
pub struct HistoryStore(pub Mutex<Vec<InterviewHistoryEntry>>);

fn history_file_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("failed to resolve app data directory: {e}"))?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create app data directory: {e}"))?;
    Ok(dir.join("interview_history.json"))
}

pub fn load(app: &AppHandle) -> Vec<InterviewHistoryEntry> {
    let path = match history_file_path(app) {
        Ok(p) => p,
        Err(err) => {
            log::warn!("interview history: {err}");
            return Vec::new();
        }
    };
    match fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

fn save(app: &AppHandle, entries: &[InterviewHistoryEntry]) -> Result<(), String> {
    let path = history_file_path(app)?;
    let json = serde_json::to_string_pretty(entries)
        .map_err(|e| format!("failed to serialize interview history: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("failed to write interview history: {e}"))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("hist-{:x}-{:x}", now_ms(), n)
}

#[tauri::command]
pub fn list_interview_history(
    app: AppHandle,
    store: tauri::State<'_, HistoryStore>,
) -> Vec<InterviewHistoryEntry> {
    let mut guard = store.0.lock().unwrap();
    if guard.is_empty() {
        *guard = load(&app);
    }
    let mut entries = guard.clone();
    entries.sort_by(|a, b| b.started_at_ms.cmp(&a.started_at_ms));
    entries
}

/// Archives one finished Interview Mode session. Called only when the user
/// closes the overlay and confirms the interview is over — an empty `turns`
/// list (nothing was ever asked) is skipped so idle opens don't clutter
/// history.
#[tauri::command]
pub fn archive_interview_session(
    app: AppHandle,
    store: tauri::State<'_, HistoryStore>,
    started_at_ms: u64,
    role: Option<String>,
    company: Option<String>,
    turns: Vec<HistoryTurn>,
) -> Result<Option<InterviewHistoryEntry>, String> {
    if turns.is_empty() {
        return Ok(None);
    }
    let entry = InterviewHistoryEntry {
        id: new_id(),
        started_at_ms,
        ended_at_ms: now_ms(),
        role: role.filter(|s| !s.trim().is_empty()),
        company: company.filter(|s| !s.trim().is_empty()),
        turns,
    };

    let mut guard = store.0.lock().unwrap();
    if guard.is_empty() {
        *guard = load(&app);
    }
    guard.push(entry.clone());
    save(&app, &guard)?;
    Ok(Some(entry))
}

#[tauri::command]
pub fn delete_interview_history_entry(
    app: AppHandle,
    store: tauri::State<'_, HistoryStore>,
    id: String,
) -> Result<(), String> {
    let mut guard = store.0.lock().unwrap();
    if guard.is_empty() {
        *guard = load(&app);
    }
    guard.retain(|e| e.id != id);
    save(&app, &guard)
}
