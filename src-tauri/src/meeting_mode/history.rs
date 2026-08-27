//! On-disk archive of past Meeting Mode meetings — load/save against its own
//! file, backed by a `Mutex<Vec<_>>` in memory.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Manager};

use crate::backend::MeetingSummary;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeetingHistoryTurn {
    pub speaker: String,
    pub text: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MeetingHistoryEntry {
    pub id: String,
    pub started_at_ms: u64,
    pub ended_at_ms: u64,
    #[serde(default)]
    pub meeting_title: Option<String>,
    #[serde(default)]
    pub participants: Option<String>,
    pub turns: Vec<MeetingHistoryTurn>,
    pub summary: MeetingSummary,
}

#[derive(Default)]
pub struct MeetingHistoryStore(pub Mutex<Vec<MeetingHistoryEntry>>);

fn history_file_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("failed to resolve app data directory: {e}"))?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create app data directory: {e}"))?;
    Ok(dir.join("meeting_history.json"))
}

pub fn load(app: &AppHandle) -> Vec<MeetingHistoryEntry> {
    let path = match history_file_path(app) {
        Ok(p) => p,
        Err(err) => {
            log::warn!("meeting history: {err}");
            return Vec::new();
        }
    };
    match fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

fn save(app: &AppHandle, entries: &[MeetingHistoryEntry]) -> Result<(), String> {
    let path = history_file_path(app)?;
    let json = serde_json::to_string_pretty(entries).map_err(|e| e.to_string())?;
    fs::write(&path, json).map_err(|e| e.to_string())
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
    format!("meeting-{:x}-{:x}", now_ms(), n)
}

#[tauri::command]
pub fn list_meeting_history(app: AppHandle, store: tauri::State<'_, MeetingHistoryStore>) -> Vec<MeetingHistoryEntry> {
    let mut guard = store.0.lock().unwrap();
    if guard.is_empty() {
        *guard = load(&app);
    }
    let mut entries = guard.clone();
    entries.sort_by(|a, b| b.started_at_ms.cmp(&a.started_at_ms));
    entries
}

#[tauri::command]
pub fn archive_meeting(
    app: AppHandle,
    store: tauri::State<'_, MeetingHistoryStore>,
    started_at_ms: u64,
    meeting_title: Option<String>,
    participants: Option<String>,
    turns: Vec<MeetingHistoryTurn>,
    summary: MeetingSummary,
) -> Result<Option<MeetingHistoryEntry>, String> {
    if turns.is_empty() {
        return Ok(None);
    }
    let entry = MeetingHistoryEntry {
        id: new_id(),
        started_at_ms,
        ended_at_ms: now_ms(),
        meeting_title: meeting_title.filter(|s| !s.trim().is_empty()),
        participants: participants.filter(|s| !s.trim().is_empty()),
        turns,
        summary,
    };

    let mut guard = store.0.lock().map_err(|e| e.to_string())?;
    if guard.is_empty() {
        *guard = load(&app);
    }
    guard.push(entry.clone());
    save(&app, &guard)?;
    Ok(Some(entry))
}

#[tauri::command]
pub fn delete_meeting_history_entry(
    app: AppHandle,
    store: tauri::State<'_, MeetingHistoryStore>,
    id: String,
) -> Result<(), String> {
    let mut guard = store.0.lock().map_err(|e| e.to_string())?;
    if guard.is_empty() {
        *guard = load(&app);
    }
    guard.retain(|e| e.id != id);
    save(&app, &guard)
}
