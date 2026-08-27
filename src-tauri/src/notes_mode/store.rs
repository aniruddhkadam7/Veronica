//! On-disk store for Notes: unlike the other modes, Notes has no start/stop
//! "session" lifecycle — `notes.json` holds the live note records themselves,
//! following the same load/save/Mutex<Vec<_>> pattern as `history.rs`.

use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Note {
    pub id: String,
    pub title: String,
    pub body: String,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
    #[serde(default)]
    pub linked_note_ids: Vec<String>,
}

#[derive(Default)]
pub struct NotesStore(pub Mutex<Vec<Note>>);

fn notes_file_path(app: &AppHandle) -> Result<PathBuf, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("failed to resolve app data directory: {e}"))?;
    fs::create_dir_all(&dir).map_err(|e| format!("failed to create app data directory: {e}"))?;
    Ok(dir.join("notes.json"))
}

pub fn load(app: &AppHandle) -> Vec<Note> {
    let path = match notes_file_path(app) {
        Ok(p) => p,
        Err(err) => {
            log::warn!("notes store: {err}");
            return Vec::new();
        }
    };
    match fs::read_to_string(&path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

fn save(app: &AppHandle, notes: &[Note]) -> Result<(), String> {
    let path = notes_file_path(app)?;
    let json = serde_json::to_string_pretty(notes).map_err(|e| e.to_string())?;
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
    format!("note-{:x}-{:x}", now_ms(), n)
}

fn ensure_loaded(app: &AppHandle, guard: &mut Vec<Note>) {
    if guard.is_empty() {
        *guard = load(app);
    }
}

#[tauri::command]
pub fn list_notes(app: AppHandle, store: tauri::State<'_, NotesStore>) -> Vec<Note> {
    let mut guard = store.0.lock().unwrap();
    ensure_loaded(&app, &mut guard);
    let mut notes = guard.clone();
    notes.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
    notes
}

#[tauri::command]
pub fn get_note(app: AppHandle, store: tauri::State<'_, NotesStore>, id: String) -> Option<Note> {
    let mut guard = store.0.lock().unwrap();
    ensure_loaded(&app, &mut guard);
    guard.iter().find(|n| n.id == id).cloned()
}

#[tauri::command]
pub fn create_note(
    app: AppHandle,
    store: tauri::State<'_, NotesStore>,
    title: String,
    body: String,
    project: Option<String>,
    tags: Option<Vec<String>>,
) -> Result<Note, String> {
    let mut guard = store.0.lock().map_err(|e| e.to_string())?;
    ensure_loaded(&app, &mut guard);

    let now = now_ms();
    let note = Note {
        id: new_id(),
        title,
        body,
        project: project.filter(|s| !s.trim().is_empty()),
        tags: tags.unwrap_or_default(),
        created_at_ms: now,
        updated_at_ms: now,
        linked_note_ids: Vec::new(),
    };
    guard.push(note.clone());
    save(&app, &guard)?;
    Ok(note)
}

#[tauri::command]
pub fn update_note(
    app: AppHandle,
    store: tauri::State<'_, NotesStore>,
    id: String,
    title: Option<String>,
    body: Option<String>,
    project: Option<String>,
    tags: Option<Vec<String>>,
    linked_note_ids: Option<Vec<String>>,
) -> Result<Note, String> {
    let mut guard = store.0.lock().map_err(|e| e.to_string())?;
    ensure_loaded(&app, &mut guard);

    let note = guard.iter_mut().find(|n| n.id == id).ok_or_else(|| "note not found".to_string())?;
    if let Some(title) = title {
        note.title = title;
    }
    if let Some(body) = body {
        note.body = body;
    }
    if let Some(project) = project {
        note.project = Some(project).filter(|s| !s.trim().is_empty());
    }
    if let Some(tags) = tags {
        note.tags = tags;
    }
    if let Some(linked_note_ids) = linked_note_ids {
        note.linked_note_ids = linked_note_ids;
    }
    note.updated_at_ms = now_ms();
    let updated = note.clone();

    save(&app, &guard)?;
    Ok(updated)
}

#[tauri::command]
pub fn delete_note(app: AppHandle, store: tauri::State<'_, NotesStore>, id: String) -> Result<(), String> {
    let mut guard = store.0.lock().map_err(|e| e.to_string())?;
    ensure_loaded(&app, &mut guard);
    guard.retain(|n| n.id != id);
    save(&app, &guard)
}

/// Simple client-side-style substring search over title/body/tags/project —
/// deliberately not RAG-backed (notes are typically small and this needs to
/// feel instant as the user types).
#[tauri::command]
pub fn search_notes(app: AppHandle, store: tauri::State<'_, NotesStore>, query: String) -> Vec<Note> {
    let mut guard = store.0.lock().unwrap();
    ensure_loaded(&app, &mut guard);
    let needle = query.trim().to_lowercase();
    if needle.is_empty() {
        let mut notes = guard.clone();
        notes.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
        return notes;
    }
    let mut matches: Vec<Note> = guard
        .iter()
        .filter(|n| {
            n.title.to_lowercase().contains(&needle)
                || n.body.to_lowercase().contains(&needle)
                || n.project.as_deref().unwrap_or("").to_lowercase().contains(&needle)
                || n.tags.iter().any(|t| t.to_lowercase().contains(&needle))
        })
        .cloned()
        .collect();
    matches.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
    matches
}
