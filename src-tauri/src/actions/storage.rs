//! Storage: read-only queries (list/search/largest-files/disk-usage) plus
//! mutating operations on EXISTING paths (delete, move/rename) — split from
//! `filesystem.rs` because this is where the risk tier changes: everything
//! here except the queries can destroy or move something the user didn't
//! just create, so `DeleteFile`/`MoveOrRename` are classified
//! `Destructive`/`Sensitive` in `registry.rs`.
//!
//! `delete_file` uses the Shell's `IFileOperation` (via the simpler,
//! still-supported `SHFileOperationW`) with `FOF_ALLOWUNDO` so a delete goes
//! to the Recycle Bin, never `std::fs::remove_file` (permanent, no native
//! "prefer Shell APIs" story) — this is the concrete form the plan's
//! "prefer native Shell APIs over blunt filesystem calls" requirement takes
//! for a destructive operation.
//!
//! `largest_files`/`search_files` accept a `&CancelToken` and check it
//! periodically during their recursive walk — the orchestrator's
//! per-iteration/per-tool-call cancellation check can't preempt a single
//! long-running `.await`ed call once it's started (see
//! `orchestrator::run_agent_loop`'s doc), so a large/slow walk must
//! self-check instead.

use std::path::{Path, PathBuf};

use windows::core::{BOOL, PCWSTR};
use windows::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;
use windows::Win32::UI::Shell::{SHFileOperationW, FOF_ALLOWUNDO, FOF_NOCONFIRMATION, FOF_NO_UI, FO_DELETE, SHFILEOPSTRUCTW};

use crate::state::CancelToken;

/// Caps how many entries `list_folder` will describe before summarizing the
/// rest — keeps a tool result bounded even for a folder with thousands of
/// files.
const LIST_FOLDER_MAX_ENTRIES: usize = 200;

/// How many filesystem entries a bounded walk (`search_files`/
/// `largest_files`) visits between cancellation checks.
const WALK_CANCEL_CHECK_INTERVAL: usize = 500;

/// Hard ceiling on how many entries a single walk will visit at all, so
/// scanning `C:\` can't hang a turn indefinitely even with no cancellation —
/// results past this point are reported as a partial scan, not silently
/// dropped.
const WALK_MAX_ENTRIES: usize = 200_000;

pub fn list_folder(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no folder path given".into());
    }
    let entries = std::fs::read_dir(trimmed).map_err(|e| format!("couldn't list \"{trimmed}\": {e}"))?;
    let mut names: Vec<String> = Vec::new();
    let mut total = 0usize;
    for entry in entries.flatten() {
        total += 1;
        if names.len() < LIST_FOLDER_MAX_ENTRIES {
            let kind = if entry.path().is_dir() { "folder" } else { "file" };
            names.push(format!("{} ({kind})", entry.file_name().to_string_lossy()));
        }
    }
    if names.is_empty() {
        return Ok(format!("\"{trimmed}\" is empty."));
    }
    let mut result = format!("Contents of \"{trimmed}\": {}", names.join(", "));
    if total > names.len() {
        result.push_str(&format!(" ...and {} more.", total - names.len()));
    }
    Ok(result)
}

fn walk(root: &Path, depth: u8, visited: &mut usize, cancel: &CancelToken, visit: &mut dyn FnMut(&Path, u64)) -> Result<bool, String> {
    if depth > 32 || *visited >= WALK_MAX_ENTRIES {
        return Ok(false);
    }
    let Ok(entries) = std::fs::read_dir(root) else { return Ok(true) };
    for entry in entries.flatten() {
        *visited += 1;
        if *visited % WALK_CANCEL_CHECK_INTERVAL == 0 && cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        if *visited >= WALK_MAX_ENTRIES {
            return Ok(false);
        }
        let path = entry.path();
        if path.is_dir() {
            if !walk(&path, depth + 1, visited, cancel, visit)? {
                return Ok(false);
            }
        } else if let Ok(meta) = entry.metadata() {
            visit(&path, meta.len());
        }
    }
    Ok(true)
}

pub fn search_files(root: &str, query: &str, max_results: Option<u32>, cancel: &CancelToken) -> Result<String, String> {
    let trimmed_root = root.trim();
    let trimmed_query = query.trim().to_lowercase();
    if trimmed_root.is_empty() || trimmed_query.is_empty() {
        return Err("search_files needs both a root folder and a query".into());
    }
    let cap = max_results.unwrap_or(50).max(1) as usize;
    let mut matches: Vec<PathBuf> = Vec::new();
    let mut visited = 0usize;
    let mut visit = |path: &Path, _size: u64| {
        if matches.len() < cap {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.to_lowercase().contains(&trimmed_query) {
                    matches.push(path.to_path_buf());
                }
            }
        }
    };
    let complete = walk(Path::new(trimmed_root), 0, &mut visited, cancel, &mut visit)?;
    if matches.is_empty() {
        return Ok(format!("No files matching \"{query}\" found under \"{trimmed_root}\"{}.", if complete { "" } else { " (scanned a subset — this may not be exhaustive)" }));
    }
    let listed = matches.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n");
    let mut result = format!("Found {} matching \"{query}\":\n{listed}", matches.len());
    if !complete {
        result.push_str("\n(scanned a subset — this may not be exhaustive)");
    }
    Ok(result)
}

pub fn largest_files(root: &str, top_n: u8, cancel: &CancelToken) -> Result<String, String> {
    let trimmed = root.trim();
    if trimmed.is_empty() {
        return Err("no root folder given".into());
    }
    let top_n = top_n.clamp(1, 100) as usize;
    let mut sized: Vec<(PathBuf, u64)> = Vec::new();
    let mut visited = 0usize;
    let mut visit = |path: &Path, size: u64| {
        sized.push((path.to_path_buf(), size));
    };
    let complete = walk(Path::new(trimmed), 0, &mut visited, cancel, &mut visit)?;
    sized.sort_by(|a, b| b.1.cmp(&a.1));
    sized.truncate(top_n);
    if sized.is_empty() {
        return Ok(format!("No files found under \"{trimmed}\"."));
    }
    let listed = sized
        .iter()
        .map(|(p, size)| format!("{} — {:.1} MB", p.display(), *size as f64 / 1_048_576.0))
        .collect::<Vec<_>>()
        .join("\n");
    let mut result = format!("Largest files under \"{trimmed}\":\n{listed}");
    if !complete {
        result.push_str("\n(scanned a subset — this may not be exhaustive)");
    }
    Ok(result)
}

pub fn disk_usage(drive: Option<&str>) -> Result<String, String> {
    let drive = drive.unwrap_or("C:\\");
    let drive = if drive.ends_with('\\') { drive.to_string() } else { format!("{drive}\\") };
    let wide: Vec<u16> = drive.encode_utf16().chain(std::iter::once(0)).collect();

    let mut free_bytes = 0u64;
    let mut total_bytes = 0u64;
    unsafe {
        GetDiskFreeSpaceExW(PCWSTR(wide.as_ptr()), None, Some(&mut total_bytes), Some(&mut free_bytes))
            .map_err(|e| format!("couldn't read disk usage for \"{drive}\": {e}"))?;
    }
    let used = total_bytes.saturating_sub(free_bytes);
    let gb = |b: u64| b as f64 / 1_073_741_824.0;
    Ok(format!(
        "Drive {drive} — {:.1} GB used of {:.1} GB total ({:.1} GB free).",
        gb(used),
        gb(total_bytes),
        gb(free_bytes)
    ))
}

/// Recycle-bin-safe delete via the Shell's `SHFileOperationW` with
/// `FOF_ALLOWUNDO` — never `std::fs::remove_file`, which is permanent.
pub fn delete_file(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no path given".into());
    }
    if !Path::new(trimmed).exists() {
        return Err(format!("\"{trimmed}\" doesn't exist."));
    }
    // SHFileOperationW's pFrom must be a double-null-terminated list of
    // paths (even for a single path).
    let mut wide: Vec<u16> = trimmed.encode_utf16().collect();
    wide.push(0);
    wide.push(0);

    let mut op = SHFILEOPSTRUCTW {
        hwnd: windows::Win32::Foundation::HWND(std::ptr::null_mut()),
        wFunc: FO_DELETE,
        pFrom: PCWSTR(wide.as_ptr()),
        pTo: PCWSTR::null(),
        fFlags: (FOF_ALLOWUNDO.0 | FOF_NOCONFIRMATION.0 | FOF_NO_UI.0) as u16,
        fAnyOperationsAborted: BOOL(0),
        hNameMappings: std::ptr::null_mut(),
        lpszProgressTitle: PCWSTR::null(),
    };
    let result = unsafe { SHFileOperationW(&mut op) };
    if result == 0 && op.fAnyOperationsAborted.0 == 0 {
        Ok(format!("Sent \"{trimmed}\" to the Recycle Bin."))
    } else {
        Err(format!("couldn't delete \"{trimmed}\" (Shell error code {result})"))
    }
}

pub fn move_or_rename(from: &str, to: &str) -> Result<String, String> {
    let (from, to) = (from.trim(), to.trim());
    if from.is_empty() || to.is_empty() {
        return Err("move_or_rename needs both a source and destination path".into());
    }
    if let Some(parent) = Path::new(to).parent() {
        if !parent.as_os_str().is_empty() {
            let _ = std::fs::create_dir_all(parent);
        }
    }
    match std::fs::rename(from, to) {
        Ok(()) => Ok(format!("Moved \"{from}\" to \"{to}\".")),
        Err(_) => {
            // Cross-volume rename fails on Windows — fall back to copy, then
            // recycle-bin-delete the original rather than a permanent
            // remove, keeping the same "never destroy without a safety net"
            // guarantee as `delete_file`.
            std::fs::copy(from, to).map_err(|e| format!("couldn't move \"{from}\" to \"{to}\": {e}"))?;
            delete_file(from)?;
            Ok(format!("Moved \"{from}\" to \"{to}\"."))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("veronica_test_storage_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn largest_files_returns_top_n_sorted_descending() {
        let dir = scratch_dir("largest");
        fs::write(dir.join("small.txt"), vec![0u8; 10]).unwrap();
        fs::write(dir.join("big.txt"), vec![0u8; 10_000]).unwrap();
        fs::write(dir.join("medium.txt"), vec![0u8; 1_000]).unwrap();

        let cancel = CancelToken::new();
        let result = largest_files(dir.to_str().unwrap(), 2, &cancel).unwrap();
        let big_pos = result.find("big.txt").unwrap();
        let medium_pos = result.find("medium.txt").unwrap();
        assert!(big_pos < medium_pos, "big.txt should be listed before medium.txt");
        assert!(!result.contains("small.txt"), "top_n=2 should exclude the smallest file");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_files_finds_by_substring_recursively() {
        let dir = scratch_dir("search");
        let nested = dir.join("sub");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("auth_handler.rs"), "").unwrap();
        fs::write(dir.join("readme.md"), "").unwrap();

        let cancel = CancelToken::new();
        let result = search_files(dir.to_str().unwrap(), "auth", None, &cancel).unwrap();
        assert!(result.contains("auth_handler.rs"));
        assert!(!result.contains("readme.md"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn search_files_with_no_match_reports_none_found_not_an_error() {
        let dir = scratch_dir("search_empty");
        let cancel = CancelToken::new();
        let result = search_files(dir.to_str().unwrap(), "nonexistent_xyz", None, &cancel).unwrap();
        assert!(result.contains("No files matching"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn cancelled_token_stops_a_walk_in_progress() {
        let dir = scratch_dir("cancel_walk");
        for i in 0..(WALK_CANCEL_CHECK_INTERVAL * 2) {
            fs::write(dir.join(format!("f{i}.txt")), "").unwrap();
        }
        let cancel = CancelToken::new();
        cancel.cancel();
        let result = search_files(dir.to_str().unwrap(), "f", None, &cancel);
        assert!(result.is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_file_on_a_missing_path_is_a_clear_error() {
        let result = delete_file(r"C:\this\path\does\not\exist_veronica_test.txt");
        assert!(result.is_err());
    }

    #[test]
    fn delete_file_actually_removes_the_file() {
        let dir = scratch_dir("delete");
        let file = dir.join("to_delete.txt");
        fs::write(&file, "x").unwrap();
        assert!(file.exists());
        delete_file(file.to_str().unwrap()).unwrap();
        assert!(!file.exists(), "file should be gone after delete_file (Recycle Bin, not in place)");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn move_or_rename_moves_the_file() {
        let dir = scratch_dir("move");
        let from = dir.join("source.txt");
        let to = dir.join("dest.txt");
        fs::write(&from, "content").unwrap();
        move_or_rename(from.to_str().unwrap(), to.to_str().unwrap()).unwrap();
        assert!(!from.exists());
        assert!(to.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn disk_usage_for_c_drive_does_not_error() {
        assert!(disk_usage(Some("C:")).is_ok());
    }
}
