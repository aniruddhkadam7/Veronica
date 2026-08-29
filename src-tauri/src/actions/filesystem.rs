//! Filesystem creation/write/read — plain `std::fs` is sufficient here (no
//! Win32 API adds anything for creating/writing/reading a file or folder on
//! a local path), matching this module's `Result<String, String>` /
//! `.map_err` convention. Mutating an EXISTING path (delete, move/rename)
//! lives in `storage.rs` instead — see that module's doc for why the split
//! matches the risk-tier boundary.

use std::fs;
use std::path::Path;

/// Cap on how much of a file's content is echoed back in a `ReadFile`
/// result — keeps a huge log/text file from blowing out the tool-result
/// payload handed back to the model.
const READ_FILE_MAX_BYTES: usize = 64 * 1024;

pub fn create_folder(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no folder path given".into());
    }
    fs::create_dir_all(trimmed).map_err(|e| format!("couldn't create folder \"{trimmed}\": {e}"))?;
    Ok(format!("Created folder \"{trimmed}\"."))
}

pub fn create_file(path: &str, content: Option<&str>) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no file path given".into());
    }
    if let Some(parent) = Path::new(trimmed).parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("couldn't create parent folder for \"{trimmed}\": {e}"))?;
        }
    }
    fs::write(trimmed, content.unwrap_or("")).map_err(|e| format!("couldn't create file \"{trimmed}\": {e}"))?;
    Ok(format!("Created file \"{trimmed}\"."))
}

pub fn write_file(path: &str, content: &str, append: bool) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no file path given".into());
    }
    if append {
        use std::io::Write;
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(trimmed)
            .map_err(|e| format!("couldn't open \"{trimmed}\" for appending: {e}"))?;
        file.write_all(content.as_bytes()).map_err(|e| format!("couldn't append to \"{trimmed}\": {e}"))?;
        Ok(format!("Appended {} bytes to \"{trimmed}\".", content.len()))
    } else {
        fs::write(trimmed, content).map_err(|e| format!("couldn't write \"{trimmed}\": {e}"))?;
        Ok(format!("Wrote {} bytes to \"{trimmed}\".", content.len()))
    }
}

pub fn read_file(path: &str) -> Result<String, String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return Err("no file path given".into());
    }
    let bytes = fs::read(trimmed).map_err(|e| format!("couldn't read \"{trimmed}\": {e}"))?;
    let truncated = bytes.len() > READ_FILE_MAX_BYTES;
    let slice = &bytes[..bytes.len().min(READ_FILE_MAX_BYTES)];
    let mut text = String::from_utf8_lossy(slice).into_owned();
    if truncated {
        text.push_str(&format!("\n...(truncated — file is {} bytes, showing the first {READ_FILE_MAX_BYTES})", bytes.len()));
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("veronica_test_{name}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn create_folder_then_file_then_write_succeeds() {
        let dir = scratch_dir("fs_chain");
        let folder = dir.join("my_project");
        let file = folder.join("notes.txt");

        create_folder(folder.to_str().unwrap()).unwrap();
        assert!(folder.exists());

        create_file(file.to_str().unwrap(), None).unwrap();
        assert!(file.exists());

        write_file(file.to_str().unwrap(), "hello world", false).unwrap();
        let read_back = read_file(file.to_str().unwrap()).unwrap();
        assert_eq!(read_back, "hello world");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_file_append_true_adds_without_truncating() {
        let dir = scratch_dir("fs_append");
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("log.txt");

        write_file(file.to_str().unwrap(), "first ", false).unwrap();
        write_file(file.to_str().unwrap(), "second", true).unwrap();
        assert_eq!(read_file(file.to_str().unwrap()).unwrap(), "first second");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_file_on_missing_path_is_a_clear_error_not_a_panic() {
        let result = read_file(r"C:\this\path\definitely\does\not\exist_12345.txt");
        assert!(result.is_err());
    }

    #[test]
    fn empty_path_is_rejected_for_every_op() {
        assert!(create_folder("").is_err());
        assert!(create_file("  ", None).is_err());
        assert!(write_file("", "x", false).is_err());
        assert!(read_file("").is_err());
    }
}
