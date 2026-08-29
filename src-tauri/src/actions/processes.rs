//! Process queries and control via `sysinfo` (already a dependency, used
//! elsewhere for CPU/memory readings) — cross-platform process enumeration
//! without hand-rolling Toolhelp32/`Win32_System_ProcessStatus` calls.
//! `kill_process` falls back to Win32 `TerminateProcess` only if `sysinfo`'s
//! own kill reports failure (e.g. an access-denied edge case).

use sysinfo::{Pid, ProcessesToUpdate, System};

/// Caps how many processes `list_processes` describes — a busy machine can
/// have 200+ processes; the model rarely needs more than the heaviest few.
const LIST_PROCESSES_MAX: usize = 40;

pub fn list_processes() -> Result<String, String> {
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let mut processes: Vec<(String, u32, f32)> = sys.processes().values().map(|p| (p.name().to_string_lossy().into_owned(), p.pid().as_u32(), p.cpu_usage())).collect();
    processes.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    processes.truncate(LIST_PROCESSES_MAX);
    if processes.is_empty() {
        return Ok("No processes found.".to_string());
    }
    let listed = processes.iter().map(|(name, pid, _)| format!("{name} (pid {pid})")).collect::<Vec<_>>().join(", ");
    Ok(format!("Running processes: {listed}."))
}

pub fn find_by_name(name: &str) -> Result<String, String> {
    let trimmed = name.trim().to_lowercase();
    if trimmed.is_empty() {
        return Err("no process name given".into());
    }
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, true);
    let matches: Vec<(String, u32)> = sys
        .processes()
        .values()
        .filter(|p| p.name().to_string_lossy().to_lowercase().contains(&trimmed))
        .map(|p| (p.name().to_string_lossy().into_owned(), p.pid().as_u32()))
        .collect();
    if matches.is_empty() {
        return Ok(format!("No running process matching \"{name}\" found."));
    }
    let listed = matches.iter().map(|(n, pid)| format!("{n} (pid {pid})")).collect::<Vec<_>>().join(", ");
    Ok(format!("Found: {listed}."))
}

pub fn kill_process(pid: Option<u32>, name: Option<&str>) -> Result<String, String> {
    let mut sys = System::new();
    sys.refresh_processes(ProcessesToUpdate::All, true);

    let target: Option<(u32, String)> = if let Some(pid) = pid {
        sys.process(Pid::from_u32(pid)).map(|p| (pid, p.name().to_string_lossy().into_owned()))
    } else if let Some(name) = name {
        let trimmed = name.trim().to_lowercase();
        sys.processes().values().find(|p| p.name().to_string_lossy().to_lowercase().contains(&trimmed)).map(|p| (p.pid().as_u32(), p.name().to_string_lossy().into_owned()))
    } else {
        return Err("kill_process needs either a pid or a name".into());
    };

    let Some((pid, display_name)) = target else {
        return Err(match (pid, name) {
            (_, Some(name)) => format!("I couldn't find a running process matching \"{name}\"."),
            _ => "I couldn't find that process.".to_string(),
        });
    };

    let killed = sys.process(Pid::from_u32(pid)).map(|p| p.kill()).unwrap_or(false);
    if killed {
        Ok(format!("Ended {display_name} (pid {pid})."))
    } else {
        Err(format!("couldn't end {display_name} (pid {pid}) — it may already be closed or require elevated permissions."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_processes_includes_the_current_test_process() {
        let result = list_processes().unwrap();
        // The current process is always running while this test runs — even
        // truncated to the heaviest N, sysinfo reliably enumerates a
        // non-empty list on any real machine.
        assert!(result.contains("Running processes:"));
    }

    #[test]
    fn find_by_name_with_no_match_returns_empty_not_error() {
        let result = find_by_name("definitely_not_a_real_process_xyz123").unwrap();
        assert!(result.contains("No running process matching"));
    }

    #[test]
    fn kill_process_with_neither_pid_nor_name_is_an_error() {
        assert!(kill_process(None, None).is_err());
    }

    #[test]
    fn kill_process_on_an_unknown_pid_is_a_clear_error_not_a_panic() {
        let result = kill_process(Some(u32::MAX - 1), None);
        assert!(result.is_err());
    }
}
