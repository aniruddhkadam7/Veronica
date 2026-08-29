//! The Action Registry + Safety Layer: every `Capability` must pass through
//! `risk_level_for_capability` before `execute_tool` will run it.
//!
//! Three tiers, not a binary allow/deny:
//!   - `Safe` — executes immediately, no confirmation.
//!   - `Sensitive` — recoverable but consequential (killing a process,
//!     moving a file, scheduling/watching something) — Veronica asks once
//!     ("Are you sure you want to...?") and waits for a yes/no before
//!     running it.
//!   - `Destructive` — hard or impossible to undo (deleting a file, running
//!     an unrecognized or destructive terminal command) — same
//!     confirm-before-run flow as `Sensitive`, phrased with extra emphasis.
//!
//! `risk_level_for_capability` remains an EXHAUSTIVE match over `Capability`
//! — adding a new variant is a compile error here until it's explicitly
//! classified. That compile-time discipline, not a runtime blocklist, is the
//! actual safety guarantee: nothing can reach `execute_tool` without having
//! been assigned a tier by a human reviewing this file.
//!
//! BLOCKLIST CHECKLIST — read this before adding a new `Capability` variant.
//! If you're adding one that resembles any of these, it must be classified
//! `RiskLevel::Destructive` (never `Safe`, and Sensitive only if it's
//! genuinely low-consequence and easily reversible) — and probably needs a
//! product conversation first:
//!   - Deleting files/folders (`StorageOp::DeleteFile` — Destructive)
//!   - Disk formatting, partitioning
//!   - Registry or other system-configuration modification
//!   - Security-setting changes
//!   - Credential/token access
//!   - Privilege escalation
//!   - Arbitrary PowerShell/CMD execution (`TerminalOp::RunCommand` — risk
//!     computed per-command, see `classify_command_risk`; unrecognized
//!     commands float to at least `Sensitive`, never `Safe`)
//!   - Software installation/uninstallation
//!   - Shutdown/restart/sleep
//!   - Killing a process (`ProcessOp::Kill` — Sensitive)
//!   - Bulk destructive operations (e.g. "close everything", "clear my downloads")
//!   - Sending consequential external messages/actions (email, chat, purchases)

use super::capability::{Capability, SchedulerOp, StorageOp, TerminalOp, WatcherOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Safe,
    Sensitive,
    Destructive,
}

/// Classifies every `Capability` variant. A new variant is a compile error
/// here until explicitly matched — see the module doc.
pub fn risk_level_for_capability(capability: &Capability) -> RiskLevel {
    match capability {
        Capability::LaunchOrFocusApp(_)
        | Capability::LaunchAppWithArg { .. }
        | Capability::OpenPath(_)
        | Capability::WindowOp { .. }
        | Capability::WindowQuery(_)
        | Capability::SystemInfo(_)
        | Capability::VolumeOp(_)
        | Capability::Clipboard(_)
        | Capability::TaskControl(_)
        | Capability::CaptureScreen
        | Capability::SearchKnowledgeBase(_)
        | Capability::FileOp(_) => RiskLevel::Safe,

        Capability::StorageQuery(_) => RiskLevel::Safe,

        Capability::StorageOp(op) => match op {
            StorageOp::DeleteFile { .. } => RiskLevel::Destructive,
            StorageOp::MoveOrRename { .. } => RiskLevel::Sensitive,
        },

        Capability::ProcessQuery(_) => RiskLevel::Safe,
        Capability::ProcessOp(_) => RiskLevel::Sensitive,
        Capability::NetworkQuery(_) => RiskLevel::Safe,

        Capability::TerminalOp(TerminalOp::RunCommand { command, .. }) => classify_command_risk(command),

        Capability::SchedulerOp(op) => match op {
            SchedulerOp::ListScheduled => RiskLevel::Safe,
            SchedulerOp::ScheduleOnce { .. } | SchedulerOp::CancelScheduled { .. } => RiskLevel::Sensitive,
        },
        Capability::WatcherOp(op) => match op {
            WatcherOp::ListWatches => RiskLevel::Safe,
            WatcherOp::WatchPath { .. } | WatcherOp::StopWatch { .. } => RiskLevel::Sensitive,
        },
    }
}

/// Closed, hand-maintained keyword lists — same discipline as the
/// BLOCKLIST CHECKLIST doc comment above: a new destructive command
/// discovered later must be added to `DESTRUCTIVE_KEYWORDS`, not left to the
/// default floor. Matched against the leading verb/command word (after
/// stripping a `cmd /c`/`powershell -command` wrapper), case-insensitive.
const SAFE_KEYWORDS: &[&str] = &[
    "dir", "ls", "type", "cat", "echo", "whoami", "hostname", "ipconfig", "ping", "tasklist", "systeminfo", "where", "find", "netstat", "git status", "git log", "git diff",
    "docker ps", "docker logs", "docker inspect", "docker images", "docker version",
];
const DESTRUCTIVE_KEYWORDS: &[&str] = &[
    "del", "erase", "rd", "rmdir", "format", "diskpart", "shutdown", "taskkill /f", "reg delete", "cipher /w", "docker rm", "docker system prune", "git reset --hard", "git clean -f",
    "net user", "/delete",
];

/// Classifies a raw terminal command's risk from its text — see the module
/// doc's "Terminal" entry in the BLOCKLIST CHECKLIST. Chained commands
/// (`&&`, `;`, `|`) escalate to the maximum risk of any segment. An
/// unrecognized leading verb floats to `Sensitive` at minimum — never `Safe`
/// — per the explicit requirement that unknown commands always require
/// confirmation.
pub fn classify_command_risk(command: &str) -> RiskLevel {
    let segments: Vec<&str> = command.split(['&', ';', '|']).map(str::trim).filter(|s| !s.is_empty()).collect();
    if segments.is_empty() {
        return RiskLevel::Sensitive;
    }
    segments.iter().map(|segment| classify_single_command(segment)).max_by_key(|risk| match risk {
        RiskLevel::Safe => 0,
        RiskLevel::Sensitive => 1,
        RiskLevel::Destructive => 2,
    }).unwrap_or(RiskLevel::Sensitive)
}

fn classify_single_command(segment: &str) -> RiskLevel {
    let lower = segment.to_lowercase();
    // Strip a leading `cmd /c`/`powershell -command`/`powershell.exe -c` wrapper.
    let stripped = ["cmd /c ", "cmd.exe /c ", "powershell -command ", "powershell.exe -command ", "powershell -c ", "pwsh -c "]
        .iter()
        .find_map(|prefix| lower.strip_prefix(prefix))
        .unwrap_or(&lower)
        .trim();

    if DESTRUCTIVE_KEYWORDS.iter().any(|kw| stripped.contains(kw)) {
        return RiskLevel::Destructive;
    }
    if SAFE_KEYWORDS.iter().any(|kw| stripped == *kw || stripped.starts_with(&format!("{kw} "))) {
        return RiskLevel::Safe;
    }
    // Recognized-but-not-safe (e.g. "docker restart", "npm install", "mkdir",
    // "git commit") and genuinely unrecognized commands both land here —
    // "unknown commands must require confirmation" means the floor for
    // anything not explicitly Safe is Sensitive, never Safe.
    RiskLevel::Sensitive
}

/// Natural-language yes/no question spoken (and shown in the overlay dialog)
/// when `execute_tool` withholds a `Sensitive`/`Destructive` capability. One
/// arm per non-`Safe` variant/op — a `Safe` capability never reaches this
/// function since `execute_tool` only calls it after checking the risk tier.
pub fn confirmation_prompt_for(capability: &Capability) -> String {
    match capability {
        Capability::StorageOp(StorageOp::DeleteFile { path }) => {
            format!("Are you sure you want to delete \"{path}\"? I'll send it to the Recycle Bin, but I want to check first.")
        }
        Capability::StorageOp(StorageOp::MoveOrRename { from, to }) => {
            format!("Should I move \"{from}\" to \"{to}\"?")
        }
        Capability::ProcessOp(super::capability::ProcessOp::Kill { pid, name }) => match (pid, name) {
            (Some(pid), _) => format!("Are you sure you want me to end process {pid}?"),
            (None, Some(name)) => format!("Are you sure you want me to end \"{name}\"?"),
            (None, None) => "Are you sure you want me to end that process?".to_string(),
        },
        Capability::TerminalOp(TerminalOp::RunCommand { command, .. }) => match classify_command_risk(command) {
            RiskLevel::Destructive => format!("Are you sure you want me to run this command? It could make destructive changes: {command}"),
            RiskLevel::Sensitive => format!("I don't recognize this command, so I want to check before running it: {command}"),
            RiskLevel::Safe => format!("Should I run this command? {command}"), // unreachable in practice — Safe never reaches confirmation
        },
        Capability::SchedulerOp(SchedulerOp::ScheduleOnce { description, .. }) => format!("Should I schedule this: {description}?"),
        Capability::SchedulerOp(SchedulerOp::CancelScheduled { .. }) => "Should I cancel that scheduled action?".to_string(),
        Capability::WatcherOp(WatcherOp::WatchPath { path, .. }) => format!("Should I start watching \"{path}\" for changes?"),
        Capability::WatcherOp(WatcherOp::StopWatch { .. }) => "Should I stop that watch?".to_string(),
        other => format!("Are you sure you want me to do this: {other:?}?"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::capability::{FileOp, StorageQuery, SystemInfoKind, TaskControlOp};

    #[test]
    fn every_capability_variant_has_a_risk_level() {
        assert_eq!(risk_level_for_capability(&Capability::LaunchOrFocusApp("Notepad".to_string())), RiskLevel::Safe);
        assert_eq!(risk_level_for_capability(&Capability::SystemInfo(SystemInfoKind::Cpu)), RiskLevel::Safe);
        assert_eq!(risk_level_for_capability(&Capability::TaskControl(TaskControlOp::Pause)), RiskLevel::Safe);
        assert_eq!(risk_level_for_capability(&Capability::CaptureScreen), RiskLevel::Safe);
        assert_eq!(
            risk_level_for_capability(&Capability::FileOp(FileOp::CreateFolder { path: "C:\\scratch".to_string() })),
            RiskLevel::Safe
        );
        assert_eq!(
            risk_level_for_capability(&Capability::StorageQuery(StorageQuery::ListFolder { path: "C:\\".to_string() })),
            RiskLevel::Safe
        );
        assert_eq!(
            risk_level_for_capability(&Capability::StorageOp(StorageOp::DeleteFile { path: "x.txt".to_string() })),
            RiskLevel::Destructive
        );
        assert_eq!(
            risk_level_for_capability(&Capability::StorageOp(StorageOp::MoveOrRename { from: "a".to_string(), to: "b".to_string() })),
            RiskLevel::Sensitive
        );
        assert_eq!(risk_level_for_capability(&Capability::ProcessQuery(crate::actions::capability::ProcessQuery::List)), RiskLevel::Safe);
        assert_eq!(
            risk_level_for_capability(&Capability::ProcessOp(crate::actions::capability::ProcessOp::Kill { pid: Some(1), name: None })),
            RiskLevel::Sensitive
        );
        assert_eq!(risk_level_for_capability(&Capability::NetworkQuery(crate::actions::capability::NetworkQuery::Status)), RiskLevel::Safe);
    }

    #[test]
    fn confirmation_prompt_mentions_the_path_being_deleted() {
        let prompt = confirmation_prompt_for(&Capability::StorageOp(StorageOp::DeleteFile { path: "report.pdf".to_string() }));
        assert!(prompt.contains("report.pdf"));
    }

    #[test]
    fn classify_command_risk_recognizes_safe_diagnostic_commands() {
        for cmd in ["dir", "ls -la", "whoami", "ipconfig", "ping 8.8.8.8", "tasklist", "docker ps", "docker logs mycontainer", "git status"] {
            assert_eq!(classify_command_risk(cmd), RiskLevel::Safe, "expected Safe for {cmd:?}");
        }
    }

    #[test]
    fn classify_command_risk_recognizes_destructive_commands() {
        for cmd in ["del file.txt", "rd /s /q C:\\temp", "format C:", "shutdown /s", "taskkill /f /pid 123", "docker rm mycontainer", "docker system prune", "git reset --hard"] {
            assert_eq!(classify_command_risk(cmd), RiskLevel::Destructive, "expected Destructive for {cmd:?}");
        }
    }

    #[test]
    fn classify_command_risk_floors_recognized_nondestructive_commands_at_sensitive() {
        for cmd in ["docker restart mycontainer", "npm install", "mkdir newfolder", "git commit -m x"] {
            assert_eq!(classify_command_risk(cmd), RiskLevel::Sensitive, "expected Sensitive for {cmd:?}");
        }
    }

    #[test]
    fn classify_command_risk_floors_unrecognized_commands_at_sensitive_never_safe() {
        assert_eq!(classify_command_risk("some_totally_unknown_tool --flag"), RiskLevel::Sensitive);
    }

    #[test]
    fn classify_command_risk_chained_commands_escalate_to_the_max_segment() {
        assert_eq!(classify_command_risk("echo hi && del file.txt"), RiskLevel::Destructive);
        assert_eq!(classify_command_risk("dir && whoami"), RiskLevel::Safe);
    }

    #[test]
    fn terminal_op_risk_flows_through_risk_level_for_capability() {
        let capability = Capability::TerminalOp(crate::actions::capability::TerminalOp::RunCommand { command: "del x.txt".to_string(), working_dir: None });
        assert_eq!(risk_level_for_capability(&capability), RiskLevel::Destructive);
    }
}
