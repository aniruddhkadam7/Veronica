//! Veronica's action-taking system: `veronica::ask_veronica` recognizes when
//! the model's response is an `ACTION: <NAME> | <target>` directive (see
//! `personal::prompts::veronica`'s ACTION-TAKING section) instead of a
//! normal answer, parses it into an `Intent` via `parse_action_line`, and
//! calls `execute` here — checked against a hardcoded safety table first.
//!
//! The LLM never executes anything itself — it only ever returns one of six
//! fixed intent names (the ACTION line format has no slot for an arbitrary
//! command), and `Intent` has no variant that could represent a destructive
//! action (delete, format, registry/security change, credential access,
//! shutdown, arbitrary shell execution, bulk destructive ops, or a
//! consequential external send) — so there is no code path, not even a
//! guarded one, that could run any of those from a request to Veronica.
//! Anything the model doesn't map to the safe six is never wrapped as an
//! `Intent` at all (see `parse_action_line`) and is treated as a normal
//! answer instead.

mod native;
mod registry;
mod router;

pub use registry::RiskLevel;

/// The ONLY vocabulary the router/executor ever see.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Intent {
    OpenApp(String),
    OpenFile(String),
    OpenFolder(String),
    OpenUrl(String),
    QuerySystemInfo(String),
}

/// Parses one line of the shape `ACTION: <NAME> | <target>` (the model's
/// entire response, when it's taking an action — see
/// `personal::prompts::veronica::SYSTEM_PROMPT`'s ACTION-TAKING section)
/// into an `Intent`. Returns `None` for anything that doesn't match this
/// exact shape or whose `<NAME>` isn't one of the six recognized values —
/// both cases mean "not an action," handled by the caller as a normal
/// answer, never as a partially-understood command.
pub fn parse_action_line(line: &str) -> Option<Intent> {
    let rest = line.trim().strip_prefix("ACTION:")?;
    let (name, target) = rest.split_once('|')?;
    let name = name.trim();
    let target = target.trim();
    if target.is_empty() {
        return None;
    }

    match name {
        "OPEN_APP" => Some(Intent::OpenApp(target.to_string())),
        "OPEN_FILE" => Some(Intent::OpenFile(target.to_string())),
        "OPEN_FOLDER" => Some(Intent::OpenFolder(target.to_string())),
        "OPEN_URL" => Some(Intent::OpenUrl(target.to_string())),
        "QUERY_SYSTEM_INFO" => Some(Intent::QuerySystemInfo(target.to_string())),
        _ => None,
    }
}

/// Checks the safety registry, then runs the action through the fastest-
/// method router. Called by `veronica::ask_veronica` once it's recognized
/// and parsed an `ACTION:` line.
pub async fn execute(intent: Intent) -> String {
    match registry::risk_level(&intent) {
        RiskLevel::Blocked => registry::refusal_message(&intent),
        RiskLevel::Safe => router::execute(&intent).await.unwrap_or_else(|e| e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_valid_action_line() {
        assert_eq!(parse_action_line("ACTION: OPEN_APP | Notepad"), Some(Intent::OpenApp("Notepad".to_string())));
    }

    #[test]
    fn rejects_non_action_text() {
        assert_eq!(parse_action_line("Sure, here's how RAG works..."), None);
    }

    #[test]
    fn rejects_unknown_action_name() {
        assert_eq!(parse_action_line("ACTION: DELETE_FILE | C:\\important.txt"), None);
    }

    #[test]
    fn rejects_empty_target() {
        assert_eq!(parse_action_line("ACTION: OPEN_APP | "), None);
    }
}
