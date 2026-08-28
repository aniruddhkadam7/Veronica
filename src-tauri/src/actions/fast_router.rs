//! The Fast Router: deterministic, in-process, zero-network capability
//! matching that runs *before* RAG or any LLM call. A match here means the
//! turn never touches the agent loop at all — see `veronica::ask_veronica`'s
//! new dispatch order.
//!
//! This is capability/verb-family matching, not a table of literal phrases:
//! `verb_family` recognizes a small, fixed set of *verbs* (open, close,
//! focus, minimize, mute, pause, ...), and the surrounding words are only
//! ever inspected for a small set of *keyword buckets* (a path-shaped
//! target, a system-info topic, a volume direction). Any phrasing built
//! from those verbs matches — "open Chrome", "would you open Chrome for
//! me", "please launch Chrome" — without a new code path per phrasing.
//! Anything whose leading verb isn't recognized returns `None` and falls
//! through to the agent loop, which is exactly the intended behavior for
//! genuinely open-ended requests ("find the bug and fix it" has no
//! recognized leading verb here).

use super::capability::{Capability, ClipboardOp, SystemInfoKind, TaskControlOp, VolumeOp, WindowOp};

/// Politeness/filler words that precede the real verb in natural speech —
/// stripped before verb matching so "could you please open Chrome" matches
/// exactly like "open Chrome".
const LEADING_FILLER: &[&str] = &[
    "hey veronica", "veronica", "please", "could you please", "could you", "can you please", "can you", "would you please",
    "would you", "just", "go ahead and", "i want you to", "i need you to",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerbFamily {
    OpenLaunch,
    Close,
    Focus,
    Minimize,
    Maximize,
    VolumeSet,
    VolumeMute,
    VolumeUnmute,
    ClipboardCopy,
    ClipboardPaste,
    SystemQuery,
    TaskPause,
    TaskResume,
    TaskCancel,
}

/// Strips one leading filler phrase (at most one — voice input rarely stacks
/// more than one) and returns the remainder, trimmed.
fn strip_filler(text: &str) -> &str {
    let lowered_len_ok = |candidate: &str, text: &str| text.len() >= candidate.len();
    let lower = text.to_lowercase();
    for filler in LEADING_FILLER {
        if lowered_len_ok(filler, &lower) && lower.starts_with(filler) {
            let rest = text[filler.len()..].trim_start_matches(|c: char| c == ',' || c.is_whitespace());
            if !rest.is_empty() {
                return rest;
            }
        }
    }
    text
}

/// Splits into a lowercased leading word/two-word phrase (for multi-word
/// verbs like "switch to"/"go to"/"turn up") and the rest of the string.
fn split_verb(text: &str) -> (String, String, String) {
    let trimmed = text.trim();
    let mut words = trimmed.split_whitespace();
    let first = words.next().unwrap_or("").to_lowercase();
    let rest_after_first = trimmed.splitn(2, char::is_whitespace).nth(1).unwrap_or("").to_string();
    let second = words.next().unwrap_or("").to_lowercase();
    (first, second, rest_after_first)
}

fn verb_family(first: &str, second: &str) -> Option<(VerbFamily, bool)> {
    // (family, consumed_second_word)
    match first {
        "open" | "launch" | "start" | "run" => Some((VerbFamily::OpenLaunch, false)),
        "close" | "quit" | "exit" | "kill" | "stop" if second == "task" || !second.is_empty() => {
            // Bare "stop" alone is ambiguous with TaskControl's "stop that" —
            // handled separately below before this table is consulted.
            Some((VerbFamily::Close, false))
        }
        "focus" => Some((VerbFamily::Focus, false)),
        "switch" if second == "to" => Some((VerbFamily::Focus, true)),
        "go" if second == "to" => Some((VerbFamily::Focus, true)),
        "minimize" | "minimise" => Some((VerbFamily::Minimize, false)),
        "maximize" | "maximise" => Some((VerbFamily::Maximize, false)),
        "mute" => Some((VerbFamily::VolumeMute, false)),
        "unmute" => Some((VerbFamily::VolumeUnmute, false)),
        "copy" => Some((VerbFamily::ClipboardCopy, false)),
        "paste" => Some((VerbFamily::ClipboardPaste, false)),
        "pause" | "hold" => Some((VerbFamily::TaskPause, false)),
        "resume" | "continue" | "proceed" | "unpause" => Some((VerbFamily::TaskResume, false)),
        "cancel" | "abandon" => Some((VerbFamily::TaskCancel, false)),
        "check" | "what's" | "whats" | "what" | "how's" | "hows" | "tell" => Some((VerbFamily::SystemQuery, false)),
        "turn" | "raise" | "lower" | "increase" | "decrease" | "set" if !second.is_empty() => Some((VerbFamily::VolumeSet, false)),
        _ => None,
    }
}

fn classify_open_target(rest: &str) -> Capability {
    let trimmed = rest.trim().trim_end_matches(['.', '!', '?']);
    let looks_like_path = trimmed.contains('\\')
        || trimmed.contains('/')
        || trimmed.to_lowercase().starts_with("http://")
        || trimmed.to_lowercase().starts_with("https://")
        || trimmed.to_lowercase().starts_with("www.")
        || has_file_extension(trimmed);
    if looks_like_path {
        Capability::OpenPath(trimmed.to_string())
    } else {
        Capability::LaunchOrFocusApp(trimmed.to_string())
    }
}

/// Whole-word-ish check for a short alphanumeric extension at the end of the
/// last token (`report.pdf`, `notes.docx`) — deliberately narrow (a
/// trailing dot-plus-letters with no spaces after it) rather than a real MIME
/// database, since the fast router only needs "does this look like a file",
/// not full validation.
fn has_file_extension(text: &str) -> bool {
    let Some(last_word) = text.split_whitespace().last() else { return false };
    let Some(dot) = last_word.rfind('.') else { return false };
    let ext = &last_word[dot + 1..];
    !ext.is_empty() && ext.len() <= 5 && ext.chars().all(|c| c.is_ascii_alphanumeric())
}

fn classify_system_query(rest: &str) -> Option<Capability> {
    let lower = rest.to_lowercase();
    const CPU_WORDS: &[&str] = &["cpu", "processor"];
    const MEMORY_WORDS: &[&str] = &["memory", "ram"];
    const BATTERY_WORDS: &[&str] = &["battery"];
    const TIME_WORDS: &[&str] = &["time", "clock"];
    const VOLUME_WORDS: &[&str] = &["volume", "sound level"];

    if CPU_WORDS.iter().any(|w| lower.contains(w)) {
        Some(Capability::SystemInfo(SystemInfoKind::Cpu))
    } else if MEMORY_WORDS.iter().any(|w| lower.contains(w)) {
        Some(Capability::SystemInfo(SystemInfoKind::Memory))
    } else if BATTERY_WORDS.iter().any(|w| lower.contains(w)) {
        Some(Capability::SystemInfo(SystemInfoKind::Battery))
    } else if VOLUME_WORDS.iter().any(|w| lower.contains(w)) {
        Some(Capability::SystemInfo(SystemInfoKind::Volume))
    } else if TIME_WORDS.iter().any(|w| lower.contains(w)) {
        Some(Capability::SystemInfo(SystemInfoKind::Time))
    } else {
        None
    }
}

/// Extracts a trailing percent amount ("to 40%", "40 percent") for a
/// `VolumeOp::SetPercent`, if present.
fn extract_percent(text: &str) -> Option<u8> {
    let digits: String = text.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u8>().ok().map(|v| v.min(100))
}

fn classify_volume_set(rest: &str) -> Option<Capability> {
    let lower = rest.to_lowercase();
    if !lower.contains("volume") && !lower.contains("sound") {
        return None;
    }
    if lower.contains("mute") {
        return Some(Capability::VolumeOp(VolumeOp::Mute));
    }
    // Look for a percent-shaped number anywhere in the phrase.
    for word in lower.split(|c: char| !c.is_ascii_digit() && c != '%') {
        if let Some(pct) = extract_percent(word) {
            if !word.is_empty() {
                return Some(Capability::VolumeOp(VolumeOp::SetPercent(pct)));
            }
        }
    }
    if lower.contains("up") || lower.contains("higher") || lower.contains("louder") || lower.contains("increase") {
        Some(Capability::VolumeOp(VolumeOp::Up(None)))
    } else if lower.contains("down") || lower.contains("lower") || lower.contains("quieter") || lower.contains("decrease") {
        Some(Capability::VolumeOp(VolumeOp::Down(None)))
    } else {
        None
    }
}

fn non_empty(s: &str) -> Option<String> {
    let t = s.trim().trim_end_matches(['.', '!', '?']);
    if t.is_empty() { None } else { Some(t.to_string()) }
}

/// Attempts to deterministically match `text` to a `Capability`. `None`
/// means "not a fast-router case" — the caller falls through to the agent
/// loop, never an error.
pub fn try_match(text: &str) -> Option<Capability> {
    let stripped = strip_filler(text.trim());
    let lower_stripped = stripped.trim().to_lowercase();
    if lower_stripped == "stop" || lower_stripped == "stop that" || lower_stripped == "hold on" {
        return Some(Capability::TaskControl(TaskControlOp::Pause));
    }

    let (first, second, rest_after_first) = split_verb(stripped);
    if first.is_empty() {
        return None;
    }
    let (family, consumed_second) = verb_family(&first, &second)?;
    let rest = if consumed_second {
        rest_after_first.trim_start().splitn(2, char::is_whitespace).nth(1).unwrap_or("").to_string()
    } else {
        rest_after_first
    };

    match family {
        VerbFamily::OpenLaunch => Some(classify_open_target(&rest)),
        VerbFamily::Close => Some(Capability::WindowOp { op: WindowOp::Close, target: non_empty(&rest) }),
        VerbFamily::Focus => Some(Capability::WindowOp { op: WindowOp::Focus, target: non_empty(&rest) }),
        VerbFamily::Minimize => Some(Capability::WindowOp { op: WindowOp::Minimize, target: non_empty(&rest) }),
        VerbFamily::Maximize => Some(Capability::WindowOp { op: WindowOp::Maximize, target: non_empty(&rest) }),
        VerbFamily::VolumeMute => Some(Capability::VolumeOp(VolumeOp::Mute)),
        VerbFamily::VolumeUnmute => Some(Capability::VolumeOp(VolumeOp::Unmute)),
        VerbFamily::ClipboardCopy => Some(Capability::Clipboard(ClipboardOp::Read)),
        VerbFamily::ClipboardPaste => non_empty(&rest).map(|t| Capability::Clipboard(ClipboardOp::Write(t))),
        VerbFamily::SystemQuery => classify_system_query(&rest),
        VerbFamily::TaskPause => Some(Capability::TaskControl(TaskControlOp::Pause)),
        VerbFamily::TaskResume => Some(Capability::TaskControl(TaskControlOp::Resume)),
        VerbFamily::TaskCancel => Some(Capability::TaskControl(TaskControlOp::Cancel)),
        VerbFamily::VolumeSet => classify_volume_set(&format!("{first} {second} {rest}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_open_app() {
        assert_eq!(try_match("Open VS Code."), Some(Capability::LaunchOrFocusApp("VS Code".to_string())));
    }

    #[test]
    fn matches_open_app_with_politeness_filler() {
        assert_eq!(try_match("Could you please open Chrome"), Some(Capability::LaunchOrFocusApp("Chrome".to_string())));
    }

    #[test]
    fn matches_open_path_by_extension() {
        assert_eq!(try_match("open report.pdf"), Some(Capability::OpenPath("report.pdf".to_string())));
    }

    #[test]
    fn matches_open_url() {
        assert_eq!(try_match("open github.com"), Some(Capability::OpenPath("github.com".to_string())));
        assert_eq!(try_match("open https://github.com"), Some(Capability::OpenPath("https://github.com".to_string())));
    }

    #[test]
    fn matches_open_folder_path() {
        assert_eq!(try_match(r"open C:\Users\me\Downloads"), Some(Capability::OpenPath(r"C:\Users\me\Downloads".to_string())));
    }

    #[test]
    fn matches_system_info_cpu() {
        assert_eq!(try_match("What's my CPU usage?"), Some(Capability::SystemInfo(SystemInfoKind::Cpu)));
    }

    #[test]
    fn matches_system_info_battery() {
        assert_eq!(try_match("check my battery"), Some(Capability::SystemInfo(SystemInfoKind::Battery)));
    }

    #[test]
    fn matches_close_app() {
        assert_eq!(
            try_match("close Spotify"),
            Some(Capability::WindowOp { op: WindowOp::Close, target: Some("Spotify".to_string()) })
        );
    }

    #[test]
    fn bare_stop_is_task_pause_not_window_close() {
        assert_eq!(try_match("stop"), Some(Capability::TaskControl(TaskControlOp::Pause)));
        assert_eq!(try_match("stop that"), Some(Capability::TaskControl(TaskControlOp::Pause)));
    }

    #[test]
    fn matches_task_resume() {
        assert_eq!(try_match("resume"), Some(Capability::TaskControl(TaskControlOp::Resume)));
        assert_eq!(try_match("continue"), Some(Capability::TaskControl(TaskControlOp::Resume)));
    }

    #[test]
    fn matches_volume_up_and_down() {
        assert_eq!(try_match("turn the volume up"), Some(Capability::VolumeOp(VolumeOp::Up(None))));
        assert_eq!(try_match("turn the volume down"), Some(Capability::VolumeOp(VolumeOp::Down(None))));
    }

    #[test]
    fn matches_volume_mute() {
        assert_eq!(try_match("mute"), Some(Capability::VolumeOp(VolumeOp::Mute)));
    }

    #[test]
    fn matches_focus_window() {
        assert_eq!(
            try_match("switch to Chrome"),
            Some(Capability::WindowOp { op: WindowOp::Focus, target: Some("Chrome".to_string()) })
        );
    }

    #[test]
    fn open_ended_complex_request_does_not_fast_route() {
        assert_eq!(try_match("Find the authentication problem in my project and fix it."), None);
    }

    #[test]
    fn screen_understanding_request_does_not_fast_route() {
        assert_eq!(try_match("Explain what is on my screen."), None);
    }

    #[test]
    fn unrecognized_verb_falls_through() {
        assert_eq!(try_match("Tell me a joke"), None);
        assert_eq!(try_match("Why is the sky blue?"), None);
    }
}
