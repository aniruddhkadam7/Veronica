//! Classifies a spoken request (already stripped of its "Veronica, ..." wake
//! phrase by the overlay — see InterviewOverlay.tsx's `tryVeronicaAction`)
//! into one of a small, fixed set of safe actions. This is deliberately the
//! ONLY vocabulary the model can express: the JSON schema below has no slot
//! for an arbitrary command or shell string, so there is no way for a model
//! response — however it's worded — to become anything other than one of
//! these six intents. See `crate::actions` for what happens after parsing.

use super::extract_json_object;

/// The fixed intent vocabulary. `Unknown` is the model's explicit escape
/// hatch for anything it can't confidently classify — treated upstream
/// exactly like today's un-matched voice-command regex (not an error, just
/// "no action recognized").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParsedIntent {
    OpenApp(String),
    OpenFile(String),
    OpenFolder(String),
    OpenUrl(String),
    QuerySystemInfo(String),
    Unknown,
}

pub const SYSTEM_INSTRUCTIONS: &str = "You classify a single spoken user request into a fixed action vocabulary for a desktop assistant.

You must respond with valid JSON only, matching exactly the schema described in the user message. Do not include any text outside the JSON object.

Only use the six intent names given to you. Never invent a new intent name. If the request does not clearly match one of them, respond with intent \"UNKNOWN\".

Do not attempt to interpret, plan, or justify system commands, file deletions, formatting, installs, shutdowns, registry or security changes, credential access, or anything destructive — none of those have a valid intent in your vocabulary, so any such request must be classified \"UNKNOWN\".

The \"target\" field is always the plain subject of the request as the user said it (an app name, a file or folder path or name, a URL, or a system-info kind) — never a command, flag, or instruction.";

/// Builds the (system, user) prompt pair for one classification call.
/// `utterance` is the text AFTER the wake phrase has already been stripped
/// (e.g. "open my VS Code project", not "Veronica, open my VS Code project").
pub fn build_intent_prompt(utterance: &str) -> (String, String) {
    let user_prompt = format!(
        "USER REQUEST\n{utterance}\n\nRespond with a single JSON object matching exactly this schema:\n{{\n  \"intent\": \"OPEN_APP\" | \"OPEN_FILE\" | \"OPEN_FOLDER\" | \"OPEN_URL\" | \"QUERY_SYSTEM_INFO\" | \"UNKNOWN\",\n  \"target\": string\n}}\n\nOPEN_APP: target is the application's name (e.g. \"Notepad\", \"Visual Studio Code\", \"Spotify\").\nOPEN_FILE: target is the file path or name the user described.\nOPEN_FOLDER: target is the folder path or name the user described.\nOPEN_URL: target is the URL or site name the user described.\nQUERY_SYSTEM_INFO: target is one of \"time\", \"battery\", or \"volume\" — whichever the user is asking about.\nUNKNOWN: target may be an empty string."
    );
    (SYSTEM_INSTRUCTIONS.to_string(), user_prompt)
}

/// Parses the model's raw response into a `ParsedIntent`. Any failure to
/// find/parse JSON, an unrecognized `intent` value, or a missing/empty
/// `target` where one is required all fall back to `Unknown` rather than
/// propagating an error — a classification call that comes back malformed
/// should read as "I didn't understand that," never crash the request.
pub fn parse_intent(raw: &str) -> ParsedIntent {
    let Ok(json_text) = extract_json_object(raw) else {
        return ParsedIntent::Unknown;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&json_text) else {
        return ParsedIntent::Unknown;
    };

    let intent = value.get("intent").and_then(|v| v.as_str()).unwrap_or("UNKNOWN");
    let target = value
        .get("target")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    match intent {
        "OPEN_APP" if !target.is_empty() => ParsedIntent::OpenApp(target),
        "OPEN_FILE" if !target.is_empty() => ParsedIntent::OpenFile(target),
        "OPEN_FOLDER" if !target.is_empty() => ParsedIntent::OpenFolder(target),
        "OPEN_URL" if !target.is_empty() => ParsedIntent::OpenUrl(target),
        "QUERY_SYSTEM_INFO" if !target.is_empty() => ParsedIntent::QuerySystemInfo(target),
        _ => ParsedIntent::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_open_app() {
        let raw = r#"{"intent": "OPEN_APP", "target": "Notepad"}"#;
        assert_eq!(parse_intent(raw), ParsedIntent::OpenApp("Notepad".to_string()));
    }

    #[test]
    fn unknown_intent_string_falls_back_to_unknown() {
        let raw = r#"{"intent": "DELETE_FILE", "target": "C:\\important.txt"}"#;
        assert_eq!(parse_intent(raw), ParsedIntent::Unknown);
    }

    #[test]
    fn missing_target_falls_back_to_unknown() {
        let raw = r#"{"intent": "OPEN_APP", "target": ""}"#;
        assert_eq!(parse_intent(raw), ParsedIntent::Unknown);
    }

    #[test]
    fn malformed_json_falls_back_to_unknown() {
        assert_eq!(parse_intent("not json at all"), ParsedIntent::Unknown);
    }
}

