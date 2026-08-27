//! Ports `apps/backend/app/services/notes_service.py` — summarizing a note
//! and answering a question with notes as optional context. No live-call
//! streaming flow (matches the reference — Notes Mode is plain request/
//! response only).

use crate::backend::{NoteContext, NoteSummary, NotesAskRequest, NotesSummaryRequest};
use super::{extract_json_object, ChatMessage};

pub const SUMMARY_SYSTEM_PROMPT: &str = "You extract structure from a personal/work note. Given the note's title and body, produce a short summary plus any tasks, decisions, and key points it contains.

Only extract what is actually present in the note — do not invent tasks or decisions that aren't there. If the note has no tasks, or no decisions, return an empty list for that field rather than inventing one.

You must respond with valid JSON only, matching exactly the schema described in the user message. Do not include any text outside the JSON object.";

pub const ASK_SYSTEM_PROMPT: &str = "You answer questions using the user's own notes as context, when relevant. The notes are supporting material, not a hard boundary: if they contain the answer, use it directly and naturally (never say \"according to your notes\" or similar); if they don't, answer from your own general knowledge instead of saying you don't have enough information.

Keep answers concise and directly useful — a few sentences, or a short list when the question calls for multiple points. No preamble.";

/// Notes ask has no length/style/humanization knobs and no per-request
/// provider override in the schema — always the same hardcoded budget.
pub const ASK_MAX_TOKENS: u32 = 250;

fn format_note(note: &NoteContext) -> String {
    let title = note.title.as_deref().unwrap_or("(untitled)");
    format!("[{title}]\n{}", note.body)
}

pub fn build_ask_messages(request: &NotesAskRequest) -> Vec<ChatMessage> {
    let mut parts = Vec::new();
    if !request.notes.is_empty() {
        let notes_block = request.notes.iter().map(format_note).collect::<Vec<_>>().join("\n\n");
        parts.push(format!("NOTES:\n{notes_block}"));
    }
    parts.push(format!("Question:\n{}", request.question));

    vec![ChatMessage::system(ASK_SYSTEM_PROMPT), ChatMessage::user(parts.join("\n\n---\n\n"))]
}

pub fn build_summary_messages(request: &NotesSummaryRequest) -> Vec<ChatMessage> {
    let title = request.title.as_deref().unwrap_or("(untitled)");
    let user_prompt = format!(
        "TITLE: {title}\n\nBODY\n{}\n\nRespond with a single JSON object matching exactly this schema:\n{{\n  \"summary\": \"string — 1-3 sentence summary\",\n  \"tasks\": [\"string\", \"...\"],\n  \"decisions\": [\"string\", \"...\"],\n  \"key_points\": [\"string\", \"...\"]\n}}",
        request.body,
    );
    vec![ChatMessage::system(SUMMARY_SYSTEM_PROMPT), ChatMessage::user(user_prompt)]
}

pub fn parse_summary(raw: &str) -> Result<NoteSummary, String> {
    let json_str = extract_json_object(raw)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
    let get_strs = |key: &str| -> Vec<String> {
        parsed.get(key).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default()
    };
    Ok(NoteSummary {
        summary: parsed.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        tasks: get_strs("tasks"),
        decisions: get_strs("decisions"),
        key_points: get_strs("key_points"),
        message: String::new(),
    })
}

/// On parse/validation OR provider failure, all list fields default to
/// empty (unlike meeting summarize, there is no client-tracked fallback
/// data to echo here).
pub fn summary_failure(error_message: &str) -> NoteSummary {
    NoteSummary { summary: String::new(), tasks: vec![], decisions: vec![], key_points: vec![], message: format!("AI summary could not be generated ({error_message}).") }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ask_messages_omit_notes_block_when_no_notes() {
        let request = NotesAskRequest { question: "What's my plan?".to_string(), notes: vec![] };
        let messages = build_ask_messages(&request);
        assert!(!messages[1].content.contains("NOTES:"));
    }

    #[test]
    fn ask_messages_format_untitled_notes() {
        let request = NotesAskRequest { question: "q".to_string(), notes: vec![NoteContext { title: None, body: "buy milk".to_string() }] };
        let messages = build_ask_messages(&request);
        assert!(messages[1].content.contains("[(untitled)]\nbuy milk"));
    }

    #[test]
    fn summary_prompt_uses_untitled_fallback() {
        let request = NotesSummaryRequest { title: None, body: "Meeting notes here".to_string() };
        let messages = build_summary_messages(&request);
        assert!(messages[1].content.contains("TITLE: (untitled)"));
    }

    #[test]
    fn parse_summary_defaults_missing_fields_to_empty() {
        let summary = parse_summary(r#"{"summary": "short"}"#).unwrap();
        assert_eq!(summary.summary, "short");
        assert!(summary.tasks.is_empty());
    }

    #[test]
    fn summary_failure_has_no_fallback_data() {
        let failure = summary_failure("boom");
        assert!(failure.tasks.is_empty());
        assert!(failure.message.contains("boom"));
    }
}
