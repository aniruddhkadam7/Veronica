//! Ports `apps/backend/app/services/meeting_service.py` +
//! `prompt_builder_meeting.py` — Meeting Mode's live ask flow and
//! end-of-meeting structured summary.

use crate::backend::{MeetingAskRequest, MeetingSummary, MeetingSummaryRequest};
use super::{extract_json_object, ChatMessage};

pub const ASK_SYSTEM_PROMPT: &str = "You are a real-time meeting assistant. You are NOT a participant speaking in the meeting — you are a quiet aide the user consults silently while the meeting is happening live.

HOW TO ANSWER

The user is asking you for help with something that just came up in the meeting — a quick recap of a point, a clarifying answer, a fact from the reference material, or a suggestion for what to say or ask next. Give them something they can use or say immediately, not a lecture.

- Speak TO the user, giving them the information or wording they can use right now.
- Ground answers in whatever meeting title/participants context is given, and in the agenda/reference documents when relevant — but never mention \"the documents\" or \"retrieved context\" to the user; just use the information naturally, as if you already knew it.
- If the user asks a factual question and the background context has the answer, give the answer directly and concisely — no hedging, no \"based on the provided information\".
- If background is thin or absent, give solid general guidance for the situation instead of stalling or asking for more information — the user is in a live meeting and needs something usable immediately.
- Keep it SHORT. This is being read in real time during a live meeting: 1-4 sentences, or a short bulleted list of 2-3 points when the question calls for options. Never write an essay.
- No preamble (\"Great question!\"), no meta-commentary about being an AI assistant. Go straight to the answer.
- Plain, natural language. Markdown bullets are fine for multiple points; otherwise plain sentences.
";

pub const SUMMARY_SYSTEM_PROMPT: &str = "You are an expert meeting notes analyst. Summarize a completed meeting from its transcript and the items the user tracked live during the meeting (key points, decisions, action items).

Be concrete and grounded only in what's in the transcript/tracked items — do not invent decisions, owners, or deadlines that weren't mentioned.

You must respond with valid JSON only, matching exactly the schema described in the user message. Do not include any text outside the JSON object.";

fn style_instruction(response_style: &str) -> &'static str {
    match response_style {
        "technical" => "Lean into concrete technical/factual detail where it strengthens the point.",
        "concise" => "Be maximally direct — the shortest usable answer, no filler.",
        _ => "Sound like a calm, well-prepared colleague.",
    }
}

fn length_instruction(answer_length: &str) -> &'static str {
    match answer_length {
        "brief" => "Ceiling: 1-2 sentences or a 2-item bullet list.",
        "detailed" => "Ceiling: roughly 120 words — only use the extra room if the question genuinely needs it.",
        _ => "Ceiling: roughly 40-90 words.",
    }
}

fn humanization_instruction(humanization: &str) -> &'static str {
    match humanization {
        "conversational" => "Lean more conversational: contractions are fine, a touch of personality is fine, as if quietly messaging a colleague rather than reciting a rehearsed answer.",
        "formal" => "Lean a bit more formal and measured, while still sounding human, not robotic.",
        _ => "Sound like a real colleague talking, not a generated answer.",
    }
}

fn context_blocks(request: &MeetingAskRequest) -> Vec<String> {
    let mut blocks = Vec::new();
    if let Some(title) = &request.meeting_title {
        blocks.push(format!("Meeting: {title}"));
    }
    if let Some(participants) = &request.participants {
        blocks.push(format!("Participants: {participants}"));
    }
    if !request.retrieved_context.is_empty() {
        let chunks = request.retrieved_context.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n\n");
        blocks.push(format!("Agenda/reference background:\n{chunks}"));
    }
    blocks
}

pub fn build_ask_messages(request: &MeetingAskRequest) -> Vec<ChatMessage> {
    let mut messages = vec![ChatMessage::system(ASK_SYSTEM_PROMPT)];

    for turn in &request.conversation_history {
        messages.push(ChatMessage::user(turn.question.clone()));
        messages.push(ChatMessage::assistant(turn.answer.clone()));
    }

    let mut parts = Vec::new();
    let blocks = context_blocks(request);
    if !blocks.is_empty() {
        parts.push(format!("CONTEXT:\n{}", blocks.join("\n\n")));
    }
    parts.push(format!("The user asks:\n{}", request.question));
    parts.push(format!(
        "{} {}\n{}\nReply with only the answer — no headers, no restating the question.",
        length_instruction(&request.answer_length),
        style_instruction(&request.response_style),
        humanization_instruction(&request.humanization),
    ));

    messages.push(ChatMessage::user(parts.join("\n\n---\n\n")));
    messages
}

pub fn max_tokens(request: &MeetingAskRequest) -> u32 {
    match request.answer_length.as_str() {
        "brief" => 120,
        "detailed" => 300,
        _ => 180,
    }
}

fn bullet_list(items: &[String]) -> String {
    if items.is_empty() {
        "(none noted)".to_string()
    } else {
        items.iter().map(|i| format!("- {i}")).collect::<Vec<_>>().join("\n")
    }
}

pub fn build_summary_prompt(request: &MeetingSummaryRequest) -> (String, String) {
    let mut header = Vec::new();
    if let Some(title) = &request.meeting_title {
        header.push(format!("MEETING\n{title}"));
    }
    if let Some(participants) = &request.participants {
        header.push(format!("PARTICIPANTS\n{participants}"));
    }
    let header_block = if header.is_empty() { "(No meeting title/participants provided.)".to_string() } else { header.join("\n\n") };

    let transcript_block = {
        let lines = request.turns.iter().map(|t| format!("{}: {}", t.speaker, t.text)).collect::<Vec<_>>().join("\n");
        if lines.is_empty() { "(No transcript captured.)".to_string() } else { lines }
    };

    let tracked_block = format!(
        "Key points noted during the meeting:\n{}\n\nDecisions made during the meeting:\n{}\n\nAction items raised during the meeting:\n{}",
        bullet_list(&request.key_points),
        bullet_list(&request.decisions),
        bullet_list(&request.action_items),
    );

    let user_prompt = format!(
        "{header_block}\n\nMEETING TRANSCRIPT\n{transcript_block}\n\nTRACKED ITEMS\n{tracked_block}\n\nRespond with a single JSON object matching exactly this schema:\n{{\n  \"summary\": \"string — 2-4 sentence overview of the meeting\",\n  \"key_points\": [\"string\", \"...\"],\n  \"decisions\": [\"string\", \"...\"],\n  \"action_items\": [\"string\", \"...\"],\n  \"next_steps\": [\"string\", \"...\"]\n}}"
    );

    (SUMMARY_SYSTEM_PROMPT.to_string(), user_prompt)
}

/// If `request.turns` is empty, no LLM call is made at all — matches the
/// Python short-circuit.
pub fn empty_turns_summary() -> MeetingSummary {
    MeetingSummary {
        summary: "No conversation was captured for this meeting.".to_string(),
        key_points: vec![],
        decisions: vec![],
        action_items: vec![],
        next_steps: vec![],
        message: "No transcript turns were provided.".to_string(),
    }
}

/// Parses the raw model response into a `MeetingSummary`. On parse/
/// validation failure OR provider failure, callers fall back to
/// `tracked_items_fallback` (both Python except-branches produce the same
/// shape, so this function only needs the happy path plus a Result).
pub fn parse_summary(raw: &str) -> Result<MeetingSummary, String> {
    let json_str = extract_json_object(raw)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| e.to_string())?;
    let get_strs = |key: &str| -> Vec<String> {
        parsed.get(key).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default()
    };
    Ok(MeetingSummary {
        summary: parsed.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        key_points: get_strs("key_points"),
        decisions: get_strs("decisions"),
        action_items: get_strs("action_items"),
        next_steps: get_strs("next_steps"),
        message: String::new(),
    })
}

/// Falls back to echoing the client's own locally-tracked items rather than
/// losing them — `next_steps` stays empty (no client-tracked equivalent
/// exists for it).
pub fn tracked_items_fallback(request: &MeetingSummaryRequest, error_message: &str) -> MeetingSummary {
    MeetingSummary {
        summary: String::new(),
        key_points: request.key_points.clone(),
        decisions: request.decisions.clone(),
        action_items: request.action_items.clone(),
        next_steps: vec![],
        message: format!("AI summary could not be generated ({error_message}); showing tracked items only."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{MeetingConversationTurn, MeetingTurnIn};

    fn base_ask_request(question: &str) -> MeetingAskRequest {
        MeetingAskRequest {
            question: question.to_string(),
            conversation_history: vec![],
            retrieved_context: vec![],
            meeting_title: None,
            participants: None,
            answer_length: "default".to_string(),
            response_style: "natural".to_string(),
            humanization: "natural".to_string(),
            llm_provider: None,
        }
    }

    #[test]
    fn max_tokens_matches_fixed_lookup() {
        let mut req = base_ask_request("q");
        assert_eq!(max_tokens(&req), 180);
        req.answer_length = "brief".to_string();
        assert_eq!(max_tokens(&req), 120);
        req.answer_length = "detailed".to_string();
        assert_eq!(max_tokens(&req), 300);
    }

    #[test]
    fn ask_messages_omit_context_block_when_empty() {
        let req = base_ask_request("What did we agree on pricing?");
        let messages = build_ask_messages(&req);
        let last = messages.last().unwrap();
        assert!(!last.content.contains("CONTEXT:"));
    }

    #[test]
    fn ask_messages_replay_history_as_real_turns() {
        let mut req = base_ask_request("And the deadline?");
        req.conversation_history = vec![MeetingConversationTurn { question: "What's the budget?".to_string(), answer: "$10k".to_string() }];
        let messages = build_ask_messages(&req);
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content, "What's the budget?");
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[2].content, "$10k");
    }

    #[test]
    fn empty_turns_short_circuits_without_llm_call() {
        let summary = empty_turns_summary();
        assert_eq!(summary.summary, "No conversation was captured for this meeting.");
    }

    #[test]
    fn summary_prompt_includes_transcript_and_tracked_items() {
        let request = MeetingSummaryRequest {
            turns: vec![MeetingTurnIn { speaker: "ME".to_string(), text: "Let's ship Friday".to_string() }],
            key_points: vec!["ship Friday".to_string()],
            decisions: vec![],
            action_items: vec![],
            meeting_title: Some("Standup".to_string()),
            participants: None,
        };
        let (_sys, user) = build_summary_prompt(&request);
        assert!(user.contains("ME: Let's ship Friday"));
        assert!(user.contains("- ship Friday"));
        assert!(user.contains("(none noted)")); // decisions/action_items empty
        assert!(user.contains("Standup"));
    }

    #[test]
    fn tracked_items_fallback_echoes_client_state_with_empty_next_steps() {
        let request = MeetingSummaryRequest {
            turns: vec![],
            key_points: vec!["kp".to_string()],
            decisions: vec!["d".to_string()],
            action_items: vec!["a".to_string()],
            meeting_title: None,
            participants: None,
        };
        let fallback = tracked_items_fallback(&request, "boom");
        assert_eq!(fallback.key_points, vec!["kp".to_string()]);
        assert!(fallback.next_steps.is_empty());
        assert!(fallback.message.contains("boom"));
    }
}
