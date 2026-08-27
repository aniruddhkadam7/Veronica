//! Direct Gemini generateContent calls — ports
//! `apps/backend/app/services/llm/gemini_provider.py`. Two details that are
//! easy to get wrong and MUST be preserved:
//!
//! 1. System instruction is a separate config field, not a message in
//!    `contents`.
//! 2. `thinking_config.thinking_level = MINIMAL` must be set on every call —
//!    without it, this model line burns most of `max_output_tokens` on
//!    invisible reasoning tokens and returns truncated/empty answers, since
//!    Gemini's `max_output_tokens` caps thinking AND the visible answer
//!    together (unlike OpenAI/Anthropic, where `max_tokens` only limits the
//!    visible answer).

use std::time::Duration;

use futures_util::StreamExt;

use crate::personal::prompts::ChatMessage;
use super::REQUEST_TIMEOUT_SECS;

const DEFAULT_MODEL: &str = "gemini-3.6-flash";

fn resolve_model(model: &str) -> &str {
    if model.starts_with("gemini-") {
        model
    } else {
        DEFAULT_MODEL
    }
}

/// Gemini uses "model" rather than "assistant" as the non-user role name,
/// and takes the system instruction as a separate config field.
fn split_system(messages: &[ChatMessage]) -> (Option<String>, Vec<serde_json::Value>) {
    let system_parts: Vec<&str> = messages.iter().filter(|m| m.role == "system").map(|m| m.content.as_str()).collect();
    let turns: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| {
            let role = if m.role == "assistant" { "model" } else { "user" };
            serde_json::json!({"role": role, "parts": [{"text": m.content}]})
        })
        .collect();
    let system = if system_parts.is_empty() { None } else { Some(system_parts.join("\n\n")) };
    (system, turns)
}

fn build_request_body(messages: &[ChatMessage], temperature: f32, max_tokens: u32) -> serde_json::Value {
    let (system, turns) = split_system(messages);
    let mut config = serde_json::json!({
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "thinkingConfig": {"thinkingLevel": "MINIMAL"},
    });
    if let Some(system) = system {
        config["systemInstruction"] = serde_json::json!({"parts": [{"text": system}]});
    }
    serde_json::json!({"contents": turns, "generationConfig": config})
}

fn extract_generation_config(mut body: serde_json::Value) -> serde_json::Value {
    // generateContent's REST shape nests systemInstruction at the top level,
    // not inside generationConfig — split it back out here so
    // build_request_body can stay a single simple constructor for both
    // generate() and stream()'s (identical) body shape.
    if let Some(system_instruction) = body.get_mut("generationConfig").and_then(|c| c.as_object_mut()).and_then(|c| c.remove("systemInstruction")) {
        body["systemInstruction"] = system_instruction;
    }
    body
}

pub async fn generate(api_key: &str, model: &str, messages: &[ChatMessage], temperature: f32, max_tokens: u32) -> Result<String, String> {
    let client = reqwest::Client::new();
    let body = extract_generation_config(build_request_body(messages, temperature, max_tokens));
    let url = format!("https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}", resolve_model(model), api_key);

    let response = client
        .post(&url)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("failed to reach Gemini: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Gemini returned {status}: {text}"));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| format!("Gemini returned an unexpected response: {e}"))?;
    Ok(extract_text(&json))
}

pub async fn stream<F>(api_key: &str, model: &str, messages: &[ChatMessage], temperature: f32, max_tokens: u32, mut on_delta: F) -> Result<(), String>
where
    F: FnMut(&str),
{
    let client = reqwest::Client::new();
    let body = extract_generation_config(build_request_body(messages, temperature, max_tokens));
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
        resolve_model(model),
        api_key
    );

    let response = client
        .post(&url)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("failed to reach Gemini: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Gemini returned {status}: {text}"));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("stream read error: {e}"))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        for delta in drain_text_deltas(&mut buffer) {
            on_delta(&delta);
        }
    }

    Ok(())
}

fn extract_text(json: &serde_json::Value) -> String {
    json.get("candidates")
        .and_then(|c| c.as_array())
        .and_then(|c| c.first())
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.as_array())
        .map(|parts| parts.iter().filter_map(|p| p.get("text").and_then(|t| t.as_str())).collect::<String>())
        .unwrap_or_default()
}

/// Extracts every complete `data: {...}` SSE frame from `buffer` (`alt=sse`
/// query param makes Gemini emit standard SSE framing) and returns any
/// non-empty text found in each chunk, matching the Python provider's
/// `if chunk.text: yield chunk.text`.
fn drain_text_deltas(buffer: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(frame_end) = buffer.find("\n\n") {
        let frame = buffer[..frame_end].to_string();
        buffer.drain(..frame_end + 2);

        for line in frame.lines() {
            let Some(data) = line.strip_prefix("data:") else { continue };
            let data = data.trim();
            let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else { continue };
            let text = extract_text(&json);
            if !text.is_empty() {
                out.push(text);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_keeps_valid_gemini_model() {
        assert_eq!(resolve_model("gemini-3.6-flash"), "gemini-3.6-flash");
    }

    #[test]
    fn resolve_model_substitutes_default_for_non_gemini_model() {
        assert_eq!(resolve_model("gpt-4o-mini"), DEFAULT_MODEL);
    }

    #[test]
    fn split_system_maps_assistant_role_to_model() {
        let messages = vec![ChatMessage::system("sys"), ChatMessage::assistant("prior answer"), ChatMessage::user("question")];
        let (system, turns) = split_system(&messages);
        assert_eq!(system, Some("sys".to_string()));
        assert_eq!(turns[0]["role"], "model");
        assert_eq!(turns[1]["role"], "user");
    }

    #[test]
    fn request_body_always_sets_thinking_level_minimal() {
        let body = extract_generation_config(build_request_body(&[ChatMessage::user("hi")], 0.5, 100));
        assert_eq!(body["generationConfig"]["thinkingConfig"]["thinkingLevel"], "MINIMAL");
    }

    #[test]
    fn system_instruction_moved_to_top_level() {
        let body = extract_generation_config(build_request_body(&[ChatMessage::system("sys"), ChatMessage::user("hi")], 0.5, 100));
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "sys");
        assert!(body["generationConfig"].get("systemInstruction").is_none());
    }

    #[test]
    fn extracts_text_from_generate_content_response() {
        let json = serde_json::json!({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]});
        assert_eq!(extract_text(&json), "Hello");
    }

    #[test]
    fn parses_streamed_sse_deltas() {
        let mut buffer = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\n".to_string();
        let deltas = drain_text_deltas(&mut buffer);
        assert_eq!(deltas, vec!["Hi".to_string()]);
    }

    #[test]
    fn leaves_trailing_partial_frame_in_buffer() {
        let mut buffer = "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hi\"}]}}]}\n\ndata: {\"cand".to_string();
        let deltas = drain_text_deltas(&mut buffer);
        assert_eq!(deltas, vec!["Hi".to_string()]);
        assert!(buffer.contains("data: {\"cand"));
    }
}
