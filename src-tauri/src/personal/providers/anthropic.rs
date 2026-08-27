//! Direct Anthropic Messages API calls — ports
//! `apps/backend/app/services/llm/anthropic_provider.py`. Anthropic takes
//! `system` as a separate top-level field, not a message in the array.

use std::time::Duration;

use futures_util::StreamExt;

use crate::personal::prompts::ChatMessage;
use super::REQUEST_TIMEOUT_SECS;

const MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// `settings.llm_model`/`settings.ask_model` default to an OpenAI model
/// string in the reference backend — if the configured model string isn't a
/// real Claude model id, substitute Anthropic's own default rather than
/// sending an OpenAI model name to this API.
const DEFAULT_MODEL: &str = "claude-sonnet-5";

fn resolve_model(model: &str) -> &str {
    if model.starts_with("claude-") {
        model
    } else {
        DEFAULT_MODEL
    }
}

/// Pulls system-role messages out of the list and joins them — Anthropic's
/// API has no "system" role inside `messages`.
fn split_system(messages: &[ChatMessage]) -> (Option<String>, Vec<serde_json::Value>) {
    let system_parts: Vec<&str> = messages.iter().filter(|m| m.role == "system").map(|m| m.content.as_str()).collect();
    let turns: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| m.role != "system")
        .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
        .collect();
    let system = if system_parts.is_empty() { None } else { Some(system_parts.join("\n\n")) };
    (system, turns)
}

pub async fn generate(api_key: &str, model: &str, messages: &[ChatMessage], temperature: f32, max_tokens: u32) -> Result<String, String> {
    let client = reqwest::Client::new();
    let (system, turns) = split_system(messages);
    let mut body = serde_json::json!({
        "model": resolve_model(model),
        "messages": turns,
        "temperature": temperature,
        "max_tokens": max_tokens,
    });
    if let Some(system) = system {
        body["system"] = serde_json::Value::String(system);
    }

    let response = client
        .post(MESSAGES_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("failed to reach Anthropic: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Anthropic returned {status}: {text}"));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| format!("Anthropic returned an unexpected response: {e}"))?;
    let content = json.get("content").and_then(|c| c.as_array()).cloned().unwrap_or_default();
    let text = content
        .iter()
        .filter(|block| block.get("type").and_then(|t| t.as_str()) == Some("text"))
        .filter_map(|block| block.get("text").and_then(|t| t.as_str()))
        .collect::<String>();
    Ok(text)
}

pub async fn stream<F>(api_key: &str, model: &str, messages: &[ChatMessage], temperature: f32, max_tokens: u32, mut on_delta: F) -> Result<(), String>
where
    F: FnMut(&str),
{
    let client = reqwest::Client::new();
    let (system, turns) = split_system(messages);
    let mut body = serde_json::json!({
        "model": resolve_model(model),
        "messages": turns,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": true,
    });
    if let Some(system) = system {
        body["system"] = serde_json::Value::String(system);
    }

    let response = client
        .post(MESSAGES_URL)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("failed to reach Anthropic: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Anthropic returned {status}: {text}"));
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

/// Anthropic's stream sends `event: content_block_delta` frames with
/// `data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}`
/// among various other event types (message_start, content_block_start,
/// ping, message_delta, message_stop) which this only needs to skip, not
/// specially handle. Frames are separated by a blank line, same SSE framing
/// as the other two providers.
fn drain_text_deltas(buffer: &mut String) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(frame_end) = buffer.find("\n\n") {
        let frame = buffer[..frame_end].to_string();
        buffer.drain(..frame_end + 2);

        for line in frame.lines() {
            let Some(data) = line.strip_prefix("data:") else { continue };
            let data = data.trim();
            let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else { continue };
            if json.get("type").and_then(|t| t.as_str()) != Some("content_block_delta") {
                continue;
            }
            if let Some(text) = json.get("delta").and_then(|d| d.get("text")).and_then(|t| t.as_str()) {
                out.push(text.to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_model_keeps_valid_claude_model() {
        assert_eq!(resolve_model("claude-sonnet-5"), "claude-sonnet-5");
    }

    #[test]
    fn resolve_model_substitutes_default_for_non_claude_model() {
        assert_eq!(resolve_model("gpt-4o-mini"), DEFAULT_MODEL);
    }

    #[test]
    fn split_system_pulls_out_and_joins_system_messages() {
        let messages = vec![ChatMessage::system("sys1"), ChatMessage::user("hello")];
        let (system, turns) = split_system(&messages);
        assert_eq!(system, Some("sys1".to_string()));
        assert_eq!(turns.len(), 1);
    }

    #[test]
    fn parses_content_block_delta_and_skips_other_events() {
        let mut buffer = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n".to_string();
        let deltas = drain_text_deltas(&mut buffer);
        assert_eq!(deltas, vec!["Hello".to_string()]);
    }

    #[test]
    fn leaves_trailing_partial_frame_in_buffer() {
        let mut buffer = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\nevent: content_block_delta\ndata: {\"type\"".to_string();
        let deltas = drain_text_deltas(&mut buffer);
        assert_eq!(deltas, vec!["Hi".to_string()]);
        assert!(buffer.contains("content_block_delta"));
    }
}
