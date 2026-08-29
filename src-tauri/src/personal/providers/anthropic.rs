//! Direct Anthropic Messages API calls — ports
//! `apps/backend/app/services/llm/anthropic_provider.py`. Anthropic takes
//! `system` as a separate top-level field, not a message in the array.

use std::time::Duration;

use futures_util::StreamExt;

use crate::personal::agent::orchestrator::AgenticProvider;
use crate::personal::agent::tool_schema::ToolSpec;
use crate::personal::agent::types::{AgentContent, AgentEvent, AgentMessage, AgentRole, StopReason};
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
    let client = crate::http_client::shared_async_client();
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
    let client = crate::http_client::shared_async_client();
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

// ---------------------------------------------------------------------
// Agent loop support: tool-calling, streamed the same way as `stream`
// above but parsing Anthropic's block/index-based SSE shape (content_block_
// start/delta/stop, message_delta's stop_reason) instead of only text
// deltas. Kept separate from `stream`/`drain_text_deltas` above — those
// remain exactly as they were for the non-agentic call sites
// (analyze/notes_ask/notes_summarize/analyze_setup), which never need tools.
// ---------------------------------------------------------------------

/// One Anthropic client, holding the resolved API key/model — this is what
/// `personal::client::DirectLlmClient` constructs and hands to
/// `agent::orchestrator::run_agent_loop` as a `&dyn AgenticProvider`.
pub struct AnthropicAgent {
    pub api_key: String,
    pub model: String,
}

impl AgenticProvider for AnthropicAgent {
    fn stream_agentic<'a>(
        &'a self,
        messages: &'a [AgentMessage],
        tools: &'a [ToolSpec],
        cancel: &'a crate::state::CancelToken,
        on_event: &'a mut (dyn FnMut(AgentEvent) + Send),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
        Box::pin(stream_agentic(&self.api_key, &self.model, messages, tools, cancel, on_event))
    }
}

fn tool_specs_to_anthropic(tools: &[ToolSpec]) -> serde_json::Value {
    serde_json::Value::Array(
        tools.iter().map(|t| serde_json::json!({ "name": t.name, "description": t.description, "input_schema": t.parameters })).collect(),
    )
}

/// One `AgentContent` -> one Anthropic content block. `ToolResult.image` (a
/// `capture_screen` result) nests an `image` content block directly inside
/// the `tool_result` block's own `content` array — Anthropic's native way
/// to hand a tool result's image back to the model in the same turn.
fn content_to_anthropic(content: &AgentContent) -> serde_json::Value {
    match content {
        AgentContent::Text(text) => serde_json::json!({ "type": "text", "text": text }),
        AgentContent::Image { media_type, data_base64 } => {
            serde_json::json!({ "type": "image", "source": { "type": "base64", "media_type": media_type, "data": data_base64 } })
        }
        AgentContent::ToolUse { id, name, input } => serde_json::json!({ "type": "tool_use", "id": id, "name": name, "input": input }),
        AgentContent::ToolResult { tool_use_id, text, image, is_error } => {
            let mut inner = vec![serde_json::json!({ "type": "text", "text": text })];
            if let Some((media_type, data_base64)) = image {
                inner.push(serde_json::json!({ "type": "image", "source": { "type": "base64", "media_type": media_type, "data": data_base64 } }));
            }
            serde_json::json!({ "type": "tool_result", "tool_use_id": tool_use_id, "content": inner, "is_error": is_error })
        }
    }
}

fn messages_to_anthropic(messages: &[AgentMessage]) -> (Option<String>, Vec<serde_json::Value>) {
    let system_parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == AgentRole::System)
        .flat_map(|m| m.content.iter())
        .filter_map(|c| if let AgentContent::Text(t) = c { Some(t.clone()) } else { None })
        .collect();
    let turns: Vec<serde_json::Value> = messages
        .iter()
        .filter(|m| m.role != AgentRole::System)
        .map(|m| {
            let role = if m.role == AgentRole::Assistant { "assistant" } else { "user" };
            let content: Vec<serde_json::Value> = m.content.iter().map(content_to_anthropic).collect();
            serde_json::json!({ "role": role, "content": content })
        })
        .collect();
    let system = if system_parts.is_empty() { None } else { Some(system_parts.join("\n\n")) };
    (system, turns)
}

/// Accumulates one in-progress content block's tool-call state (Anthropic
/// streams a tool call's `input` JSON incrementally across several
/// `content_block_delta` frames, all sharing the same block `index`, and
/// only reports the block complete at `content_block_stop`).
#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    partial_json: String,
}

async fn stream_agentic(
    api_key: &str,
    model: &str,
    messages: &[AgentMessage],
    tools: &[ToolSpec],
    cancel: &crate::state::CancelToken,
    on_event: &mut (dyn FnMut(AgentEvent) + Send),
) -> Result<(), String> {
    let client = crate::http_client::shared_async_client();
    let (system, turns) = messages_to_anthropic(messages);
    let mut body = serde_json::json!({
        "model": resolve_model(model),
        "messages": turns,
        "temperature": 0.4,
        "max_tokens": 1024,
        "stream": true,
        "tools": tool_specs_to_anthropic(tools),
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
    let mut pending: std::collections::HashMap<u64, PendingToolCall> = std::collections::HashMap::new();
    let mut stop_reason = StopReason::EndTurn;

    while let Some(chunk) = stream.next().await {
        // Checked per network chunk, not only between orchestrator
        // iterations — see `AgenticProvider::stream_agentic`'s doc: a turn
        // cancelled mid-stream (barge-in, or a fast follow-up superseding
        // this one) must stop emitting deltas promptly, not once this whole
        // HTTP response happens to finish.
        //
        // Returns `Err` (not `Ok(())`) deliberately: an `Ok` here previously
        // made `run_agent_loop` treat a mid-stream-cancelled turn as a
        // NORMAL completed answer (no `Done` event ever fired, so
        // `stop_reason` stayed at its `EndTurn` default and whatever partial
        // text had streamed so far got spoken/returned as if it were the
        // real, final answer) — a stale turn racing a newer one instead of
        // being cleanly superseded. Matching the orchestrator's own
        // iteration-level cancellation error shape (`"cancelled"`) makes
        // this the single, consistent way any layer of the loop reports
        // "this turn was interrupted, not completed."
        if cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        let chunk = chunk.map_err(|e| format!("stream read error: {e}"))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(frame_end) = buffer.find("\n\n") {
            let frame = buffer[..frame_end].to_string();
            buffer.drain(..frame_end + 2);

            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data:") else { continue };
                let Ok(json) = serde_json::from_str::<serde_json::Value>(data.trim()) else { continue };
                let event_type = json.get("type").and_then(|t| t.as_str()).unwrap_or("");

                match event_type {
                    "content_block_start" => {
                        let index = json.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                        if let Some(block) = json.get("content_block") {
                            if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                                let id = block.get("id").and_then(|i| i.as_str()).unwrap_or_default().to_string();
                                let name = block.get("name").and_then(|n| n.as_str()).unwrap_or_default().to_string();
                                pending.insert(index, PendingToolCall { id, name, partial_json: String::new() });
                            }
                        }
                    }
                    "content_block_delta" => {
                        let index = json.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                        let Some(delta) = json.get("delta") else { continue };
                        match delta.get("type").and_then(|t| t.as_str()) {
                            Some("text_delta") => {
                                if let Some(text) = delta.get("text").and_then(|t| t.as_str()) {
                                    on_event(AgentEvent::TextDelta(text.to_string()));
                                }
                            }
                            Some("input_json_delta") => {
                                if let Some(partial) = delta.get("partial_json").and_then(|p| p.as_str()) {
                                    if let Some(entry) = pending.get_mut(&index) {
                                        entry.partial_json.push_str(partial);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                    "content_block_stop" => {
                        let index = json.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                        if let Some(call) = pending.remove(&index) {
                            let input = if call.partial_json.trim().is_empty() {
                                serde_json::json!({})
                            } else {
                                serde_json::from_str(&call.partial_json).unwrap_or(serde_json::json!({}))
                            };
                            on_event(AgentEvent::ToolCallReady { id: call.id, name: call.name, input });
                        }
                    }
                    "message_delta" => {
                        if let Some(reason) = json.get("delta").and_then(|d| d.get("stop_reason")).and_then(|r| r.as_str()) {
                            stop_reason = match reason {
                                "tool_use" => StopReason::ToolUse,
                                "max_tokens" => StopReason::MaxTokens,
                                _ => StopReason::EndTurn,
                            };
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    on_event(AgentEvent::Done { stop_reason });
    Ok(())
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
