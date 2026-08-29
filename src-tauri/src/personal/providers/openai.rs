//! Direct OpenAI Chat Completions calls — ports
//! `apps/backend/app/services/llm/openai_provider.py`. OpenAI keeps the
//! system message inline in the `messages` array (no special-casing needed,
//! unlike Anthropic/Gemini).

use std::time::Duration;

use futures_util::StreamExt;

use crate::personal::agent::orchestrator::AgenticProvider;
use crate::personal::agent::tool_schema::ToolSpec;
use crate::personal::agent::types::{AgentContent, AgentEvent, AgentMessage, AgentRole, StopReason};
use crate::personal::prompts::ChatMessage;
use super::REQUEST_TIMEOUT_SECS;

const CHAT_COMPLETIONS_URL: &str = "https://api.openai.com/v1/chat/completions";

fn build_request_body(model: &str, messages: &[ChatMessage], temperature: f32, max_tokens: u32, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "messages": messages.iter().map(|m| serde_json::json!({"role": m.role, "content": m.content})).collect::<Vec<_>>(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    })
}

pub async fn generate(api_key: &str, model: &str, messages: &[ChatMessage], temperature: f32, max_tokens: u32) -> Result<String, String> {
    let client = crate::http_client::shared_async_client();
    let body = build_request_body(model, messages, temperature, max_tokens, false);

    let response = client
        .post(CHAT_COMPLETIONS_URL)
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("failed to reach OpenAI: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("OpenAI returned {status}: {text}"));
    }

    let json: serde_json::Value = response.json().await.map_err(|e| format!("OpenAI returned an unexpected response: {e}"))?;
    Ok(json
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string())
}

pub async fn stream<F>(api_key: &str, model: &str, messages: &[ChatMessage], temperature: f32, max_tokens: u32, mut on_delta: F) -> Result<(), String>
where
    F: FnMut(&str),
{
    let client = crate::http_client::shared_async_client();
    let body = build_request_body(model, messages, temperature, max_tokens, true);

    let response = client
        .post(CHAT_COMPLETIONS_URL)
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("failed to reach OpenAI: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("OpenAI returned {status}: {text}"));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("stream read error: {e}"))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        for delta in drain_sse_deltas(&mut buffer) {
            if let Some(text) = delta {
                on_delta(&text);
            }
        }
    }

    Ok(())
}

/// Extracts every complete `data: {...}` SSE line currently in `buffer`,
/// returning `Some(delta_text)` for a normal content chunk, `None` for a
/// chunk with no choices or an empty delta (both skipped, matching the
/// Python provider's `if not chunk.choices: continue` / `if delta:` guards),
/// and stopping (without emitting further items) once `data: [DONE]` is
/// seen. Leaves any trailing partial line in `buffer` for the next network
/// chunk.
fn drain_sse_deltas(buffer: &mut String) -> Vec<Option<String>> {
    let mut out = Vec::new();
    while let Some(newline_pos) = buffer.find('\n') {
        let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
        buffer.drain(..=newline_pos);

        let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            break;
        }

        let parsed: Result<serde_json::Value, _> = serde_json::from_str(data);
        match parsed {
            Ok(json) => {
                let choices = json.get("choices").and_then(|c| c.as_array());
                let Some(choices) = choices else {
                    out.push(None);
                    continue;
                };
                if choices.is_empty() {
                    out.push(None);
                    continue;
                }
                let delta_text = choices[0].get("delta").and_then(|d| d.get("content")).and_then(|c| c.as_str());
                match delta_text {
                    Some(text) if !text.is_empty() => out.push(Some(text.to_string())),
                    _ => out.push(None),
                }
            }
            Err(_) => out.push(None),
        }
    }
    out
}

// ---------------------------------------------------------------------
// Agent loop support — see anthropic.rs's equivalent section doc; same
// separation from `generate`/`stream` above applies here.
// ---------------------------------------------------------------------

pub struct OpenAiAgent {
    pub api_key: String,
    pub model: String,
}

impl AgenticProvider for OpenAiAgent {
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

fn tool_specs_to_openai(tools: &[ToolSpec]) -> serde_json::Value {
    serde_json::Value::Array(
        tools
            .iter()
            .map(|t| serde_json::json!({ "type": "function", "function": { "name": t.name, "description": t.description, "parameters": t.parameters } }))
            .collect(),
    )
}

/// Converts the shared `AgentMessage` list into OpenAI's own message array.
/// Two shapes need expansion into more than one OpenAI message each:
/// - An assistant turn with tool-use blocks becomes one message with a
///   `tool_calls` array (OpenAI's own shape for "the model asked to call
///   these").
/// - A tool-results turn becomes one `role: "tool"` message per result
///   (OpenAI has no single message carrying several tool results, unlike
///   Anthropic), and — when a result carries an image (`capture_screen`) —
///   OpenAI's `tool` role can't hold an image part directly, so a follow-up
///   `role: "user"` message with an `image_url` content part is appended
///   right after it, which the model reads as "here's what that tool call
///   returned to look at."
fn messages_to_openai(messages: &[AgentMessage]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for message in messages {
        let is_tool_results = message.role == AgentRole::User && message.content.iter().all(|c| matches!(c, AgentContent::ToolResult { .. }));
        if is_tool_results {
            for content in &message.content {
                let AgentContent::ToolResult { tool_use_id, text, image, .. } = content else { continue };
                out.push(serde_json::json!({ "role": "tool", "tool_call_id": tool_use_id, "content": text }));
                if let Some((media_type, data_base64)) = image {
                    out.push(serde_json::json!({
                        "role": "user",
                        "content": [
                            { "type": "text", "text": "(image result from the previous tool call)" },
                            { "type": "image_url", "image_url": { "url": format!("data:{media_type};base64,{data_base64}") } },
                        ],
                    }));
                }
            }
            continue;
        }

        match message.role {
            AgentRole::System => {
                let text = text_only(&message.content);
                out.push(serde_json::json!({ "role": "system", "content": text }));
            }
            AgentRole::User => {
                let text = text_only(&message.content);
                out.push(serde_json::json!({ "role": "user", "content": text }));
            }
            AgentRole::Assistant => {
                let text = text_only(&message.content);
                let tool_calls: Vec<serde_json::Value> = message
                    .content
                    .iter()
                    .filter_map(|c| match c {
                        AgentContent::ToolUse { id, name, input } => {
                            Some(serde_json::json!({ "id": id, "type": "function", "function": { "name": name, "arguments": input.to_string() } }))
                        }
                        _ => None,
                    })
                    .collect();
                let mut obj = serde_json::json!({ "role": "assistant", "content": if text.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(text) } });
                if !tool_calls.is_empty() {
                    obj["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                out.push(obj);
            }
        }
    }
    out
}

fn text_only(content: &[AgentContent]) -> String {
    content
        .iter()
        .filter_map(|c| if let AgentContent::Text(t) = c { Some(t.as_str()) } else { None })
        .collect::<Vec<_>>()
        .join("\n")
}

/// One in-progress tool call, keyed by its `index` in the `tool_calls`
/// delta array — OpenAI sends `id`/`function.name` once (typically the
/// first chunk for that index) and streams `function.arguments` as
/// incremental string fragments across further chunks at the same index.
#[derive(Default)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
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
    let body = serde_json::json!({
        "model": model,
        "messages": messages_to_openai(messages),
        "temperature": 0.4,
        "max_tokens": 1024,
        "stream": true,
        "tools": tool_specs_to_openai(tools),
    });

    let response = client
        .post(CHAT_COMPLETIONS_URL)
        .bearer_auth(api_key)
        .timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("failed to reach OpenAI: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("OpenAI returned {status}: {text}"));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut pending: std::collections::HashMap<u64, PendingToolCall> = std::collections::HashMap::new();
    let mut stop_reason = StopReason::EndTurn;

    while let Some(chunk) = stream.next().await {
        // See anthropic.rs's `stream_agentic` for why this must be `Err`,
        // not `Ok(())` — a swallowed-as-success cancellation let a stale,
        // truncated turn's partial answer get spoken/returned as if it were
        // the real final answer.
        if cancel.is_cancelled() {
            return Err("cancelled".to_string());
        }
        let chunk = chunk.map_err(|e| format!("stream read error: {e}"))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(newline_pos) = buffer.find('\n') {
            let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
            buffer.drain(..=newline_pos);

            let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else { continue };
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            if data == "[DONE]" {
                break;
            }
            let Ok(json) = serde_json::from_str::<serde_json::Value>(data) else { continue };
            let Some(choice) = json.get("choices").and_then(|c| c.as_array()).and_then(|c| c.first()) else { continue };

            if let Some(delta) = choice.get("delta") {
                if let Some(text) = delta.get("content").and_then(|c| c.as_str()) {
                    if !text.is_empty() {
                        on_event(AgentEvent::TextDelta(text.to_string()));
                    }
                }
                if let Some(tool_call_deltas) = delta.get("tool_calls").and_then(|t| t.as_array()) {
                    for tc in tool_call_deltas {
                        let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
                        let entry = pending.entry(index).or_default();
                        if let Some(id) = tc.get("id").and_then(|i| i.as_str()) {
                            entry.id = id.to_string();
                        }
                        if let Some(function) = tc.get("function") {
                            if let Some(name) = function.get("name").and_then(|n| n.as_str()) {
                                entry.name.push_str(name);
                            }
                            if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                                entry.arguments.push_str(args);
                            }
                        }
                    }
                }
            }

            if let Some(finish_reason) = choice.get("finish_reason").and_then(|f| f.as_str()) {
                stop_reason = match finish_reason {
                    "tool_calls" => StopReason::ToolUse,
                    "length" => StopReason::MaxTokens,
                    _ => StopReason::EndTurn,
                };
            }
        }
    }

    // Every accumulated tool call is only known complete once the stream
    // ends (OpenAI has no per-call "stop" event like Anthropic's
    // content_block_stop — the whole response finishing is the signal).
    for (_, call) in pending {
        let input = if call.arguments.trim().is_empty() {
            serde_json::json!({})
        } else {
            serde_json::from_str(&call.arguments).unwrap_or(serde_json::json!({}))
        };
        on_event(AgentEvent::ToolCallReady { id: call.id, name: call.name, input });
    }

    on_event(AgentEvent::Done { stop_reason });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_delta_chunk() {
        let mut buffer = "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n".to_string();
        let deltas: Vec<String> = drain_sse_deltas(&mut buffer).into_iter().flatten().collect();
        assert_eq!(deltas, vec!["Hello".to_string()]);
    }

    #[test]
    fn stops_at_done_marker() {
        let mut buffer = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: [DONE]\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ignored\"}}]}\n\n".to_string();
        let deltas: Vec<String> = drain_sse_deltas(&mut buffer).into_iter().flatten().collect();
        assert_eq!(deltas, vec!["Hi".to_string()]);
    }

    #[test]
    fn skips_chunks_with_no_choices() {
        let mut buffer = "data: {\"choices\":[]}\n\n".to_string();
        let deltas = drain_sse_deltas(&mut buffer);
        assert_eq!(deltas, vec![None]);
    }

    #[test]
    fn leaves_trailing_partial_line_in_buffer() {
        let mut buffer = "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\ndata: {\"choic".to_string();
        let _ = drain_sse_deltas(&mut buffer);
        assert!(buffer.contains("data: {\"choic"));
    }
}
