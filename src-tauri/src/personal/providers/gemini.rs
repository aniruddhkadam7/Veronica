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

use crate::personal::agent::orchestrator::AgenticProvider;
use crate::personal::agent::tool_schema::ToolSpec;
use crate::personal::agent::types::{AgentContent, AgentEvent, AgentMessage, AgentRole, StopReason};
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
    let client = crate::http_client::shared_async_client();
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
    let client = crate::http_client::shared_async_client();
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

// ---------------------------------------------------------------------
// Agent loop support — see anthropic.rs's equivalent section doc. Gemini's
// function-calling protocol is the most different of the three: it has no
// per-call ID at all (a `functionCall` part carries only a `name` and
// `args`), and — unlike Anthropic/OpenAI's incremental argument streaming —
// each `functionCall` arrives as one complete, already-parsed JSON value in
// a single chunk, needing no accumulation across deltas.
//
// Since a turn can call the same tool more than once (no natural id to
// disambiguate), a synthetic id of the shape "<name>#<position>" is used
// internally (see `synthesize_id`/`real_tool_name`) — round-tripped through
// `AgentContent::ToolUse.id`/`ToolResult.tool_use_id` unchanged by the
// orchestrator, and unpacked back to the plain name when building the
// `functionResponse` part Gemini expects.
// ---------------------------------------------------------------------

pub struct GeminiAgent {
    pub api_key: String,
    pub model: String,
}

impl AgenticProvider for GeminiAgent {
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

fn synthesize_id(name: &str, position: usize) -> String {
    format!("{name}#{position}")
}

fn real_tool_name(id: &str) -> &str {
    id.split('#').next().unwrap_or(id)
}

fn tool_specs_to_gemini(tools: &[ToolSpec]) -> serde_json::Value {
    let declarations: Vec<serde_json::Value> =
        tools.iter().map(|t| serde_json::json!({ "name": t.name, "description": t.description, "parameters": t.parameters })).collect();
    serde_json::json!([{ "functionDeclarations": declarations }])
}

fn messages_to_gemini(messages: &[AgentMessage]) -> (Option<String>, Vec<serde_json::Value>) {
    let system_parts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == AgentRole::System)
        .flat_map(|m| m.content.iter())
        .filter_map(|c| if let AgentContent::Text(t) = c { Some(t.clone()) } else { None })
        .collect();

    let mut turns = Vec::new();
    for message in messages.iter().filter(|m| m.role != AgentRole::System) {
        let is_tool_results = message.role == AgentRole::User && message.content.iter().all(|c| matches!(c, AgentContent::ToolResult { .. }));
        if is_tool_results {
            let mut parts = Vec::new();
            for content in &message.content {
                let AgentContent::ToolResult { tool_use_id, text, image, .. } = content else { continue };
                parts.push(serde_json::json!({
                    "functionResponse": { "name": real_tool_name(tool_use_id), "response": { "result": text } }
                }));
                if let Some((media_type, data_base64)) = image {
                    parts.push(serde_json::json!({ "inlineData": { "mimeType": media_type, "data": data_base64 } }));
                }
            }
            turns.push(serde_json::json!({ "role": "user", "parts": parts }));
            continue;
        }

        let role = if message.role == AgentRole::Assistant { "model" } else { "user" };
        let parts: Vec<serde_json::Value> = message
            .content
            .iter()
            .map(|c| match c {
                AgentContent::Text(t) => serde_json::json!({ "text": t }),
                AgentContent::Image { media_type, data_base64 } => serde_json::json!({ "inlineData": { "mimeType": media_type, "data": data_base64 } }),
                AgentContent::ToolUse { name, input, .. } => serde_json::json!({ "functionCall": { "name": name, "args": input } }),
                AgentContent::ToolResult { .. } => serde_json::Value::Null, // unreachable: handled by the is_tool_results branch above
            })
            .collect();
        turns.push(serde_json::json!({ "role": role, "parts": parts }));
    }

    let system = if system_parts.is_empty() { None } else { Some(system_parts.join("\n\n")) };
    (system, turns)
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
    let (system, turns) = messages_to_gemini(messages);
    let mut config = serde_json::json!({
        "temperature": 0.4,
        "maxOutputTokens": 1024,
        "thinkingConfig": { "thinkingLevel": "MINIMAL" },
    });
    if let Some(system) = &system {
        config["systemInstruction"] = serde_json::json!({ "parts": [{ "text": system }] });
    }
    let system_instruction = config.as_object_mut().and_then(|c| c.remove("systemInstruction"));
    let mut body = serde_json::json!({ "contents": turns, "generationConfig": config, "tools": tool_specs_to_gemini(tools) });
    if let Some(system_instruction) = system_instruction {
        body["systemInstruction"] = system_instruction;
    }

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
        resolve_model(model),
        api_key
    );

    let response = client.post(&url).timeout(Duration::from_secs(REQUEST_TIMEOUT_SECS)).json(&body).send().await.map_err(|e| format!("failed to reach Gemini: {e}"))?;

    let status = response.status();
    if !status.is_success() {
        let text = response.text().await.unwrap_or_default();
        return Err(format!("Gemini returned {status}: {text}"));
    }

    let mut stream = response.bytes_stream();
    let mut buffer = String::new();
    let mut saw_tool_call = false;
    let mut finish_reason_seen: Option<String> = None;
    let mut tool_call_position = 0usize;

    while let Some(chunk) = stream.next().await {
        if cancel.is_cancelled() {
            return Ok(());
        }
        let chunk = chunk.map_err(|e| format!("stream read error: {e}"))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(frame_end) = buffer.find("\n\n") {
            let frame = buffer[..frame_end].to_string();
            buffer.drain(..frame_end + 2);

            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data:") else { continue };
                let Ok(json) = serde_json::from_str::<serde_json::Value>(data.trim()) else { continue };
                let Some(candidate) = json.get("candidates").and_then(|c| c.as_array()).and_then(|c| c.first()) else { continue };

                if let Some(reason) = candidate.get("finishReason").and_then(|r| r.as_str()) {
                    finish_reason_seen = Some(reason.to_string());
                }

                let Some(parts) = candidate.get("content").and_then(|c| c.get("parts")).and_then(|p| p.as_array()) else { continue };
                for part in parts {
                    if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                        if !text.is_empty() {
                            on_event(AgentEvent::TextDelta(text.to_string()));
                        }
                    }
                    if let Some(call) = part.get("functionCall") {
                        let name = call.get("name").and_then(|n| n.as_str()).unwrap_or_default().to_string();
                        let args = call.get("args").cloned().unwrap_or(serde_json::json!({}));
                        let id = synthesize_id(&name, tool_call_position);
                        tool_call_position += 1;
                        saw_tool_call = true;
                        on_event(AgentEvent::ToolCallReady { id, name, input: args });
                    }
                }
            }
        }
    }

    let stop_reason = if saw_tool_call {
        StopReason::ToolUse
    } else {
        match finish_reason_seen.as_deref() {
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        }
    };
    on_event(AgentEvent::Done { stop_reason });
    Ok(())
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
