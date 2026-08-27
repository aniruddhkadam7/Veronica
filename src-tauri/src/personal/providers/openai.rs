//! Direct OpenAI Chat Completions calls — ports
//! `apps/backend/app/services/llm/openai_provider.py`. OpenAI keeps the
//! system message inline in the `messages` array (no special-casing needed,
//! unlike Anthropic/Gemini).

use std::time::Duration;

use futures_util::StreamExt;

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
    let client = reqwest::Client::new();
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
    let client = reqwest::Client::new();
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
