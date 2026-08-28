//! Provider-agnostic message/event types for the agent loop
//! (`personal::agent::orchestrator`). Deliberately separate from
//! `personal::prompts::ChatMessage` (a plain string-content type used by
//! `analyze`/`notes_ask`/`notes_summarize`/`analyze_setup`, none of which
//! need tool calls or images) — those call sites are untouched by this
//! module.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    System,
    User,
    Assistant,
}

/// One piece of one message's content. A single `AgentMessage` can carry
/// several of these (e.g. an assistant turn with both spoken text AND a
/// tool call, or a tool-result turn with both a text summary and an image).
#[derive(Debug, Clone)]
pub enum AgentContent {
    Text(String),
    /// An image tool result — base64-encoded, with its MIME type. Only ever
    /// produced by `Capability::CaptureScreen` today.
    Image { media_type: &'static str, data_base64: String },
    /// The model's request to call one tool — `id` is echoed back in the
    /// matching `ToolResult` so multi-tool-call turns stay correlated.
    ToolUse { id: String, name: String, input: serde_json::Value },
    /// The result of running one tool call, keyed back to its `ToolUse.id`.
    /// `image` is `Some` only for `capture_screen`'s result — see each
    /// provider adapter for how it's embedded into that provider's own
    /// tool-result wire shape (they differ: Anthropic nests an image content
    /// block directly inside the tool_result block; OpenAI/Gemini need a
    /// separate image part alongside the text result — see those modules).
    ToolResult { tool_use_id: String, text: String, image: Option<(&'static str, String)>, is_error: bool },
}

#[derive(Debug, Clone)]
pub struct AgentMessage {
    pub role: AgentRole,
    pub content: Vec<AgentContent>,
}

impl AgentMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: AgentRole::System, content: vec![AgentContent::Text(text.into())] }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Self { role: AgentRole::User, content: vec![AgentContent::Text(text.into())] }
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Self { role: AgentRole::Assistant, content: vec![AgentContent::Text(text.into())] }
    }

    pub fn assistant(content: Vec<AgentContent>) -> Self {
        Self { role: AgentRole::Assistant, content }
    }

    /// One turn carrying every tool result from one round of tool calls —
    /// each provider adapter decides for itself whether that means one wire
    /// message or several (Anthropic: one `user` message with N
    /// `tool_result` blocks; OpenAI: N separate `tool` messages; Gemini: N
    /// `functionResponse` parts in one turn).
    pub fn tool_results(results: Vec<AgentContent>) -> Self {
        Self { role: AgentRole::User, content: results }
    }
}

/// One decoded event from a provider's streaming response — every provider
/// adapter (`personal::providers::{anthropic,openai,gemini}::stream_agentic`)
/// translates its own wire format into this shared shape, so
/// `orchestrator::run_agent_loop` never needs to know which provider it's
/// talking to.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    TextDelta(String),
    /// A complete tool call — provider adapters only emit this once a call's
    /// arguments are fully accumulated (Anthropic: at `content_block_stop`;
    /// OpenAI: at the end of stream, arguments assembled from incremental
    /// deltas; Gemini: as soon as the `functionCall` part arrives, since
    /// Gemini sends it as one atomic JSON value, not incremental deltas).
    ToolCallReady { id: String, name: String, input: serde_json::Value },
    Done { stop_reason: StopReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}
