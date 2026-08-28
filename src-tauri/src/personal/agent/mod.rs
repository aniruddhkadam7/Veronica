//! The agent loop: provider-agnostic tool-calling, replacing the old hidden
//! `ACTION: <NAME> | <target>` text convention (see
//! `personal::prompts::veronica`'s former ACTION-TAKING section, and
//! `actions::parse_action_line`, both removed) with real structured tool
//! calls against all three direct providers.
//!
//! - `types`: the shared `AgentMessage`/`AgentContent`/`AgentEvent` shapes
//!   every provider adapter translates its own wire format into/out of.
//! - `tool_schema`: the one fixed tool list, and the parser from a
//!   completed tool call back into `actions::Capability`.
//! - `orchestrator`: the actual UNDERSTAND -> DECIDE -> EXECUTE TOOL ->
//!   OBSERVE -> DECIDE NEXT loop, generic over any `AgenticProvider`.
//!
//! `personal::providers::{anthropic,openai,gemini}` each implement
//! `orchestrator::AgenticProvider` alongside their existing plain
//! `generate`/`stream` functions (untouched) — see those modules' new
//! `stream_agentic`/`{Anthropic,OpenAi,Gemini}Agent` additions.

pub mod orchestrator;
pub mod tool_schema;
pub mod types;

pub use orchestrator::{run_agent_loop, AgentOutcome, AgenticProvider};
pub use types::{AgentContent, AgentEvent, AgentMessage, AgentRole, StopReason};
