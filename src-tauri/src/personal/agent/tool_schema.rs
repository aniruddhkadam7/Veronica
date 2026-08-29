//! The tool schema: one fixed list of tools, described once, converted into
//! each provider's own tool-definition wire shape by that provider's
//! adapter (`personal::providers::{anthropic,openai,gemini}`). A model's
//! completed tool call is parsed back into a `Capability` here too — this
//! is the single place that knows how the closed `Capability` vocabulary
//! (`actions::capability`) maps onto tool names/JSON arguments, so the
//! three provider adapters and the fast router (`actions::fast_router`)
//! never have two different ideas of what a given capability is called.

use serde_json::{json, Value};

use crate::actions::{Capability, ClipboardOp, SystemInfoKind, VolumeOp, WindowOp};

pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
    /// A JSON Schema `object` describing the tool's arguments — the same
    /// shape every provider expects (Anthropic's `input_schema`, OpenAI's
    /// `function.parameters`, Gemini's `functionDeclarations[].parameters`),
    /// so each adapter just embeds this value under its own field name
    /// rather than building it three different ways.
    pub parameters: Value,
}

/// Every tool the agent loop can call. `TaskControl` is deliberately
/// excluded — see `actions::capability`'s doc: it only ever reaches the
/// fast router, dispatched directly against `AppState.working_state`, never
/// through a tool call an LLM decides to make.
pub fn all_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "launch_or_focus_app",
            description: "Launch an application by name, or bring it to the foreground if it's already running.",
            parameters: json!({
                "type": "object",
                "properties": { "name": { "type": "string", "description": "The application's name, e.g. \"VS Code\", \"Chrome\"." } },
                "required": ["name"],
            }),
        },
        ToolSpec {
            name: "open_path",
            description: "Open a file, folder, or URL with its OS-default handler.",
            parameters: json!({
                "type": "object",
                "properties": { "target": { "type": "string", "description": "A file path, folder path, or URL." } },
                "required": ["target"],
            }),
        },
        ToolSpec {
            name: "window_op",
            description: "Minimize, maximize, close, or focus a window.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "op": { "type": "string", "enum": ["minimize", "maximize", "close", "focus"] },
                    "target": { "type": "string", "description": "Window title substring. Omit to act on the currently focused window." },
                },
                "required": ["op"],
            }),
        },
        ToolSpec {
            name: "system_info",
            description: "Read a live system value: the time, battery level, CPU usage, memory usage, or current volume.",
            parameters: json!({
                "type": "object",
                "properties": { "kind": { "type": "string", "enum": ["time", "battery", "cpu", "memory", "volume"] } },
                "required": ["kind"],
            }),
        },
        ToolSpec {
            name: "volume_op",
            description: "Adjust the system output volume.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "op": { "type": "string", "enum": ["up", "down", "mute", "unmute", "set"] },
                    "percent": { "type": "integer", "description": "Required only when op is \"set\" — target volume 0-100." },
                },
                "required": ["op"],
            }),
        },
        ToolSpec {
            name: "clipboard_op",
            description: "Read the current clipboard text, or write new text to the clipboard.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "op": { "type": "string", "enum": ["read", "write"] },
                    "text": { "type": "string", "description": "Required only when op is \"write\"." },
                },
                "required": ["op"],
            }),
        },
        ToolSpec {
            name: "capture_screen",
            description: "Capture a screenshot of the primary monitor to see what's currently displayed, so you can describe or reason about it.",
            parameters: json!({ "type": "object", "properties": {} }),
        },
        ToolSpec {
            name: "search_knowledge_base",
            description: "Search the user's attached/uploaded documents (resume, notes, project files) for context relevant to a question. Only call this when the question genuinely needs the user's own personal/uploaded information — not for general knowledge questions or system commands.",
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"],
            }),
        },
    ]
}

/// A short, human-readable "what Veronica is doing right now" line for one
/// tool call — shown to the user as a natural progress update (requirement
/// 11: "show high-level progress, not chain-of-thought"), never the raw
/// tool name or its JSON arguments. Deliberately present-tense and terse,
/// matching the style of the example in the spec ("Checking the
/// authentication code...", "Running tests...").
pub fn progress_message(name: &str, input: &Value) -> String {
    let arg = |field: &str| input.get(field).and_then(|v| v.as_str()).map(|s| s.to_string());
    match name {
        "launch_or_focus_app" => match arg("name") {
            Some(app) => format!("Opening {app}..."),
            None => "Opening that...".to_string(),
        },
        "open_path" => match arg("target") {
            Some(target) => format!("Opening {target}..."),
            None => "Opening that...".to_string(),
        },
        "window_op" => "Adjusting the window...".to_string(),
        "system_info" => "Checking that...".to_string(),
        "volume_op" => "Adjusting the volume...".to_string(),
        "clipboard_op" => "Working with the clipboard...".to_string(),
        "capture_screen" => "Looking at your screen...".to_string(),
        "search_knowledge_base" => "Checking your documents...".to_string(),
        _ => "Working on it...".to_string(),
    }
}

fn get_str(input: &Value, field: &str) -> Result<String, String> {
    input.get(field).and_then(|v| v.as_str()).map(|s| s.to_string()).ok_or_else(|| format!("tool call missing required field \"{field}\""))
}

/// Turns one completed tool call (`name` + its JSON `input`, as reported by
/// `AgentEvent::ToolCallReady`) into the `Capability` `actions::execute_tool`
/// knows how to run. `Err` for a malformed call (missing/wrong-typed field,
/// or a tool name outside the fixed list from `all_tools()` — which a
/// well-behaved provider should never send, but a malformed/hallucinated
/// call is handled as a normal tool-result error, not a panic) — the
/// orchestrator reports that back to the model as a `ToolResult` with
/// `is_error: true` so it can recover (e.g. retry with corrected arguments)
/// instead of the whole turn failing.
pub fn parse_tool_call(name: &str, input: &Value) -> Result<Capability, String> {
    match name {
        "launch_or_focus_app" => Ok(Capability::LaunchOrFocusApp(get_str(input, "name")?)),
        "open_path" => Ok(Capability::OpenPath(get_str(input, "target")?)),
        "window_op" => {
            let op = match get_str(input, "op")?.as_str() {
                "minimize" => WindowOp::Minimize,
                "maximize" => WindowOp::Maximize,
                "close" => WindowOp::Close,
                "focus" => WindowOp::Focus,
                other => return Err(format!("unknown window_op.op \"{other}\"")),
            };
            let target = input.get("target").and_then(|v| v.as_str()).map(|s| s.to_string());
            Ok(Capability::WindowOp { op, target })
        }
        "system_info" => {
            let kind = match get_str(input, "kind")?.as_str() {
                "time" => SystemInfoKind::Time,
                "battery" => SystemInfoKind::Battery,
                "cpu" => SystemInfoKind::Cpu,
                "memory" => SystemInfoKind::Memory,
                "volume" => SystemInfoKind::Volume,
                other => return Err(format!("unknown system_info.kind \"{other}\"")),
            };
            Ok(Capability::SystemInfo(kind))
        }
        "volume_op" => {
            let percent = input.get("percent").and_then(|v| v.as_u64()).map(|v| v.min(100) as u8);
            let op = match get_str(input, "op")?.as_str() {
                "up" => VolumeOp::Up(None),
                "down" => VolumeOp::Down(None),
                "mute" => VolumeOp::Mute,
                "unmute" => VolumeOp::Unmute,
                "set" => VolumeOp::SetPercent(percent.ok_or("volume_op op \"set\" requires \"percent\"")?),
                other => return Err(format!("unknown volume_op.op \"{other}\"")),
            };
            Ok(Capability::VolumeOp(op))
        }
        "clipboard_op" => {
            let op = match get_str(input, "op")?.as_str() {
                "read" => ClipboardOp::Read,
                "write" => ClipboardOp::Write(get_str(input, "text")?),
                other => return Err(format!("unknown clipboard_op.op \"{other}\"")),
            };
            Ok(Capability::Clipboard(op))
        }
        "capture_screen" => Ok(Capability::CaptureScreen),
        "search_knowledge_base" => Ok(Capability::SearchKnowledgeBase(get_str(input, "query")?)),
        other => Err(format!("unknown tool \"{other}\"")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_name_round_trips_through_parse_tool_call_with_minimal_valid_args() {
        let cases: Vec<(&str, Value)> = vec![
            ("launch_or_focus_app", json!({"name": "VS Code"})),
            ("open_path", json!({"target": "report.pdf"})),
            ("window_op", json!({"op": "focus", "target": "Chrome"})),
            ("system_info", json!({"kind": "cpu"})),
            ("volume_op", json!({"op": "set", "percent": 40})),
            ("clipboard_op", json!({"op": "read"})),
            ("capture_screen", json!({})),
            ("search_knowledge_base", json!({"query": "python experience"})),
        ];
        for (name, input) in cases {
            assert!(parse_tool_call(name, &input).is_ok(), "tool {name} should parse with valid args");
        }
    }

    #[test]
    fn missing_required_field_is_an_error_not_a_panic() {
        assert!(parse_tool_call("launch_or_focus_app", &json!({})).is_err());
        assert!(parse_tool_call("volume_op", &json!({"op": "set"})).is_err()); // missing percent
    }

    #[test]
    fn unknown_tool_name_is_an_error() {
        assert!(parse_tool_call("delete_everything", &json!({})).is_err());
    }

    #[test]
    fn progress_message_never_leaks_raw_tool_names_or_json() {
        for (name, input) in [
            ("launch_or_focus_app", json!({"name": "VS Code"})),
            ("open_path", json!({"target": "report.pdf"})),
            ("window_op", json!({"op": "focus"})),
            ("system_info", json!({"kind": "cpu"})),
            ("volume_op", json!({"op": "up"})),
            ("clipboard_op", json!({"op": "read"})),
            ("capture_screen", json!({})),
            ("search_knowledge_base", json!({"query": "resume"})),
            ("some_unknown_tool", json!({"weird": "shape"})),
        ] {
            let msg = progress_message(name, &input);
            assert!(!msg.contains('{'), "progress message for {name} must not contain raw JSON: {msg:?}");
            assert!(!msg.contains(name) || name == "launch_or_focus_app", "progress message for {name} should not echo the raw tool name: {msg:?}");
        }
    }

    #[test]
    fn all_tools_have_unique_names_matching_the_parser() {
        let tools = all_tools();
        let mut names: Vec<&str> = tools.iter().map(|t| t.name).collect();
        let original_len = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), original_len, "tool names must be unique");
        for tool in &tools {
            // Every declared tool must be parseable with an empty object at
            // least attempted (some will Err on missing fields, which is
            // fine — this only guards against a genuinely unknown name).
            let result = parse_tool_call(tool.name, &json!({}));
            if let Err(e) = result {
                assert!(!e.starts_with("unknown tool"), "tool {} is declared but not handled by parse_tool_call", tool.name);
            }
        }
    }
}
