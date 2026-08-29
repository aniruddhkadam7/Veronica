//! The tool schema: one fixed list of tools, described once, converted into
//! each provider's own tool-definition wire shape by that provider's
//! adapter (`personal::providers::{anthropic,openai,gemini}`). A model's
//! completed tool call is parsed back into a `Capability` here too — this
//! is the single place that knows how the closed `Capability` vocabulary
//! (`actions::capability`) maps onto tool names/JSON arguments, so the
//! three provider adapters and the fast router (`actions::fast_router`)
//! never have two different ideas of what a given capability is called.

use serde_json::{json, Value};

use crate::actions::{Capability, ClipboardOp, FileOp, NetworkQuery, ProcessOp, ProcessQuery, StorageOp, StorageQuery, SystemInfoKind, TerminalOp, VolumeOp, WindowOp, WindowQueryOp};
use crate::working_state::WorkingState;

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
            name: "launch_app_with_arg",
            description: "Launch (or focus) an application AND open a specific file/folder/project with it in one step, e.g. \"open VS Code and open my security project\".",
            parameters: json!({
                "type": "object",
                "properties": {
                    "app": { "type": "string", "description": "The application's name, e.g. \"VS Code\"." },
                    "arg": { "type": "string", "description": "The file/folder/project path to open with it." },
                },
                "required": ["app", "arg"],
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
            name: "window_query",
            description: "List every open window, or check which window is currently focused.",
            parameters: json!({
                "type": "object",
                "properties": { "kind": { "type": "string", "enum": ["list_open", "get_active"] } },
                "required": ["kind"],
            }),
        },
        ToolSpec {
            name: "system_info",
            description: "Read a live system value: the time, battery level, CPU usage, memory usage, current volume, disk space, or uptime.",
            parameters: json!({
                "type": "object",
                "properties": { "kind": { "type": "string", "enum": ["time", "battery", "cpu", "memory", "volume", "disk_space", "uptime"] } },
                "required": ["kind"],
            }),
        },
        ToolSpec {
            name: "process_query",
            description: "List running processes, or find one by name.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "enum": ["list", "find_by_name"] },
                    "name": { "type": "string", "description": "Required only when kind is \"find_by_name\"." },
                },
                "required": ["kind"],
            }),
        },
        ToolSpec {
            name: "kill_process",
            description: "End a running process by pid or by name. This is a sensitive action and will require the user's confirmation before it actually runs.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "pid": { "type": "integer" },
                    "name": { "type": "string" },
                },
            }),
        },
        ToolSpec {
            name: "network_query",
            description: "Check network adapter status, ping a host, or list locally listening ports.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "kind": { "type": "string", "enum": ["status", "ping", "listening_ports"] },
                    "host": { "type": "string", "description": "Required only when kind is \"ping\"." },
                },
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
        ToolSpec {
            name: "create_folder",
            description: "Create a new folder (and any missing parent folders) at the given path.",
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
            }),
        },
        ToolSpec {
            name: "create_file",
            description: "Create a new file at the given path, optionally with initial text content.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string", "description": "Optional initial text content." },
                },
                "required": ["path"],
            }),
        },
        ToolSpec {
            name: "write_file",
            description: "Write (or append) text content to an existing or new file.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Omit to write to the current file being worked on, if any." },
                    "content": { "type": "string" },
                    "append": { "type": "boolean", "description": "true to append instead of overwrite. Defaults to false." },
                },
                "required": ["content"],
            }),
        },
        ToolSpec {
            name: "read_file",
            description: "Read a text file's content.",
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Omit to read the current file being worked on, if any." } },
            }),
        },
        ToolSpec {
            name: "list_folder",
            description: "List the files and subfolders directly inside a folder.",
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Omit to list the current folder being worked on, if any." } },
            }),
        },
        ToolSpec {
            name: "search_files",
            description: "Recursively search a folder tree for files whose name contains a query substring — use this to locate a project, a file, or code by name.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "root": { "type": "string", "description": "Folder to search under, e.g. \"C:\\\\Users\\\\me\"." },
                    "query": { "type": "string", "description": "Substring to match against file names." },
                    "max_results": { "type": "integer" },
                },
                "required": ["root", "query"],
            }),
        },
        ToolSpec {
            name: "largest_files",
            description: "Find the largest files under a folder tree (e.g. a whole drive), sorted biggest first.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "root": { "type": "string", "description": "Folder or drive root to scan, e.g. \"C:\\\\\"." },
                    "top_n": { "type": "integer", "description": "How many results to return, e.g. 10." },
                },
                "required": ["root", "top_n"],
            }),
        },
        ToolSpec {
            name: "disk_usage",
            description: "Check how much space is used/free on a drive.",
            parameters: json!({
                "type": "object",
                "properties": { "drive": { "type": "string", "description": "e.g. \"C:\". Omit for the C: drive." } },
            }),
        },
        ToolSpec {
            name: "delete_file",
            description: "Delete a file or folder (sent to the Recycle Bin, not permanently erased). This is a destructive action and will require the user's confirmation before it actually runs.",
            parameters: json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "Omit to delete the current file being worked on, if any." } },
            }),
        },
        ToolSpec {
            name: "run_terminal_command",
            description: "Run a command in a terminal (cmd.exe) and return its output — use this for diagnostics, checking service/container status, or other tasks not covered by a more specific tool. Recognized destructive or unrecognized commands will require the user's confirmation before they actually run.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "working_dir": { "type": "string" },
                },
                "required": ["command"],
            }),
        },
        ToolSpec {
            name: "move_or_rename",
            description: "Move or rename a file or folder. This is a sensitive action and will require the user's confirmation before it actually runs.",
            parameters: json!({
                "type": "object",
                "properties": {
                    "from": { "type": "string", "description": "Omit to move the current file being worked on, if any." },
                    "to": { "type": "string" },
                },
                "required": ["to"],
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
        "launch_app_with_arg" => match arg("app") {
            Some(app) => format!("Opening {app}..."),
            None => "Opening that...".to_string(),
        },
        "open_path" => match arg("target") {
            Some(target) => format!("Opening {target}..."),
            None => "Opening that...".to_string(),
        },
        "window_op" => "Adjusting the window...".to_string(),
        "window_query" => "Checking your windows...".to_string(),
        "system_info" => "Checking that...".to_string(),
        "process_query" => "Checking running processes...".to_string(),
        "kill_process" => "Preparing to end that process...".to_string(),
        "network_query" => "Checking the network...".to_string(),
        "volume_op" => "Adjusting the volume...".to_string(),
        "clipboard_op" => "Working with the clipboard...".to_string(),
        "capture_screen" => "Looking at your screen...".to_string(),
        "search_knowledge_base" => "Checking your documents...".to_string(),
        "create_folder" => "Creating a folder...".to_string(),
        "create_file" => "Creating a file...".to_string(),
        "write_file" => "Writing that down...".to_string(),
        "read_file" => "Reading that file...".to_string(),
        "list_folder" => "Looking at that folder...".to_string(),
        "search_files" => "Searching for that...".to_string(),
        "largest_files" => "Checking what's taking up space...".to_string(),
        "disk_usage" => "Checking disk space...".to_string(),
        "delete_file" => "Preparing to delete that...".to_string(),
        "move_or_rename" => "Preparing to move that...".to_string(),
        "run_terminal_command" => "Running a command...".to_string(),
        _ => "Working on it...".to_string(),
    }
}

fn get_str(input: &Value, field: &str) -> Result<String, String> {
    input.get(field).and_then(|v| v.as_str()).map(|s| s.to_string()).ok_or_else(|| format!("tool call missing required field \"{field}\""))
}

fn opt_str(input: &Value, field: &str) -> Option<String> {
    input.get(field).and_then(|v| v.as_str()).map(|s| s.to_string())
}

/// Resolves a path-shaped field: the model's own argument if given, else a
/// fallback drawn from `WorkingState` (see `actions::context`'s writer side)
/// for pronoun-style follow-ups ("delete it", "read that file") where the
/// field was omitted entirely. `label` names the `WorkingState` field to use
/// ("file" or "folder").
fn resolve_path_field(input: &Value, field: &str, context: &WorkingState, label: &str) -> Result<String, String> {
    if let Some(explicit) = opt_str(input, field) {
        return Ok(explicit);
    }
    let fallback = match label {
        "file" => context.current_file.clone(),
        "folder" => context.current_folder.clone(),
        _ => None,
    };
    fallback.ok_or_else(|| format!("tool call missing required field \"{field}\" and there's no current {label} to fall back to"))
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
///
/// `context` supplies fallbacks for omitted, context-resolvable fields (see
/// `resolve_path_field`) — the concrete mechanism behind "it"/"that
/// file"/"the current folder" resolving correctly across tool calls within
/// one multi-step task.
pub fn parse_tool_call(name: &str, input: &Value, context: &WorkingState) -> Result<Capability, String> {
    match name {
        "launch_or_focus_app" => Ok(Capability::LaunchOrFocusApp(get_str(input, "name")?)),
        "launch_app_with_arg" => Ok(Capability::LaunchAppWithArg { app: get_str(input, "app")?, arg: resolve_path_field(input, "arg", context, "folder").or_else(|_| get_str(input, "arg"))? }),
        "open_path" => Ok(Capability::OpenPath(get_str(input, "target")?)),
        "window_query" => {
            let kind = match get_str(input, "kind")?.as_str() {
                "list_open" => WindowQueryOp::ListOpen,
                "get_active" => WindowQueryOp::GetActive,
                other => return Err(format!("unknown window_query.kind \"{other}\"")),
            };
            Ok(Capability::WindowQuery(kind))
        }
        "window_op" => {
            let op = match get_str(input, "op")?.as_str() {
                "minimize" => WindowOp::Minimize,
                "maximize" => WindowOp::Maximize,
                "close" => WindowOp::Close,
                "focus" => WindowOp::Focus,
                other => return Err(format!("unknown window_op.op \"{other}\"")),
            };
            let target = opt_str(input, "target");
            Ok(Capability::WindowOp { op, target })
        }
        "system_info" => {
            let kind = match get_str(input, "kind")?.as_str() {
                "time" => SystemInfoKind::Time,
                "battery" => SystemInfoKind::Battery,
                "cpu" => SystemInfoKind::Cpu,
                "memory" => SystemInfoKind::Memory,
                "volume" => SystemInfoKind::Volume,
                "disk_space" => SystemInfoKind::DiskSpace,
                "uptime" => SystemInfoKind::Uptime,
                other => return Err(format!("unknown system_info.kind \"{other}\"")),
            };
            Ok(Capability::SystemInfo(kind))
        }
        "process_query" => {
            let query = match get_str(input, "kind")?.as_str() {
                "list" => ProcessQuery::List,
                "find_by_name" => ProcessQuery::FindByName(get_str(input, "name")?),
                other => return Err(format!("unknown process_query.kind \"{other}\"")),
            };
            Ok(Capability::ProcessQuery(query))
        }
        "kill_process" => {
            let pid = input.get("pid").and_then(|v| v.as_u64()).map(|v| v as u32);
            let name = opt_str(input, "name");
            if pid.is_none() && name.is_none() {
                return Err("kill_process needs either a pid or a name".to_string());
            }
            Ok(Capability::ProcessOp(ProcessOp::Kill { pid, name }))
        }
        "network_query" => {
            let query = match get_str(input, "kind")?.as_str() {
                "status" => NetworkQuery::Status,
                "ping" => NetworkQuery::PingHost { host: get_str(input, "host")? },
                "listening_ports" => NetworkQuery::ListeningPorts,
                other => return Err(format!("unknown network_query.kind \"{other}\"")),
            };
            Ok(Capability::NetworkQuery(query))
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
        "create_folder" => Ok(Capability::FileOp(FileOp::CreateFolder { path: get_str(input, "path")? })),
        "create_file" => Ok(Capability::FileOp(FileOp::CreateFile { path: get_str(input, "path")?, content: opt_str(input, "content") })),
        "write_file" => {
            let path = resolve_path_field(input, "path", context, "file")?;
            let content = get_str(input, "content")?;
            let append = input.get("append").and_then(|v| v.as_bool()).unwrap_or(false);
            Ok(Capability::FileOp(FileOp::WriteFile { path, content, append }))
        }
        "read_file" => Ok(Capability::FileOp(FileOp::ReadFile { path: resolve_path_field(input, "path", context, "file")? })),
        "list_folder" => Ok(Capability::StorageQuery(StorageQuery::ListFolder { path: resolve_path_field(input, "path", context, "folder")? })),
        "search_files" => Ok(Capability::StorageQuery(StorageQuery::SearchFiles {
            root: get_str(input, "root")?,
            query: get_str(input, "query")?,
            max_results: input.get("max_results").and_then(|v| v.as_u64()).map(|v| v as u32),
        })),
        "largest_files" => Ok(Capability::StorageQuery(StorageQuery::LargestFiles {
            root: get_str(input, "root")?,
            top_n: input.get("top_n").and_then(|v| v.as_u64()).map(|v| v.min(100) as u8).unwrap_or(10),
        })),
        "disk_usage" => Ok(Capability::StorageQuery(StorageQuery::DiskUsage { drive: opt_str(input, "drive") })),
        "delete_file" => Ok(Capability::StorageOp(StorageOp::DeleteFile { path: resolve_path_field(input, "path", context, "file")? })),
        "move_or_rename" => Ok(Capability::StorageOp(StorageOp::MoveOrRename { from: resolve_path_field(input, "from", context, "file")?, to: get_str(input, "to")? })),
        "run_terminal_command" => {
            let working_dir = opt_str(input, "working_dir").or_else(|| context.current_folder.clone());
            Ok(Capability::TerminalOp(TerminalOp::RunCommand { command: get_str(input, "command")?, working_dir }))
        }
        other => Err(format!("unknown tool \"{other}\"")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> WorkingState {
        WorkingState::default()
    }

    #[test]
    fn every_tool_name_round_trips_through_parse_tool_call_with_minimal_valid_args() {
        let cases: Vec<(&str, Value)> = vec![
            ("launch_or_focus_app", json!({"name": "VS Code"})),
            ("launch_app_with_arg", json!({"app": "VS Code", "arg": "C:\\proj"})),
            ("open_path", json!({"target": "report.pdf"})),
            ("window_op", json!({"op": "focus", "target": "Chrome"})),
            ("window_query", json!({"kind": "list_open"})),
            ("system_info", json!({"kind": "cpu"})),
            ("volume_op", json!({"op": "set", "percent": 40})),
            ("clipboard_op", json!({"op": "read"})),
            ("capture_screen", json!({})),
            ("search_knowledge_base", json!({"query": "python experience"})),
            ("create_folder", json!({"path": "C:\\scratch"})),
            ("create_file", json!({"path": "C:\\scratch\\a.txt"})),
            ("write_file", json!({"path": "C:\\scratch\\a.txt", "content": "hi"})),
            ("read_file", json!({"path": "C:\\scratch\\a.txt"})),
            ("list_folder", json!({"path": "C:\\scratch"})),
            ("search_files", json!({"root": "C:\\", "query": "auth"})),
            ("largest_files", json!({"root": "C:\\", "top_n": 10})),
            ("disk_usage", json!({"drive": "C:"})),
            ("delete_file", json!({"path": "C:\\scratch\\a.txt"})),
            ("move_or_rename", json!({"from": "a.txt", "to": "b.txt"})),
            ("process_query", json!({"kind": "list"})),
            ("kill_process", json!({"pid": 1234})),
            ("network_query", json!({"kind": "status"})),
            ("run_terminal_command", json!({"command": "dir"})),
        ];
        for (name, input) in cases {
            assert!(parse_tool_call(name, &input, &ctx()).is_ok(), "tool {name} should parse with valid args");
        }
    }

    #[test]
    fn missing_required_field_is_an_error_not_a_panic() {
        assert!(parse_tool_call("launch_or_focus_app", &json!({}), &ctx()).is_err());
        assert!(parse_tool_call("volume_op", &json!({"op": "set"}), &ctx()).is_err()); // missing percent
    }

    #[test]
    fn unknown_tool_name_is_an_error() {
        assert!(parse_tool_call("delete_everything", &json!({}), &ctx()).is_err());
    }

    #[test]
    fn parse_tool_call_falls_back_to_working_state_for_omitted_path() {
        let mut context = WorkingState::default();
        context.note_context(None, None, Some("C:\\scratch\\notes.txt".to_string()), None);

        let capability = parse_tool_call("read_file", &json!({}), &context).unwrap();
        assert_eq!(capability, Capability::FileOp(FileOp::ReadFile { path: "C:\\scratch\\notes.txt".to_string() }));
    }

    #[test]
    fn parse_tool_call_errors_clearly_when_omitted_and_no_context_available() {
        let result = parse_tool_call("read_file", &json!({}), &WorkingState::default());
        assert!(result.is_err());
    }

    #[test]
    fn progress_message_never_leaks_raw_tool_names_or_json() {
        for (name, input) in [
            ("launch_or_focus_app", json!({"name": "VS Code"})),
            ("launch_app_with_arg", json!({"app": "VS Code", "arg": "C:\\proj"})),
            ("open_path", json!({"target": "report.pdf"})),
            ("window_op", json!({"op": "focus"})),
            ("window_query", json!({"kind": "list_open"})),
            ("system_info", json!({"kind": "cpu"})),
            ("volume_op", json!({"op": "up"})),
            ("clipboard_op", json!({"op": "read"})),
            ("capture_screen", json!({})),
            ("search_knowledge_base", json!({"query": "resume"})),
            ("create_folder", json!({"path": "C:\\scratch"})),
            ("create_file", json!({"path": "a.txt"})),
            ("write_file", json!({"content": "hi"})),
            ("read_file", json!({})),
            ("list_folder", json!({})),
            ("search_files", json!({"root": "C:\\", "query": "auth"})),
            ("largest_files", json!({"root": "C:\\", "top_n": 10})),
            ("disk_usage", json!({})),
            ("delete_file", json!({})),
            ("move_or_rename", json!({"to": "b.txt"})),
            ("process_query", json!({"kind": "list"})),
            ("kill_process", json!({"pid": 1234})),
            ("network_query", json!({"kind": "status"})),
            ("run_terminal_command", json!({"command": "dir"})),
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
            let result = parse_tool_call(tool.name, &json!({}), &ctx());
            if let Err(e) = result {
                assert!(!e.starts_with("unknown tool"), "tool {} is declared but not handled by parse_tool_call", tool.name);
            }
        }
    }

    #[test]
    fn pronoun_fallback_resolves_across_every_applicable_tool() {
        let mut context = WorkingState::default();
        context.note_context(None, None, Some("C:\\scratch\\notes.txt".to_string()), Some("C:\\scratch".to_string()));

        for (name, input) in [
            ("read_file", json!({})),
            ("write_file", json!({"content": "hi"})),
            ("delete_file", json!({})),
            ("move_or_rename", json!({"to": "b.txt"})),
            ("list_folder", json!({})),
        ] {
            assert!(parse_tool_call(name, &input, &context).is_ok(), "tool {name} should resolve its omitted path from WorkingState");
        }
    }
}
