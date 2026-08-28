//! `Capability`: the closed tool/action vocabulary both the fast router
//! (deterministic, pre-LLM matching — see `fast_router.rs`) and the agent
//! loop's tool schema (see `personal::agent::tool_schema`) are built from.
//! One enum, two consumers — this is what "no duplicate routing logic"
//! means here: a capability is defined once, and whether it gets reached
//! deterministically or via an LLM tool call is a property of how it's
//! dispatched, not a second definition of what it is.
//!
//! `CaptureScreen` and `SearchKnowledgeBase` are deliberately excluded from
//! `fast_router`'s match arms (see that module) even though they live in
//! this same enum: both always need an LLM to reason over their result
//! (an image, a set of retrieved chunks), so routing them natively would
//! only produce raw data with nothing to interpret it — they are agent-loop
//! tools, never fast-router targets.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowOp {
    Minimize,
    Maximize,
    Close,
    Focus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemInfoKind {
    Time,
    Battery,
    Cpu,
    Memory,
    Volume,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VolumeOp {
    Up(Option<u8>),
    Down(Option<u8>),
    Mute,
    Unmute,
    SetPercent(u8),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipboardOp {
    Read,
    Write(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskControlOp {
    Pause,
    Resume,
    Cancel,
}

/// The full capability vocabulary. See the module doc for why
/// `CaptureScreen`/`SearchKnowledgeBase` are here but never fast-routed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Capability {
    /// Focuses an already-running window matching `name` if one exists,
    /// otherwise launches it fresh — see `native::launch_or_focus_app`.
    LaunchOrFocusApp(String),
    /// A file, folder, or URL — resolved by the OS's own default-handler
    /// association (`ShellExecuteW`), same as today.
    OpenPath(String),
    WindowOp { op: WindowOp, target: Option<String> },
    SystemInfo(SystemInfoKind),
    VolumeOp(VolumeOp),
    Clipboard(ClipboardOp),
    /// Pause/resume/cancel the current multi-step task (see
    /// `working_state::WorkingState`) — dispatched by the caller directly
    /// against `AppState.working_state`, not through `execute_tool`, since
    /// it mutates session state rather than calling a native OS API. Still
    /// lives in this enum so it participates in fast-router matching and
    /// tool-schema generation like every other capability.
    TaskControl(TaskControlOp),
    /// Agent-loop-only: captures the primary monitor and returns it as an
    /// image tool result for the model to describe/reason over.
    CaptureScreen,
    /// Agent-loop-only: searches the user's attached documents. Replaces
    /// the old unconditional RAG prefetch — the model calls this only when
    /// it decides the question actually needs indexed personal context.
    SearchKnowledgeBase(String),
}

impl Capability {
    /// Whether this capability is ever a fast-router target. `false` for
    /// the two agent-loop-only tools (see the module doc) and for
    /// `TaskControl` isn't included here — task control DOES fast-route,
    /// it's just dispatched specially by the caller (see the field doc
    /// above) rather than through `actions::execute_tool`.
    pub fn is_agent_only(&self) -> bool {
        matches!(self, Capability::CaptureScreen | Capability::SearchKnowledgeBase(_))
    }
}

/// What running a capability produced — text for every native action, or an
/// image for `CaptureScreen` (which `personal::agent` wraps into a vision
/// content block before handing it back to the model).
#[derive(Debug, Clone)]
pub enum ToolOutcome {
    Text(String),
    Image { media_type: &'static str, png_bytes: Vec<u8> },
}

impl ToolOutcome {
    pub fn text(s: impl Into<String>) -> Self {
        ToolOutcome::Text(s.into())
    }

    /// A short, human-readable summary for logging/working-state purposes —
    /// never the raw image bytes.
    pub fn summary(&self) -> String {
        match self {
            ToolOutcome::Text(s) => s.clone(),
            ToolOutcome::Image { png_bytes, .. } => format!("[captured screen image, {} bytes]", png_bytes.len()),
        }
    }
}
