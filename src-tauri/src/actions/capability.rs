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

/// Read-only window queries — always `Safe`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowQueryOp {
    /// Every visible top-level window's title (and owning process), e.g.
    /// "what windows do I have open?".
    ListOpen,
    /// The currently focused window's title.
    GetActive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemInfoKind {
    Time,
    Battery,
    Cpu,
    Memory,
    Volume,
    DiskSpace,
    Uptime,
}

/// Read-only process queries — always `Safe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessQuery {
    List,
    FindByName(String),
}

/// Terminates a running process, by pid or by name (first case-insensitive
/// match) — `Sensitive` (see `registry.rs`): recoverable/common, but
/// consequential enough to confirm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessOp {
    Kill { pid: Option<u32>, name: Option<String> },
}

/// Read-only network queries — always `Safe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkQuery {
    Status,
    PingHost { host: String },
    ListeningPorts,
}

/// Runs a command in a shell (`cmd /C`). NOT statically classified in
/// `registry.rs` — its risk is computed per-invocation from the command text
/// itself (see `registry::classify_command_risk`), since "run a command" can
/// range from `dir` (harmless) to `del /s /q C:\` (catastrophic). An
/// unrecognized command always floats to at least `Sensitive`, never `Safe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerminalOp {
    RunCommand { command: String, working_dir: Option<String> },
}

/// Schedules, cancels, or lists a future one-off action. `ScheduleOnce`'s
/// `action` reuses the SAME closed `Capability` vocabulary (composable, not
/// a special-cased workflow) — but is restricted to `Safe`-classified inner
/// capabilities only at schedule time (see `registry.rs`), since there's no
/// live turn to confirm a `Sensitive`/`Destructive` action against when it
/// fires unattended. `ScheduleOnce`/`CancelScheduled` are `Sensitive`
/// (silently scheduling/cancelling future actions is surprising);
/// `ListScheduled` is `Safe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerOp {
    ScheduleOnce { run_at_unix_ms: i64, description: String, action: Box<Capability> },
    CancelScheduled { id: String },
    ListScheduled,
}

/// Watches a path for filesystem changes, or stops/lists active watches. A
/// watch is passive — it produces `WorkingState` notes / a
/// `veronica:watch-event`, it never itself acts — but `WatchPath`/`StopWatch`
/// are still `Sensitive` (background surveillance-adjacent, cheap to
/// confirm); `ListWatches` is `Safe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatcherOp {
    WatchPath { path: String, description: String },
    StopWatch { id: String },
    ListWatches,
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

/// A file-creation/write/read operation on a NEW or user-owned path — never
/// a mutation of something that already exists in a way that could destroy
/// data (that's `StorageOp`, classified higher — see `registry.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileOp {
    CreateFolder { path: String },
    CreateFile { path: String, content: Option<String> },
    WriteFile { path: String, content: String, append: bool },
    ReadFile { path: String },
}

/// Read-only filesystem/storage queries — always `Safe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageQuery {
    ListFolder { path: String },
    SearchFiles { root: String, query: String, max_results: Option<u32> },
    LargestFiles { root: String, top_n: u8 },
    DiskUsage { drive: Option<String> },
}

/// Mutating storage operations on EXISTING paths — `DeleteFile` is
/// `Destructive`, `MoveOrRename` is `Sensitive` (see `registry.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageOp {
    DeleteFile { path: String },
    MoveOrRename { from: String, to: String },
}

/// The full capability vocabulary. See the module doc for why
/// `CaptureScreen`/`SearchKnowledgeBase` are here but never fast-routed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Capability {
    /// Focuses an already-running window matching `name` if one exists,
    /// otherwise launches it fresh — see `native::launch_or_focus_app`.
    LaunchOrFocusApp(String),
    /// Launches (or focuses) `app` AND opens `arg` with it in one call, e.g.
    /// "open VS Code and open my security project" — distinct from
    /// `OpenPath` because the target app is named explicitly rather than
    /// resolved from the path's file association.
    LaunchAppWithArg { app: String, arg: String },
    /// A file, folder, or URL — resolved by the OS's own default-handler
    /// association (`ShellExecuteW`), same as today.
    OpenPath(String),
    WindowOp { op: WindowOp, target: Option<String> },
    /// Read-only window queries — see `WindowQueryOp`. Always `Safe`.
    WindowQuery(WindowQueryOp),
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
    /// Create/write/read a file or folder — see `FileOp`. Always `Safe`.
    FileOp(FileOp),
    /// Read-only filesystem/disk queries — see `StorageQuery`. Always `Safe`.
    StorageQuery(StorageQuery),
    /// Mutating storage operations — see `StorageOp`. `Sensitive`/`Destructive`.
    StorageOp(StorageOp),
    /// Read-only process queries — see `ProcessQuery`. Always `Safe`.
    ProcessQuery(ProcessQuery),
    /// Kill a process — see `ProcessOp`. `Sensitive`.
    ProcessOp(ProcessOp),
    /// Read-only network queries — see `NetworkQuery`. Always `Safe`.
    NetworkQuery(NetworkQuery),
    /// Run a terminal command — see `TerminalOp`. Risk computed
    /// per-invocation from the command text, see `registry::classify_command_risk`.
    TerminalOp(TerminalOp),
    /// Schedule/cancel/list a future one-off action — see `SchedulerOp`.
    SchedulerOp(SchedulerOp),
    /// Watch/stop/list filesystem watches — see `WatcherOp`.
    WatcherOp(WatcherOp),
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

/// What running a capability produced — text for every native action, an
/// image for `CaptureScreen` (which `personal::agent` wraps into a vision
/// content block before handing it back to the model), or a withheld
/// execution pending user confirmation (see `registry::RiskLevel` and
/// `execute_tool`'s `confirmed` parameter).
#[derive(Debug, Clone)]
pub enum ToolOutcome {
    Text(String),
    Image { media_type: &'static str, png_bytes: Vec<u8> },
    /// Execution was withheld — `capability` is exactly what would run if
    /// re-invoked with `confirmed: true`. Never a "result": callers must not
    /// record this into `WorkingState.last_result` as if the action happened.
    NeedsConfirmation { capability: Capability, voice_prompt: String, risk: super::registry::RiskLevel },
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
            ToolOutcome::NeedsConfirmation { voice_prompt, .. } => voice_prompt.clone(),
        }
    }
}
