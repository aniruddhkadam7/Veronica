//! The Fastest-Method Router: for each `Intent`, dispatch straight to
//! whichever execution tier is actually fastest — no mouse movement, screen
//! search, or agent loop when a direct method exists. Priority order per the
//! product spec:
//!
//!   1. Native Windows API / local OS function
//!   2. Application API / CLI
//!   3. Direct local tool
//!   4. MCP tool, when appropriate
//!   5. UI/browser automation (last resort)
//!
//! Every arm below uses tier 1 today — the current safe-action vocabulary
//! (app launch, open file/folder/URL, simple system-info queries) never
//! needs anything past a native OS call. Tiers 4/5 have no arms here at all:
//! there is no MCP client and no UI-automation driver anywhere in this
//! codebase, and inventing either ahead of an `Intent` that actually needs
//! them would be speculative scaffolding nothing calls. When an intent is
//! added that genuinely can't be served natively, add its arm here calling
//! whatever tier fits — this `match` IS the router; it does not need a
//! trait or plugin abstraction to remain one.

use super::native;
use super::Intent;

pub async fn execute(intent: &Intent) -> Result<String, String> {
    match intent {
        Intent::OpenApp(name) => native::resolve_and_launch_app(name),
        Intent::OpenFile(target) => native::open_path_or_url(target),
        Intent::OpenFolder(target) => native::open_path_or_url(target),
        Intent::OpenUrl(target) => native::open_path_or_url(target),
        Intent::QuerySystemInfo(kind) => native::query_system_info(kind),
    }
}
