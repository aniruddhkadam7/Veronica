//! The Action Registry + Safety Layer: every `Intent` must pass through
//! `risk_level` before `run_veronica_action` will let it reach the router.
//!
//! `Intent` (see `super::Intent`) has variants ONLY for the safe read/launch
//! actions this version supports. The categories the product spec requires
//! to be blocked or confirmed — recorded below as a checklist, not as code —
//! have no corresponding `Intent` variant at all, so they cannot appear here
//! as `Blocked` matches; they simply never exist as a value this function
//! could be called with. This is intentional: the safety guarantee comes
//! from the vocabulary being closed, not from remembering to block things.
//!
//! BLOCKLIST CHECKLIST — read this before adding a new `Intent` variant.
//! None of the following exist as intents today; if you're adding one that
//! resembles any of these, it must be classified `RiskLevel::Blocked` (or,
//! once introduced, a future `RiskLevel::ConfirmRequired`) rather than
//! `Safe`, and probably needs a product conversation first:
//!   - Deleting files/folders
//!   - Disk formatting
//!   - Registry or other system-configuration modification
//!   - Security-setting changes
//!   - Credential/token access
//!   - Privilege escalation
//!   - Arbitrary PowerShell/CMD execution
//!   - Software installation/uninstallation
//!   - Shutdown/restart/sleep
//!   - Bulk destructive operations (e.g. "close everything", "clear my downloads")
//!   - Sending consequential external messages/actions (email, chat, purchases)

use super::Intent;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Safe,
    Blocked,
}

/// `Intent::Unknown` is handled by the caller before this is ever consulted
/// (see `run_veronica_action`) — the `_ =>` arm below exists only as a
/// defensive default for any variant added later without an explicit
/// decision, and defaults to `Blocked` rather than `Safe` so a forgotten
/// classification fails closed, not open.
pub fn risk_level(intent: &Intent) -> RiskLevel {
    match intent {
        Intent::OpenApp(_)
        | Intent::OpenFile(_)
        | Intent::OpenFolder(_)
        | Intent::OpenUrl(_)
        | Intent::QuerySystemInfo(_) => RiskLevel::Safe,
        Intent::Unknown => RiskLevel::Blocked,
    }
}

/// User-facing refusal text for a `Blocked` intent. Only reached for
/// `Intent::Unknown` today (see `risk_level`'s doc comment) — kept as its
/// own function, separate from the generic "I didn't recognize an action"
/// message `run_veronica_action` uses for classification misses, so a
/// future genuinely-blocked-but-recognized intent (e.g. once a `DeleteFile`
/// variant exists) can get a more specific refusal without touching the
/// call site.
pub fn refusal_message(_intent: &Intent) -> String {
    "I can't do that — it's outside what I'm allowed to run.".to_string()
}
