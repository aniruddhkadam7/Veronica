//! The Action Registry + Safety Layer: every `Intent` must pass through
//! `risk_level` before `execute` will let it reach the router.
//!
//! `Intent` (see `super::Intent`) has variants ONLY for the safe read/launch
//! actions this version supports. The categories the product spec requires
//! to be blocked or confirmed — recorded below as a checklist, not as code —
//! have no corresponding `Intent` variant at all, so they cannot appear here
//! as `Blocked` matches; they simply never exist as a value this function
//! could be called with. Anything the model's response doesn't cleanly map
//! to one of the six safe intents is filtered out earlier, by
//! `parse_action_line` returning `None` (treated as a normal answer, not
//! reaching this module at all) — so the safety guarantee here comes from
//! the vocabulary being closed, not from remembering to block things.
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

/// Every current `Intent` variant is `Safe` — none of the blocked
/// categories above have a variant to match here. Kept as a real function
/// (not a constant `true`) so adding a new `Intent` variant later forces a
/// compile error here until it's explicitly classified.
pub fn risk_level(intent: &Intent) -> RiskLevel {
    match intent {
        Intent::OpenApp(_)
        | Intent::OpenFile(_)
        | Intent::OpenFolder(_)
        | Intent::OpenUrl(_)
        | Intent::QuerySystemInfo(_) => RiskLevel::Safe,
    }
}

/// User-facing refusal text for a `Blocked` intent. Not reachable today
/// (see `risk_level`'s doc comment) — kept ready for when a future `Intent`
/// variant is classified `Blocked`.
pub fn refusal_message(_intent: &Intent) -> String {
    "I can't do that — it's outside what I'm allowed to run.".to_string()
}
