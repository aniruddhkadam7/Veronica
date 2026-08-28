//! The Action Registry + Safety Layer: every `Capability` must pass through
//! `risk_level_for_capability` before `execute_tool` will run it.
//!
//! `Capability` (see `capability.rs`) has variants ONLY for the safe read/
//! launch/adjust actions this version supports. The categories the product
//! spec requires to be blocked or confirmed — recorded below as a
//! checklist, not as code — have no corresponding `Capability` variant at
//! all, so they cannot appear here as `Blocked` matches; they simply never
//! exist as a value this function could be called with. Neither the fast
//! router (`actions::fast_router`) nor the agent loop's tool schema
//! (`personal::agent::tool_schema`) can produce anything outside this
//! vocabulary — so the safety guarantee here comes from the vocabulary
//! being closed, not from remembering to block things.
//!
//! BLOCKLIST CHECKLIST — read this before adding a new `Capability` variant.
//! None of the following exist as capabilities today; if you're adding one
//! that resembles any of these, it must be classified `RiskLevel::Blocked`
//! (or, once introduced, a future `RiskLevel::ConfirmRequired`) rather than
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

use super::capability::Capability;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskLevel {
    Safe,
    Blocked,
}

/// Every current `Capability` variant is `Safe` — none of the blocked
/// categories above have a variant to match here. Kept as a real function
/// (not a constant `true`) so adding a new `Capability` variant later forces
/// a compile error here until it's explicitly classified.
pub fn risk_level_for_capability(capability: &Capability) -> RiskLevel {
    match capability {
        Capability::LaunchOrFocusApp(_)
        | Capability::OpenPath(_)
        | Capability::WindowOp { .. }
        | Capability::SystemInfo(_)
        | Capability::VolumeOp(_)
        | Capability::Clipboard(_)
        | Capability::TaskControl(_)
        | Capability::CaptureScreen
        | Capability::SearchKnowledgeBase(_) => RiskLevel::Safe,
    }
}

/// Not reachable today — see `risk_level_for_capability`'s doc.
pub fn refusal_message_for_capability(_capability: &Capability) -> String {
    "I can't do that — it's outside what I'm allowed to run.".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::capability::{SystemInfoKind, TaskControlOp};

    #[test]
    fn every_capability_variant_is_safe() {
        assert_eq!(risk_level_for_capability(&Capability::LaunchOrFocusApp("Notepad".to_string())), RiskLevel::Safe);
        assert_eq!(risk_level_for_capability(&Capability::SystemInfo(SystemInfoKind::Cpu)), RiskLevel::Safe);
        assert_eq!(risk_level_for_capability(&Capability::TaskControl(TaskControlOp::Pause)), RiskLevel::Safe);
        assert_eq!(risk_level_for_capability(&Capability::CaptureScreen), RiskLevel::Safe);
    }
}
