//! Detects a spoken/typed INTERRUPTION command ("stop", "wait", "cancel",
//! "hold on", and obvious equivalents) so it can be handled as a dedicated
//! control signal — never as a normal conversational turn.
//!
//! This is checked BEFORE a `Turn` is ever created for the utterance (see
//! `VeronicaOverlay.tsx`'s transcript listener and `veronica_widget`'s
//! equivalent): an interruption must never appear as a "YOU: stop" message
//! in the conversation, and must never produce a visible assistant reply
//! (no "(interrupted)", no "Paused."). It only ever silences TTS, cancels
//! the in-flight turn, and returns Veronica to LISTENING.
//!
//! Deliberately narrower than `actions::fast_router`'s general verb-family
//! matching: this only recognizes a short, closed set of interruption
//! phrases as a WHOLE utterance (optionally addressed to "Veronica"), so it
//! can never misfire on a sentence that merely contains one of these words
//! as part of a longer, real request (e.g. "wait, what was that tool called
//! again?" is a real question, not a bare interruption).

/// Whole-utterance interruption phrases, after stripping a leading/trailing
/// "veronica" address and surrounding punctuation. Kept short and closed —
/// each one is something a person would say ONLY to interrupt, never as the
/// start of a real question.
const INTERRUPT_PHRASES: &[&str] = &[
    "stop",
    "stop it",
    "stop that",
    "stop veronica",
    "stop please",
    "please stop",
    "wait",
    "wait wait",
    "hold on",
    "hold up",
    "cancel",
    "cancel that",
    "never mind",
    "nevermind",
    "that's enough",
    "thats enough",
    "shush",
    "quiet",
    "enough",
];

/// Strips a leading or trailing address to Veronica ("Veronica, stop" /
/// "stop, Veronica") and surrounding punctuation/whitespace, lowercased.
fn normalize(text: &str) -> String {
    let lower = text.trim().trim_matches(|c: char| c == '.' || c == '!' || c == ',').to_lowercase();
    let lower = lower.trim();
    let without_leading = lower.strip_prefix("veronica").map(|rest| rest.trim_start_matches([',', ' '])).unwrap_or(lower);
    let without_trailing = without_leading.strip_suffix("veronica").map(|rest| rest.trim_end_matches([',', ' '])).unwrap_or(without_leading);
    without_trailing.trim().trim_matches(|c: char| c == '.' || c == '!' || c == ',').trim().to_string()
}

/// True when `text`, taken as a whole utterance, is an interruption command
/// rather than a real request. Only matches short, unambiguous phrases —
/// see the module doc for why this is intentionally narrower than the fast
/// router's general verb matching.
pub fn is_interrupt(text: &str) -> bool {
    let normalized = normalize(text);
    if normalized.is_empty() {
        return false;
    }
    INTERRUPT_PHRASES.contains(&normalized.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_stop_is_an_interrupt() {
        assert!(is_interrupt("Stop."));
        assert!(is_interrupt("stop"));
    }

    #[test]
    fn addressed_variants_are_interrupts() {
        assert!(is_interrupt("Stop Veronica"));
        assert!(is_interrupt("Veronica, stop"));
        assert!(is_interrupt("Veronica stop"));
    }

    #[test]
    fn wait_and_hold_on_and_cancel_are_interrupts() {
        assert!(is_interrupt("Wait."));
        assert!(is_interrupt("Hold on."));
        assert!(is_interrupt("Cancel."));
        assert!(is_interrupt("Never mind."));
    }

    #[test]
    fn a_real_question_containing_the_word_is_not_an_interrupt() {
        assert!(!is_interrupt("Wait, what was that tool called again?"));
        assert!(!is_interrupt("Can you cancel my subscription reminder note?"));
        assert!(!is_interrupt("Stop overthinking and just tell me the answer."));
    }

    #[test]
    fn unrelated_requests_are_not_interrupts() {
        assert!(!is_interrupt("Open VS Code."));
        assert!(!is_interrupt("What's the weather like?"));
        assert!(!is_interrupt(""));
    }
}
