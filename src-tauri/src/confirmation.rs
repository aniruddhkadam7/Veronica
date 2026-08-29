//! Classifies a user's reply to a pending confirmation ("Are you sure you
//! want to delete report.pdf?") as affirmative, negative, or unrelated.
//! Deliberately narrow and syntactic — the same "small fixed vocabulary,
//! not an LLM call" spirit as `actions::fast_router`, since resolving a
//! yes/no reply doesn't need one either. `None` means "the user said
//! something else entirely" — treated as a fresh, unrelated turn, not a
//! reply to the pending confirmation, so Veronica never gets stuck
//! swallowing the user's next real question forever waiting for a yes/no
//! that was never coming.

const AFFIRMATIVE: &[&str] = &[
    "yes", "yeah", "yep", "yup", "sure", "do it", "go ahead", "confirmed", "confirm", "please do", "okay do it", "ok do it", "affirmative",
    // Bare acknowledgements — requirement 8's confirmation-reply carve-out:
    // a bare "okay"/"right" answering a real pending confirmation must
    // resolve it, not silently drop it as unrelated (see run_turn's
    // pending-confirmation check, which calls this function).
    "okay", "ok", "right", "mhm", "yeah okay",
];
const NEGATIVE: &[&str] = &["no", "nope", "nah", "don't", "do not", "cancel", "cancel that", "stop", "never mind", "nevermind"];

/// `Some(true)` for an affirmative reply, `Some(false)` for a negative one,
/// `None` if the text doesn't look like either — the caller should then
/// silently drop the pending confirmation and treat the utterance as a new,
/// unrelated turn.
pub fn classify_reply(text: &str) -> Option<bool> {
    let trimmed = text.trim().trim_end_matches(['.', '!', '?']).to_lowercase();
    if trimmed.is_empty() {
        return None;
    }
    if AFFIRMATIVE.iter().any(|w| trimmed == *w || trimmed.starts_with(&format!("{w} "))) {
        return Some(true);
    }
    if NEGATIVE.iter().any(|w| trimmed == *w || trimmed.starts_with(&format!("{w} "))) {
        return Some(false);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_common_affirmative_replies() {
        for text in ["yes", "Yeah", "Do it.", "sure", "confirmed", "go ahead"] {
            assert_eq!(classify_reply(text), Some(true), "expected affirmative for {text:?}");
        }
    }

    #[test]
    fn classifies_common_negative_replies() {
        for text in ["no", "Nope.", "cancel that", "don't", "never mind"] {
            assert_eq!(classify_reply(text), Some(false), "expected negative for {text:?}");
        }
    }

    #[test]
    fn classifies_bare_okay_and_right_as_affirmative() {
        for text in ["okay", "Okay.", "ok", "right", "mhm"] {
            assert_eq!(classify_reply(text), Some(true), "expected affirmative for {text:?}");
        }
    }

    #[test]
    fn unrelated_text_is_neither() {
        assert_eq!(classify_reply("what's the weather like"), None);
        assert_eq!(classify_reply("open Chrome instead"), None);
        assert_eq!(classify_reply(""), None);
    }
}
