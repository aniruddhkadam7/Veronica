//! English-only language policy: Veronica supports ONLY English. Enforcement
//! happens here, on the final STT transcript, BEFORE the fast router or the
//! LLM ever see it (see `veronica::run_turn`) — an utterance classified as
//! anything else is rejected immediately, with no LLM request made at all.
//!
//! Groq's Whisper transcription endpoint (`stt::groq::transcribe`) has no
//! "restrict to this language" option that also *refuses* other languages —
//! setting the `language` hint only biases recognition, it doesn't stop
//! Whisper from transcribing something else it hears. This module is
//! therefore where the actual enforcement lives: Whisper can still
//! transcribe speech in any language it recognizes, but nothing other than
//! English ever reaches the LLM or gets treated as a real command.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    English,
}

impl Language {
    pub fn code(self) -> &'static str {
        match self {
            Language::English => "en",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Supported(Language),
    /// Confident, positive evidence of a non-English language: a
    /// non-Latin/non-Devanagari-excluded script, or a real function-word
    /// marker from a specific other language. This is a genuine
    /// language-policy rejection — the user really did speak (or write)
    /// something other than English.
    Unsupported,
    /// No positive evidence either way: no English marker, no other-language
    /// marker, no other-script character. This is almost always garbled/
    /// misheard STT output (a real English utterance the transcriber
    /// mangled), NOT a real foreign-language request — see requirement 8:
    /// never claim a language violation for this case, and never guess at
    /// an invented intent from it either. Handled as an STT-quality
    /// clarification instead of the language-policy refusal.
    LowConfidence,
}

impl Decision {
    /// Wire-friendly label for the `veronica:language-detected` event and
    /// `[LANGUAGE]` log lines.
    pub fn code(self) -> &'static str {
        match self {
            Decision::Supported(lang) => lang.code(),
            Decision::Unsupported => "unsupported",
            Decision::LowConfidence => "low_confidence",
        }
    }
}

/// The fixed refusal line for a CONFIDENT non-English detection — see
/// `veronica::prompts::SYSTEM_PROMPT`'s language-policy section for the
/// matching instruction told to the model itself (defense in depth, not the
/// primary enforcement — this function's caller never lets an unsupported
/// utterance reach the LLM at all, so the model is never actually in a
/// position to violate this itself for a REJECTED utterance).
pub fn rejection_message() -> &'static str {
    "I currently support English only."
}

/// The line for `Decision::LowConfidence` — garbled/empty/unintelligible
/// audio, not a language violation. Deliberately does NOT mention language
/// support at all (requirement 8: don't repeat the language-policy message
/// unless the input actually requires it) and deliberately does not attempt
/// to answer anything, since inventing an answer from a low-confidence
/// transcript would be hallucinating the user's request.
pub fn clarification_message() -> &'static str {
    "I couldn't hear you clearly. Please say that again."
}

/// A script that is not Latin (incl. accented Latin-1, for French/Spanish/
/// German/etc. — still rejected below via the word-list check, since it's
/// visually Latin script but not English) is a high-confidence "not
/// English" signal on its own: Devanagari, Tamil, Arabic, CJK, Cyrillic,
/// and so on.
fn has_other_script(text: &str) -> bool {
    text.chars().any(|c| {
        if c.is_ascii() || c.is_whitespace() {
            return false;
        }
        // Latin-1 Supplement letters (à-ÿ, À-Ÿ) — French/Spanish/German/
        // Italian/Portuguese accented characters. Still "not English," but
        // handled via the word-list check below rather than this blunt
        // script test, since they're visually still Latin script and the
        // word list gives a more specific signal.
        if ('à'..='ÿ').contains(&c) || ('À'..='Ÿ').contains(&c) {
            return false;
        }
        true
    })
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase().split(|c: char| !c.is_alphanumeric() && c != '\'').filter(|w| !w.is_empty()).map(str::to_string).collect()
}

fn count_hits(words: &[String], markers: &[&str]) -> usize {
    words.iter().filter(|w| markers.contains(&w.as_str())).count()
}

/// Common greeting/question words from non-English languages a real English
/// utterance would essentially never contain — deliberately not exhaustive,
/// just enough to confidently reject the obvious cases (including
/// Romanized Hindi/Marathi, now unsupported) rather than let them silently
/// default to English.
const OTHER_LANGUAGE_MARKERS: &[&str] = &[
    // French
    "bonjour", "comment", "allez", "merci", "vous", "salut", "bonsoir", "revoir", "s'il",
    // Spanish
    "hola", "como", "estas", "gracias", "favor", "buenos", "dias", "buenas", "adios", "que",
    // German
    "wie", "geht", "dir", "danke", "bitte", "guten", "tag", "hallo", "tschuss", "nein",
    // Italian/Portuguese overlap
    "ciao", "grazie", "obrigado", "bom", "dia",
    // Romanized Hindi
    "hai", "hain", "kya", "kyun", "kaise", "kaisa", "kahan", "tum", "tumhara", "aap", "mujhe",
    "mera", "meri", "tera", "yeh", "voh", "woh", "nahi", "nahin", "haan", "acha", "accha",
    "theek", "matlab", "karo", "karna", "batao", "chahiye", "hona", "kitna", "kitni", "kitne",
    "bhai", "yaar", "abhi", "lekin", "baare",
    // Romanized Marathi
    "ahes", "ahe", "aahe", "kay", "kaay", "kasa", "kashi", "tula", "mala", "amhi", "tumhi",
    "pahije", "baddal", "sang", "kiti", "kuthe", "hoy",
];

const ENGLISH_MARKERS: &[&str] = &[
    "the", "is", "are", "you", "how", "what", "when", "where", "why", "who", "can", "do", "does", "i", "my", "your", "please",
    "hello", "hi", "thanks", "thank", "help", "want", "need", "would", "could", "should", "this", "that", "with", "and", "for", "open",
    "close", "check",
];

fn looks_like_plain_english(words: &[String]) -> bool {
    !words.is_empty() && count_hits(words, ENGLISH_MARKERS) > 0
}

/// Classifies `text` (the final STT transcript for one utterance) as
/// English, confidently Unsupported, or LowConfidence (no signal either
/// way — see that variant's doc). A pure text heuristic — no network call,
/// no ML model — deliberately cheap enough to run on every single utterance
/// with no perceptible latency cost, since it sits directly in the critical
/// path before any LLM request.
pub fn detect(text: &str) -> Decision {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Decision::Supported(Language::English);
    }

    if has_other_script(trimmed) {
        return Decision::Unsupported;
    }

    let words = tokenize(trimmed);
    if count_hits(&words, OTHER_LANGUAGE_MARKERS) > 0 {
        return Decision::Unsupported;
    }

    // No non-English marker fired at all. Prefer English when the text
    // plausibly contains real English, OR is short enough that a false
    // rejection (e.g. a bare "ok"/"yes"/a name) would be far more annoying
    // than an occasional wrong guess. Anything longer with truly zero
    // recognizable signal in any direction is treated as LowConfidence
    // (garbled/misheard audio), never silently guessed at AND never treated
    // as a language-policy violation — see that variant's doc.
    if looks_like_plain_english(&words) || words.len() <= 3 {
        Decision::Supported(Language::English)
    } else {
        Decision::LowConfidence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_lang(text: &str, expected: Language) {
        assert_eq!(detect(text), Decision::Supported(expected), "text: {text:?}");
    }

    fn assert_unsupported(text: &str) {
        assert_eq!(detect(text), Decision::Unsupported, "text: {text:?}");
    }

    fn assert_low_confidence(text: &str) {
        assert_eq!(detect(text), Decision::LowConfidence, "text: {text:?}");
    }

    #[test]
    fn english_plain_question() {
        assert_lang("How are you?", Language::English);
    }

    #[test]
    fn hindi_devanagari_is_rejected() {
        assert_unsupported("तुम कैसे हो?");
    }

    #[test]
    fn marathi_devanagari_is_rejected() {
        assert_unsupported("तू कसा आहेस?");
    }

    #[test]
    fn hindi_romanized_is_rejected() {
        assert_unsupported("tum kaise ho?");
    }

    #[test]
    fn marathi_romanized_is_rejected() {
        assert_unsupported("tu kasa ahes?");
    }

    #[test]
    fn mixed_hindi_english_is_rejected() {
        assert_unsupported("Home loan ka EMI kitna hona chahiye?");
    }

    #[test]
    fn french_is_rejected() {
        assert_unsupported("Bonjour, comment allez-vous?");
    }

    #[test]
    fn spanish_is_rejected() {
        assert_unsupported("Hola, como estas?");
    }

    #[test]
    fn german_is_rejected() {
        assert_unsupported("Wie geht es dir?");
    }

    #[test]
    fn tamil_script_is_rejected() {
        assert_unsupported("நீங்கள் எப்படி இருக்கிறீர்கள்?");
    }

    #[test]
    fn empty_text_defaults_to_english() {
        assert_lang("", Language::English);
    }

    #[test]
    fn short_ambiguous_reply_defaults_to_english() {
        assert_lang("ok", Language::English);
        assert_lang("yes", Language::English);
    }

    #[test]
    fn plain_english_command_still_works() {
        assert_lang("Open VS Code.", Language::English);
        assert_lang("What's my CPU usage?", Language::English);
    }

    #[test]
    fn long_zero_signal_text_is_low_confidence_not_a_language_rejection() {
        // No English/other-language markers at all, and long enough that a
        // wrong guess is worse than asking the user to repeat — but this is
        // almost always garbled STT output, not a real foreign-language
        // utterance, so it must NOT get the language-policy refusal (see
        // requirement 8: don't repeat "I only support English" for bad
        // audio).
        assert_low_confidence("xyzzy plugh zorkmid frobnitz wibble wobble");
    }

    #[test]
    fn rejection_message_is_the_exact_requested_text() {
        assert_eq!(rejection_message(), "I currently support English only.");
    }

    #[test]
    fn clarification_message_does_not_mention_language_support() {
        let msg = clarification_message();
        assert_eq!(msg, "I couldn't hear you clearly. Please say that again.");
        assert!(!msg.to_lowercase().contains("english"));
    }
}
