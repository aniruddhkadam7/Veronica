/// The three AI providers a personal build can call directly. Uses the same
/// wire strings already sent today by `AskRequest.llm_provider` /
/// `MeetingAskRequest.llm_provider` (see `backend/types.rs`) and stored by
/// the frontend's `llmProviderSetting.ts` — a personal build reuses that
/// exact "which provider is active" selector rather than inventing a second
/// one; this enum only exists so `personal::client`/`personal::api_key_store`
/// have a small typed value instead of passing raw strings around internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    OpenAi,
    Anthropic,
    Gemini,
}

impl LlmProvider {
    /// Parses the wire strings already used by `AskRequest.llm_provider`
    /// etc. ("openai" | "anthropic" | "gemini"). "deepseek" and any other
    /// string return `None` — "deepseek" has no backend implementation
    /// today (the header dropdown keeps it disabled), so a personal build
    /// has nothing to call for it either.
    pub fn from_wire_str(s: &str) -> Option<Self> {
        match s {
            "openai" => Some(Self::OpenAi),
            "anthropic" => Some(Self::Anthropic),
            "gemini" => Some(Self::Gemini),
            _ => None,
        }
    }

    pub fn as_wire_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_valid_wire_strings() {
        for s in ["openai", "anthropic", "gemini"] {
            let provider = LlmProvider::from_wire_str(s).expect("should parse");
            assert_eq!(provider.as_wire_str(), s);
        }
    }

    #[test]
    fn rejects_unknown_or_unimplemented_providers() {
        assert_eq!(LlmProvider::from_wire_str("deepseek"), None);
        assert_eq!(LlmProvider::from_wire_str("bogus"), None);
        assert_eq!(LlmProvider::from_wire_str(""), None);
    }
}
