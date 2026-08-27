pub mod openai;
pub mod anthropic;
pub mod gemini;

/// Request timeout for direct provider calls — matches
/// `apps/backend/app/services/llm/*_provider.py`'s
/// `_REQUEST_TIMEOUT_SECONDS = 30.0` (a hung upstream call must not hang the
/// desktop app indefinitely).
pub const REQUEST_TIMEOUT_SECS: u64 = 30;
