//! Tauri commands exposed to the frontend for the personal-build API-key
//! Settings panel. Never logs a key; `personal_get_api_key` returning the
//! raw stored value to fill an edit field is fine since it is the user's own
//! local input, never anything that reached a network call's logs.
//!
//! `provider` is any non-empty identifier the frontend chooses — not
//! restricted to `LlmProvider`'s three variants, since this panel also
//! manages the Groq (STT) and Deepgram (TTS) keys, which have no place in
//! that LLM-specific enum (see `provider.rs`'s doc). Validation here is
//! deliberately minimal (non-empty only): the credential-store entry name
//! is namespaced (`personal-api-key-<provider>`, see `api_key_store.rs`),
//! so an unrecognized name just becomes its own harmless, unused entry
//! rather than a security or correctness problem.

use super::api_key_store;

fn validate_provider(provider: &str) -> Result<&str, String> {
    if provider.trim().is_empty() {
        return Err("provider name must not be empty".to_string());
    }
    Ok(provider)
}

#[tauri::command]
pub fn personal_get_api_key(provider: String) -> Result<Option<String>, String> {
    api_key_store::load_key(validate_provider(&provider)?)
}

#[tauri::command]
pub fn personal_set_api_key(provider: String, key: String) -> Result<(), String> {
    api_key_store::store_key(validate_provider(&provider)?, &key)
}

#[tauri::command]
pub fn personal_clear_api_key(provider: String) -> Result<(), String> {
    api_key_store::clear_key(validate_provider(&provider)?)
}
