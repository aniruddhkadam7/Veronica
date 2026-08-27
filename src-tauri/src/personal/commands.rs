//! Tauri commands exposed to the frontend for the personal-build API-key
//! Settings panel. Never logs a key; `personal_get_api_key` returning the
//! raw stored value to fill an edit field is fine since it is the user's own
//! local input, never anything that reached a network call's logs.

use super::api_key_store;
use super::provider::LlmProvider;

fn parse_provider(provider: &str) -> Result<LlmProvider, String> {
    LlmProvider::from_wire_str(provider).ok_or_else(|| format!("unknown provider: {provider}"))
}

#[tauri::command]
pub fn personal_get_api_key(provider: String) -> Result<Option<String>, String> {
    api_key_store::load_key(parse_provider(&provider)?)
}

#[tauri::command]
pub fn personal_set_api_key(provider: String, key: String) -> Result<(), String> {
    api_key_store::store_key(parse_provider(&provider)?, &key)
}

#[tauri::command]
pub fn personal_clear_api_key(provider: String) -> Result<(), String> {
    api_key_store::clear_key(parse_provider(&provider)?)
}
