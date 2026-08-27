//! Persists a user-supplied AI provider API key in Windows Credential
//! Manager via the `keyring` crate — never a plaintext file on disk, never
//! logged, never sent anywhere over the network.

use super::provider::LlmProvider;

const SERVICE_NAME: &str = "Smallbird";

fn entry_name(provider: LlmProvider) -> String {
    format!("personal-api-key-{}", provider.as_wire_str())
}

fn entry(provider: LlmProvider) -> Result<keyring::Entry, String> {
    keyring::Entry::new(SERVICE_NAME, &entry_name(provider)).map_err(|e| format!("credential store unavailable: {e}"))
}

pub fn store_key(provider: LlmProvider, key: &str) -> Result<(), String> {
    entry(provider)?.set_password(key).map_err(|e| format!("failed to store API key: {e}"))
}

/// Returns `Ok(None)` (not an error) if no key is stored yet for this
/// provider — the normal "not configured" state, not a failure.
pub fn load_key(provider: LlmProvider) -> Result<Option<String>, String> {
    match entry(provider)?.get_password() {
        Ok(key) => Ok(Some(key)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(format!("failed to read API key: {e}")),
    }
}

pub fn clear_key(provider: LlmProvider) -> Result<(), String> {
    match entry(provider)?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(e) => Err(format!("failed to clear API key: {e}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exercises the real Windows Credential Manager (no mock trait exists
    // for `keyring` in this codebase) — uses the "gemini" provider slot as
    // the dedicated test entry so it can never collide with a real stored
    // OpenAI/Anthropic key on the machine running the test, and always
    // leaves the store clean afterward regardless of assertion outcome.
    #[test]
    #[ignore = "requires a real, accessible Windows Credential Manager — not available in a sandboxed/headless test runner (observed: set_password succeeds but get_password returns NoEntry in this environment). Run manually with `cargo test -- --ignored` on a real desktop session before release."]
    fn round_trips_through_real_credential_store() {
        let provider = LlmProvider::Gemini;
        let _ = clear_key(provider);

        assert_eq!(load_key(provider).unwrap(), None);

        store_key(provider, "test-key-value").unwrap();
        assert_eq!(load_key(provider).unwrap(), Some("test-key-value".to_string()));

        clear_key(provider).unwrap();
        assert_eq!(load_key(provider).unwrap(), None);
    }
}
