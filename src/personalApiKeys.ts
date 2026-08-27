import { invoke } from "@tauri-apps/api/core";

// Thin wrappers around the personal API-key Tauri commands (see
// src-tauri/src/personal/commands.rs). This is a personal build — every
// call here is always meaningful, unlike in the SaaS repo this was split
// from, where these were gated behind a PERSONAL_BUILD compile-time flag.

export type PersonalLlmProvider = "openai" | "anthropic" | "gemini";

export function getPersonalApiKey(provider: PersonalLlmProvider): Promise<string | null> {
  return invoke<string | null>("personal_get_api_key", { provider });
}

export function setPersonalApiKey(provider: PersonalLlmProvider, key: string): Promise<void> {
  return invoke<void>("personal_set_api_key", { provider, key });
}

export function clearPersonalApiKey(provider: PersonalLlmProvider): Promise<void> {
  return invoke<void>("personal_clear_api_key", { provider });
}
