//! The personal AI client: talks directly to your configured AI provider
//! (OpenAI/Anthropic/Gemini) using a manually-entered API key. This repo is
//! permanently the personal build (no SaaS backend, no Supabase sign-in, no
//! entitlement/session-authority gating) — see `client::DirectLlmClient` for
//! the direct-provider call path used at every AI call site.

pub mod agent;
pub mod api_key_store;
pub mod client;
pub mod commands;
pub mod prompts;
pub mod provider;
pub mod providers;

pub use client::DirectLlmClient;
