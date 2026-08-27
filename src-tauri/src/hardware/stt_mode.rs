//! STT_MODE (auto/local/cloud) preference parsing and resolution.
//!
//! Separate from `SttEngineKind` (`crate::stt::sidecar`) — that already
//! selects between two *local* engines (the production StreamingAsr
//! sidecar vs. the PocketSphinx comparison fallback), a different concern
//! from local-vs-cloud provider selection. Also kept out of `manager.rs`,
//! which is explicitly performance-tuning-only (thread counts, RAG batch
//! sizes) and never engine selection — this module reads the tier
//! `PerformanceManager` has already detected, without doing any new
//! hardware probing itself.
//!
//! No cloud STT provider exists yet (see `docs/architecture.md` — STT is
//! 100% local). `resolve()` is written so it is IMPOSSIBLE to return a
//! value implying a real cloud attempt should be made: `Auto` always
//! resolves to `Local`, and an explicit `Cloud` preference resolves to
//! `CloudUnavailable`, a variant with no attached "go call this provider"
//! behavior anywhere. This makes "never attempt a network call to a
//! nonexistent cloud STT provider" true by construction, not convention.

use super::tier::HardwareTier;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SttModePreference {
    Auto,
    Local,
    Cloud,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResolvedSttMode {
    /// Use the existing local STT sidecar (StreamingAsr/PocketSphinx per
    /// `SttEngineKind`) — the only mode that has ever run in this app.
    Local,
    /// The caller asked for cloud STT, but no provider is implemented yet.
    /// Callers must treat this exactly like `Local` (stay on local
    /// transcription) while surfacing that the request was not honored —
    /// never attempt any network call on the strength of this variant.
    CloudUnavailable,
}

/// Reads `STT_MODE` the same way `SttEngineKind::from_env` reads
/// `STT_ENGINE` — case-insensitive, unset or unrecognized falls back to the
/// safe default (`Auto`), never an error.
pub fn preference_from_env() -> SttModePreference {
    match std::env::var("STT_MODE").ok().as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("local") => SttModePreference::Local,
        Some("cloud") => SttModePreference::Cloud,
        _ => SttModePreference::Auto,
    }
}

/// `tier` is accepted so `Auto`'s behavior is documented as intentionally
/// tier-independent (see module doc) rather than looking like an oversight
/// — a future real cloud integration would branch on it here.
pub fn resolve(preference: SttModePreference, _tier: HardwareTier) -> ResolvedSttMode {
    match preference {
        SttModePreference::Local | SttModePreference::Auto => ResolvedSttMode::Local,
        SttModePreference::Cloud => ResolvedSttMode::CloudUnavailable,
    }
}

/// Purely informational — mirrors `hardware::commands::PerformanceModeInfo`'s
/// `is_below_recommended` advisory posture (never itself a gate, never
/// blocks anything). Every tier except the lowest is considered adequate for
/// local STT today; this only exists so a future UI could show a "your
/// hardware is below our recommendation for local transcription" notice.
pub fn local_stt_recommended(tier: HardwareTier) -> bool {
    !matches!(tier, HardwareTier::Entry)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_never_returns_a_real_cloud_attempt_variant() {
        for preference in [SttModePreference::Auto, SttModePreference::Local, SttModePreference::Cloud] {
            for tier in [
                HardwareTier::Entry,
                HardwareTier::Standard,
                HardwareTier::Performance,
                HardwareTier::HighPerformance,
            ] {
                let resolved = resolve(preference, tier);
                assert!(matches!(resolved, ResolvedSttMode::Local | ResolvedSttMode::CloudUnavailable));
            }
        }
    }

    #[test]
    fn auto_always_resolves_to_local_regardless_of_tier() {
        for tier in [
            HardwareTier::Entry,
            HardwareTier::Standard,
            HardwareTier::Performance,
            HardwareTier::HighPerformance,
        ] {
            assert_eq!(resolve(SttModePreference::Auto, tier), ResolvedSttMode::Local);
        }
    }

    #[test]
    fn local_preference_always_resolves_to_local() {
        assert_eq!(resolve(SttModePreference::Local, HardwareTier::Entry), ResolvedSttMode::Local);
        assert_eq!(resolve(SttModePreference::Local, HardwareTier::HighPerformance), ResolvedSttMode::Local);
    }

    #[test]
    fn cloud_preference_resolves_to_cloud_unavailable_not_local() {
        // Explicit opt-in to cloud must be visibly distinct from local, even
        // though both currently result in local transcription running — the
        // caller needs to know the request wasn't honored.
        assert_eq!(resolve(SttModePreference::Cloud, HardwareTier::HighPerformance), ResolvedSttMode::CloudUnavailable);
    }

    #[test]
    fn local_stt_recommended_is_false_only_for_entry_tier() {
        assert!(!local_stt_recommended(HardwareTier::Entry));
        assert!(local_stt_recommended(HardwareTier::Standard));
        assert!(local_stt_recommended(HardwareTier::Performance));
        assert!(local_stt_recommended(HardwareTier::HighPerformance));
    }

    #[test]
    fn preference_from_env_defaults_to_auto_when_unset() {
        std::env::remove_var("STT_MODE");
        assert_eq!(preference_from_env(), SttModePreference::Auto);
    }

    #[test]
    fn preference_from_env_parses_known_values_case_insensitively() {
        std::env::set_var("STT_MODE", "LOCAL");
        assert_eq!(preference_from_env(), SttModePreference::Local);
        std::env::set_var("STT_MODE", "Cloud");
        assert_eq!(preference_from_env(), SttModePreference::Cloud);
        std::env::remove_var("STT_MODE");
    }

    #[test]
    fn preference_from_env_falls_back_to_auto_for_unrecognized_value() {
        std::env::set_var("STT_MODE", "quantum");
        assert_eq!(preference_from_env(), SttModePreference::Auto);
        std::env::remove_var("STT_MODE");
    }
}
