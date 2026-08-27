//! Single source of truth for STT/RAG performance configuration. Detects
//! hardware once at app launch, combines it with the user's persisted mode
//! preference, and exposes one `effective_config()` that every consumer
//! (STT sidecar spawn, RAG sidecar spawn, retrieval planner) reads from —
//! no hardware/tier logic is duplicated at any call site.
//!
//! `config_for_tier`'s values below were corrected by Milestone 4a's real
//! benchmark sweeps
//! (`packages/stt-bench/scripts/run_thread_sweep.py`,
//! `packages/rag-bench/bench.py`) on one machine (i5-13400, 10c/16t,
//! 15.6GB RAM, RTX 3050 8GB — see docs/performance-tuning.md once Milestone
//! 7 writes it). Two findings from that sweep reshaped this table from its
//! original "more resources = better" assumption:
//!
//! 1. **STT thread count barely affects latency** on this model/hardware —
//!    first-partial and finalize latency were flat (within ~15ms, i.e.
//!    noise) from 1 to 6 threads, and measurably *worse* at 8 threads,
//!    while CPU-of-machine scaled almost linearly (1.8% → 92.4%) with
//!    thread count. More STT threads is a pure CPU-cost knob here, not a
//!    latency lever — so `stt_num_threads` is now the *same modest value
//!    across every tier* rather than scaling up for higher tiers, and the
//!    CPU headroom a high-end machine has is spent on RAG concurrency
//!    instead (see point 2), where the data shows it actually helps.
//! 2. **RAG embedding scales close to linearly with torch thread count**
//!    (~66-82ms/chunk at 1 thread down to ~24-25ms/chunk at 8 threads,
//!    consistent across batch sizes) but is nearly insensitive to batch
//!    size at a fixed thread count (noise-level differences between batch
//!    8/16/32/64) — batch size mainly costs RSS, not latency, at this
//!    corpus scale. So `rag_torch_threads` is the tier-scaled lever;
//!    `rag_embed_batch_size` only needs to be "big enough," not maximized.
//!
//! What this sweep could NOT validate on the one available machine: how
//! ENTRY/STANDARD tiers behave on genuinely lower-core-count hardware
//! (thread counts below 1 aren't meaningful, so the low end of the sweep is
//! a proxy, not a real low-end measurement); RAG search latency at a
//! realistic multi-thousand-chunk knowledge base (this sweep used 150
//! synthetic chunks, where sqlite-vec's exact KNN was too fast — ~1-2ms —
//! to be a useful discriminator); and STT-under-concurrent-RAG-load
//! contention (this sweep measured STT and RAG in isolation from each
//! other, never running both at once).

use super::pressure::PressureTracker;
use super::profile::HardwareProfile;
use super::tier::{self, HardwareTier};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PerformanceMode {
    Adaptive,
    MaximumPerformance,
    BatterySaver,
}

impl Default for PerformanceMode {
    fn default() -> Self {
        Self::Adaptive
    }
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PerformanceConfig {
    pub stt_num_threads: u32,
    pub rag_top_k: u32,
    pub rag_max_context_chars: usize,
    pub rag_similarity_threshold: f64,
    pub rag_retrieval_timeout_ms: u64,
    pub rag_embed_batch_size: u32,
    pub rag_torch_threads: u32,
}

/// Tier → config lookup, corrected by Milestone 4a's benchmark sweep (see
/// module doc). Still NOT wired into production spawn paths.
///
/// `stt_num_threads` intentionally does NOT scale up with tier the way it
/// did in the pre-benchmark provisional table — the sweep found no latency
/// benefit above 1 thread on this hardware/model, so every tier gets a
/// small, fixed value; what scales with tier is CPU *conservatism* (ENTRY
/// stays minimal to leave headroom for the OS/UI on constrained hardware),
/// not an assumed speed gain. `rag_torch_threads` is the lever that
/// actually showed a strong, close-to-linear latency benefit from more
/// threads, so that scales with tier as originally intended.
/// `rag_embed_batch_size` is now capped at a modest value on every tier
/// (batch size only moved RSS, not latency, in the sweep) rather than
/// scaling to 64 on HighPerformance for no measured benefit.
///
/// Not tier-varied at all (kept fixed regardless of tier, per the plan's
/// governing rule): STT model choice, STT VAD/endpointing settings, RAG
/// chunk size/overlap, and `MAX_CHUNK_CHARS` (tied to the backend's
/// validation limit, see `rag::retrieval_planner`) — none of those are
/// represented in this struct because they are not performance knobs this
/// manager controls.
/// `pub(super)` — visible to sibling modules within `hardware::` (notably
/// `pressure`, which needs the ENTRY row as the ceiling it clamps down to
/// under sustained memory pressure) without exposing this lookup function
/// outside the module; external callers go through
/// `PerformanceManager::effective_config`.
pub(super) fn config_for_tier(tier: HardwareTier, logical_cores: usize) -> PerformanceConfig {
    match tier {
        HardwareTier::Entry => PerformanceConfig {
            stt_num_threads: 1,
            rag_top_k: 2,
            rag_max_context_chars: 2000,
            rag_similarity_threshold: 0.35,
            rag_retrieval_timeout_ms: 800,
            rag_embed_batch_size: 8,
            rag_torch_threads: 1,
        },
        HardwareTier::Standard => PerformanceConfig {
            // Raised from 2 to 4 after field testing on a real low-end
            // machine (Intel i5-6300U, 2-core/4-thread, 2016 mobile ULV
            // chip) showed STT genuinely falling behind live speech —
            // backlog growing over time rather than staying flat, i.e.
            // decode not keeping up with real-time at 2 threads. The
            // original "2, not 4" choice was based on a sweep run entirely
            // on an i5-13400 (10-core/16-thread 2023 desktop CPU) — docs
            // explicitly noted that sweep couldn't validate genuinely
            // lower-core-count hardware and was "a proxy, not a real
            // low-end measurement." 4 matches this class of machine's full
            // logical core count rather than leaving half of it idle while
            // the decoder falls behind.
            stt_num_threads: 4,
            rag_top_k: 3,
            rag_max_context_chars: 3000,
            rag_similarity_threshold: 0.3,
            rag_retrieval_timeout_ms: 1200,
            rag_embed_batch_size: 16,
            rag_torch_threads: 2,
        },
        HardwareTier::Performance => PerformanceConfig {
            // stt_num_threads=4 matches today's shipped default
            // (sidecar.py DEFAULT_NUM_THREADS=4) — kept as-is for
            // continuity with the current production behavior even though
            // the sweep shows 1-2 threads would be equally fast; changing
            // the shipped STT default itself is a separate decision from
            // this tier table and is not made here (see
            // docs/performance-tuning.md open question once Milestone 7
            // writes it).
            stt_num_threads: 4,
            rag_top_k: 4,
            rag_max_context_chars: 3500,
            rag_similarity_threshold: 0.3,
            rag_retrieval_timeout_ms: 1500,
            rag_embed_batch_size: 32,
            rag_torch_threads: 4,
        },
        HardwareTier::HighPerformance => PerformanceConfig {
            // stt_num_threads is capped at 4, NOT scaled to 6/8 as the
            // pre-benchmark provisional table assumed — the sweep measured
            // 6 threads as statistically indistinguishable from 4 (within
            // ~15ms noise) and 8 threads as measurably WORSE (532ms vs
            // 486-501ms first-partial). There is no benchmark evidence to
            // justify spending more CPU here, so this tier does not.
            // rag_torch_threads is where HighPerformance's extra CPU
            // headroom actually pays off (see rag_torch_threads below).
            stt_num_threads: 4,
            rag_top_k: 5,
            rag_max_context_chars: 4000,
            rag_similarity_threshold: 0.25,
            rag_retrieval_timeout_ms: 2000,
            // 32, not 64: the sweep found batch size changes RSS (roughly
            // +90MB from batch 16 to 64 at any fixed thread count) with no
            // measurable latency benefit (ms/chunk was noise-level flat
            // across 8/16/32/64 at every thread count tested) — 64 would
            // spend memory for nothing, so this stays at the same value as
            // the Performance tier.
            rag_embed_batch_size: 32,
            // This is the one place HighPerformance's extra CPU actually
            // helps: embedding scaled close to linearly with torch thread
            // count (~66-82ms/chunk at 1 thread down to ~24-25ms/chunk at 8
            // threads). logical_cores/2 keeps headroom for STT + OS + UI
            // rather than claiming every core for embedding.
            rag_torch_threads: (logical_cores / 2).clamp(1, 8) as u32,
        },
    }
}

pub struct PerformanceManager {
    profile: HardwareProfile,
    tier: HardwareTier,
    mode: PerformanceMode,
    /// Milestone 6: sustained-memory-pressure overlay. Consulted only by
    /// `effective_config_checked` (the checkpoint-driven path callers use
    /// before spawning an STT sidecar or planning a RAG retrieval) — plain
    /// `effective_config()` stays a pure function of tier/mode, unaffected
    /// by runtime memory state, so existing callers that haven't been
    /// updated to pass a fresh RAM reading keep their prior behavior
    /// exactly.
    pressure: PressureTracker,
    /// Set by `hardware::init` (via `set_hardware_refreshed`) when the
    /// persisted hardware fingerprint didn't match this machine's — i.e.
    /// this launch invalidated a stale, explicit mode override that was
    /// chosen on a different physical PC. Purely informational (surfaced by
    /// `get_performance_mode` for the Performance panel, see
    /// `hardware::commands`); never itself affects `effective_config()`.
    hardware_refreshed: bool,
}

impl PerformanceManager {
    pub fn new(profile: HardwareProfile, mode: PerformanceMode) -> Self {
        let tier = tier::classify(&profile);
        Self { profile, tier, mode, pressure: PressureTracker::default(), hardware_refreshed: false }
    }

    pub fn profile(&self) -> &HardwareProfile {
        &self.profile
    }

    /// This machine's hardware fingerprint (see `hardware::fingerprint`) —
    /// used to tag a persisted mode change so a later launch on different
    /// hardware knows not to trust it.
    pub fn hardware_fingerprint(&self) -> String {
        super::fingerprint::compute(&self.profile)
    }

    pub fn detected_tier(&self) -> HardwareTier {
        self.tier
    }

    pub fn mode(&self) -> PerformanceMode {
        self.mode
    }

    pub fn set_hardware_refreshed(&mut self, refreshed: bool) {
        self.hardware_refreshed = refreshed;
    }

    /// See `hardware_refreshed` field doc.
    pub fn hardware_refreshed(&self) -> bool {
        self.hardware_refreshed
    }

    /// Milestone 7: current memory-pressure state, for correlating logged
    /// pipeline-stage timings with whether a downgrade was in effect. Pure
    /// read — does not itself observe a new RAM sample (that only happens
    /// via `effective_config_checked`).
    pub fn pressure_state(&self) -> super::pressure::PressureState {
        self.pressure.state()
    }

    /// Milestone 7: builds the performance-correlation context
    /// (`telemetry::PerfContext`) callers attach to a logged stage timing —
    /// current tier/mode/pressure state plus the config actually in effect
    /// (uses `effective_config()`, not `effective_config_checked`, since
    /// this is a read for logging purposes and must not itself trigger a
    /// pressure-state observation as a side effect of measuring latency).
    pub fn perf_context(&self) -> super::telemetry::PerfContext {
        super::telemetry::PerfContext::new(self.tier, self.mode, self.pressure.state(), &self.effective_config())
    }

    pub fn set_mode(&mut self, mode: PerformanceMode) {
        self.mode = mode;
    }

    /// STT/RAG scheduling coordination (Phase B — see
    /// `stt_rag_coordination` module doc). Whether active STT should
    /// currently throttle RAG indexing: `MaximumPerformance` never
    /// throttles (explicit user opt-in to prioritize resources regardless
    /// of detected tier — requirement: "do not unnecessarily reduce
    /// performance on HighPerformance machines"), `BatterySaver` always
    /// throttles (explicit user opt-in to conserve), `Adaptive` throttles
    /// only when the DETECTED tier is Entry or Standard — a HighPerformance
    /// machine left on Adaptive (the default) never throttles either.
    pub fn should_throttle_background_work(&self) -> bool {
        match self.mode {
            PerformanceMode::MaximumPerformance => false,
            PerformanceMode::BatterySaver => true,
            PerformanceMode::Adaptive => {
                matches!(self.tier, HardwareTier::Entry | HardwareTier::Standard)
            }
        }
    }

    /// The config actually in effect right now: `Adaptive` uses the
    /// detected tier's row; `MaximumPerformance`/`BatterySaver` are
    /// absolute overrides (force the HighPerformance/Entry row) regardless
    /// of detected tier — a deliberate user opt-in, not a blend with the
    /// detected tier.
    pub fn effective_config(&self) -> PerformanceConfig {
        let tier = match self.mode {
            PerformanceMode::Adaptive => self.tier,
            PerformanceMode::MaximumPerformance => HardwareTier::HighPerformance,
            PerformanceMode::BatterySaver => HardwareTier::Entry,
        };
        config_for_tier(tier, self.profile.cpu_logical_cores)
    }

    /// Same as `effective_config()`, but first feeds a freshly-measured
    /// available-RAM reading, plus the latest published CPU-utilization
    /// reading (see `profile::CpuUsageSampler`; `None` if no sample has
    /// published yet), into the sustained-pressure tracker and, if
    /// currently under pressure, clamps STT threads / RAG retrieval budget
    /// down accordingly (never touching RAG's embed batch size / torch
    /// thread count — see `pressure`'s module doc for why). Call sites that
    /// are about to spawn an STT sidecar or plan a RAG retrieval should use
    /// this instead of `effective_config()`; anything that just wants to
    /// *display* the configured tier (e.g. the Performance panel) should
    /// keep using `effective_config()` so the UI reflects the user's
    /// selected mode, not a transient runtime dip.
    ///
    /// Returns the config to use, plus `Some(reason)` if the pressure state
    /// changed on this call — callers should log that reason (see
    /// `commands.rs`/`notes_mode/dictation.rs`/`retrieval_planner` call
    /// sites), since an automatic downgrade or restoration should always be
    /// visible in the logs, not silent.
    pub fn effective_config_checked(
        &mut self,
        available_ram_mb: u64,
        cpu_usage_percent: Option<f32>,
    ) -> (PerformanceConfig, Option<String>) {
        let reason = self.pressure.observe(available_ram_mb, cpu_usage_percent, std::time::Instant::now());
        let config = self.pressure.apply(self.effective_config());
        (config, reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile_with(cores: usize, ram_gb: u64) -> HardwareProfile {
        HardwareProfile {
            cpu_physical_cores: cores / 2,
            cpu_logical_cores: cores,
            cpu_brand: "Test CPU".to_string(),
            total_ram_mb: ram_gb * 1024,
            available_ram_mb: ram_gb * 1024 / 2,
            gpu_vendor: None,
            gpu_name: None,
            gpu_vram_mb: None,
            storage_is_ssd: None,
            windows_version: "Windows 11".to_string(),
        }
    }

    // -- each tier produces its documented, benchmark-corrected config ------

    #[test]
    fn entry_tier_config_matches_the_documented_table() {
        let cfg = config_for_tier(HardwareTier::Entry, 2);
        assert_eq!(cfg.stt_num_threads, 1);
        assert_eq!(cfg.rag_top_k, 2);
        assert_eq!(cfg.rag_max_context_chars, 2000);
        assert_eq!(cfg.rag_similarity_threshold, 0.35);
        assert_eq!(cfg.rag_retrieval_timeout_ms, 800);
        assert_eq!(cfg.rag_embed_batch_size, 8);
        assert_eq!(cfg.rag_torch_threads, 1);
    }

    #[test]
    fn standard_tier_config_matches_the_documented_table() {
        let cfg = config_for_tier(HardwareTier::Standard, 6);
        // Raised from 2 to 4 after field testing on a real 2-core/4-thread
        // machine showed STT backlog growing over time at 2 threads — see
        // the comment on this tier's PerformanceConfig in config_for_tier.
        assert_eq!(cfg.stt_num_threads, 4);
        assert_eq!(cfg.rag_top_k, 3);
        assert_eq!(cfg.rag_max_context_chars, 3000);
        assert_eq!(cfg.rag_similarity_threshold, 0.3);
        assert_eq!(cfg.rag_retrieval_timeout_ms, 1200);
        assert_eq!(cfg.rag_embed_batch_size, 16);
        assert_eq!(cfg.rag_torch_threads, 2);
    }

    #[test]
    fn performance_tier_config_matches_the_documented_table_and_todays_stt_default() {
        let cfg = config_for_tier(HardwareTier::Performance, 12);
        // stt_num_threads must stay 4 — matches sidecar.py's
        // DEFAULT_NUM_THREADS, a deliberate continuity choice per the user's
        // explicit instruction not to globally change the shipped default.
        assert_eq!(cfg.stt_num_threads, 4);
        assert_eq!(cfg.rag_top_k, 4);
        assert_eq!(cfg.rag_max_context_chars, 3500);
        assert_eq!(cfg.rag_similarity_threshold, 0.3);
        assert_eq!(cfg.rag_retrieval_timeout_ms, 1500);
        assert_eq!(cfg.rag_embed_batch_size, 32);
        assert_eq!(cfg.rag_torch_threads, 4);
    }

    #[test]
    fn high_performance_tier_caps_stt_threads_at_4_not_6_or_8() {
        // Regression guard for the Milestone 4a correction: the sweep found
        // no latency benefit above 4 threads (6 was statistically
        // indistinguishable, 8 was measurably worse), so this must never
        // regress back to the pre-benchmark provisional value of 6.
        let cfg = config_for_tier(HardwareTier::HighPerformance, 16);
        assert_eq!(cfg.stt_num_threads, 4);
    }

    #[test]
    fn high_performance_tier_caps_embed_batch_size_at_32_not_64() {
        // Regression guard: batch size showed no latency benefit above 32
        // in the sweep, only extra RSS — must not regress to 64.
        let cfg = config_for_tier(HardwareTier::HighPerformance, 16);
        assert_eq!(cfg.rag_embed_batch_size, 32);
    }

    #[test]
    fn high_performance_tier_scales_torch_threads_with_logical_cores() {
        // The one lever the sweep validated as worth scaling with tier.
        assert_eq!(config_for_tier(HardwareTier::HighPerformance, 16).rag_torch_threads, 8);
        assert_eq!(config_for_tier(HardwareTier::HighPerformance, 8).rag_torch_threads, 4);
        assert_eq!(config_for_tier(HardwareTier::HighPerformance, 4).rag_torch_threads, 2);
    }

    #[test]
    fn high_performance_torch_threads_never_exceeds_8_even_on_a_very_high_core_count_machine() {
        // Clamped to [1, 8] regardless of how many cores are available —
        // unbounded scaling was never validated by the sweep (max tested
        // was 8 threads), so this must not silently extrapolate beyond it.
        let cfg = config_for_tier(HardwareTier::HighPerformance, 64);
        assert_eq!(cfg.rag_torch_threads, 8);
    }

    #[test]
    fn high_performance_torch_threads_is_never_zero_on_a_single_core_machine() {
        // logical_cores=1 would compute 1/2=0 without the clamp's lower
        // bound — torch.set_num_threads(0) is invalid, so this must floor
        // at 1 regardless of how few cores are reported.
        let cfg = config_for_tier(HardwareTier::HighPerformance, 1);
        assert_eq!(cfg.rag_torch_threads, 1);
    }

    // -- mode propagation: Adaptive / MaximumPerformance / BatterySaver -----

    #[test]
    fn adaptive_mode_uses_the_detected_tiers_config() {
        let high_end = profile_with(16, 16); // scores into HighPerformance
        let manager = PerformanceManager::new(high_end, PerformanceMode::Adaptive);
        assert_eq!(manager.detected_tier(), HardwareTier::HighPerformance);
        let cfg = manager.effective_config();
        assert_eq!(cfg.stt_num_threads, config_for_tier(HardwareTier::HighPerformance, 16).stt_num_threads);
        assert_eq!(cfg.rag_torch_threads, config_for_tier(HardwareTier::HighPerformance, 16).rag_torch_threads);
    }

    #[test]
    fn adaptive_mode_on_low_end_hardware_uses_entry_config_not_a_forced_upgrade() {
        let low_end = profile_with(2, 4); // scores into Entry
        let manager = PerformanceManager::new(low_end, PerformanceMode::Adaptive);
        assert_eq!(manager.detected_tier(), HardwareTier::Entry);
        let cfg = manager.effective_config();
        assert_eq!(cfg.stt_num_threads, 1);
        assert_eq!(cfg.rag_torch_threads, 1);
    }

    #[test]
    fn maximum_performance_forces_high_performance_config_even_on_low_end_hardware() {
        // A deliberate user opt-in to spend more resources than the
        // detected tier would normally use — must override, not blend.
        let low_end = profile_with(2, 4);
        let manager = PerformanceManager::new(low_end, PerformanceMode::MaximumPerformance);
        assert_eq!(manager.detected_tier(), HardwareTier::Entry); // detection itself is unaffected
        let cfg = manager.effective_config();
        assert_eq!(cfg.stt_num_threads, config_for_tier(HardwareTier::HighPerformance, 2).stt_num_threads);
        assert_eq!(cfg.rag_embed_batch_size, config_for_tier(HardwareTier::HighPerformance, 2).rag_embed_batch_size);
        // On only 2 logical cores, HighPerformance's torch_threads formula
        // (logical_cores/2, clamped to >=1) still floors at 1 — Maximum
        // Performance forces the HighPerformance *tier row*, it does not
        // pretend the machine has more cores than it does.
        assert_eq!(cfg.rag_torch_threads, 1);
    }

    #[test]
    fn battery_saver_forces_entry_config_even_on_high_end_hardware() {
        // A deliberate user opt-in to conserve resources despite capable
        // hardware — must override, not blend.
        let high_end = profile_with(16, 16);
        let manager = PerformanceManager::new(high_end, PerformanceMode::BatterySaver);
        assert_eq!(manager.detected_tier(), HardwareTier::HighPerformance); // detection itself is unaffected
        let cfg = manager.effective_config();
        assert_eq!(cfg.stt_num_threads, 1);
        assert_eq!(cfg.rag_top_k, 2);
        assert_eq!(cfg.rag_embed_batch_size, 8);
        assert_eq!(cfg.rag_torch_threads, 1);
    }

    #[test]
    fn changing_mode_at_runtime_changes_effective_config_without_re_detecting_hardware() {
        let profile = profile_with(16, 16);
        let mut manager = PerformanceManager::new(profile, PerformanceMode::Adaptive);
        let adaptive_cfg = manager.effective_config();
        assert_eq!(adaptive_cfg.rag_embed_batch_size, 32); // HighPerformance row

        manager.set_mode(PerformanceMode::BatterySaver);
        assert_eq!(manager.detected_tier(), HardwareTier::HighPerformance); // unchanged
        let battery_cfg = manager.effective_config();
        assert_eq!(battery_cfg.rag_embed_batch_size, 8); // Entry row now in effect

        manager.set_mode(PerformanceMode::MaximumPerformance);
        let max_cfg = manager.effective_config();
        assert_eq!(max_cfg.rag_embed_batch_size, 32); // back to HighPerformance row
    }

    #[test]
    fn performance_mode_defaults_to_adaptive() {
        assert_eq!(PerformanceMode::default(), PerformanceMode::Adaptive);
    }

    // -- hardware_refreshed: purely informational, set by hardware::init ----

    #[test]
    fn hardware_refreshed_defaults_to_false() {
        let manager = PerformanceManager::new(profile_with(8, 8), PerformanceMode::Adaptive);
        assert!(!manager.hardware_refreshed());
    }

    #[test]
    fn set_hardware_refreshed_is_observable_and_does_not_affect_effective_config() {
        let mut manager = PerformanceManager::new(profile_with(8, 8), PerformanceMode::Adaptive);
        let before = manager.effective_config().stt_num_threads;
        manager.set_hardware_refreshed(true);
        assert!(manager.hardware_refreshed());
        assert_eq!(manager.effective_config().stt_num_threads, before);
    }

    #[test]
    fn hardware_fingerprint_reflects_this_managers_profile() {
        let manager = PerformanceManager::new(profile_with(8, 8), PerformanceMode::Adaptive);
        assert_eq!(manager.hardware_fingerprint(), super::super::fingerprint::compute(manager.profile()));
    }

    // -- effective_config_checked: PerformanceManager <-> PressureTracker ---
    // (PressureTracker's own state-machine behavior — entering/exiting
    // pressure, hysteresis, cooldown — is covered exhaustively in
    // hardware::pressure::tests; these confirm the manager actually wires a
    // fresh RAM reading through to it and applies the result.)

    #[test]
    fn effective_config_checked_matches_plain_effective_config_when_ram_is_healthy() {
        let mut manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::Adaptive);
        let (checked, reason) = manager.effective_config_checked(8_000, None);
        assert!(reason.is_none());
        assert_eq!(checked.stt_num_threads, manager.effective_config().stt_num_threads);
    }

    #[test]
    fn effective_config_checked_clamps_down_after_sustained_low_readings() {
        let mut manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::Adaptive);
        let baseline = manager.effective_config();
        assert_eq!(baseline.stt_num_threads, 4); // HighPerformance row

        let (_, reason1) = manager.effective_config_checked(500, None);
        assert!(reason1.is_none()); // first low reading — not sustained yet

        let (checked, reason2) = manager.effective_config_checked(500, None);
        assert!(reason2.is_some(), "second consecutive low reading should trigger pressure");
        assert_eq!(checked.stt_num_threads, 1); // clamped to Entry's value
    }

    #[test]
    fn effective_config_checked_leaves_rag_embed_config_untouched_even_under_pressure() {
        let mut manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::Adaptive);
        let baseline = manager.effective_config();

        manager.effective_config_checked(500, None);
        let (checked, _) = manager.effective_config_checked(500, None);

        assert_eq!(checked.rag_embed_batch_size, baseline.rag_embed_batch_size);
        assert_eq!(checked.rag_torch_threads, baseline.rag_torch_threads);
    }

    #[test]
    fn effective_config_checked_clamps_down_after_sustained_high_cpu_even_with_healthy_ram() {
        let mut manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::Adaptive);
        let baseline = manager.effective_config();
        assert_eq!(baseline.stt_num_threads, 4); // HighPerformance row

        let (_, reason1) = manager.effective_config_checked(8_000, Some(96.0));
        assert!(reason1.is_none()); // first high-CPU reading — not sustained yet

        let (checked, reason2) = manager.effective_config_checked(8_000, Some(96.0));
        assert!(reason2.is_some(), "second consecutive high-CPU reading should trigger pressure");
        assert_eq!(checked.stt_num_threads, 1); // clamped to Entry's value, same ceiling as RAM pressure
    }

    #[test]
    fn effective_config_checked_with_no_cpu_reading_behaves_exactly_like_ram_only_pressure() {
        // Mirrors the pre-CPU-pressure behavior exactly when the sampler
        // hasn't published a reading yet — a regression guard that adding
        // CPU pressure didn't change existing RAM-only call sites'
        // behavior when they simply don't have a CPU reading available.
        let mut manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::Adaptive);
        manager.effective_config_checked(500, None);
        let (checked, reason) = manager.effective_config_checked(500, None);
        assert!(reason.is_some());
        assert_eq!(checked.stt_num_threads, 1);
    }

    // -- should_throttle_background_work: STT/RAG scheduling (Phase B) ------

    #[test]
    fn adaptive_mode_throttles_on_entry_tier_hardware() {
        let manager = PerformanceManager::new(profile_with(2, 4), PerformanceMode::Adaptive);
        assert_eq!(manager.detected_tier(), HardwareTier::Entry);
        assert!(manager.should_throttle_background_work());
    }

    #[test]
    fn adaptive_mode_throttles_on_standard_tier_hardware() {
        let manager = PerformanceManager::new(profile_with(6, 16), PerformanceMode::Adaptive);
        assert_eq!(manager.detected_tier(), HardwareTier::Standard);
        assert!(manager.should_throttle_background_work());
    }

    #[test]
    fn adaptive_mode_does_not_throttle_on_high_performance_hardware() {
        let manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::Adaptive);
        assert_eq!(manager.detected_tier(), HardwareTier::HighPerformance);
        assert!(
            !manager.should_throttle_background_work(),
            "must not unnecessarily reduce performance on HighPerformance machines"
        );
    }

    #[test]
    fn adaptive_mode_does_not_throttle_on_performance_tier_hardware() {
        let manager = PerformanceManager::new(profile_with(12, 16), PerformanceMode::Adaptive);
        assert_eq!(manager.detected_tier(), HardwareTier::Performance);
        assert!(!manager.should_throttle_background_work());
    }

    #[test]
    fn maximum_performance_mode_never_throttles_even_on_entry_hardware() {
        let manager = PerformanceManager::new(profile_with(2, 4), PerformanceMode::MaximumPerformance);
        assert_eq!(manager.detected_tier(), HardwareTier::Entry); // detection itself unaffected
        assert!(
            !manager.should_throttle_background_work(),
            "explicit MaximumPerformance opt-in must never throttle, regardless of detected tier"
        );
    }

    #[test]
    fn battery_saver_mode_always_throttles_even_on_high_end_hardware() {
        let manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::BatterySaver);
        assert_eq!(manager.detected_tier(), HardwareTier::HighPerformance); // detection itself unaffected
        assert!(
            manager.should_throttle_background_work(),
            "explicit BatterySaver opt-in must always throttle, regardless of detected tier"
        );
    }

    #[test]
    fn throttle_decision_updates_immediately_when_mode_changes() {
        let mut manager = PerformanceManager::new(profile_with(16, 16), PerformanceMode::Adaptive);
        assert!(!manager.should_throttle_background_work());

        manager.set_mode(PerformanceMode::BatterySaver);
        assert!(manager.should_throttle_background_work());

        manager.set_mode(PerformanceMode::MaximumPerformance);
        assert!(!manager.should_throttle_background_work());
    }
}
