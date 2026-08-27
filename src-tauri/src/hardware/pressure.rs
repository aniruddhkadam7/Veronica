//! Runtime memory- and CPU-pressure adaptation: a conservative,
//! evidence-gated overlay on top of the hardware-detected tier
//! (`tier.rs`/`manager.rs`).
//!
//! Goal (per explicit product decision): prevent a low-memory OR
//! CPU-saturated system from becoming unresponsive — NOT continuously
//! re-tune settings. This is deliberately narrow in scope:
//!
//! - Only ever *reduces* local workload (STT thread count for the next
//!   recording session, RAG retrieval top_k/context budget/timeout for the
//!   next question) below what the detected tier/user mode would otherwise
//!   select. It never increases beyond the tier/mode's own config.
//! - Never touches RAG's embedding batch size / torch thread count — those
//!   are fixed at RAG-sidecar-process-spawn time (see `manager.rs`'s module
//!   doc and `rag::process::RagServiceHandle::spawn`'s doc comment), and
//!   restarting that process is a deliberate, user-initiated action
//!   (`set_performance_mode`), never an automatic side effect of a memory
//!   reading. This module holds no reference to `RagServiceHandle` at all.
//! - Never changes STT model choice, VAD, endpointing, or anything on the
//!   cloud LLM/backend path — those are out of scope for hardware-adaptive
//!   tuning entirely (see `manager.rs`'s module doc).
//! - Judges "pressure" from sustained readings, not one instantaneous
//!   sample, and applies hysteresis so entering and exiting pressure state
//!   both require the signal to hold for a while — see `PressureTracker`'s
//!   doc comment for the exact rule.
//!
//! This is checkpoint-driven (sampled when a caller asks, e.g. before
//! spawning an STT sidecar or planning a RAG retrieval), not a continuous
//! background poll — no timer thread, no periodic wakeups. A checkpoint
//! that never fires (the app sits idle) simply never re-evaluates, which is
//! correct: there is nothing to protect if nothing is about to consume
//! memory.

use std::time::{Duration, Instant};

use super::manager::PerformanceConfig;
use super::tier::HardwareTier;

/// Available RAM below this is treated as a candidate pressure signal.
/// Anchored to STT's own measured footprint (228MB, see
/// docs/stt-benchmark.md) — 1024MB is roughly 4x that, leaving headroom for
/// RAG's embedding model (~90MB weights + torch/transformers overhead) and
/// the OS/webview on top, without being so tight that ordinary memory
/// fluctuation from an unrelated application falsely triggers it.
const LOW_MEMORY_THRESHOLD_MB: u64 = 1024;

/// System-wide CPU utilization at/above this is treated as a candidate
/// pressure signal. 90%, not 100%, because the failure mode being protected
/// against is the OS scheduler starving Smallbird's own STT/RAG threads of
/// timeslices — that starts to bite before every last percent is claimed,
/// and waiting for a literal 100% reading (which real hardware rarely
/// reports with perfect precision even when saturated) would react too late
/// on a machine already struggling.
const HIGH_CPU_THRESHOLD_PERCENT: f32 = 90.0;

/// CPU usage this far below the threshold counts as "comfortably
/// recovered" — mirrors `RECOVERY_MARGIN_MB`'s purpose for RAM: avoids a
/// reading sitting right at the boundary from flapping the state back and
/// forth as ordinary noise.
const CPU_RECOVERY_MARGIN_PERCENT: f32 = 15.0;

/// A single low reading is not "sustained" — require at least this many
/// consecutive checkpoint samples below the threshold before entering
/// pressure state. Two, not one: the smallest number that distinguishes
/// "briefly dipped, already recovering" from "still low next time we
/// checked," without requiring a long observation window that would delay
/// protecting a genuinely struggling machine.
const CONSECUTIVE_LOW_SAMPLES_TO_ENTER: u32 = 2;

/// Once in pressure state, available RAM must read comfortably above the
/// threshold (with margin, see `RECOVERY_MARGIN_MB`) for at least this long
/// before restoring normal config — the cooldown. Prevents flapping back
/// and forth if memory hovers right at the threshold (e.g. another
/// application's own working set oscillating). 90s is long enough to ride
/// out a brief recovery blip, short enough that a genuine, sustained
/// recovery (the other application closing, a one-time spike passing)
/// isn't stuck in reduced mode for an unreasonable stretch of a session.
const COOLDOWN: Duration = Duration::from_secs(90);

/// Recovery must clear the threshold by more than just returning to
/// exactly the trigger point — a small margin (not just `> THRESHOLD`)
/// avoids a value sitting right at the boundary from repeatedly crossing
/// it in both directions as ordinary noise.
const RECOVERY_MARGIN_MB: u64 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureState {
    Normal,
    UnderPressure,
}

/// Tracks sustained memory pressure across checkpoint samples and decides,
/// with hysteresis, when to enter/exit a reduced-workload state.
///
/// State machine:
///
/// ```text
/// Normal --(N consecutive low samples)--> UnderPressure
/// UnderPressure --(available RAM > threshold+margin for >= COOLDOWN)--> Normal
/// ```
///
/// A sample outside the "comfortably recovered" band while under pressure
/// resets the cooldown clock (the recovery timer only counts *continuous*
/// time spent above the recovery line) — a single good reading followed by
/// a bad one does not restore normal config.
pub struct PressureTracker {
    state: PressureState,
    consecutive_low_samples: u32,
    /// When the most recent continuous stretch of "comfortably recovered"
    /// readings began, if we're currently in one. `None` means the last
    /// sample was NOT comfortably recovered (at or below threshold+margin),
    /// so there is no active recovery stretch to measure a cooldown against.
    recovery_since: Option<Instant>,
    last_sample_mb: Option<u64>,
    /// Same sustained/hysteresis bookkeeping as the RAM fields above, but
    /// for CPU utilization — a fully independent signal (see `observe`'s
    /// doc: either signal alone can enter pressure, both must recover for
    /// exit).
    consecutive_high_cpu_samples: u32,
    cpu_recovery_since: Option<Instant>,
    last_cpu_sample_percent: Option<f32>,
}

impl Default for PressureTracker {
    fn default() -> Self {
        Self {
            state: PressureState::Normal,
            consecutive_low_samples: 0,
            recovery_since: None,
            last_sample_mb: None,
            consecutive_high_cpu_samples: 0,
            cpu_recovery_since: None,
            last_cpu_sample_percent: None,
        }
    }
}

impl PressureTracker {
    pub fn state(&self) -> PressureState {
        self.state
    }

    /// Feeds one fresh available-RAM reading (always) and, if available, a
    /// fresh CPU-utilization reading (from `hardware::profile::
    /// CpuUsageSampler`, which may not have published its first sample yet
    /// — `None` in that window) into the tracker, and returns
    /// `Some(reason)` if the overall state changed (for the caller to log).
    /// Callers should call this at a natural checkpoint (session/job start)
    /// with freshly-measured values, not on a timer.
    ///
    /// RAM and CPU are independent signals evaluated with the same
    /// sustained/hysteresis rule each (see `evaluate_signal`): **either**
    /// signal alone sustaining past its threshold is enough to enter
    /// `UnderPressure` (a machine can be strangled by either resource), but
    /// **both** must independently finish their own recovery cooldown
    /// before the tracker returns to `Normal` — recovering RAM while CPU is
    /// still pegged (or vice versa) must not prematurely restore full
    /// workload.
    pub fn observe(&mut self, available_ram_mb: u64, cpu_usage_percent: Option<f32>, now: Instant) -> Option<String> {
        self.last_sample_mb = Some(available_ram_mb);
        let ram_low = available_ram_mb < LOW_MEMORY_THRESHOLD_MB;
        let ram_recovered = available_ram_mb > LOW_MEMORY_THRESHOLD_MB + RECOVERY_MARGIN_MB;

        let cpu_high = cpu_usage_percent.map(|p| p >= HIGH_CPU_THRESHOLD_PERCENT).unwrap_or(false);
        let cpu_recovered = cpu_usage_percent
            .map(|p| p < HIGH_CPU_THRESHOLD_PERCENT - CPU_RECOVERY_MARGIN_PERCENT)
            .unwrap_or(true); // no CPU reading yet: don't let an absent signal block RAM-only recovery
        if let Some(p) = cpu_usage_percent {
            self.last_cpu_sample_percent = Some(p);
        }

        match self.state {
            PressureState::Normal => {
                self.consecutive_low_samples = if ram_low { self.consecutive_low_samples + 1 } else { 0 };
                self.consecutive_high_cpu_samples =
                    if cpu_high { self.consecutive_high_cpu_samples + 1 } else { 0 };

                let ram_triggered = self.consecutive_low_samples >= CONSECUTIVE_LOW_SAMPLES_TO_ENTER;
                let cpu_triggered = self.consecutive_high_cpu_samples >= CONSECUTIVE_LOW_SAMPLES_TO_ENTER;

                if ram_triggered || cpu_triggered {
                    self.state = PressureState::UnderPressure;
                    self.recovery_since = None;
                    self.cpu_recovery_since = None;
                    let reason = if ram_triggered && cpu_triggered {
                        format!(
                            "entering reduced-workload mode: available RAM {available_ram_mb}MB stayed below {LOW_MEMORY_THRESHOLD_MB}MB AND CPU usage stayed at/above {HIGH_CPU_THRESHOLD_PERCENT:.0}% for {} consecutive checks",
                            self.consecutive_low_samples.max(self.consecutive_high_cpu_samples)
                        )
                    } else if ram_triggered {
                        format!(
                            "entering reduced-workload mode: available RAM {available_ram_mb}MB stayed below {LOW_MEMORY_THRESHOLD_MB}MB for {} consecutive checks",
                            self.consecutive_low_samples
                        )
                    } else {
                        format!(
                            "entering reduced-workload mode: CPU usage stayed at/above {HIGH_CPU_THRESHOLD_PERCENT:.0}% for {} consecutive checks",
                            self.consecutive_high_cpu_samples
                        )
                    };
                    return Some(reason);
                }
                None
            }
            PressureState::UnderPressure => {
                let ram_cooldown_done = Self::advance_cooldown(&mut self.recovery_since, ram_recovered, now);
                let cpu_cooldown_done = Self::advance_cooldown(&mut self.cpu_recovery_since, cpu_recovered, now);

                if ram_cooldown_done && cpu_cooldown_done {
                    self.state = PressureState::Normal;
                    self.consecutive_low_samples = 0;
                    self.consecutive_high_cpu_samples = 0;
                    self.recovery_since = None;
                    self.cpu_recovery_since = None;
                    let reason = format!(
                        "restoring normal performance mode: RAM and CPU both stayed within healthy range for >= {}s cooldown",
                        COOLDOWN.as_secs()
                    );
                    return Some(reason);
                }
                None
            }
        }
    }

    /// Shared recovery-cooldown bookkeeping for one signal (RAM or CPU):
    /// `recovered` reflects whether *this checkpoint's* reading is
    /// comfortably past that signal's recovery line. Returns `true` only
    /// once this signal has been continuously recovered for at least
    /// `COOLDOWN`. A reading that is not comfortably recovered resets the
    /// stretch, matching the original RAM-only behavior exactly.
    fn advance_cooldown(since: &mut Option<Instant>, recovered: bool, now: Instant) -> bool {
        if !recovered {
            *since = None;
            return false;
        }
        let started = *since.get_or_insert(now);
        now.duration_since(started) >= COOLDOWN
    }

    /// Applies this tracker's current state to a tier-selected config,
    /// producing the config that should actually be used. `Normal` is a
    /// pure passthrough. `UnderPressure` clamps STT threads and RAG
    /// retrieval budget down to the ENTRY tier's values (never below what
    /// ENTRY already uses — pressure adaptation is a ceiling, not its own
    /// separate scale) while leaving `rag_embed_batch_size`/
    /// `rag_torch_threads` completely untouched, since those cannot be
    /// changed without restarting the RAG process, which this module never
    /// does (see module doc).
    pub fn apply(&self, base: PerformanceConfig) -> PerformanceConfig {
        match self.state {
            PressureState::Normal => base,
            PressureState::UnderPressure => {
                let entry = super::manager::config_for_tier(HardwareTier::Entry, 1);
                PerformanceConfig {
                    stt_num_threads: base.stt_num_threads.min(entry.stt_num_threads),
                    rag_top_k: base.rag_top_k.min(entry.rag_top_k),
                    rag_max_context_chars: base.rag_max_context_chars.min(entry.rag_max_context_chars),
                    rag_similarity_threshold: base.rag_similarity_threshold.max(entry.rag_similarity_threshold),
                    rag_retrieval_timeout_ms: base.rag_retrieval_timeout_ms.min(entry.rag_retrieval_timeout_ms),
                    // Untouched — see doc comment above.
                    rag_embed_batch_size: base.rag_embed_batch_size,
                    rag_torch_threads: base.rag_torch_threads,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(seconds_from_epoch_ish: u64) -> Instant {
        // Tests only ever compare durations between two `Instant`s produced
        // by this helper, never against wall-clock time, so an arbitrary
        // fixed base plus an offset is a valid, deterministic stand-in for
        // "time passing" without sleeping in the test.
        Instant::now() + Duration::from_secs(seconds_from_epoch_ish)
    }

    // -- entering pressure: sustained, not a single blip --------------------

    #[test]
    fn a_single_low_reading_does_not_trigger_pressure() {
        let mut tracker = PressureTracker::default();
        let changed = tracker.observe(500, None, t(0));
        assert!(changed.is_none());
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn two_consecutive_low_readings_trigger_pressure() {
        let mut tracker = PressureTracker::default();
        assert!(tracker.observe(500, None, t(0)).is_none());
        let changed = tracker.observe(400, None, t(10));
        assert!(changed.is_some());
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    #[test]
    fn a_low_reading_followed_by_a_healthy_one_resets_the_low_streak() {
        let mut tracker = PressureTracker::default();
        assert!(tracker.observe(500, None, t(0)).is_none()); // 1 low
        assert!(tracker.observe(4000, None, t(10)).is_none()); // healthy — resets streak
        assert!(tracker.observe(500, None, t(20)).is_none()); // 1 low again, not 2 in a row
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn logged_reason_names_the_actual_reading_and_threshold() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        let reason = tracker.observe(300, None, t(10)).expect("should have entered pressure");
        assert!(reason.contains("300MB"));
        assert!(reason.contains("1024MB"));
    }

    // -- staying under pressure without oscillating --------------------------

    #[test]
    fn a_reading_right_at_the_threshold_does_not_recover() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));
        assert_eq!(tracker.state(), PressureState::UnderPressure);
        // Exactly at LOW_MEMORY_THRESHOLD_MB, not above the recovery
        // margin — must not restore.
        let changed = tracker.observe(LOW_MEMORY_THRESHOLD_MB, None, t(200));
        assert!(changed.is_none());
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    #[test]
    fn crossing_above_threshold_but_not_the_recovery_margin_does_not_recover() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));
        // Above LOW_MEMORY_THRESHOLD_MB (1024) but not above
        // threshold+margin (1280) — must not start/complete recovery.
        let changed = tracker.observe(1100, None, t(200));
        assert!(changed.is_none());
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    #[test]
    fn recovery_reading_before_cooldown_elapses_does_not_restore() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));
        assert_eq!(tracker.state(), PressureState::UnderPressure);
        // Comfortably above the recovery line, but only 30s in — cooldown
        // is 90s.
        let changed = tracker.observe(4000, None, t(40));
        assert!(changed.is_none());
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    #[test]
    fn sustained_recovery_for_the_full_cooldown_restores_normal_mode() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));
        assert_eq!(tracker.state(), PressureState::UnderPressure);
        tracker.observe(4000, None, t(20)); // recovery stretch begins here
        let changed = tracker.observe(4000, None, t(20 + 90)); // >= 90s later
        assert!(changed.is_some());
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn a_dip_back_below_the_recovery_line_resets_the_cooldown_clock() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));
        tracker.observe(4000, None, t(20)); // recovery stretch begins
        tracker.observe(4000, None, t(60)); // 40s into recovery, still under 90s cooldown
        // Dips back down — breaks the continuous recovery stretch.
        tracker.observe(1100, None, t(70));
        assert_eq!(tracker.state(), PressureState::UnderPressure);
        // Even though 90s has now passed since the FIRST recovery reading
        // (t=20 -> t=115 would be >=90s), the dip at t=70 reset the clock,
        // so this must NOT have recovered yet.
        let changed = tracker.observe(4000, None, t(115));
        assert!(changed.is_none(), "a broken recovery stretch must not count toward cooldown");
        assert_eq!(tracker.state(), PressureState::UnderPressure);
        // But continuing to recover from t=115 for the full cooldown does
        // eventually restore.
        let changed = tracker.observe(4000, None, t(115 + 90));
        assert!(changed.is_some());
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn after_restoring_normal_mode_pressure_can_trigger_again() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));
        tracker.observe(4000, None, t(20));
        tracker.observe(4000, None, t(20 + 90));
        assert_eq!(tracker.state(), PressureState::Normal);

        // A fresh sustained-low episode later in the same session.
        tracker.observe(500, None, t(1000));
        let changed = tracker.observe(500, None, t(1010));
        assert!(changed.is_some());
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    // -- CPU pressure: an independent signal, same sustained/hysteresis rule -

    #[test]
    fn a_single_high_cpu_reading_does_not_trigger_pressure() {
        let mut tracker = PressureTracker::default();
        // Healthy RAM throughout — only CPU is being exercised here.
        let changed = tracker.observe(8000, Some(95.0), t(0));
        assert!(changed.is_none());
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn two_consecutive_high_cpu_readings_trigger_pressure_even_with_healthy_ram() {
        let mut tracker = PressureTracker::default();
        assert!(tracker.observe(8000, Some(95.0), t(0)).is_none());
        let changed = tracker.observe(8000, Some(97.0), t(10));
        assert!(changed.is_some(), "sustained high CPU alone must be enough to enter pressure");
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    #[test]
    fn a_reading_below_the_cpu_threshold_resets_the_high_cpu_streak() {
        let mut tracker = PressureTracker::default();
        assert!(tracker.observe(8000, Some(95.0), t(0)).is_none()); // 1 high
        assert!(tracker.observe(8000, Some(50.0), t(10)).is_none()); // healthy — resets streak
        assert!(tracker.observe(8000, Some(95.0), t(20)).is_none()); // 1 high again, not 2 in a row
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn no_cpu_reading_never_triggers_cpu_pressure_on_its_own() {
        // The sampler hasn't published its first reading yet (or failed to
        // start) — must never be treated as "0% idle" or "100% saturated",
        // just absent.
        let mut tracker = PressureTracker::default();
        assert!(tracker.observe(8000, None, t(0)).is_none());
        assert!(tracker.observe(8000, None, t(10)).is_none());
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn cpu_recovery_requires_dropping_below_threshold_minus_margin_not_just_below_threshold() {
        let mut tracker = PressureTracker::default();
        tracker.observe(8000, Some(95.0), t(0));
        tracker.observe(8000, Some(95.0), t(10));
        assert_eq!(tracker.state(), PressureState::UnderPressure);
        // Below HIGH_CPU_THRESHOLD_PERCENT (90) but not below the recovery
        // line (90 - 15 = 75) — must not recover.
        let changed = tracker.observe(8000, Some(80.0), t(200));
        assert!(changed.is_none());
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    #[test]
    fn sustained_cpu_recovery_for_the_full_cooldown_restores_normal_mode() {
        let mut tracker = PressureTracker::default();
        tracker.observe(8000, Some(95.0), t(0));
        tracker.observe(8000, Some(95.0), t(10));
        assert_eq!(tracker.state(), PressureState::UnderPressure);
        tracker.observe(8000, Some(10.0), t(20)); // recovery stretch begins
        let changed = tracker.observe(8000, Some(10.0), t(20 + 90));
        assert!(changed.is_some());
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn recovery_requires_both_ram_and_cpu_to_clear_even_if_only_one_caused_entry() {
        // Entered pressure via CPU alone (RAM stays healthy throughout).
        // RAM recovering instantly must not matter — CPU must independently
        // finish its own cooldown too, since CPU is what's still saturated.
        let mut tracker = PressureTracker::default();
        tracker.observe(8000, Some(95.0), t(0));
        tracker.observe(8000, Some(95.0), t(10));
        assert_eq!(tracker.state(), PressureState::UnderPressure);

        // RAM was always healthy (8000MB), so its own signal "recovers"
        // immediately — but CPU is still pegged, so overall state must stay
        // UnderPressure.
        let changed = tracker.observe(8000, Some(95.0), t(20 + 90));
        assert!(changed.is_none(), "must not restore normal mode while CPU is still saturated");
        assert_eq!(tracker.state(), PressureState::UnderPressure);

        // Now CPU also recovers and completes its own cooldown.
        tracker.observe(8000, Some(10.0), t(300));
        let changed = tracker.observe(8000, Some(10.0), t(300 + 90));
        assert!(changed.is_some());
        assert_eq!(tracker.state(), PressureState::Normal);
    }

    #[test]
    fn either_signal_alone_sustaining_is_enough_to_enter_pressure() {
        // RAM low, CPU healthy — must still enter pressure (RAM-only case,
        // regression guard that adding CPU didn't break the original
        // RAM-only trigger path).
        let mut tracker = PressureTracker::default();
        tracker.observe(500, Some(10.0), t(0));
        let changed = tracker.observe(400, Some(10.0), t(10));
        assert!(changed.is_some());
        assert_eq!(tracker.state(), PressureState::UnderPressure);
    }

    // -- applying pressure to config: STT/RAG-retrieval only, never embed ---

    fn sample_high_performance_config() -> PerformanceConfig {
        super::super::manager::config_for_tier(HardwareTier::HighPerformance, 16)
    }

    #[test]
    fn normal_state_passes_the_base_config_through_unchanged() {
        let tracker = PressureTracker::default();
        let base = sample_high_performance_config();
        let applied = tracker.apply(base.clone());
        assert_eq!(applied.stt_num_threads, base.stt_num_threads);
        assert_eq!(applied.rag_embed_batch_size, base.rag_embed_batch_size);
        assert_eq!(applied.rag_torch_threads, base.rag_torch_threads);
    }

    #[test]
    fn under_pressure_clamps_stt_threads_and_retrieval_budget_to_entry_tier() {
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));
        assert_eq!(tracker.state(), PressureState::UnderPressure);

        let base = sample_high_performance_config(); // stt_num_threads=4, rag_top_k=5, etc.
        let applied = tracker.apply(base);
        let entry = super::super::manager::config_for_tier(HardwareTier::Entry, 1);

        assert_eq!(applied.stt_num_threads, entry.stt_num_threads);
        assert_eq!(applied.rag_top_k, entry.rag_top_k);
        assert_eq!(applied.rag_max_context_chars, entry.rag_max_context_chars);
        assert_eq!(applied.rag_retrieval_timeout_ms, entry.rag_retrieval_timeout_ms);
    }

    #[test]
    fn under_pressure_never_touches_rag_embed_batch_size_or_torch_threads() {
        // Regression guard for the explicit product decision: pressure
        // adaptation must never imply a RAG process restart.
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));

        let base = sample_high_performance_config();
        let applied = tracker.apply(base.clone());
        assert_eq!(applied.rag_embed_batch_size, base.rag_embed_batch_size);
        assert_eq!(applied.rag_torch_threads, base.rag_torch_threads);
    }

    #[test]
    fn under_pressure_never_increases_a_config_that_was_already_more_conservative_than_entry() {
        // If the base config (e.g. the user is already on Battery Saver /
        // Entry tier) is already at or below Entry's values, applying
        // pressure must not raise it back up — `.min`/`.max` as used in
        // `apply` are a ceiling/floor, not a forced overwrite.
        let mut tracker = PressureTracker::default();
        tracker.observe(500, None, t(0));
        tracker.observe(500, None, t(10));

        let already_minimal = super::super::manager::config_for_tier(HardwareTier::Entry, 1);
        let applied = tracker.apply(already_minimal.clone());
        assert_eq!(applied.stt_num_threads, already_minimal.stt_num_threads);
        assert_eq!(applied.rag_top_k, already_minimal.rag_top_k);
    }
}
