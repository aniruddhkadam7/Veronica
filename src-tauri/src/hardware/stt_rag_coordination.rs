//! STT/RAG scheduling coordination (Phase B of the STT low-end optimization
//! pass — see docs/stt-performance-phase2.md).
//!
//! Goal: while STT is actively running on Entry/Standard-tier hardware (see
//! `PerformanceManager::should_throttle_background_work`), signal the RAG
//! service to yield its CPU-heavy embedding step so it doesn't compete with
//! real-time transcription. Retrieval (`/search`) is never affected — only
//! the RAG-side embedding step checks the signal (see
//! `packages/rag/app/throttle.py`).
//!
//! Two independent STT sessions can be active concurrently in this app —
//! the main recording session (`AppState.capture`, system audio + mic) and
//! Notes' separate dictation session (`AppState.notes_dictation`) — so this
//! is reference-counted, not a boolean: the throttle stays active as long
//! as at least one session is running, and only clears once the last one
//! stops. The counting decision (`RefCount`) is a plain, dependency-free
//! struct kept separate from the Tauri/async plumbing below specifically so
//! it can be unit-tested directly (see the `tests` module) without needing
//! a real Tauri app instance or async runtime.
//!
//! No new dependency: uses only `std::sync::atomic` + the `tokio` "time"
//! feature already present (no `tokio::sync` needed) — the refresh loop
//! polls a stop flag once per sleep interval rather than using a channel,
//! which bounds the shutdown-notice delay to one `REFRESH_INTERVAL` (3s),
//! comfortably under the RAG-side throttle's own 10s TTL either way.
//!
//! Safety against abnormal termination (a crash mid-recording that skips
//! `stop_audio_capture` entirely): the RAG-side throttle has its own TTL
//! and auto-expires if never refreshed (see throttle.py) — this module's
//! periodic refresh loop is what keeps it alive during a real session, so
//! if the refresh loop itself dies (process crash) the RAG side recovers on
//! its own within that TTL, never requiring an explicit "STT ended"
//! notification to have fired.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{AppHandle, Manager};

use crate::rag::RagClient;

/// How often to refresh the RAG-side throttle while at least one STT
/// session is active, and the polling granularity for noticing "stop was
/// requested". Must be comfortably shorter than throttle.py's `_TTL_S`
/// (10s) so a normal scheduling delay here never causes a spurious
/// auto-expiry mid-session.
const REFRESH_INTERVAL: Duration = Duration::from_secs(3);

/// Pure reference-counting decision: how many concurrent STT sessions are
/// currently asking for the RAG throttle to stay active, and whether the
/// caller of `begin()`/`end()` was the one responsible for starting/
/// stopping the underlying refresh task. Never negative — `end()` on an
/// already-zero count is a documented no-op, matching
/// "on_stt_session_ended is safe to call even if on_stt_session_started
/// was never called for this session."
#[derive(Debug, Default, PartialEq, Eq)]
struct RefCount(usize);

impl RefCount {
    /// Returns `true` if this call brought the count from 0 to 1 — i.e. the
    /// caller is responsible for spawning the refresh task.
    fn begin(&mut self) -> bool {
        self.0 += 1;
        self.0 == 1
    }

    /// Returns `true` if this call brought the count from 1 to 0 — i.e. the
    /// caller is responsible for stopping the refresh task. Returns `false`
    /// (and leaves the count at 0) if it was already 0 — never underflows.
    fn end(&mut self) -> bool {
        if self.0 == 0 {
            return false;
        }
        self.0 -= 1;
        self.0 == 0
    }
}

struct CoordinationState {
    count: RefCount,
    stop_flag: Option<Arc<AtomicBool>>,
}

impl Default for CoordinationState {
    fn default() -> Self {
        Self { count: RefCount::default(), stop_flag: None }
    }
}

/// Tauri-managed state — registered via `.manage(...)` in `lib.rs`, mirroring
/// `PerformanceState`'s pattern.
pub struct SttRagCoordination(Mutex<CoordinationState>);

impl Default for SttRagCoordination {
    fn default() -> Self {
        Self(Mutex::new(CoordinationState::default()))
    }
}

/// Call when an STT session starts (system audio, mic capture, or Notes
/// dictation) — after the sidecar has been spawned, before returning to the
/// caller. Cheap and safe to call even when throttling won't apply (checks
/// `should_throttle_background_work()` internally and no-ops if not).
pub fn on_stt_session_started(app: &AppHandle) {
    if !crate::hardware::should_throttle_background_work(app) {
        return;
    }

    let coordination = app.state::<SttRagCoordination>();
    let mut guard = coordination.0.lock().unwrap();
    let is_first_session = guard.count.begin();

    if is_first_session {
        // First session to start: spawn the refresh loop. Subsequent
        // concurrent sessions just increment the count (handled by
        // `begin()` above) — the existing loop already keeps the throttle
        // alive for all of them.
        let stop_flag = Arc::new(AtomicBool::new(false));
        guard.stop_flag = Some(stop_flag.clone());

        tauri::async_runtime::spawn(async move {
            let client = RagClient::new();
            if let Err(err) = client.set_throttle(true).await {
                log::warn!("stt_rag_coordination: failed to activate RAG throttle: {err}");
            }

            while !stop_flag.load(Ordering::Relaxed) {
                tokio::time::sleep(REFRESH_INTERVAL).await;
                if stop_flag.load(Ordering::Relaxed) {
                    break;
                }
                if let Err(err) = client.set_throttle(true).await {
                    log::warn!("stt_rag_coordination: failed to refresh RAG throttle: {err}");
                }
            }

            // Last session ended: send one final deactivate — best-effort,
            // same as the activate/refresh calls above; if the RAG service
            // is unreachable the TTL-based auto-expiry on its side still
            // recovers within a few seconds regardless.
            if let Err(err) = client.set_throttle(false).await {
                log::warn!("stt_rag_coordination: failed to deactivate RAG throttle: {err}");
            }
        });
    }
}

/// Call when an STT session ends — on clean stop (`stop_audio_capture`,
/// `stop_note_dictation`) only; abnormal termination is handled by the
/// RAG-side TTL, not by this function (see module doc). Safe to call even
/// if `on_stt_session_started` was never called for this session (e.g.
/// throttling wasn't applicable on this hardware tier) — becomes a no-op.
pub fn on_stt_session_ended(app: &AppHandle) {
    let coordination = app.state::<SttRagCoordination>();
    let mut guard = coordination.0.lock().unwrap();

    let was_last_session = guard.count.end();
    if was_last_session {
        if let Some(stop_flag) = guard.stop_flag.take() {
            stop_flag.store(true, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- RefCount: pure logic, no Tauri/async runtime needed ----------------

    #[test]
    fn single_session_begin_reports_first_and_end_reports_last() {
        let mut count = RefCount::default();
        assert!(count.begin(), "the only session starting must be reported as first");
        assert!(count.end(), "the only session ending must be reported as last");
    }

    #[test]
    fn second_concurrent_session_is_not_reported_as_first() {
        let mut count = RefCount::default();
        assert!(count.begin());
        assert!(!count.begin(), "a second concurrent session must not re-trigger 'first session' setup");
    }

    #[test]
    fn ending_one_of_two_sessions_is_not_reported_as_last() {
        let mut count = RefCount::default();
        count.begin();
        count.begin();
        assert!(!count.end(), "one of two sessions ending must not tear down the shared refresh task");
        assert!(count.end(), "the second (final) session ending must be reported as last");
    }

    #[test]
    fn end_on_an_already_zero_count_is_a_safe_no_op() {
        let mut count = RefCount::default();
        assert!(!count.end(), "ending with nothing active must not report 'last' or panic");
        assert_eq!(count, RefCount(0), "count must never go negative");
    }

    #[test]
    fn repeated_sessions_each_correctly_report_first_and_last() {
        // Mirrors the brief's explicit "repeated STT sessions" test case —
        // start/stop/start/stop/start/stop must behave identically each
        // time, not leak state between cycles.
        let mut count = RefCount::default();
        for _ in 0..3 {
            assert!(count.begin(), "each fresh cycle's session must be reported as first");
            assert!(count.end(), "each fresh cycle's session must be reported as last");
        }
    }

    #[test]
    fn three_overlapping_sessions_only_first_and_last_are_flagged() {
        let mut count = RefCount::default();
        assert!(count.begin()); // session A starts -> first
        assert!(!count.begin()); // session B starts -> not first
        assert!(!count.begin()); // session C starts -> not first
        assert!(!count.end()); // session B ends -> not last (A, C still running)
        assert!(!count.end()); // session A ends -> not last (C still running)
        assert!(count.end()); // session C ends -> last
    }

    #[test]
    fn stt_failure_mid_session_is_still_a_normal_end_call() {
        // "STT failure/termination" from the brief: as far as this module
        // is concerned, an abnormal stop that still reaches
        // on_stt_session_ended (e.g. an error path that cleans up handles
        // before returning) behaves identically to a normal stop — the
        // TRULY abnormal case (process crash, on_stt_session_ended never
        // called at all) is handled by the RAG-side TTL, not by this logic
        // (see module doc) — there is nothing for RefCount itself to do
        // differently for a "failure" vs a "clean stop", which is itself
        // the point: one code path handles both.
        let mut count = RefCount::default();
        count.begin();
        assert!(count.end(), "a session ending via an error path must still correctly report 'last'");
    }
}
