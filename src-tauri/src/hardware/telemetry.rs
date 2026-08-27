//! Milestone 7: lightweight end-to-end latency instrumentation for the
//! STT -> RAG retrieval -> cloud LLM -> answer pipeline.
//!
//! Scope and safety rules (all deliberate, not oversights):
//!
//! - **Timing only.** Every value logged here is a duration in milliseconds,
//!   a stage name, a performance-tier/mode enum, or a small numeric count
//!   (e.g. answer length in characters, used only to note that a timing
//!   number exists — never the text itself). Nothing here ever logs a
//!   question, an answer, a transcript segment, retrieved chunk text, a
//!   filename, an API key, a JWT, or any request/response body. Call sites
//!   pass `&str`/text values into this module's `stage` labels only as
//!   fixed string literals chosen by the caller (e.g. `"stt_final"`), never
//!   as user content — there is no code path here that could accidentally
//!   log content because the API only accepts `Duration`s and enum/string
//!   *labels*, not arbitrary user data.
//! - **No new persistence.** Everything here goes through the existing
//!   `log` crate (the same sink STT/RAG/backend code already logs through)
//!   — no new file, no new database table, no new telemetry queue. This is
//!   strictly for a developer/support person reading Smallbird's existing log
//!   output, not a new data-collection feature.
//! - **Lightweight.** One `log::info!` call per pipeline stage boundary,
//!   gated behind the existing `log`/`env_logger` setup (silent at the
//!   default log level, same as the rest of the app's `log::info!` calls) —
//!   no per-sample/per-chunk logging, no background aggregation thread, no
//!   metrics server.
//! - **Does not change behavior.** This module only measures and logs; it
//!   never influences a decision (that stays entirely in
//!   `hardware::manager`/`hardware::pressure`, unchanged by this milestone).

use std::time::{Duration, Instant};

use super::manager::{PerformanceConfig, PerformanceMode};
use super::pressure::PressureState;
use super::tier::HardwareTier;

/// A single pipeline-stage duration, ready to log. Constructed via
/// `Stopwatch::stop`, never by hand with an arbitrary label — this keeps
/// every stage name one of the fixed set below rather than a caller-chosen
/// string that could accidentally carry content.
pub struct StageTiming {
    pub stage: PipelineStage,
    pub duration: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStage {
    /// Start Interview click -> WASAPI loopback capture confirmed
    /// initialized and streaming (not merely "capture thread spawned" —
    /// see `audio::system_capture`'s readiness handshake).
    AudioCaptureReady,
    /// Start Interview click -> STT sidecar signaled `{"type":"ready"}`
    /// (model loaded, accepting audio) — the number that actually gates
    /// when the UI is allowed to report "Recording" (see
    /// `commands::start_system_audio_capture`).
    SttReady,
    /// Audio capture start -> STT sidecar process spawned and ready. Kept
    /// as the umbrella "total startup" measurement alongside the more
    /// granular `AudioCaptureReady`/`SttSpawned`/`SttReady` breakdown above.
    SttSessionStart,
    /// Speech start -> first partial transcript text.
    SttFirstPartial,
    /// Speech end -> finalized transcript segment.
    SttFinal,
    /// One `RetrievalPlanner::plan_for_question` call: the RAG HTTP round
    /// trip plus local filter/dedup/truncate — see `retrieval_planner.rs`'s
    /// `filter_and_limit`, which is in-process and fast enough (sub-ms at
    /// realistic result counts) that splitting it out as its own separately
    /// logged number would not be a meaningful additional measurement, so
    /// it is folded into this one stage rather than invented as a second
    /// one that would just restate "basically zero."
    RagRetrieval,
    /// Cloud LLM request dispatched -> first streamed answer token/delta
    /// received (time-to-first-token).
    LlmFirstToken,
    /// Cloud LLM request dispatched -> stream complete (full answer).
    LlmTotal,
    /// User's question submitted -> final answer complete: the number the
    /// user actually experiences end to end for one "ask" interaction.
    QuestionToAnswer,
}

impl PipelineStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::AudioCaptureReady => "audio_capture_ready",
            Self::SttReady => "stt_ready",
            Self::SttSessionStart => "stt_session_start",
            Self::SttFirstPartial => "stt_first_partial",
            Self::SttFinal => "stt_final",
            Self::RagRetrieval => "rag_retrieval",
            Self::LlmFirstToken => "llm_first_token",
            Self::LlmTotal => "llm_total",
            Self::QuestionToAnswer => "question_to_answer",
        }
    }
}

/// Starts timing one pipeline stage. `Instant`-based (monotonic), never
/// wall-clock — matches `hardware::pressure`'s existing choice for the same
/// reason (immune to system clock adjustments).
pub struct Stopwatch {
    started: Instant,
}

impl Stopwatch {
    pub fn start() -> Self {
        Self { started: Instant::now() }
    }

    pub fn stop(self, stage: PipelineStage) -> StageTiming {
        StageTiming { stage, duration: self.started.elapsed() }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

/// Captures a time-to-first-token measurement out of a streaming callback
/// that the caller (e.g. `BackendClient::ask_stream`) owns and never hands
/// back — the callback closure is moved into the streaming call and only
/// its side effects (here, one shared-cell write) are observable
/// afterward. Wraps the `Arc<Mutex<Option<u128>>>` pattern so each of the
/// four live-answer call sites (Interview/Sales/Consulting/Agent-ask)
/// doesn't hand-roll it.
///
/// ```ignore
/// let first_token = FirstTokenTracker::new();
/// let recorder = first_token.recorder();
/// client.ask_stream(&request, move |delta| {
///     recorder.mark();
///     // ... forward delta ...
/// }).await?;
/// if let Some(ms) = first_token.elapsed_ms() { /* log it */ }
/// ```
#[derive(Clone)]
pub struct FirstTokenTracker {
    started: Instant,
    ms: std::sync::Arc<std::sync::Mutex<Option<u128>>>,
}

impl FirstTokenTracker {
    pub fn new() -> Self {
        Self { started: Instant::now(), ms: std::sync::Arc::new(std::sync::Mutex::new(None)) }
    }

    /// A cheap `Clone` to move into the streaming closure — call `.mark()`
    /// on the first delta only (subsequent calls are no-ops).
    pub fn recorder(&self) -> Self {
        self.clone()
    }

    /// Records elapsed time since `new()`, but only on the first call —
    /// safe to call on every delta without re-locking-and-overwriting.
    pub fn mark(&self) {
        let mut slot = self.ms.lock().unwrap();
        if slot.is_none() {
            *slot = Some(self.started.elapsed().as_millis());
        }
    }

    /// `None` if `mark()` was never called (e.g. the stream produced no
    /// deltas before erroring out).
    pub fn elapsed_ms(&self) -> Option<u128> {
        *self.ms.lock().unwrap()
    }
}

impl Default for FirstTokenTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// The performance-correlation context to attach to a logged timing —
/// exactly the fields the product brief asked for (tier, mode, pressure
/// state, STT thread count, RAG retrieval config), nothing else. All of
/// these are either enums or small numbers, never anything derived from
/// user content.
#[derive(Debug, Clone, Copy)]
pub struct PerfContext {
    pub tier: HardwareTier,
    pub mode: PerformanceMode,
    pub pressure: PressureState,
    pub stt_num_threads: u32,
    pub rag_top_k: u32,
    pub rag_max_context_chars: usize,
}

impl PerfContext {
    pub fn new(tier: HardwareTier, mode: PerformanceMode, pressure: PressureState, config: &PerformanceConfig) -> Self {
        Self {
            tier,
            mode,
            pressure,
            stt_num_threads: config.stt_num_threads,
            rag_top_k: config.rag_top_k,
            rag_max_context_chars: config.rag_max_context_chars,
        }
    }
}

/// Logs one stage timing at `info` level with its performance context, in a
/// single structured line — safe to leave enabled by default (no per-sample
/// spam: one line per pipeline stage boundary, the same cadence as the
/// existing `log::info!("STT sidecar ready ...")`-style lines elsewhere in
/// this codebase).
pub fn log_stage(timing: &StageTiming, ctx: &PerfContext) {
    log_stage_ms(timing.stage, timing.duration.as_millis(), ctx);
}

/// Same as `log_stage`, for the case where the elapsed time was already
/// computed as a plain duration/millis value rather than via a live
/// `Stopwatch` — e.g. time-to-first-token, captured inside a synchronous
/// streaming callback where holding onto a `Stopwatch` across a `move`
/// closure is awkward; callers there capture just the starting `Instant`
/// and compute `.elapsed()` themselves.
pub fn log_stage_ms(stage: PipelineStage, ms: u128, ctx: &PerfContext) {
    log::info!(
        "perf: stage={} ms={} tier={:?} mode={:?} pressure={:?} stt_threads={} rag_top_k={} rag_ctx_chars={}",
        stage.label(),
        ms,
        ctx.tier,
        ctx.mode,
        ctx.pressure,
        ctx.stt_num_threads,
        ctx.rag_top_k,
        ctx.rag_max_context_chars,
    );
}

/// Convenience for the common case: stop a stopwatch, log it, and return
/// the elapsed duration in case the caller wants to fold it into a
/// higher-level stage (e.g. one leg of `QuestionToAnswer`).
pub fn finish(stopwatch: Stopwatch, stage: PipelineStage, ctx: &PerfContext) -> Duration {
    let timing = stopwatch.stop(stage);
    let elapsed = timing.duration;
    log_stage(&timing, ctx);
    elapsed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::manager::PerformanceConfig;

    fn sample_config() -> PerformanceConfig {
        PerformanceConfig {
            stt_num_threads: 4,
            rag_top_k: 4,
            rag_max_context_chars: 3500,
            rag_similarity_threshold: 0.3,
            rag_retrieval_timeout_ms: 1500,
            rag_embed_batch_size: 32,
            rag_torch_threads: 4,
        }
    }

    #[test]
    fn stopwatch_measures_a_nonzero_elapsed_duration() {
        let sw = Stopwatch::start();
        std::thread::sleep(Duration::from_millis(5));
        let timing = sw.stop(PipelineStage::SttFinal);
        assert!(timing.duration >= Duration::from_millis(5));
        assert_eq!(timing.stage, PipelineStage::SttFinal);
    }

    #[test]
    fn every_pipeline_stage_has_a_fixed_non_empty_label() {
        // Guards against a future stage variant being added without a
        // corresponding label — `label()`'s match is exhaustive, so this
        // mostly documents the full set for readers of this test file.
        let stages = [
            PipelineStage::AudioCaptureReady,
            PipelineStage::SttReady,
            PipelineStage::SttSessionStart,
            PipelineStage::SttFirstPartial,
            PipelineStage::SttFinal,
            PipelineStage::RagRetrieval,
            PipelineStage::LlmFirstToken,
            PipelineStage::LlmTotal,
            PipelineStage::QuestionToAnswer,
        ];
        for stage in stages {
            assert!(!stage.label().is_empty());
            assert!(stage.label().chars().all(|c| c.is_ascii_lowercase() || c == '_'));
        }
    }

    #[test]
    fn perf_context_carries_exactly_the_requested_correlation_fields() {
        let ctx = PerfContext::new(
            HardwareTier::HighPerformance,
            PerformanceMode::Adaptive,
            PressureState::Normal,
            &sample_config(),
        );
        assert_eq!(ctx.tier, HardwareTier::HighPerformance);
        assert_eq!(ctx.mode, PerformanceMode::Adaptive);
        assert_eq!(ctx.pressure, PressureState::Normal);
        assert_eq!(ctx.stt_num_threads, 4);
        assert_eq!(ctx.rag_top_k, 4);
        assert_eq!(ctx.rag_max_context_chars, 3500);
    }

    #[test]
    fn finish_returns_the_same_duration_it_logs() {
        let ctx = PerfContext::new(HardwareTier::Entry, PerformanceMode::BatterySaver, PressureState::Normal, &sample_config());
        let sw = Stopwatch::start();
        std::thread::sleep(Duration::from_millis(2));
        let elapsed = finish(sw, PipelineStage::RagRetrieval, &ctx);
        assert!(elapsed >= Duration::from_millis(2));
    }

    // -- FirstTokenTracker: mirrors how a streaming ask_stream callback uses it

    #[test]
    fn first_token_tracker_has_no_elapsed_time_before_any_mark() {
        let tracker = FirstTokenTracker::new();
        assert!(tracker.elapsed_ms().is_none());
    }

    #[test]
    fn first_token_tracker_records_elapsed_time_on_first_mark() {
        let tracker = FirstTokenTracker::new();
        std::thread::sleep(Duration::from_millis(5));
        tracker.mark();
        let ms = tracker.elapsed_ms().expect("mark() should have recorded a value");
        assert!(ms >= 5);
    }

    #[test]
    fn first_token_tracker_ignores_marks_after_the_first() {
        // Mirrors a streaming callback invoked once per delta — only the
        // FIRST delta's arrival time is the time-to-first-token number;
        // later deltas must not overwrite it.
        let tracker = FirstTokenTracker::new();
        tracker.mark();
        let first_reading = tracker.elapsed_ms().unwrap();
        std::thread::sleep(Duration::from_millis(20));
        tracker.mark(); // second delta arrives much later
        let still_reading = tracker.elapsed_ms().unwrap();
        assert_eq!(first_reading, still_reading, "a later mark() must not overwrite the first");
    }

    #[test]
    fn first_token_tracker_recorder_clones_share_the_same_underlying_cell() {
        // This is exactly the pattern the ask_stream call sites use: a
        // `.recorder()` clone is moved into the streaming closure, and the
        // original `first_token` handle (kept outside the closure) must see
        // the mark it records.
        let tracker = FirstTokenTracker::new();
        let recorder = tracker.recorder();
        recorder.mark();
        assert!(tracker.elapsed_ms().is_some());
    }

    #[test]
    fn first_token_tracker_default_starts_with_no_elapsed_time() {
        let tracker = FirstTokenTracker::default();
        assert!(tracker.elapsed_ms().is_none());
    }

    // -- composed pipeline flow: the exact pattern the four ask-question call
    // sites (Interview/Sales/Consulting/Agent-ask) use around
    // BackendClient::ask_stream, with a fake streaming closure standing in
    // for the real network call — no real cloud API request happens here,
    // so this test cannot be flaky on network/API latency. This is the
    // reusable unit under test; the four call sites themselves are thin,
    // mechanical wiring around it, verified by cargo check plus a live
    // smoke test rather than duplicated here as four near-identical
    // integration tests.

    /// Simulates `BackendClient::ask_stream`'s shape (owns the closure,
    /// calls it once per "delta", the caller never gets the closure back)
    /// without any real HTTP — a fixed delay before the first delta and a
    /// fixed delay before the stream "completes", both short so the test
    /// suite stays fast.
    fn fake_ask_stream<F: FnMut(&str)>(mut on_delta: F) -> String {
        std::thread::sleep(Duration::from_millis(3)); // stand-in for network + LLM latency to first token
        on_delta("Hello");
        std::thread::sleep(Duration::from_millis(2)); // stand-in for the rest of the stream
        on_delta(" world");
        "Hello world".to_string()
    }

    #[test]
    fn composed_question_to_answer_flow_measures_retrieval_first_token_and_total_correctly() {
        let ctx = PerfContext::new(HardwareTier::Performance, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());

        let question_to_answer = Stopwatch::start();

        // Simulated RAG retrieval leg.
        let retrieval_timer = Stopwatch::start();
        std::thread::sleep(Duration::from_millis(4));
        let retrieval_ms = finish(retrieval_timer, PipelineStage::RagRetrieval, &ctx).as_millis();
        assert!(retrieval_ms >= 4);

        // Simulated LLM leg — mirrors the real call sites exactly.
        let llm_timer = Stopwatch::start();
        let first_token = FirstTokenTracker::new();
        let recorder = first_token.recorder();
        let answer = fake_ask_stream(move |_delta| {
            recorder.mark();
        });
        assert_eq!(answer, "Hello world"); // the fake "network call" still completed normally

        let first_token_ms = first_token.elapsed_ms().expect("first delta should have been marked");
        assert!(first_token_ms >= 3, "first token should reflect the delay before the first delta");

        let llm_total_ms = finish(llm_timer, PipelineStage::LlmTotal, &ctx).as_millis();
        assert!(llm_total_ms >= 5, "total should reflect delay-to-first-delta plus delay-to-completion");
        assert!(llm_total_ms >= first_token_ms, "total must never be less than time-to-first-token");

        let question_to_answer_ms = finish(question_to_answer, PipelineStage::QuestionToAnswer, &ctx).as_millis();
        assert!(
            question_to_answer_ms >= retrieval_ms + llm_total_ms.min(question_to_answer_ms),
            "end-to-end must be at least as long as its slowest known sub-leg"
        );
    }

    #[test]
    fn a_stream_with_no_deltas_before_erroring_leaves_first_token_unset_but_does_not_panic() {
        // Mirrors ask_stream returning an Err before any delta arrived
        // (e.g. connection refused) — first_token.elapsed_ms() must stay
        // None, and nothing here should panic on an absent value.
        let ctx = PerfContext::new(HardwareTier::Entry, PerformanceMode::BatterySaver, PressureState::Normal, &sample_config());
        let first_token = FirstTokenTracker::new();
        // No .mark() call at all — simulates zero deltas received.
        assert!(first_token.elapsed_ms().is_none());
        // The real call sites guard this with `if let Some(ms) = ...` —
        // confirm that pattern is exactly what a "no deltas" case hits.
        if let Some(ms) = first_token.elapsed_ms() {
            log_stage_ms(PipelineStage::LlmFirstToken, ms, &ctx);
            panic!("should not have reached here — no mark() was called");
        }
    }

    // -- content-safety guard: the logging API cannot accept arbitrary text --

    #[test]
    fn log_stage_api_only_accepts_fixed_enum_labels_and_numeric_durations() {
        // Not a runtime assertion (there is nothing to assert at runtime —
        // this is a compile-time property) but a documented regression
        // guard: if this test still compiles, `log_stage`/`log_stage_ms`'s
        // signatures still take only `PipelineStage` (a closed enum),
        // `Duration`/`u128`, and `PerfContext` (enums + small numbers) —
        // there is no `&str`/`String` parameter a future edit could
        // accidentally wire a question/answer/transcript into. If someone
        // adds a free-text parameter to either function, this comment (not
        // a compiler error) is the only thing that will catch it, so the
        // call-site audit in docs/performance-tuning.md's Milestone 7
        // section is the real guard — this test just documents the intent
        // at the type level.
        let ctx = PerfContext::new(HardwareTier::Standard, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());
        log_stage_ms(PipelineStage::LlmFirstToken, 42, &ctx);
        let sw = Stopwatch::start();
        log_stage(&sw.stop(PipelineStage::LlmTotal), &ctx);
    }
}
