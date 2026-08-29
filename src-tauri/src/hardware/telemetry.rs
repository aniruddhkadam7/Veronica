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

    // -- Full per-turn voice-pipeline stages (see `TurnTelemetry`). These are
    // additive: none of the stages above are removed or renumbered, so any
    // existing call site keeps working unchanged.
    /// First audio chunk observed from WASAPI capture this turn.
    MicDetected,
    /// Local VAD engine's first `partial` line after a silence/session
    /// boundary — the moment speech is judged to have started.
    SpeechStarted,
    /// Local VAD engine's `final` line — the moment speech is judged to have
    /// ended (this is the real end-of-utterance signal; nothing downstream
    /// waits on a fixed timer after this).
    SpeechEnded,
    /// The Groq transcription HTTP request for this utterance is dispatched.
    SttStarted,
    /// First result available from the STT provider. For Groq (a batch
    /// endpoint — see `stt::groq`'s module doc) this is always equal to
    /// `SttFinal`, logged separately anyway so the two are never conflated
    /// and a future streaming STT provider (see that module's isolated
    /// interface) has somewhere real to report an earlier value.
    SttFirstResult,
    /// The fast router (`actions::fast_router`) begins matching the final
    /// transcript against the deterministic capability table.
    RouterStarted,
    /// The fast router has decided: matched (a capability, executed with no
    /// LLM call) or fell through to the agent loop.
    RouterDecision,
    /// The agent loop's first LLM request for this turn is dispatched
    /// (distinct from `LlmFirstToken`/`LlmTotal`, which measure one single
    /// provider call — a multi-step agent turn can dispatch several).
    LlmStarted,
    /// The agent loop has produced its final answer text (after any tool
    /// calls have all been executed and observed).
    LlmComplete,
    /// The TTS session's first `Speak` for this turn is sent to Flux.
    TtsStarted,
    /// First raw PCM audio byte received back from Flux for this turn.
    TtsFirstAudio,
    /// First PCM chunk appended to the playback sink for this turn (audio is
    /// now actually reaching the speakers).
    PlaybackStarted,
    /// The whole turn is done — either the fast-router path's confirmation
    /// finished playing, or the agent loop + TTS both settled.
    TurnComplete,
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
            Self::MicDetected => "mic_detected",
            Self::SpeechStarted => "speech_started",
            Self::SpeechEnded => "speech_ended",
            Self::SttStarted => "stt_started",
            Self::SttFirstResult => "stt_first_result",
            Self::RouterStarted => "router_started",
            Self::RouterDecision => "router_decision",
            Self::LlmStarted => "llm_started",
            Self::LlmComplete => "llm_complete",
            Self::TtsStarted => "tts_started",
            Self::TtsFirstAudio => "tts_first_audio",
            Self::PlaybackStarted => "playback_started",
            Self::TurnComplete => "turn_complete",
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

/// One full voice turn's worth of stage timestamps (mic detected through
/// turn complete — see `PipelineStage`'s new variants), and the six derived
/// latency numbers requested for observability: speech -> STT final, STT
/// final -> router decision, STT final -> LLM first token, LLM first token
/// -> TTS first audio, speech end -> first Veronica audio, and total turn
/// latency. `mark()` is idempotent per stage (first call wins, later calls
/// for the same stage are ignored) so a call site can call it defensively
/// without worrying about double-marking skewing a duration — mirrors
/// `FirstTokenTracker`'s existing "first delta only" behavior above.
///
/// Every timestamp is `Instant`-based (monotonic), matching this module's
/// existing rule. `finish()` logs one summary line with every delta that had
/// both of its endpoints marked — a turn that took the fast-router path
/// (no LLM/TTS stages) still gets a useful summary with just its stages,
/// rather than a line full of "n/a".
pub struct TurnTelemetry {
    /// Correlates every `[TAG] turn_id=... ` log line this turn produces
    /// (see `veronica::ask_veronica`, `personal::agent::orchestrator`) back
    /// to one real conversational turn — required for requirement 13's
    /// logging (and for making sense of overlapping turns: a barge-in means
    /// two `TurnTelemetry`s can legitimately be alive at once, briefly).
    id: String,
    stages: std::sync::Mutex<std::collections::HashMap<&'static str, Instant>>,
}

/// Generates a turn id unique within this process run — a monotonic counter
/// suffixed onto the current time, not a full UUID dependency, matching this
/// crate's existing `transcript::mod`'s `uuid_like_id` convention for the
/// same "good enough to grep for, not globally unique across machines"
/// need.
fn new_turn_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    format!("t-{now_ms:x}-{n:x}")
}

impl TurnTelemetry {
    pub fn new() -> Self {
        Self { id: new_turn_id(), stages: std::sync::Mutex::new(std::collections::HashMap::new()) }
    }

    /// This turn's id — include it in every log line for this turn (see
    /// `veronica::ask_veronica`'s `[TURN]`/`[STATE]`/`[ERROR]` logging and
    /// `personal::agent::orchestrator`'s `[LLM_*]` logging) so the whole
    /// lifecycle of one turn can be grepped out of interleaved logs from
    /// concurrent/overlapping turns (a barge-in briefly has two).
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Records `stage` as having happened right now, unless it was already
    /// marked earlier in this turn.
    pub fn mark(&self, stage: PipelineStage) {
        let mut guard = self.stages.lock().unwrap();
        guard.entry(stage.label()).or_insert_with(Instant::now);
    }

    fn get(&self, stage: PipelineStage) -> Option<Instant> {
        self.stages.lock().unwrap().get(stage.label()).copied()
    }

    fn delta_ms(&self, from: PipelineStage, to: PipelineStage) -> Option<i64> {
        let (from, to) = (self.get(from)?, self.get(to)?);
        Some(to.saturating_duration_since(from).as_millis() as i64)
    }

    /// Logs every per-stage line (via `log_stage_ms`, for whichever stages
    /// were actually marked this turn) plus one summary line with the six
    /// requested end-to-end deltas — call once, when the turn is fully done
    /// (audio finished playing, or the fast-router path's confirmation
    /// finished).
    pub fn finish(&self, ctx: &PerfContext) {
        let deltas = self.compute_deltas();
        let ordered = self.ordered_stage_offsets();
        for (label, ms_since_start) in &ordered {
            log::info!("perf: turn_id={} turn_stage={label} ms_since_turn_start={ms_since_start}", self.id);
        }

        log::info!(
            "perf: turn_id={} turn_summary speech_to_stt_final_ms={} stt_final_to_router_decision_ms={} stt_final_to_llm_first_token_ms={} llm_first_token_to_tts_first_audio_ms={} speech_end_to_first_audio_ms={} total_turn_latency_ms={} tier={:?} mode={:?} pressure={:?}",
            self.id,
            fmt_opt(deltas.speech_to_stt_final),
            fmt_opt(deltas.stt_final_to_router_decision),
            fmt_opt(deltas.stt_final_to_llm_first_token),
            fmt_opt(deltas.llm_first_token_to_tts_first_audio),
            fmt_opt(deltas.speech_end_to_first_audio),
            fmt_opt(deltas.total_turn_latency),
            ctx.tier,
            ctx.mode,
            ctx.pressure,
        );
    }

    /// Every stage actually marked this turn, as `(label, ms since the
    /// turn's start marker)` pairs, sorted chronologically. The "start
    /// marker" is `MicDetected`, falling back to `SpeechStarted` — same rule
    /// `finish()` has always used. A stage marked before `start` (should not
    /// happen in practice, but not asserted against) still gets a value via
    /// `saturating_duration_since`, never a negative/underflowed one.
    fn ordered_stage_offsets(&self) -> Vec<(&'static str, u128)> {
        let start = self.get(PipelineStage::MicDetected).or_else(|| self.get(PipelineStage::SpeechStarted));
        let Some(start) = start else { return Vec::new() };
        let stages = self.stages.lock().unwrap();
        let mut ordered: Vec<(&'static str, Instant)> = stages.iter().map(|(k, v)| (*k, *v)).collect();
        drop(stages);
        ordered.sort_by_key(|(_, at)| *at);
        ordered.into_iter().map(|(label, at)| (label, at.saturating_duration_since(start).as_millis())).collect()
    }

    /// The six requested end-to-end deltas, computed once so `finish()` and
    /// `snapshot()` can never disagree on the numbers — each `None` unless
    /// BOTH of its endpoint stages were actually marked this turn.
    fn compute_deltas(&self) -> TurnDeltas {
        TurnDeltas {
            speech_to_stt_final: self.delta_ms(PipelineStage::SpeechEnded, PipelineStage::SttFinal),
            stt_final_to_router_decision: self.delta_ms(PipelineStage::SttFinal, PipelineStage::RouterDecision),
            stt_final_to_llm_first_token: self.delta_ms(PipelineStage::SttFinal, PipelineStage::LlmFirstToken),
            llm_first_token_to_tts_first_audio: self.delta_ms(PipelineStage::LlmFirstToken, PipelineStage::TtsFirstAudio),
            speech_end_to_first_audio: self.delta_ms(PipelineStage::SpeechEnded, PipelineStage::PlaybackStarted),
            total_turn_latency: self.delta_ms(PipelineStage::SpeechEnded, PipelineStage::TurnComplete),
        }
    }

    /// A point-in-time, read-only capture of this turn's stages and derived
    /// deltas — for the latency dashboard's in-memory history. Never logs
    /// (that stays `finish()`'s job) and never mutates `self`, so it is safe
    /// to call from a non-voice-path command handler at any time, including
    /// concurrently with `finish()` on the same turn.
    pub fn snapshot(&self, ctx: &PerfContext) -> TurnSnapshot {
        let deltas = self.compute_deltas();
        let stage_ms_since_start =
            self.ordered_stage_offsets().into_iter().map(|(label, ms)| (label.to_string(), ms as i64)).collect();
        // A turn is "interrupted/incomplete" from the dashboard's point of
        // view exactly when TurnComplete's own end-to-end delta could not be
        // computed (SpeechEnded and/or TurnComplete never got marked) even
        // though this snapshot is being taken — i.e. the turn was recorded
        // without ever reaching a normal, fully-timed completion. This is
        // read directly off already-marked stages, never inferred from a
        // duration or threshold.
        let interrupted = deltas.total_turn_latency.is_none();
        let recorded_at_ms =
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
        TurnSnapshot {
            turn_id: self.id.clone(),
            stage_ms_since_start,
            speech_to_stt_final_ms: deltas.speech_to_stt_final,
            stt_final_to_router_decision_ms: deltas.stt_final_to_router_decision,
            stt_final_to_llm_first_token_ms: deltas.stt_final_to_llm_first_token,
            llm_first_token_to_tts_first_audio_ms: deltas.llm_first_token_to_tts_first_audio,
            speech_end_to_first_audio_ms: deltas.speech_end_to_first_audio,
            total_turn_latency_ms: deltas.total_turn_latency,
            tier: format!("{:?}", ctx.tier),
            mode: format!("{:?}", ctx.mode),
            pressure: format!("{:?}", ctx.pressure),
            recorded_at_ms,
            interrupted,
        }
    }
}

/// The six requested end-to-end latency deltas, each `None` unless both of
/// its endpoint stages were actually marked — shared by `finish()`'s log
/// line and `snapshot()`'s dashboard record so the two can never drift.
struct TurnDeltas {
    speech_to_stt_final: Option<i64>,
    stt_final_to_router_decision: Option<i64>,
    stt_final_to_llm_first_token: Option<i64>,
    llm_first_token_to_tts_first_audio: Option<i64>,
    speech_end_to_first_audio: Option<i64>,
    total_turn_latency: Option<i64>,
}

/// A point-in-time capture of one turn's real, measured telemetry — every
/// field here is either a value read directly off a marked `Instant`
/// (`stage_ms_since_start`), a delta between two such `Instant`s (the six
/// `..._ms` fields, `None` when either endpoint was never marked), or
/// `PerfContext`/id metadata already computed elsewhere. Nothing here is
/// estimated, interpolated, or defaulted to zero.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnSnapshot {
    pub turn_id: String,
    /// `(stage label, milliseconds since this turn's start marker)`,
    /// chronologically sorted, containing ONLY stages that were actually
    /// marked — a stage the pipeline never reached for this turn (e.g. no
    /// LLM call on a fast-router turn) simply does not appear here, rather
    /// than appearing with a fabricated value.
    pub stage_ms_since_start: Vec<(String, i64)>,
    pub speech_to_stt_final_ms: Option<i64>,
    pub stt_final_to_router_decision_ms: Option<i64>,
    pub stt_final_to_llm_first_token_ms: Option<i64>,
    pub llm_first_token_to_tts_first_audio_ms: Option<i64>,
    pub speech_end_to_first_audio_ms: Option<i64>,
    pub total_turn_latency_ms: Option<i64>,
    pub tier: String,
    pub mode: String,
    pub pressure: String,
    /// Wall-clock capture time in ms since the Unix epoch — for ordering
    /// and export display ONLY. Every duration above stays `Instant`-based
    /// (monotonic); this field is never used in any duration computation.
    pub recorded_at_ms: u64,
    /// `true` when this turn never reached a normally-timed completion
    /// (`total_turn_latency_ms` is `None` because `SpeechEnded` and/or
    /// `TurnComplete` was never marked — e.g. a barge-in superseded it).
    /// Dashboard aggregate stats must exclude these turns from their
    /// populations rather than silently averaging in a missing value.
    pub interrupted: bool,
}

/// A bounded, in-memory ring buffer of recent turns' telemetry snapshots for
/// the latency dashboard. Deliberately capped (not unbounded growth) and
/// deliberately NOT persisted to disk — see this module's top-level "No new
/// persistence" rule; this stores the same kind of data `finish()` already
/// logs, just retained in memory long enough for the dashboard to read it.
/// `push`/`clear` are a `Mutex` lock plus a `VecDeque` operation — cheap and
/// non-blocking, safe to call from the voice-turn-completion path.
const HISTORY_CAPACITY: usize = 200;

pub struct TurnHistory {
    turns: std::sync::Mutex<std::collections::VecDeque<TurnSnapshot>>,
}

impl TurnHistory {
    pub fn new() -> Self {
        Self { turns: std::sync::Mutex::new(std::collections::VecDeque::with_capacity(HISTORY_CAPACITY)) }
    }

    /// Records one turn's snapshot, evicting the oldest entry once at
    /// capacity — never grows unbounded.
    pub fn push(&self, snapshot: TurnSnapshot) {
        let mut turns = self.turns.lock().unwrap();
        if turns.len() >= HISTORY_CAPACITY {
            turns.pop_back();
        }
        turns.push_front(snapshot);
    }

    /// All recorded turns, newest first — a plain clone, no computation.
    pub fn snapshot_all(&self) -> Vec<TurnSnapshot> {
        self.turns.lock().unwrap().iter().cloned().collect()
    }

    pub fn clear(&self) {
        self.turns.lock().unwrap().clear();
    }
}

impl Default for TurnHistory {
    fn default() -> Self {
        Self::new()
    }
}

fn fmt_opt(ms: Option<i64>) -> String {
    match ms {
        Some(ms) => ms.to_string(),
        None => "n/a".to_string(),
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
            PipelineStage::MicDetected,
            PipelineStage::SpeechStarted,
            PipelineStage::SpeechEnded,
            PipelineStage::SttStarted,
            PipelineStage::SttFirstResult,
            PipelineStage::RouterStarted,
            PipelineStage::RouterDecision,
            PipelineStage::LlmStarted,
            PipelineStage::LlmComplete,
            PipelineStage::TtsStarted,
            PipelineStage::TtsFirstAudio,
            PipelineStage::PlaybackStarted,
            PipelineStage::TurnComplete,
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

    // -- TurnTelemetry: the six requested per-turn latency deltas --

    #[test]
    fn turn_telemetry_computes_all_six_requested_deltas_when_fully_marked() {
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::MicDetected);
        turn.mark(PipelineStage::SpeechEnded);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::SttFinal);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::RouterDecision);
        turn.mark(PipelineStage::LlmFirstToken);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::TtsFirstAudio);
        turn.mark(PipelineStage::PlaybackStarted);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::TurnComplete);

        assert!(turn.delta_ms(PipelineStage::SpeechEnded, PipelineStage::SttFinal).unwrap() >= 2);
        assert!(turn.delta_ms(PipelineStage::SttFinal, PipelineStage::RouterDecision).unwrap() >= 2);
        assert!(turn.delta_ms(PipelineStage::LlmFirstToken, PipelineStage::TtsFirstAudio).unwrap() >= 2);
        assert!(turn.delta_ms(PipelineStage::SpeechEnded, PipelineStage::TurnComplete).unwrap() >= 6);
    }

    #[test]
    fn turn_telemetry_missing_stage_yields_none_not_a_panic() {
        // A fast-router turn never marks LlmFirstToken/TtsFirstAudio at all —
        // those deltas must come back None, not panic or report a bogus 0.
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::SpeechEnded);
        turn.mark(PipelineStage::RouterDecision);
        turn.mark(PipelineStage::TurnComplete);
        assert!(turn.delta_ms(PipelineStage::LlmFirstToken, PipelineStage::TtsFirstAudio).is_none());
        assert!(turn.delta_ms(PipelineStage::SpeechEnded, PipelineStage::RouterDecision).is_some());
    }

    #[test]
    fn turn_telemetry_mark_is_idempotent_first_call_wins() {
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::SpeechEnded);
        let first = turn.get(PipelineStage::SpeechEnded).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        turn.mark(PipelineStage::SpeechEnded);
        let second = turn.get(PipelineStage::SpeechEnded).unwrap();
        assert_eq!(first, second, "a later mark() for the same stage must not overwrite the first");
    }

    #[test]
    fn each_turn_telemetry_gets_a_distinct_non_empty_id() {
        let a = TurnTelemetry::new();
        let b = TurnTelemetry::new();
        assert!(!a.id().is_empty());
        assert_ne!(a.id(), b.id(), "concurrent/overlapping turns (e.g. a barge-in) must be distinguishable in logs");
    }

    #[test]
    fn turn_telemetry_finish_does_not_panic_on_a_partially_marked_turn() {
        let ctx = PerfContext::new(HardwareTier::Standard, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::SpeechEnded);
        turn.finish(&ctx); // must not panic even though most stages/deltas are unmarked
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

    // -- TurnSnapshot / TurnHistory: the dashboard's read path --

    #[test]
    fn snapshot_computes_the_same_deltas_finish_would_log() {
        let ctx = PerfContext::new(HardwareTier::Performance, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::MicDetected);
        turn.mark(PipelineStage::SpeechEnded);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::SttFinal);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::RouterDecision);
        turn.mark(PipelineStage::LlmFirstToken);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::TtsFirstAudio);
        turn.mark(PipelineStage::PlaybackStarted);
        std::thread::sleep(Duration::from_millis(2));
        turn.mark(PipelineStage::TurnComplete);

        let snap = turn.snapshot(&ctx);
        assert_eq!(snap.turn_id, turn.id());
        assert_eq!(snap.speech_to_stt_final_ms, turn.delta_ms(PipelineStage::SpeechEnded, PipelineStage::SttFinal));
        assert_eq!(
            snap.stt_final_to_router_decision_ms,
            turn.delta_ms(PipelineStage::SttFinal, PipelineStage::RouterDecision)
        );
        assert_eq!(
            snap.total_turn_latency_ms,
            turn.delta_ms(PipelineStage::SpeechEnded, PipelineStage::TurnComplete)
        );
        assert!(!snap.interrupted, "a turn with TurnComplete marked from SpeechEnded must not be flagged interrupted");
        // Every marked stage shows up in the ordered list, none fabricated.
        let labels: Vec<&str> = snap.stage_ms_since_start.iter().map(|(l, _)| l.as_str()).collect();
        assert!(labels.contains(&"speech_ended"));
        assert!(labels.contains(&"turn_complete"));
        assert!(!labels.contains(&"stt_started"), "an unmarked stage must never appear in the snapshot");
    }

    #[test]
    fn snapshot_leaves_unmarked_deltas_as_none_never_zero() {
        // Mirrors a fast-router turn: no LLM/TTS stages ever get marked.
        let ctx = PerfContext::new(HardwareTier::Entry, PerformanceMode::BatterySaver, PressureState::Normal, &sample_config());
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::SpeechEnded);
        turn.mark(PipelineStage::RouterDecision);
        turn.mark(PipelineStage::TurnComplete);

        let snap = turn.snapshot(&ctx);
        assert!(snap.stt_final_to_llm_first_token_ms.is_none());
        assert!(snap.llm_first_token_to_tts_first_audio_ms.is_none());
        assert!(snap.stt_final_to_router_decision_ms.is_none(), "SttFinal was never marked in this scenario either");
        assert!(snap.stt_final_to_router_decision_ms != Some(0));
    }

    #[test]
    fn snapshot_flags_interrupted_when_total_latency_cannot_be_computed() {
        let ctx = PerfContext::new(HardwareTier::Standard, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::SpeechEnded);
        // TurnComplete never marked — e.g. superseded by a barge-in.
        let snap = turn.snapshot(&ctx);
        assert!(snap.interrupted);
        assert!(snap.total_turn_latency_ms.is_none());
    }

    #[test]
    fn snapshot_does_not_consume_state_finish_can_still_be_called_after() {
        let ctx = PerfContext::new(HardwareTier::Standard, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::SpeechEnded);
        turn.mark(PipelineStage::TurnComplete);
        let _ = turn.snapshot(&ctx);
        turn.finish(&ctx); // must not panic — snapshot() only reads, never mutates
    }

    #[test]
    fn turn_history_returns_newest_first_and_evicts_oldest_at_capacity() {
        let history = TurnHistory::new();
        for i in 0..(HISTORY_CAPACITY + 5) {
            let turn = TurnTelemetry::new();
            turn.mark(PipelineStage::SpeechEnded);
            turn.mark(PipelineStage::TurnComplete);
            let ctx = PerfContext::new(HardwareTier::Standard, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());
            let mut snap = turn.snapshot(&ctx);
            snap.turn_id = format!("turn-{i}");
            history.push(snap);
        }
        let all = history.snapshot_all();
        assert_eq!(all.len(), HISTORY_CAPACITY, "must never grow past the fixed capacity");
        assert_eq!(all[0].turn_id, format!("turn-{}", HISTORY_CAPACITY + 4), "newest turn must be first");
    }

    #[test]
    fn turn_history_clear_empties_it() {
        let history = TurnHistory::new();
        let ctx = PerfContext::new(HardwareTier::Standard, PerformanceMode::Adaptive, PressureState::Normal, &sample_config());
        let turn = TurnTelemetry::new();
        turn.mark(PipelineStage::SpeechEnded);
        history.push(turn.snapshot(&ctx));
        assert_eq!(history.snapshot_all().len(), 1);
        history.clear();
        assert!(history.snapshot_all().is_empty());
    }
}
