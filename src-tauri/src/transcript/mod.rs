//! TranscriptManager
//!
//! Maintains an `InterviewSession` made of `TranscriptSegment`s produced by the STT
//! pipeline. No segment is ever classified as a "question" or otherwise interpreted —
//! this module only accumulates text plus its timing/source metadata.

use crate::audio::AudioSource;
use crate::stt::{SttEvent, SttEventKind};

#[derive(Debug, Clone, serde::Serialize)]
pub struct TranscriptSegment {
    pub id: String,
    /// Wall-clock time the segment was finalized, ms since Unix epoch.
    pub timestamp: u64,
    pub source: AudioSource,
    /// Latest in-progress (not yet finalized) text for this segment, if any.
    pub partial_text: Option<String>,
    /// Finalized text, set once the STT engine ends the utterance.
    pub final_text: Option<String>,
    pub start_time: Option<f64>,
    pub end_time: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum RecordingState {
    Idle,
    /// Start requested: audio capture is being initialized and/or the STT
    /// sidecar is starting and has not yet signaled ready. A second Start
    /// request seen in this state is rejected as a duplicate (see
    /// `commands::start_system_audio_capture`) — real audio/transcript
    /// flow has not begun yet, so the UI must not treat this as Recording.
    Starting,
    Recording,
    Paused,
    /// Stop requested: capture/sidecar teardown is in progress (threads
    /// being joined, sidecar process being flushed and shut down). A Start
    /// request seen in this state waits briefly for teardown to finish
    /// rather than racing a new session against the old one's still-live
    /// process/device handles (see `commands::start_system_audio_capture`).
    Stopping,
    Stopped,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct InterviewSession {
    pub id: String,
    pub started_at_ms: u64,
    pub ended_at_ms: Option<u64>,
    pub segments: Vec<TranscriptSegment>,
    /// Total milliseconds spent paused so far, excluded from elapsed/duration.
    pub paused_ms_total: u64,
    /// Wall-clock time the current pause began, if currently paused.
    #[serde(skip)]
    pub pause_started_at_ms: Option<u64>,
}

impl InterviewSession {
    pub fn new() -> Self {
        Self {
            id: uuid_like_id(),
            started_at_ms: now_ms(),
            ended_at_ms: None,
            segments: Vec::new(),
            paused_ms_total: 0,
            pause_started_at_ms: None,
        }
    }

    pub fn word_count(&self) -> usize {
        self.segments
            .iter()
            .filter_map(|s| s.final_text.as_ref())
            .map(|t| t.split_whitespace().count())
            .sum()
    }

    pub fn full_transcript_text(&self) -> String {
        self.segments
            .iter()
            .filter_map(|s| s.final_text.as_deref())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Elapsed recording time in milliseconds, excluding any paused duration.
    /// While actively paused, time is frozen at the moment the pause began.
    pub fn elapsed_ms(&self) -> u64 {
        let end = self
            .ended_at_ms
            .or(self.pause_started_at_ms)
            .unwrap_or_else(now_ms);
        end.saturating_sub(self.started_at_ms)
            .saturating_sub(self.paused_ms_total)
    }

    pub fn mark_paused(&mut self) {
        if self.pause_started_at_ms.is_none() {
            self.pause_started_at_ms = Some(now_ms());
        }
    }

    pub fn mark_resumed(&mut self) {
        if let Some(paused_at) = self.pause_started_at_ms.take() {
            self.paused_ms_total = self
                .paused_ms_total
                .saturating_add(now_ms().saturating_sub(paused_at));
        }
    }

    pub fn mark_stopped(&mut self) {
        self.mark_resumed();
        self.ended_at_ms = Some(now_ms());
    }
}

impl Default for InterviewSession {
    fn default() -> Self {
        Self::new()
    }
}

/// Accumulates STT events into an `InterviewSession`. One live "in-progress" segment
/// is kept per audio source; a `Final` event closes it out and a fresh one begins
/// on the next `Partial`/`Final` for that source.
pub struct TranscriptManager {
    session: InterviewSession,
    in_progress: std::collections::HashMap<AudioSource, TranscriptSegment>,
}

impl TranscriptManager {
    pub fn new() -> Self {
        Self {
            session: InterviewSession::new(),
            in_progress: std::collections::HashMap::new(),
        }
    }

    pub fn session(&self) -> &InterviewSession {
        &self.session
    }

    pub fn take_session(self) -> InterviewSession {
        self.session
    }

    pub fn clear(&mut self) {
        self.session = InterviewSession::new();
        self.in_progress.clear();
    }

    pub fn mark_paused(&mut self) {
        self.session.mark_paused();
    }

    pub fn mark_resumed(&mut self) {
        self.session.mark_resumed();
    }

    pub fn mark_stopped(&mut self) {
        self.session.mark_stopped();
    }

    /// Applies one STT event, returning the segment that changed (for emitting a
    /// UI update event) if any.
    pub fn apply_event(&mut self, event: SttEvent) -> Option<TranscriptSegment> {
        match event.kind {
            SttEventKind::Partial => {
                let segment = self
                    .in_progress
                    .entry(event.source)
                    .or_insert_with(|| new_segment(event.source));
                segment.partial_text = Some(event.text);
                Some(segment.clone())
            }
            SttEventKind::Final => {
                let mut segment = self
                    .in_progress
                    .remove(&event.source)
                    .unwrap_or_else(|| new_segment(event.source));
                segment.partial_text = None;
                segment.final_text = Some(event.text);
                segment.start_time = event.start_time;
                segment.end_time = event.end_time;
                segment.timestamp = now_ms();
                self.session.segments.push(segment.clone());
                Some(segment)
            }
        }
    }
}

impl Default for TranscriptManager {
    fn default() -> Self {
        Self::new()
    }
}

fn new_segment(source: AudioSource) -> TranscriptSegment {
    TranscriptSegment {
        id: uuid_like_id(),
        timestamp: now_ms(),
        source,
        partial_text: None,
        final_text: None,
        start_time: None,
        end_time: None,
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn uuid_like_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    format!("{:x}-{:x}", now_ms(), n)
}
