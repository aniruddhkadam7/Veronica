//! Fills the holes WASAPI loopback leaves in the audio timeline.
//!
//! Measured behaviour (see `src/bin/audio_probe.rs`): a loopback capture on an
//! *idle* render endpoint delivers no packets at all. Not silent packets —
//! nothing. Per-second delivery during a probe run looked like this, with
//! speech between t=5s and t=11s:
//!
//! ```text
//!   t= 2s  0.000s        <- nothing playing, no packets
//!   t= 6s  1.005s        <- speech, full rate
//!   t=12s  0.000s        <- speech stopped, no packets again
//! ```
//!
//! Downstream this is corrosive in a way that is easy to miss, because the
//! audio that *does* arrive is perfectly intact. The damage is to the timeline:
//! every speech-endpointing algorithm — PocketSphinx's `Endpointer`,
//! sherpa-onnx's `is_endpoint`, any VAD — decides an utterance has ended by
//! observing *trailing silence*. If silence is never delivered, that trigger
//! never fires. The last utterance stays open until the speaker happens to say
//! something else, at which point the two run together.
//!
//! That single fact accounts for a family of symptoms usually blamed on the
//! recognizer: finals arriving seconds late or not at all, the last words of a
//! sentence going missing, and separate questions being glued into one segment.
//!
//! The fix is to reconstruct the missing time. This tracks how much audio
//! *should* have been produced by now against how much actually was, and emits
//! real silence samples to close the difference, so whatever consumes the
//! stream always sees a continuous 16 kHz timeline.

use std::time::Instant;

/// Don't synthesize for gaps smaller than this. Normal packet jitter is tens of
/// milliseconds; treating that as a hole would inject a steady dribble of
/// silence into otherwise continuous speech and could clip word endings.
const MIN_GAP_MS: u64 = 120;

/// Cap on a single synthesized burst, so returning from a long idle period
/// hands downstream a bounded chunk instead of a multi-megabyte allocation.
const MAX_BURST_MS: u64 = 1_000;

pub struct SilenceGapFiller {
    sample_rate: u32,
    started: Instant,
    samples_accounted: u64,
}

impl SilenceGapFiller {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            started: Instant::now(),
            samples_accounted: 0,
        }
    }

    /// Call when real captured samples are delivered downstream.
    pub fn on_samples(&mut self, count: usize) {
        self.samples_accounted += count as u64;
    }

    /// Returns silence to emit to bring the stream back up to wall-clock time,
    /// or `None` when the stream is already current. Call on a timer (and after
    /// each real chunk) — it is cheap and allocation-free when there is no gap.
    pub fn take_silence(&mut self) -> Option<Vec<f32>> {
        let expected = (self.started.elapsed().as_secs_f64() * self.sample_rate as f64) as u64;
        if expected <= self.samples_accounted {
            return None;
        }
        let deficit = expected - self.samples_accounted;
        let min_gap = self.sample_rate as u64 * MIN_GAP_MS / 1000;
        if deficit < min_gap {
            return None;
        }
        let max_burst = self.sample_rate as u64 * MAX_BURST_MS / 1000;
        let emit = deficit.min(max_burst) as usize;
        self.samples_accounted += emit as u64;
        Some(vec![0.0f32; emit])
    }

    /// Seconds of silence synthesized so far is not tracked directly; this
    /// exposes total accounted stream time for diagnostics.
    pub fn stream_seconds(&self) -> f64 {
        self.samples_accounted as f64 / self.sample_rate as f64
    }

    /// Re-anchors the stream clock to now, discarding any accumulated deficit.
    ///
    /// Required when resuming from pause. While paused no audio is forwarded,
    /// so the deficit grows for the whole pause; without this, resuming would
    /// dump the entire paused duration into the decoder as one burst of
    /// silence — finalizing spuriously and, for a long pause, stalling the
    /// pipeline while it chewed through minutes of synthesized nothing.
    pub fn resync(&mut self) {
        self.started = Instant::now();
        self.samples_accounted = 0;
    }
}
