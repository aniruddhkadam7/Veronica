//! The capture → gap-fill → STT loop.
//!
//! Extracted from `commands::start_system_audio_capture` so that the exact code
//! the application runs can also be driven headlessly by
//! `src/bin/pipeline_test.rs`. Verifying a *copy* of this loop would prove
//! nothing about what actually ships, so there is only one copy and both
//! callers use it.
//!
//! Deliberately free of Tauri types: the caller supplies closures for audio
//! levels and STT events. That is what lets the test binary run it without a
//! window, and it keeps the Tauri command surface unchanged.

use std::time::Duration;

use crossbeam_channel::Receiver;

use super::{AudioChunk, PauseSignal, SilenceGapFiller, TARGET_SAMPLE_RATE};
use crate::stt::SttSidecar;

/// How long to wait for a chunk before checking whether silence needs
/// synthesizing. When the render endpoint is idle no chunks arrive at all, so a
/// blocking receive would never wake up to close the gap.
const RECV_TIMEOUT: Duration = Duration::from_millis(50);

/// Runs until the audio channel closes, then flushes and shuts down the
/// sidecar. Consumes the sidecar because shutdown is part of the contract.
pub fn run_stt_pipeline<L>(
    audio_rx: Receiver<AudioChunk>,
    mut sidecar: SttSidecar,
    pause: PauseSignal,
    mut on_level: L,
) where
    L: FnMut(&AudioChunk),
{
    // WASAPI loopback delivers *no packets at all* while the render endpoint is
    // idle — not silent packets, nothing (measured; see docs/stt-benchmark.md).
    // Every speech endpointer decides an utterance is over by observing
    // trailing silence, so without reconstructing that silence the final for a
    // question never fires until the interviewer happens to speak again.
    let mut gap_filler = SilenceGapFiller::new(TARGET_SAMPLE_RATE);
    let mut was_paused = false;

    loop {
        let chunk = match audio_rx.recv_timeout(RECV_TIMEOUT) {
            Ok(chunk) => Some(chunk),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        };

        // While paused, keep draining the channel (so the WASAPI capture
        // thread's send never blocks) but forward nothing to STT or the UI
        // meter — this preserves the sidecar process and the transcript-so-far
        // untouched, per the Pause requirements.
        if pause.is_paused() {
            if !was_paused {
                // Entering pause: finalize whatever was already said, so a
                // half-decoded question is committed rather than left sitting
                // in the decoder until resume.
                if let Err(err) = sidecar.flush() {
                    log::warn!("failed to flush STT sidecar on pause: {err}");
                }
                was_paused = true;
            }
            continue;
        }
        if was_paused {
            // Drop the deficit accumulated across the pause rather than
            // flushing the whole paused duration into the decoder as one burst.
            gap_filler.resync();
            was_paused = false;
        }

        if let Some(chunk) = chunk {
            on_level(&chunk);
            gap_filler.on_samples(chunk.samples.len());
            if let Err(err) = sidecar.send_samples(&chunk.samples) {
                log::warn!("failed to send audio to STT sidecar: {err}");
            }
        }

        if let Some(silence) = gap_filler.take_silence() {
            if let Err(err) = sidecar.send_samples(&silence) {
                log::warn!("failed to send gap-fill silence to STT sidecar: {err}");
            }
        }
    }

    // Channel closed (capture stopped): flush so any in-progress utterance is
    // finalized rather than dropped, then shut the sidecar down, which closes
    // its stdout and lets the event reader thread exit.
    let _ = sidecar.flush();
    sidecar.shutdown();
}
