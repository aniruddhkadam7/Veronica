//! Instrumentation for the capture pipeline.
//!
//! Added to answer a specific question with data instead of a guess: *is the
//! existing audio path dropping samples or producing chunks too small for an
//! ASR engine to work with?* Every counter here is an atomic bumped on the
//! capture thread, so reading them from another thread costs nothing and the
//! capture loop pays only an uncontended fetch_add per packet.
//!
//! The counter that actually settles the dropped-audio question is
//! `discontinuities`: WASAPI sets AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY when
//! the capture buffer overflowed and samples were lost before we read them.
//! A non-zero value there is a real, unambiguous glitch. Comparing
//! `samples_emitted` against elapsed wall time catches the subtler case where
//! nothing reports an error but audio is quietly going missing anyway.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct CaptureMetrics {
    /// WASAPI packets read from the device.
    pub packets: AtomicU64,
    /// Device frames read, at the device's native sample rate.
    pub frames_in: AtomicU64,
    /// Samples pushed downstream, at TARGET_SAMPLE_RATE mono.
    pub samples_emitted: AtomicU64,
    /// Packets flagged AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY — audio was lost
    /// before we read it. Any non-zero value is a real dropout.
    pub discontinuities: AtomicU64,
    /// Packets flagged silent (WASAPI's optimization for a silent mix).
    pub silent_packets: AtomicU64,
    /// Packets flagged with an unreliable device timestamp.
    pub timestamp_errors: AtomicU64,
    /// Total microseconds spent in downmix+resample, to show whether the
    /// conversion is anywhere near being a bottleneck.
    pub resample_micros: AtomicU64,
    /// Chunks that reached the consumer with zero samples (the FFT resampler
    /// buffers internally and returns nothing until a full block accumulates).
    pub empty_chunks: AtomicU64,
}

impl CaptureMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            packets: self.packets.load(Ordering::Relaxed),
            frames_in: self.frames_in.load(Ordering::Relaxed),
            samples_emitted: self.samples_emitted.load(Ordering::Relaxed),
            discontinuities: self.discontinuities.load(Ordering::Relaxed),
            silent_packets: self.silent_packets.load(Ordering::Relaxed),
            timestamp_errors: self.timestamp_errors.load(Ordering::Relaxed),
            resample_micros: self.resample_micros.load(Ordering::Relaxed),
            empty_chunks: self.empty_chunks.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MetricsSnapshot {
    pub packets: u64,
    pub frames_in: u64,
    pub samples_emitted: u64,
    pub discontinuities: u64,
    pub silent_packets: u64,
    pub timestamp_errors: u64,
    pub resample_micros: u64,
    pub empty_chunks: u64,
}

impl MetricsSnapshot {
    /// Audio actually delivered downstream, in seconds.
    pub fn emitted_seconds(&self, target_rate: u32) -> f64 {
        self.samples_emitted as f64 / target_rate as f64
    }

    /// Ratio of audio delivered to wall time elapsed. 1.0 means nothing was
    /// lost; meaningfully below 1.0 means samples are going missing somewhere
    /// between the device and the STT engine.
    pub fn capture_ratio(&self, elapsed_s: f64, target_rate: u32) -> f64 {
        if elapsed_s <= 0.0 {
            return 0.0;
        }
        self.emitted_seconds(target_rate) / elapsed_s
    }

    pub fn mean_packet_frames(&self) -> f64 {
        if self.packets == 0 {
            return 0.0;
        }
        self.frames_in as f64 / self.packets as f64
    }

    /// Share of a single core spent on downmix+resample.
    pub fn resample_load(&self, elapsed_s: f64) -> f64 {
        if elapsed_s <= 0.0 {
            return 0.0;
        }
        (self.resample_micros as f64 / 1e6) / elapsed_s
    }
}
