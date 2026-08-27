//! AudioCapture
//! ├── SystemAudioCapture  (WASAPI loopback capture of the default render/output device)
//! ├── MicrophoneCapture   (WASAPI capture of a default/selected input device)
//! └── AudioDeviceManager  (device enumeration)
//!
//! Captured audio is consumed by `stt`: a local sherpa-onnx engine detects
//! speech/utterance boundaries, and each detected segment's raw audio is
//! sent to Groq Cloud's Whisper API for transcription — the only point in
//! this app where captured audio leaves the machine, and only that
//! already-endpointed segment, never a continuous stream.

mod device_manager;
mod gap_fill;
mod metrics;
mod mic_capture;
mod pipeline;
mod resample;
mod system_capture;

pub use device_manager::{AudioDeviceInfo, AudioDeviceManager};
pub use gap_fill::SilenceGapFiller;
pub use metrics::{CaptureMetrics, MetricsSnapshot};
pub use pipeline::run_stt_pipeline;
pub use mic_capture::MicrophoneCapture;
pub use system_capture::SystemAudioCapture;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Target format that both capture sources resample/convert into before handing
/// samples to the STT pipeline: 16kHz, mono, 32-bit float.
pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const TARGET_CHANNELS: u16 = 1;

/// A chunk of captured, resampled audio plus a coarse level measurement for the
/// UI's live level visualizer.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub source: AudioSource,
    pub samples: Vec<f32>,
    /// Root-mean-square level of this chunk, 0.0-1.0, for the UI meter.
    pub rms_level: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AudioSource {
    SystemAudio,
    Microphone,
}

pub fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
    ((sum_sq / samples.len() as f64).sqrt() as f32).min(1.0)
}

/// Shared stop flag handed to a capture thread so it can be told to shut down
/// cleanly from the Tauri command layer.
#[derive(Clone, Default)]
pub struct StopSignal(Arc<AtomicBool>);

impl StopSignal {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn stop(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_stopped(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// Shared pause flag. Unlike `StopSignal`, this does not stop WASAPI capture (the
/// capture thread keeps draining the device buffer so it never overflows) — it only
/// tells the audio/STT pipeline thread to stop forwarding samples to the STT
/// sidecar and to stop emitting level/transcript-affecting events, so the STT
/// process and the transcript-so-far are preserved untouched across a pause.
#[derive(Clone, Default)]
pub struct PauseSignal(Arc<AtomicBool>);

impl PauseSignal {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn set_paused(&self, paused: bool) {
        self.0.store(paused, Ordering::SeqCst);
    }

    pub fn is_paused(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
