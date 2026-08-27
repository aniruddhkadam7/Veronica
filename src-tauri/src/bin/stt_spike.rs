//! Audio source for the live STT spike.
//!
//! Captures WASAPI loopback exactly as the app does, patches the timeline holes
//! the probe found (see `audio::gap_fill`), and writes raw 16 kHz mono PCM16 to
//! stdout. The Python side pipes that into an `STTEngine` and prints interim and
//! final text as it arrives:
//!
//! ```text
//! cargo run --bin stt_spike | \
//!   packages/stt-bench/.venv/Scripts/python.exe \
//!   packages/stt-bench/scripts/live_spike.py nemo-80ms
//! ```
//!
//! Splitting it this way keeps the spike honest: the audio really does come
//! from the production capture path, and the recognizer really is the one the
//! benchmark scored, with no file-replay shortcut in between.
//!
//! stdout is binary PCM and nothing else. Diagnostics go to stderr.

use std::io::Write;
use std::time::{Duration, Instant};

use desktop_lib::audio::{
    CaptureMetrics, SilenceGapFiller, StopSignal, SystemAudioCapture, TARGET_SAMPLE_RATE,
};

fn write_samples(out: &mut impl Write, samples: &[f32]) -> std::io::Result<()> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        bytes.extend_from_slice(&((clamped * i16::MAX as f32) as i16).to_le_bytes());
    }
    out.write_all(&bytes)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn"))
        .target(env_logger::Target::Stderr)
        .init();

    let seconds: Option<u64> = std::env::args().nth(1).and_then(|a| a.parse().ok());

    let (tx, rx) = crossbeam_channel::unbounded();
    let stop = StopSignal::new();
    let metrics = CaptureMetrics::new();

    let handle = match SystemAudioCapture::start_with_metrics(tx, stop.clone(), metrics.clone()) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("failed to start loopback capture: {err}");
            std::process::exit(1);
        }
    };

    eprintln!(
        "stt_spike: capturing loopback -> {} Hz mono PCM16 on stdout",
        TARGET_SAMPLE_RATE
    );

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut filler = SilenceGapFiller::new(TARGET_SAMPLE_RATE);
    let started = Instant::now();
    let mut synthesized_samples: u64 = 0;

    loop {
        if let Some(limit) = seconds {
            if started.elapsed() >= Duration::from_secs(limit) {
                break;
            }
        }

        // Short timeout so that when the endpoint goes idle and no packets
        // arrive at all, we still wake up regularly to synthesize the silence
        // that keeps the stream continuous.
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(chunk) => {
                filler.on_samples(chunk.samples.len());
                if write_samples(&mut out, &chunk.samples).is_err() {
                    break;
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }

        if let Some(silence) = filler.take_silence() {
            synthesized_samples += silence.len() as u64;
            if write_samples(&mut out, &silence).is_err() {
                break;
            }
        }

        if out.flush().is_err() {
            break;
        }
    }

    stop.stop();
    let _ = handle.join();
    let _ = out.flush();

    let elapsed = started.elapsed().as_secs_f64();
    let snapshot = metrics.snapshot();
    eprintln!(
        "stt_spike: {elapsed:.1}s wall, {:.1}s captured, {:.1}s synthesized silence, \
         {} discontinuities",
        snapshot.emitted_seconds(TARGET_SAMPLE_RATE),
        synthesized_samples as f64 / TARGET_SAMPLE_RATE as f64,
        snapshot.discontinuities
    );
}
