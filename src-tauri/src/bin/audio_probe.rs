//! Measures the real WASAPI capture pipeline.
//!
//!     cargo run --bin audio_probe -- [seconds]
//!
//! Play audio through the default output device while this runs (any source —
//! a video, a meeting, music). It drives `SystemAudioCapture` exactly as the
//! app does and reports what actually comes out the other end, so the answer to
//! "is the audio path dropping samples or producing chunks that are too small"
//! is measured rather than assumed.
//!
//! Reported, per the spec's list:
//!   sample rate / channel count  — device mix format and the converted output
//!   buffer size / chunk duration — distribution of chunk sizes reaching STT
//!   processing time              — time spent in downmix + resample
//!   queue depth                  — backlog in the capture->consumer channel
//!   dropped audio frames         — WASAPI discontinuity flags, plus the ratio
//!                                  of delivered audio to elapsed wall time

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use desktop_lib::audio::{
    AudioDeviceManager, CaptureMetrics, StopSignal, SystemAudioCapture, TARGET_SAMPLE_RATE,
};

fn percentile(sorted: &[usize], pct: f64) -> usize {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let seconds: u64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(20);

    println!("=== WASAPI capture pipeline probe ===");
    match AudioDeviceManager::list_output_devices() {
        Ok(devices) => {
            println!("Output devices:");
            for device in &devices {
                let marker = if device.is_default { "*" } else { " " };
                println!("  {marker} {}", device.name);
            }
        }
        Err(err) => println!("  (could not enumerate output devices: {err})"),
    }
    println!();
    println!("Capturing loopback for {seconds}s — play some audio now.");
    println!();

    let (tx, rx) = crossbeam_channel::unbounded();
    let stop = StopSignal::new();
    let metrics = CaptureMetrics::new();

    let handle = match SystemAudioCapture::start_with_metrics(tx, stop.clone(), metrics.clone()) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("failed to start capture: {err}");
            std::process::exit(1);
        }
    };

    let started = Instant::now();
    let deadline = started + Duration::from_secs(seconds);

    let mut chunk_sizes: Vec<usize> = Vec::new();
    let mut queue_depths: Vec<usize> = Vec::new();
    let mut gaps_ms: Vec<f64> = Vec::new();
    let mut nonsilent_chunks = 0usize;
    let mut last_arrival: Option<Instant> = None;
    // Per-second delivered-audio buckets. Averages hide the shape of the loss:
    // a uniform 50% shortfall and a stream that runs perfectly then stalls for
    // four seconds have very different causes and very different fixes.
    let mut buckets: Vec<u64> = vec![0; seconds as usize + 1];

    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(remaining.min(Duration::from_millis(500))) {
            Ok(chunk) => {
                let now = Instant::now();
                if let Some(previous) = last_arrival {
                    gaps_ms.push((now - previous).as_secs_f64() * 1000.0);
                }
                last_arrival = Some(now);

                let bucket = (now - started).as_secs() as usize;
                if bucket < buckets.len() {
                    buckets[bucket] += chunk.samples.len() as u64;
                }

                chunk_sizes.push(chunk.samples.len());
                // Depth *after* our recv, i.e. how much backlog the consumer is
                // failing to keep up with. Anything consistently above zero
                // means the downstream stage is slower than capture.
                queue_depths.push(rx.len());
                if chunk.rms_level > 0.001 {
                    nonsilent_chunks += 1;
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
        }
    }

    stop.stop();
    let _ = handle.join();
    let elapsed = started.elapsed().as_secs_f64();
    let snapshot = metrics.snapshot();

    chunk_sizes.sort_unstable();
    queue_depths.sort_unstable();
    gaps_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let total_samples: usize = chunk_sizes.iter().sum();
    let mean_chunk = if chunk_sizes.is_empty() {
        0.0
    } else {
        total_samples as f64 / chunk_sizes.len() as f64
    };
    let to_ms = |samples: usize| samples as f64 * 1000.0 / TARGET_SAMPLE_RATE as f64;

    println!("--- format ---");
    println!("  device mix format    : see 'system audio loopback started' log line above");
    println!(
        "  pipeline output      : {} Hz, {} channel (f32)",
        TARGET_SAMPLE_RATE,
        desktop_lib::audio::TARGET_CHANNELS
    );
    println!("  mean device packet   : {:.1} frames", snapshot.mean_packet_frames());

    println!();
    println!("--- chunks delivered to STT ---");
    println!("  count                : {}", chunk_sizes.len());
    println!("  non-silent           : {nonsilent_chunks}");
    println!(
        "  size  min/p50/p99/max: {} / {} / {} / {} samples",
        chunk_sizes.first().copied().unwrap_or(0),
        percentile(&chunk_sizes, 50.0),
        percentile(&chunk_sizes, 99.0),
        chunk_sizes.last().copied().unwrap_or(0)
    );
    println!(
        "  duration  p50 / mean : {:.1} ms / {:.1} ms",
        to_ms(percentile(&chunk_sizes, 50.0)),
        to_ms(mean_chunk as usize)
    );
    if !gaps_ms.is_empty() {
        println!(
            "  arrival gap p50/p99  : {:.1} ms / {:.1} ms",
            gaps_ms[gaps_ms.len() / 2],
            gaps_ms[(gaps_ms.len() * 99 / 100).min(gaps_ms.len() - 1)]
        );
    }

    println!();
    println!("--- audio delivered per second of wall time ---");
    println!("  (each row: seconds of audio delivered during that 1s window)");
    for (index, samples) in buckets.iter().enumerate() {
        if index as f64 >= elapsed {
            break;
        }
        let delivered = *samples as f64 / TARGET_SAMPLE_RATE as f64;
        let bars = (delivered * 40.0).round() as usize;
        println!(
            "  t={index:>3}s  {:.3}s  {}",
            delivered,
            "#".repeat(bars.min(60))
        );
    }

    println!();
    println!("--- queue depth (backlog after each recv) ---");
    println!(
        "  p50 / p99 / max      : {} / {} / {}",
        percentile(&queue_depths, 50.0),
        percentile(&queue_depths, 99.0),
        queue_depths.last().copied().unwrap_or(0)
    );

    println!();
    println!("--- processing cost ---");
    println!(
        "  downmix+resample     : {:.1} ms total over {:.1}s ({:.3}% of one core)",
        snapshot.resample_micros as f64 / 1000.0,
        elapsed,
        snapshot.resample_load(elapsed) * 100.0
    );

    println!();
    println!("--- dropped audio ---");
    println!("  WASAPI packets       : {}", snapshot.packets);
    println!(
        "  discontinuities      : {}  <-- non-zero means audio was genuinely lost",
        snapshot.discontinuities
    );
    println!("  timestamp errors     : {}", snapshot.timestamp_errors);
    println!("  silent packets       : {}", snapshot.silent_packets);
    println!(
        "  empty chunks         : {}  (resampler withholding a partial block)",
        snapshot.empty_chunks
    );

    let ratio = snapshot.capture_ratio(elapsed, TARGET_SAMPLE_RATE);
    println!(
        "  audio delivered      : {:.2}s of {:.2}s elapsed  (ratio {:.4})",
        snapshot.emitted_seconds(TARGET_SAMPLE_RATE),
        elapsed,
        ratio
    );

    println!();
    // The resampler holds back up to one 1024-sample block plus startup latency,
    // so a ratio a hair under 1.0 is expected and harmless. A ratio meaningfully
    // below 1.0 is the signature of real sample loss.
    if snapshot.discontinuities > 0 {
        println!("VERDICT: audio IS being dropped ({} discontinuities).", snapshot.discontinuities);
    } else if ratio < 0.98 {
        println!(
            "VERDICT: no WASAPI dropouts, but only {:.1}% of elapsed audio was delivered \
             — samples are going missing in conversion.",
            ratio * 100.0
        );
    } else {
        println!("VERDICT: no dropped audio. Capture is delivering a complete stream.");
    }

    let _ = metrics.packets.load(Ordering::Relaxed);
}
