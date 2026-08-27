use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::Sender;
use wasapi::{Direction, SampleType, StreamMode, WaveFormat};

use super::metrics::CaptureMetrics;
use super::resample::AudioResampler;
use super::{compute_rms, AudioChunk, AudioSource, StopSignal};

/// How long `start`/`start_with_metrics` will wait for the capture thread to
/// confirm WASAPI initialization before giving up. Generous relative to how
/// long device enumeration + `IAudioClient::Initialize` + `Start` normally
/// take (well under a second) — this exists to bound the wait if the thread
/// never gets to send at all (e.g. panics before reaching the send point),
/// not to accommodate a slow-but-working init.
const INIT_TIMEOUT: Duration = Duration::from_secs(10);

/// Captures Windows system/output audio via WASAPI loopback on the default render
/// device. This works regardless of which application is producing sound (Chrome,
/// Edge, Teams, Zoom, Meet, ...) because it taps the shared mix, not any specific
/// app's output.
pub struct SystemAudioCapture;

impl SystemAudioCapture {
    /// Spawns a dedicated OS thread that captures loopback audio and pushes
    /// resampled 16kHz mono `AudioChunk`s onto `tx` until `stop` is signalled.
    ///
    /// Blocks (up to `INIT_TIMEOUT`) until the capture thread has confirmed
    /// WASAPI device enumeration + client init + `Start()` all succeeded, or
    /// returns the specific error if any of them failed. Earlier versions of
    /// this function returned as soon as the thread was merely *spawned*,
    /// which let a real-but-transient WASAPI init failure (a driver hiccup,
    /// a device busy/exclusive-mode conflict, no default render device) pass
    /// completely silently — the caller believed capture had started while
    /// the thread had actually already exited, so the UI reported
    /// "Recording" with zero audio ever flowing and no visible error. See
    /// `docs/performance-tuning.md`'s STT-start-reliability section.
    pub fn start(
        tx: Sender<AudioChunk>,
        stop: StopSignal,
    ) -> Result<JoinHandle<()>, String> {
        Self::start_with_metrics(tx, stop, CaptureMetrics::new())
    }

    /// Same as `start`, but the caller keeps a handle on the pipeline counters.
    /// Used by the `audio_probe` binary to measure the real capture path; the
    /// normal `start` path allocates a throwaway handle so the counters cost
    /// the same either way.
    pub fn start_with_metrics(
        tx: Sender<AudioChunk>,
        stop: StopSignal,
        metrics: Arc<CaptureMetrics>,
    ) -> Result<JoinHandle<()>, String> {
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
        let handle = std::thread::Builder::new()
            .name("system-audio-capture".into())
            .spawn(move || {
                if let Err(err) = run_capture_loop(tx, stop, metrics, ready_tx) {
                    log::error!("system audio capture stopped with error: {err}");
                }
            })
            .map_err(|e| e.to_string())?;

        match ready_rx.recv_timeout(INIT_TIMEOUT) {
            Ok(Ok(())) => Ok(handle),
            Ok(Err(err)) => {
                // The thread already returned (init failed before the
                // capture loop starts) — join it to avoid leaking the
                // handle, then surface the real error to the caller instead
                // of a fake success.
                let _ = handle.join();
                Err(err)
            }
            Err(_) => {
                // Timed out waiting — the thread never reached the
                // send point (hung or panicked without unwinding through
                // the send). Don't block the caller forever; report a clear,
                // actionable timeout instead of silently returning "started".
                Err("system audio capture did not initialize in time".to_string())
            }
        }
    }
}

fn run_capture_loop(
    tx: Sender<AudioChunk>,
    stop: StopSignal,
    metrics: Arc<CaptureMetrics>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let init_result: Result<_, String> = (|| {
        wasapi::initialize_mta().ok().map_err(|e| e.to_string())?;

        let enumerator = wasapi::DeviceEnumerator::new().map_err(|e| e.to_string())?;
        let device = enumerator
            .get_default_device(&Direction::Render)
            .map_err(|e| format!("no default output device: {e}"))?;

        let mut audio_client = device.get_iaudioclient().map_err(|e| e.to_string())?;
        let mix_format = audio_client.get_mixformat().map_err(|e| e.to_string())?;
        let in_rate = mix_format.get_samplespersec();
        let in_channels = mix_format.get_nchannels();

        // Request float samples at the device's native mix format; WASAPI shared-mode
        // loopback requires matching the mix format (autoconvert handles edge cases).
        let desired_format = WaveFormat::new(32, 32, &SampleType::Float, in_rate as usize, in_channels as usize, None);

        let buffer_duration_hns = 200_000; // 20ms
        let mode = StreamMode::EventsShared {
            autoconvert: true,
            buffer_duration_hns,
        };

        audio_client
            .initialize_client(&desired_format, &Direction::Capture, &mode)
            .map_err(|e| format!("failed to initialize loopback client: {e}"))?;

        let event_handle = audio_client.set_get_eventhandle().map_err(|e| e.to_string())?;
        let capture_client = audio_client.get_audiocaptureclient().map_err(|e| e.to_string())?;

        audio_client.start_stream().map_err(|e| e.to_string())?;

        Ok((audio_client, capture_client, event_handle, in_rate, in_channels))
    })();

    let (audio_client, capture_client, event_handle, in_rate, in_channels) = match init_result {
        Ok(bits) => {
            let _ = ready_tx.send(Ok(()));
            bits
        }
        Err(err) => {
            let _ = ready_tx.send(Err(err.clone()));
            return Err(err);
        }
    };

    let mut resampler = AudioResampler::new(in_rate, in_channels);
    let bytes_per_frame = (in_channels as usize) * 4; // 32-bit float
    let mut byte_buf: Vec<u8> = Vec::new();

    log::info!(
        "system audio loopback started: {in_rate} Hz, {in_channels} ch -> {} Hz mono",
        super::TARGET_SAMPLE_RATE
    );

    while !stop.is_stopped() {
        if event_handle.wait_for_event(200).is_err() {
            continue;
        }

        loop {
            let packet_frames = match capture_client.get_next_packet_size() {
                Ok(Some(frames)) => frames,
                Ok(None) => break,
                Err(_) => break,
            };
            if packet_frames == 0 {
                break;
            }

            byte_buf.resize(packet_frames as usize * bytes_per_frame, 0);
            let (frames_read, buffer_info) = capture_client
                .read_from_device(&mut byte_buf)
                .map_err(|e| e.to_string())?;
            if frames_read == 0 {
                break;
            }

            metrics.packets.fetch_add(1, Ordering::Relaxed);
            metrics
                .frames_in
                .fetch_add(frames_read as u64, Ordering::Relaxed);
            if buffer_info.flags.data_discontinuity {
                // The device buffer overflowed and audio was lost before we
                // read it. This is the definitive dropped-audio signal.
                metrics.discontinuities.fetch_add(1, Ordering::Relaxed);
            }
            if buffer_info.flags.timestamp_error {
                metrics.timestamp_errors.fetch_add(1, Ordering::Relaxed);
            }

            let resample_started = std::time::Instant::now();
            let mono_resampled = if buffer_info.flags.silent {
                metrics.silent_packets.fetch_add(1, Ordering::Relaxed);
                let mono_len = frames_read as usize;
                resampler.process(&vec![0.0f32; mono_len * in_channels as usize])
            } else {
                let sample_count = frames_read as usize * in_channels as usize;
                let interleaved: Vec<f32> = byte_buf[..sample_count * 4]
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                resampler.process(&interleaved)
            };
            metrics
                .resample_micros
                .fetch_add(resample_started.elapsed().as_micros() as u64, Ordering::Relaxed);

            if mono_resampled.is_empty() {
                metrics.empty_chunks.fetch_add(1, Ordering::Relaxed);
            } else {
                metrics
                    .samples_emitted
                    .fetch_add(mono_resampled.len() as u64, Ordering::Relaxed);
            }
            emit_chunk(&tx, mono_resampled);
        }
    }

    let _ = audio_client.stop_stream();
    log::info!("system audio loopback stopped");
    Ok(())
}

fn emit_chunk(tx: &Sender<AudioChunk>, samples: Vec<f32>) {
    if samples.is_empty() {
        return;
    }
    let rms_level = compute_rms(&samples);
    let _ = tx.send(AudioChunk {
        source: AudioSource::SystemAudio,
        samples,
        rms_level,
    });
}
