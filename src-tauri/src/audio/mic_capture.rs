use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::Sender;
use wasapi::{Direction, SampleType, StreamMode, WaveFormat};

use super::resample::AudioResampler;
use super::{compute_rms, AudioChunk, AudioSource, StopSignal};

/// How long to wait before re-opening the default input device after
/// `run_capture_loop` exits with an error — covers both "the device was
/// unplugged/switched, and WASAPI needs a moment before the new default is
/// queryable" and "there is genuinely no input device right now" (e.g. the
/// user unplugged their only mic), where retrying instantly would spin-loop
/// pointlessly until one reappears.
const REOPEN_DELAY: Duration = Duration::from_millis(500);

/// Captures microphone input via WASAPI. Secondary source for Phase 1 — system
/// audio (the interviewer's side, and shared meeting audio) is the primary source.
pub struct MicrophoneCapture;

impl MicrophoneCapture {
    /// `device_id`: a WASAPI endpoint ID (see `AudioDeviceInfo::id`) to open
    /// instead of the system default input device, or `None` for the
    /// default — see `state::SelectedDevices`'s doc. Captured once at start,
    /// not re-read per retry: a mid-session device change (the selection or
    /// the physical device itself) is picked up the next time the mic
    /// assistant is started, mirroring how a plain default-device change
    /// already only self-heals within the currently-open device class.
    pub fn start(tx: Sender<AudioChunk>, stop: StopSignal, device_id: Option<String>) -> Result<JoinHandle<()>, String> {
        let handle = std::thread::Builder::new()
            .name("microphone-capture".into())
            .spawn(move || {
                // `run_capture_loop` returns `Err` not only on a genuine
                // failure to start but also when the OPEN device is
                // invalidated mid-session (unplugged, or the user switched
                // audio devices — e.g. disconnecting a Bluetooth headset
                // that was providing input) via `read_from_device`'s error
                // path. Previously this thread simply exited on that first
                // error, permanently and silently killing STT for the rest
                // of the session (see the audit: capture "stopped working"
                // after a device change, with no user-facing signal at
                // all). Looping here — re-opening the selected (or default)
                // device via a fresh `run_capture_loop` call — means a
                // mid-session device swap self-heals onto whatever is now
                // available, exactly like `StopSignal` already lets a
                // deliberate Stop interrupt this loop cleanly between
                // attempts. If a specifically-selected device was unplugged,
                // this will keep retrying that same device ID rather than
                // silently falling back to a different physical device the
                // user didn't choose.
                while !stop.is_stopped() {
                    if let Err(err) = run_capture_loop(tx.clone(), stop.clone(), device_id.as_deref()) {
                        log::error!("microphone capture error, retrying against the current device: {err}");
                        if stop.is_stopped() {
                            break;
                        }
                        std::thread::sleep(REOPEN_DELAY);
                    }
                }
            })
            .map_err(|e| e.to_string())?;
        Ok(handle)
    }
}

fn run_capture_loop(tx: Sender<AudioChunk>, stop: StopSignal, device_id: Option<&str>) -> Result<(), String> {
    wasapi::initialize_mta().ok().map_err(|e| e.to_string())?;

    let enumerator = wasapi::DeviceEnumerator::new().map_err(|e| e.to_string())?;
    let device = match device_id {
        Some(id) => enumerator
            .get_device(id)
            .map_err(|e| format!("selected input device unavailable: {e}"))?,
        None => enumerator
            .get_default_device(&Direction::Capture)
            .map_err(|e| format!("no default input device: {e}"))?,
    };

    let mut audio_client = device.get_iaudioclient().map_err(|e| e.to_string())?;
    let mix_format = audio_client.get_mixformat().map_err(|e| e.to_string())?;
    let in_rate = mix_format.get_samplespersec();
    let in_channels = mix_format.get_nchannels();

    let desired_format = WaveFormat::new(32, 32, &SampleType::Float, in_rate as usize, in_channels as usize, None);

    let buffer_duration_hns = 200_000;
    let mode = StreamMode::EventsShared {
        autoconvert: true,
        buffer_duration_hns,
    };

    audio_client
        .initialize_client(&desired_format, &Direction::Capture, &mode)
        .map_err(|e| format!("failed to initialize mic client: {e}"))?;

    let event_handle = audio_client.set_get_eventhandle().map_err(|e| e.to_string())?;
    let capture_client = audio_client.get_audiocaptureclient().map_err(|e| e.to_string())?;

    audio_client.start_stream().map_err(|e| e.to_string())?;

    let mut resampler = AudioResampler::new(in_rate, in_channels);
    let bytes_per_frame = (in_channels as usize) * 4;
    let mut byte_buf: Vec<u8> = Vec::new();

    log::info!(
        "microphone capture started: {in_rate} Hz, {in_channels} ch -> {} Hz mono",
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
            let (frames_read, _buffer_info) = capture_client
                .read_from_device(&mut byte_buf)
                .map_err(|e| e.to_string())?;
            if frames_read == 0 {
                break;
            }

            let sample_count = frames_read as usize * in_channels as usize;
            let interleaved: Vec<f32> = byte_buf[..sample_count * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            let mono_resampled = resampler.process(&interleaved);
            if !mono_resampled.is_empty() {
                let rms_level = compute_rms(&mono_resampled);
                let _ = tx.send(AudioChunk {
                    source: AudioSource::Microphone,
                    samples: mono_resampled,
                    rms_level,
                });
            }
        }
    }

    let _ = audio_client.stop_stream();
    log::info!("microphone capture stopped");
    Ok(())
}
