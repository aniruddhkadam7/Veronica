//! Captures real microphone audio through the *production* path and writes it
//! to a WAV file so the exact samples entering the STT recognizer can be
//! inspected independently.
//!
//!     cargo run --bin mic_probe -- [seconds] [out.wav]
//!
//! Drives `MicrophoneCapture` — the same WASAPI capture + downmix + resample
//! code `voice_command::start_mic_assistant` feeds the sidecar with — and tees
//! every chunk to disk as 16kHz mono PCM16, byte-identical to what
//! `SttSidecar::send_samples` would encode and send over stdin.
//!
//! Reports level statistics (peak, RMS, clipping, silence ratio) so an input
//! that is too quiet, clipped, or gated by third-party audio software is
//! obvious from the console alone without opening the WAV.

use std::io::Write;
use std::time::Duration;

use desktop_lib::audio::{MicrophoneCapture, StopSignal};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    let seconds: u64 = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let out_path = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "mic_probe.wav".to_string());

    println!("=== microphone probe (production capture path) ===");
    println!("duration: {seconds}s");
    println!("output  : {out_path}");
    println!("\nSpeak now.\n");

    let (tx, rx) = crossbeam_channel::unbounded();
    let stop = StopSignal::new();

    let handle = match MicrophoneCapture::start(tx, stop.clone(), None) {
        Ok(h) => h,
        Err(err) => {
            eprintln!("mic capture failed: {err}");
            std::process::exit(1);
        }
    };

    let collector = std::thread::spawn(move || {
        let mut all: Vec<f32> = Vec::new();
        for chunk in rx.iter() {
            all.extend_from_slice(&chunk.samples);
        }
        all
    });

    std::thread::sleep(Duration::from_secs(seconds));
    stop.stop();
    let _ = handle.join();
    let samples = collector.join().unwrap_or_default();

    if samples.is_empty() {
        eprintln!("\nNO AUDIO CAPTURED — the capture thread produced zero samples.");
        std::process::exit(2);
    }

    // Statistics that distinguish "mic is fine" from the common failure modes:
    // too quiet to decode, clipped/distorted, or mostly gated to silence.
    let peak = samples.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    let rms = (samples.iter().map(|s| (*s as f64) * (*s as f64)).sum::<f64>() / samples.len() as f64).sqrt();
    let clipped = samples.iter().filter(|s| s.abs() >= 0.999).count();
    let near_silent = samples.iter().filter(|s| s.abs() < 0.001).count();
    let duration_s = samples.len() as f64 / 16_000.0;

    println!("\n=== capture statistics ===");
    println!("samples       : {} ({duration_s:.2}s at 16kHz)", samples.len());
    println!("peak amplitude: {peak:.4}");
    println!("rms amplitude : {rms:.4}");
    println!("clipped       : {clipped} ({:.2}%)", 100.0 * clipped as f64 / samples.len() as f64);
    println!("near-silent   : {near_silent} ({:.2}%)", 100.0 * near_silent as f64 / samples.len() as f64);

    if peak < 0.05 {
        println!("\nWARNING: peak below 0.05 — input is very quiet. Recognition will suffer.");
    }
    if clipped > samples.len() / 1000 {
        println!("\nWARNING: significant clipping — input is too hot and distorted.");
    }

    if let Err(err) = write_wav(&out_path, &samples) {
        eprintln!("failed to write {out_path}: {err}");
        std::process::exit(1);
    }
    println!("\nwrote {out_path}");
}

/// Writes 16kHz mono PCM16 — the exact encoding `SttSidecar::send_samples`
/// produces, so the file contains precisely what the recognizer receives.
fn write_wav(path: &str, samples: &[f32]) -> std::io::Result<()> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let v = (clamped * i16::MAX as f32) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }

    let mut f = std::fs::File::create(path)?;
    let data_len = pcm.len() as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?; // PCM fmt chunk size
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // mono
    f.write_all(&16_000u32.to_le_bytes())?; // sample rate
    f.write_all(&32_000u32.to_le_bytes())?; // byte rate = 16000 * 1 * 2
    f.write_all(&2u16.to_le_bytes())?; // block align
    f.write_all(&16u16.to_le_bytes())?; // bits per sample
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    f.write_all(&pcm)?;
    Ok(())
}
