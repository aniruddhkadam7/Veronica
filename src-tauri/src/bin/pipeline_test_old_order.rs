//! Headless end-to-end test of the production recording pipeline.
//!
//!     cargo run --bin pipeline_test -- [seconds]
//!     cargo run --bin pipeline_test -- 40 --lifecycle
//!
//! Drives the *same* code the Tauri command runs — `SystemAudioCapture`,
//! `run_stt_pipeline` (gap-fill included), `SttSidecar`, `TranscriptManager` —
//! with console callbacks in place of Tauri events. The only thing not
//! exercised is the window itself.
//!
//! Play audio through the default output device while it runs. Partials are
//! rewritten in place; finals print on their own line with the latency from
//! end-of-speech, which is the number the product cares about.
//!
//! `--lifecycle` additionally exercises Start → Pause → Resume → Stop on a
//! timer and asserts the transcript survives the pause untouched.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use desktop_lib::audio::{run_stt_pipeline, PauseSignal, StopSignal, SystemAudioCapture};
use desktop_lib::stt::{SttEventKind, SttSidecar};
use desktop_lib::transcript::TranscriptManager;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args: Vec<String> = std::env::args().collect();
    let seconds: u64 = args
        .get(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(30);
    let lifecycle = args.iter().any(|a| a == "--lifecycle");

    println!("=== production STT pipeline test ===");
    println!("duration : {seconds}s");
    println!("lifecycle: {}", if lifecycle { "Start/Pause/Resume/Stop" } else { "Start/Stop only" });
    println!("Play audio through your speakers now.\n");

    let (audio_tx, audio_rx) = crossbeam_channel::unbounded();
    let (stt_tx, stt_rx) = std::sync::mpsc::channel();
    let stop = StopSignal::new();
    let pause = PauseSignal::new();

    let audio_thread = match SystemAudioCapture::start(audio_tx, stop.clone()) {
        Ok(handle) => handle,
        Err(err) => {
            eprintln!("capture failed: {err}");
            std::process::exit(1);
        }
    };

    // `None`: this headless binary has no `PerformanceManager`/`AppHandle` —
    // preserves the pre-existing STT_NUM_THREADS env-var pass-through
    // behavior unchanged, same as before this parameter existed.
    let sidecar = match SttSidecar::spawn(desktop_lib::audio::AudioSource::SystemAudio, stt_tx, None, None) {
        Ok(sidecar) => sidecar,
        Err(err) => {
            eprintln!("sidecar failed: {err}");
            std::process::exit(1);
        }
    };

    let transcript = Arc::new(Mutex::new(TranscriptManager::new()));
    let partial_count = Arc::new(AtomicUsize::new(0));
    let final_count = Arc::new(AtomicUsize::new(0));

    let started = Instant::now();
    let transcript_for_events = transcript.clone();
    let partials = partial_count.clone();
    let finals = final_count.clone();

    // Mirrors the `stt-events-forwarder` thread in commands.rs, including
    // running every event through the real TranscriptManager.
    let events_thread = std::thread::spawn(move || {
        // Tracks when audio last looked like speech, so finalization latency is
        // measured from end-of-speech rather than from process start.
        let mut last_partial_at: Option<Instant> = None;
        for event in stt_rx.iter() {
            let at = started.elapsed().as_secs_f64();
            match event.kind {
                SttEventKind::Partial => {
                    partials.fetch_add(1, Ordering::Relaxed);
                    last_partial_at = Some(Instant::now());
                    print!("\r\x1b[K[{at:6.2}s] … {}", event.text);
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
                SttEventKind::Final => {
                    finals.fetch_add(1, Ordering::Relaxed);
                    let settle = last_partial_at
                        .map(|t| t.elapsed().as_millis())
                        .unwrap_or(0);
                    println!("\r\x1b[K[{at:6.2}s] FINAL (+{settle}ms after last partial)");
                    println!("           {}", event.text);
                    last_partial_at = None;
                }
            }
            if let Ok(mut manager) = transcript_for_events.lock() {
                manager.apply_event(event);
            }
        }
    });

    let pause_for_pipeline = pause.clone();
    let pipeline_thread = std::thread::spawn(move || {
        run_stt_pipeline(audio_rx, sidecar, pause_for_pipeline, None, |_chunk| {});
    });

    if lifecycle {
        let quarter = seconds / 4;
        std::thread::sleep(Duration::from_secs(quarter));

        println!("\n--- PAUSE ---");
        pause.set_paused(true);
        if let Ok(mut m) = transcript.lock() {
            m.mark_paused();
        }
        let segments_at_pause = transcript
            .lock()
            .map(|m| m.session().segments.len())
            .unwrap_or(0);
        println!("    segments at pause: {segments_at_pause}");
        std::thread::sleep(Duration::from_secs(quarter));

        let segments_after_pause = transcript
            .lock()
            .map(|m| m.session().segments.len())
            .unwrap_or(0);
        // The transcript must not shrink across a pause. It may grow by one if
        // the flush-on-pause finalized a question that was mid-decode, which is
        // the intended behaviour.
        if segments_after_pause < segments_at_pause {
            println!("    FAIL: transcript lost segments during pause \
                      ({segments_at_pause} -> {segments_after_pause})");
        } else {
            println!("    OK: transcript intact during pause ({segments_after_pause} segments)");
        }

        println!("--- RESUME ---");
        pause.set_paused(false);
        if let Ok(mut m) = transcript.lock() {
            m.mark_resumed();
        }
        std::thread::sleep(Duration::from_secs(seconds - 2 * quarter));
    } else {
        std::thread::sleep(Duration::from_secs(seconds));
    }

    println!("\n--- STOP ---");
    stop.stop();
    let _ = audio_thread.join();
    let _ = pipeline_thread.join();
    let _ = events_thread.join();

    if let Ok(mut m) = transcript.lock() {
        m.mark_stopped();
    }

    let manager = transcript.lock().unwrap();
    let session = manager.session();
    println!("\n=== RESULT ===");
    println!("partial events : {}", partial_count.load(Ordering::Relaxed));
    println!("final events   : {}", final_count.load(Ordering::Relaxed));
    println!("segments       : {}", session.segments.len());
    println!("words          : {}", session.word_count());
    println!("\n--- transcript ---");
    for segment in &session.segments {
        if let Some(text) = segment.final_text.as_deref() {
            println!("  {text}");
        }
    }
}
