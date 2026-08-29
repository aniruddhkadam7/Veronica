//! Text-to-speech: Deepgram Flux (`flux-sienna-en`, over a streaming `wss://`
//! session) is the only provider. There is no local model and no fallback —
//! if the Deepgram API key is missing or the session fails (connect
//! failure, network error), the rest of that answer is simply not spoken
//! and a warning is logged; the text answer itself is never affected, since
//! TTS is purely an add-on to the existing LLM -> text pipeline (see
//! `veronica::ask_veronica`).
//!
//! `TtsSession` is now a persistent, cross-turn object — created once when
//! the mic-assistant session activates and held in `AppState` for the whole
//! session's lifetime, not recreated per answer. Each turn calls
//! `begin_turn()` then feeds text via `speak()`/`speak_now()` (one sentence
//! at a time as `SentenceChunker` completes them, or immediately for a short
//! response with no chunker involved) and `finish()` once the turn's text is
//! done. Consecutive turns reuse the same underlying Flux WebSocket
//! connection whenever the server has kept it open (see
//! `deepgram_flux::FluxSession::is_alive`), rather than paying a fresh
//! TCP+TLS+WebSocket handshake on every single turn — only a hard `stop()`
//! (barge-in) or a genuine connection failure opens a new one.
//!
//! The *local* playback device is the one part of this NOT kept open across
//! turns: `begin_turn()` also reopens the output stream/sink every turn (see
//! `player::PlaybackHandle::reopen`'s doc) — a persistent WASAPI stream with
//! no audio flowing through it during the silent gap between turns is
//! exactly the condition that drops a Bluetooth output device's audio
//! session, and `rodio` 0.20 gives no way to detect that from here. This is
//! a cheap local device reopen, not a network round trip, so it doesn't
//! reintroduce the per-turn latency the persistent Flux *connection* above
//! was specifically introduced to remove.
//!
//! `TtsSession` is `Send`/`Sync` (safe to store in `AppState`, a Tauri
//! `State`) — see `player`'s module doc for why that required keeping
//! `rodio`'s actual output stream confined to its own dedicated thread
//! rather than held here directly.

mod chunker;
mod deepgram_flux;
mod player;

pub use chunker::SentenceChunker;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use deepgram_flux::FluxSession;
use player::PlaybackHandle;

/// RMS of a raw linear16 (little-endian i16) PCM chunk, scaled to 0.0-1.0 —
/// the same 0.0-1.0 meter convention as `audio::compute_rms` (mic capture),
/// but operating directly on the raw bytes Flux sends rather than decoded
/// `f32` samples, since this runs on every chunk in the hot `on_audio` path
/// and a UI level meter doesn't need a second full decode pass or to persist
/// a leftover odd trailing byte across calls (`player::PcmDecoder` already
/// does that correctly for the audio actually played; dropping one trailing
/// byte here, at most once per chunk, is imperceptible for a level meter).
fn pcm_i16le_rms(bytes: &[u8]) -> f32 {
    let pairs = bytes.chunks_exact(2);
    let mut sum_sq = 0f64;
    let mut count = 0usize;
    for pair in pairs {
        let sample = i16::from_le_bytes([pair[0], pair[1]]) as f64 / i16::MAX as f64;
        sum_sq += sample * sample;
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    ((sum_sq / count as f64).sqrt() as f32).min(1.0)
}

/// Ground truth for "is Veronica's own voice coming out of the speakers
/// right now" — checked by `voice_command::mod`'s mic-assistant pump before
/// forwarding audio to STT, so Veronica's own TTS output (picked up
/// acoustically by the mic, no cable involved, no toggle needed to trigger
/// it — just a mic and speakers in the same room) doesn't get transcribed
/// and answered as if the user said it.
///
/// `true` for as long as EITHER this answer's Flux session hasn't been
/// flushed/closed yet, OR the player's audio sink hasn't finished playing
/// everything already appended — both conditions matter independently: a
/// long pause between the session opening and its first audio chunk
/// arriving must not flip this to `false` for even a moment, and a
/// still-in-the-sink final sentence must keep it `true` after the session
/// itself has already closed. Combining both into one type (rather than
/// checking them separately at each call site) is what makes that
/// combination correct in one place, not something every caller has to get
/// right.
#[derive(Clone, Default)]
pub struct TtsSpeakingSignal {
    /// Whether an answer's Flux session is currently open (from
    /// `TtsSession::start` until `stop()`/the LLM stream ending and the
    /// session finishing). `AtomicBool`, not a counter: unlike the old
    /// per-sentence-HTTP-request model where multiple concurrent requests
    /// could be in flight at once, there is now at most one Flux session
    /// per `TtsSession`, so a single flag is enough.
    session_open: Arc<AtomicBool>,
    /// Whether the player's sink has audio queued or playing right now.
    /// Distinct from `session_open` — see the struct doc.
    sink_active: Arc<AtomicBool>,
}

impl TtsSpeakingSignal {
    fn set_session_open(&self, open: bool) {
        self.session_open.store(open, Ordering::SeqCst);
    }

    fn set_sink_active(&self, active: bool) {
        self.sink_active.store(active, Ordering::SeqCst);
    }

    /// Whether the mic-assistant pump should currently withhold audio from
    /// STT. See the struct doc for why both conditions are checked.
    pub fn is_speaking(&self) -> bool {
        self.session_open.load(Ordering::SeqCst) || self.sink_active.load(Ordering::SeqCst)
    }

    /// Forces both conditions clear — used by `TtsSession::stop()` so an
    /// interrupted answer un-mutes the mic immediately rather than waiting
    /// for the Flux session to notice it was stopped.
    fn force_clear(&self) {
        self.session_open.store(false, Ordering::SeqCst);
        self.sink_active.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone)]
pub struct TtsSession {
    player: PlaybackHandle,
    /// The live Flux session for this answer, created lazily on the first
    /// `speak()` call rather than in `start()` — `start()` must succeed
    /// whenever the audio device opens, even with no API key configured
    /// yet or before the answer's first sentence exists to speak (mirrors
    /// the old per-sentence client's "never touches the network until
    /// there's something to say" behavior). `Mutex<Option<..>>` because
    /// `TtsSession` is cheaply `Clone`d (see `veronica::ask_veronica`,
    /// which holds two clones of the same session) and every clone must
    /// share the same one-session-per-answer state, not create its own.
    flux: Arc<Mutex<Option<FluxSession>>>,
    stopped: Arc<AtomicBool>,
    speaking: TtsSpeakingSignal,
    /// Only used to emit `veronica:error` when this answer's Flux session
    /// fails (see `speak()`) — the orb widgets' only path to learning about
    /// a TTS failure, which previously had no way to reach the frontend at
    /// all (a background thread, no command in flight to reject). `None` in
    /// the `#[cfg(test)]` unit test below, which has no `AppHandle` and
    /// simply emits nothing on failure, exactly as before this field
    /// existed.
    app: Option<tauri::AppHandle>,
    /// Fired at most once per turn, on the first raw PCM chunk received
    /// back from Flux for that turn — real per-turn `tts_first_audio`
    /// telemetry (see `hardware::telemetry::PipelineStage::TtsFirstAudio`),
    /// not an approximation. Set via `set_turn_audio_hook` at the start of
    /// each turn (typically right after `begin_turn()`); taken (and so
    /// cleared) the moment it fires, so a later chunk in the same turn — or
    /// any chunk in a turn that never set a hook — does nothing here.
    /// `PlaybackStarted` is deliberately not tracked as a separate signal:
    /// `on_audio` below calls `player.enqueue()` synchronously right after
    /// this fires, and rodio begins playing newly-appended data essentially
    /// immediately, so the gap between "PCM bytes received" and "audio
    /// reaching the speakers" is sub-millisecond — not a real latency worth
    /// its own instrumentation point, unlike the network/synthesis gap this
    /// hook does measure (LLM first token -> TTS first audio).
    on_first_audio_this_turn: Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>,
}

impl TtsSession {
    /// Opens the default audio output device (on its own dedicated thread —
    /// see `player`'s module doc) and starts the background PCM relay.
    /// Fails only if the device itself can't be opened (no speakers, driver
    /// issue) — never touches the network or the Deepgram API key here, so
    /// a missing/invalid key surfaces on the first `speak()` call instead of
    /// failing session creation outright.
    ///
    /// `speaking` is threaded through from `AppState` (see
    /// `veronica::ask_veronica`) rather than created fresh here: it must be
    /// the SAME signal the mic-assistant pump reads, shared across the
    /// whole app for the app's whole lifetime, not scoped to one answer's
    /// session — a new `TtsSession` is created per answer, but there is only
    /// ever one mic-mute signal.
    pub fn start(speaking: TtsSpeakingSignal, app: Option<tauri::AppHandle>) -> Result<Self, String> {
        let device_name = app
            .as_ref()
            .and_then(|app| tauri::Manager::state::<crate::state::AppState>(app).selected_devices.output());
        let player = PlaybackHandle::start(speaking.clone(), device_name)?;
        Ok(Self {
            player,
            flux: Arc::new(Mutex::new(None)),
            stopped: Arc::new(AtomicBool::new(false)),
            speaking,
            app,
            on_first_audio_this_turn: Arc::new(Mutex::new(None)),
        })
    }

    /// Registers a one-shot callback for this turn's first received audio
    /// chunk — see the `on_first_audio_this_turn` field doc. Call once per
    /// turn, after `begin_turn()`, only when the caller actually wants this
    /// signal (real callers pass a closure that marks `TurnTelemetry`;
    /// nothing is scheduled/polled when no hook is set).
    pub fn set_turn_audio_hook(&self, hook: impl FnOnce() + Send + 'static) {
        *self.on_first_audio_this_turn.lock().unwrap() = Some(Box::new(hook));
    }

    /// Streams one sentence/chunk of text into the current Flux turn,
    /// opening the WebSocket session on the very first call (see the `flux`
    /// field doc) or reusing whatever session is already open from a prior
    /// turn — `TtsSession` is now a per-app-session, cross-turn object (see
    /// `veronica::ask_veronica`), not a fresh one per answer, so most calls
    /// reuse an already-connected socket rather than paying a new
    /// TCP+TLS+WebSocket handshake on every turn. If the previously-open
    /// session's I/O thread has since exited (the server closed the
    /// connection after the last turn's `Flush` — some turn-based streaming
    /// protocols do this) `FluxSession::is_alive()` catches that and a fresh
    /// session is opened transparently, exactly as if this were the first
    /// call — see that method's doc.
    ///
    /// Any connect/send failure is logged and the rest of this answer is
    /// silently skipped — never a fallback to another engine (there is
    /// none), never an error surfaced to the answer text itself.
    pub fn speak(&self, text: &str) {
        if self.stopped.load(Ordering::SeqCst) {
            return;
        }
        // Set on every call, not only when a session is freshly created —
        // reusing an already-open connection is still "speaking" for the
        // mic-mute signal's purposes; this used to only be set inside the
        // creation branch below, which was correct back when every turn
        // always created a fresh session, but would otherwise silently never
        // mute the mic on a turn that reuses an existing connection.
        self.speaking.set_session_open(true);
        let mut guard = self.flux.lock().unwrap();
        let needs_new_session = !matches!(guard.as_ref(), Some(session) if session.is_alive());
        if needs_new_session {
            *guard = None;
            let player = self.player.clone();
            let stopped = self.stopped.clone();
            let speaking = self.speaking.clone();
            let app = self.app.clone();
            let stopped_for_audio = stopped.clone();
            let audio_hook = self.on_first_audio_this_turn.clone();
            let app_for_level = self.app.clone();
            let on_audio = move |chunk: &[u8]| {
                if stopped_for_audio.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(hook) = audio_hook.lock().unwrap().take() {
                    hook();
                }
                // Emitted before `enqueue` (not after, and not from the
                // player thread) so this is the lowest-latency point for the
                // orb's "speaking" animation to react to real TTS output —
                // the instant Deepgram's raw PCM arrives over the WebSocket,
                // same reasoning as `on_first_audio_this_turn` above. A
                // dedicated event rather than reusing `audio:level`/
                // `AudioSource`: those tag microphone/system *capture*
                // sources for STT, and TTS playback isn't a capture source.
                if let Some(app) = app_for_level.as_ref() {
                    use tauri::Emitter;
                    let rms_level = pcm_i16le_rms(chunk);
                    let _ = app.emit("tts:audio-level", rms_level);
                }
                player.enqueue(chunk.to_vec());
            };
            let speaking_for_error = speaking.clone();
            let on_error = move |err: deepgram_flux::FluxError| {
                log::warn!("Deepgram Flux TTS failed, rest of this answer won't be spoken: {err}");
                if let Some(app) = app.as_ref() {
                    use tauri::Emitter;
                    let _ = app.emit("veronica:error", format!("Voice output failed: {err}"));
                }
                speaking_for_error.set_session_open(false);
            };
            match FluxSession::start(on_audio, on_error) {
                Ok(session) => {
                    *guard = Some(session);
                }
                Err(err) => {
                    log::warn!("Deepgram Flux TTS unavailable for this answer: {err}");
                    if let Some(app) = self.app.as_ref() {
                        use tauri::Emitter;
                        let _ = app.emit("veronica:error", format!("Voice output failed: {err}"));
                    }
                    return;
                }
            }
        }
        if let Some(session) = guard.as_ref() {
            session.speak(text);
        }
    }

    /// Marks the end of this answer's text — call once after the LLM
    /// stream ends (and any trailing chunk has been sent via `speak()`), so
    /// Flux knows to finalize the turn. A no-op if `speak()` was never
    /// called (nothing was ever spoken, so there's no session to flush).
    /// Deliberately does not close/clear the underlying connection — see
    /// this module's doc: the same `FluxSession` is reused by the next
    /// turn's `speak()` call whenever the server has kept it open.
    pub fn finish(&self) {
        if let Some(session) = self.flux.lock().unwrap().as_ref() {
            session.flush();
        }
        self.speaking.set_session_open(false);
    }

    /// Resets this session for a new turn — call once at the start of every
    /// new answer (fast-router confirmation or agent-loop response), before
    /// any `speak()`/`speak_now()` for it. Required now that `TtsSession` is
    /// a persistent, cross-turn object (see the module doc) rather than a
    /// fresh one per answer: without this, a `stop()` from an earlier turn's
    /// barge-in would leave `stopped` permanently set and silently suppress
    /// every later turn's speech too.
    pub fn begin_turn(&self) {
        self.stopped.store(false, Ordering::SeqCst);
        // Clears any hook a turn set but never actually triggered (e.g. a
        // turn that decided not to speak after all) — otherwise it would
        // incorrectly fire on some LATER turn's first chunk instead of
        // never firing at all.
        self.on_first_audio_this_turn.lock().unwrap().take();
        // Reopens the local playback device every turn — see
        // `PlaybackHandle::reopen`'s doc for why a persistent cross-turn
        // session otherwise silently loses audio on a Bluetooth output
        // device after an idle gap. Cheap (a local device open, not a
        // network call), so this doesn't reintroduce the latency the
        // persistent-session redesign removed.
        self.player.reopen();
    }

    /// Speaks `text` immediately and finalizes the turn in one call — for
    /// short, already-complete responses (a fast-router confirmation like
    /// "Opening VS Code.", or a short final answer) that have no reason to
    /// go through `SentenceChunker`'s sentence-boundary buffering. Skips the
    /// chunker's punctuation-boundary wait entirely, so a short response
    /// with no trailing punctuation is still guaranteed to be spoken (see
    /// the audit finding this fixes: a chunk that never hits a sentence
    /// boundary was previously never released to `speak()` at all).
    pub fn speak_now(&self, text: &str) {
        self.speak(text);
        self.finish();
    }

    /// Stops playback immediately and cancels any in-flight Flux
    /// synthesis — used when a new question arrives while this answer is
    /// still being spoken (barge-in).
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Some(session) = self.flux.lock().unwrap().take() {
            session.interrupt();
        }
        self.player.stop();
        // Forced rather than waiting for the Flux session to reach its own
        // close: stop() must unmute the mic immediately (the user is about
        // to speak their next question), not whenever an already-abandoned
        // session happens to notice `stopped` and unwind — that could be
        // seconds away on a slow connection.
        self.speaking.force_clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `TtsSession::start()` opens a real audio device, which is not
    // reliably available in a sandboxed/headless CI runner — mirrors
    // `api_key_store`'s real-Credential-Manager test being `#[ignore]`'d
    // for the same class of reason. Run manually with `cargo test --
    // --ignored` on a real desktop session.
    #[test]
    #[ignore = "requires a real audio output device — not available in a sandboxed/headless test runner"]
    fn session_starts_and_stops_without_panicking() {
        let speaking = TtsSpeakingSignal::default();
        let session = TtsSession::start(speaking, None).expect("failed to start session on a machine with real audio");
        session.speak("this will fail without a real Deepgram API key, which is fine for this test");
        session.stop();
    }

    #[test]
    #[ignore = "requires a real audio output device — not available in a sandboxed/headless test runner"]
    fn stop_does_not_permanently_silence_a_later_turn_once_begin_turn_is_called() {
        // Regression test for the persistent-session bug: `stopped` used to
        // be a one-way flag, correct only when a fresh `TtsSession` was
        // created per answer. Now that one `TtsSession` spans many turns
        // (barge-in on turn 1 must not silence turn 2), `begin_turn()` must
        // reset it.
        let speaking = TtsSpeakingSignal::default();
        let session = TtsSession::start(speaking, None).expect("failed to start session on a machine with real audio");
        session.begin_turn();
        session.speak("turn one");
        session.stop(); // barge-in
        session.begin_turn(); // next turn starts
        session.speak("turn two — must not be silently suppressed");
        session.finish();
    }

    #[test]
    fn speaking_signal_defaults_to_not_speaking() {
        let signal = TtsSpeakingSignal::default();
        assert!(!signal.is_speaking());
    }

    #[test]
    fn speaking_signal_true_while_session_is_open() {
        let signal = TtsSpeakingSignal::default();
        signal.set_session_open(true);
        assert!(signal.is_speaking());
        signal.set_session_open(false);
        assert!(!signal.is_speaking());
    }

    #[test]
    fn speaking_signal_true_while_sink_is_active_even_with_session_closed() {
        // Models the case the Flux session has already been flushed/closed,
        // but the player's sink is still playing the last sentence's audio.
        let signal = TtsSpeakingSignal::default();
        signal.set_sink_active(true);
        assert!(signal.is_speaking());
        signal.set_sink_active(false);
        assert!(!signal.is_speaking());
    }

    #[test]
    fn speaking_signal_true_if_either_condition_holds() {
        let signal = TtsSpeakingSignal::default();
        signal.set_session_open(true);
        signal.set_sink_active(true);
        assert!(signal.is_speaking());

        signal.set_session_open(false);
        assert!(signal.is_speaking(), "sink still active, must stay true");

        signal.set_sink_active(false);
        assert!(!signal.is_speaking());
    }

    #[test]
    fn force_clear_resets_both_conditions() {
        let signal = TtsSpeakingSignal::default();
        signal.set_session_open(true);
        signal.set_sink_active(true);
        signal.force_clear();
        assert!(!signal.is_speaking());
    }
}
