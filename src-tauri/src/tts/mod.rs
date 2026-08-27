//! Text-to-speech: Deepgram Cloud (Aura-1) is the only provider. There is no
//! local model and no fallback — if `DEEPGRAM_API_KEY` is missing or a
//! request fails (timeout, rate limit, network error), that sentence is
//! simply not spoken and a warning is logged; the text answer itself is
//! never affected, since TTS is purely an add-on to the existing
//! LLM -> text pipeline (see `veronica::ask_veronica`).
//!
//! `TtsSession` owns one answer's worth of speech: created when an answer
//! with voice output enabled starts streaming, fed one sentence at a time
//! via `speak()` as `SentenceChunker` completes them, and handed off to
//! `AppState` once the LLM stream ends so playback of the last sentence(s)
//! can finish in the background after the Tauri command returns.
//!
//! `TtsSession` is `Send`/`Sync` (safe to store in `AppState`, a Tauri
//! `State`) — see `player`'s module doc for why that required keeping
//! `rodio`'s actual output stream confined to its own dedicated thread
//! rather than held here directly.

mod chunker;
mod deepgram;
mod player;

pub use chunker::SentenceChunker;

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use player::PlaybackHandle;

/// Ground truth for "is Veronica's own voice coming out of the speakers
/// right now" — checked by `voice_command::mod`'s mic-assistant pump before
/// forwarding audio to STT, so Veronica's own TTS output (picked up
/// acoustically by the mic, no cable involved, no toggle needed to trigger
/// it — just a mic and speakers in the same room) doesn't get transcribed
/// and answered as if the user said it.
///
/// `true` for as long as EITHER any `speak()` call's Deepgram request hasn't
/// finished yet, OR the player's audio sink hasn't finished playing
/// everything already appended — both conditions matter independently:
/// a long pause between sentence 1 finishing and sentence 2's (still
/// synthesizing) first chunk arriving must not flip this to `false` for
/// even a moment, and a still-in-the-sink final sentence must keep it
/// `true` after every `speak()` call has already returned. Combining both
/// into one type (rather than checking them separately at each call site)
/// is what makes that combination correct in one place, not something every
/// caller has to get right.
#[derive(Clone, Default)]
pub struct TtsSpeakingSignal {
    /// Count of `speak()` calls whose Deepgram request thread hasn't
    /// finished yet (success or failure — either way it stops counting once
    /// that thread sends its final marker). `AtomicI64`, not `AtomicU64`:
    /// signed so a bug that double-decrements is loud (goes negative,
    /// visibly wrong) rather than wrapping to a huge unsigned value that
    /// would silently jam the mic muted forever.
    pending_sentences: Arc<AtomicI64>,
    /// Whether the player's sink has audio queued or playing right now.
    /// Distinct from `pending_sentences` — see the struct doc.
    sink_active: Arc<AtomicBool>,
}

impl TtsSpeakingSignal {
    fn on_sentence_started(&self) {
        self.pending_sentences.fetch_add(1, Ordering::SeqCst);
    }

    fn on_sentence_finished(&self) {
        self.pending_sentences.fetch_sub(1, Ordering::SeqCst);
    }

    fn set_sink_active(&self, active: bool) {
        self.sink_active.store(active, Ordering::SeqCst);
    }

    /// Whether the mic-assistant pump should currently withhold audio from
    /// STT. See the struct doc for why both conditions are checked.
    pub fn is_speaking(&self) -> bool {
        self.pending_sentences.load(Ordering::SeqCst) > 0 || self.sink_active.load(Ordering::SeqCst)
    }

    /// Forces both conditions clear — used by `TtsSession::stop()` so an
    /// interrupted answer un-mutes the mic immediately rather than waiting
    /// for in-flight request threads to notice they were stopped.
    fn force_clear(&self) {
        self.pending_sentences.store(0, Ordering::SeqCst);
        self.sink_active.store(false, Ordering::SeqCst);
    }
}

#[derive(Clone)]
pub struct TtsSession {
    player: PlaybackHandle,
    stopped: Arc<AtomicBool>,
    speaking: TtsSpeakingSignal,
    /// Assigns each `speak()` call a sequence number in call order — see
    /// `player`'s module doc for why: two sentences' Deepgram requests run
    /// concurrently for latency, so their PCM chunks can arrive at the
    /// player interleaved rather than one sentence fully at a time. The
    /// player uses `seq` to always append audio to the sink in sentence
    /// order regardless of arrival order. `Arc<AtomicU64>` (not a plain
    /// counter) because `TtsSession` is cheaply `Clone`d (see
    /// `veronica::ask_veronica`, which holds two clones of the same
    /// session) — every clone must share one counter, not restart its own.
    next_seq: Arc<AtomicU64>,
    /// Only used to emit `veronica:error` when a sentence's Deepgram request
    /// fails (see `speak()`) — the orb widgets' only path to learning about a
    /// per-sentence TTS failure, which previously had no way to reach the
    /// frontend at all (a background thread, no command in flight to reject).
    /// `None` in the `#[cfg(test)]` unit test below, which has no `AppHandle`
    /// and simply emits nothing on failure, exactly as before this field
    /// existed.
    app: Option<tauri::AppHandle>,
}

impl TtsSession {
    /// Opens the default audio output device (on its own dedicated thread —
    /// see `player`'s module doc) and starts the background PCM relay.
    /// Fails only if the device itself can't be opened (no speakers, driver
    /// issue) — never touches the network or `DEEPGRAM_API_KEY` here, so a
    /// missing/invalid key surfaces per sentence in `speak()` instead of
    /// failing session creation outright.
    ///
    /// `speaking` is threaded through from `AppState` (see
    /// `veronica::ask_veronica`) rather than created fresh here: it must be
    /// the SAME signal the mic-assistant pump reads, shared across the
    /// whole app for the app's whole lifetime, not scoped to one answer's
    /// session — a new `TtsSession` is created per answer, but there is only
    /// ever one mic-mute signal.
    pub fn start(speaking: TtsSpeakingSignal, app: Option<tauri::AppHandle>) -> Result<Self, String> {
        let player = PlaybackHandle::start(speaking.clone())?;
        Ok(Self {
            player,
            stopped: Arc::new(AtomicBool::new(false)),
            speaking,
            next_seq: Arc::new(AtomicU64::new(0)),
            app,
        })
    }

    /// Synthesizes and queues one sentence. Runs the Deepgram request on a
    /// dedicated thread (see `tts::deepgram`'s module doc for why: the
    /// caller is a synchronous closure inside an async streaming loop, and
    /// must not block on network I/O) so multiple sentences from the same
    /// answer can be in flight to Deepgram concurrently for lower latency,
    /// while the player (see `player`'s module doc) still guarantees
    /// sentence-ordered, non-interleaved playback via each call's `seq`.
    ///
    /// Any Deepgram failure is logged and that one sentence is silently
    /// skipped — never a fallback to another engine (there is none), never
    /// an error surfaced to the answer text itself. A failed/skipped
    /// sentence's `seq` still gets an `is_last` marker sent (see below) so
    /// the player doesn't stall forever waiting for a sentence that will
    /// never otherwise report completion.
    pub fn speak(&self, text: &str) {
        if self.stopped.load(Ordering::SeqCst) {
            return;
        }
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let text = text.to_string();
        let player = self.player.clone();
        let stopped = self.stopped.clone();
        let speaking = self.speaking.clone();
        let app = self.app.clone();
        speaking.on_sentence_started();
        std::thread::Builder::new()
            .name("tts-deepgram-request".into())
            .spawn(move || {
                let result = deepgram::speak_streaming(&text, |chunk| {
                    if stopped.load(Ordering::SeqCst) {
                        return;
                    }
                    player.enqueue(seq, chunk.to_vec(), false);
                });
                if let Err(err) = result {
                    log::warn!("Deepgram TTS failed for one sentence, skipping it: {err}");
                    if let Some(app) = app.as_ref() {
                        use tauri::Emitter;
                        let _ = app.emit("veronica:error", format!("Voice output failed: {err}"));
                    }
                }
                // Always sent, success or failure, unless the whole session
                // was stopped: marks this seq complete so the player can
                // advance to the next sentence. A failed/empty sentence
                // still needs this — otherwise the player would wait
                // forever for a seq that will never send real audio,
                // silently stalling every sentence queued after it.
                if !stopped.load(Ordering::SeqCst) {
                    player.enqueue(seq, Vec::new(), true);
                }
                speaking.on_sentence_finished();
            })
            .ok();
    }

    /// Stops playback immediately and prevents any in-flight `speak()` calls
    /// from queuing further audio — used when a new question arrives while
    /// this answer is still being spoken.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        self.player.stop();
        // Forced rather than waiting for in-flight request threads to reach
        // their own on_sentence_finished(): stop() must unmute the mic
        // immediately (the user is about to speak their next question), not
        // whenever an already-abandoned Deepgram request happens to notice
        // `stopped` and unwind — that could be seconds away on a slow
        // connection.
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
        session.speak("this will fail without a real DEEPGRAM_API_KEY, which is fine for this test");
        session.stop();
    }

    #[test]
    fn speaking_signal_defaults_to_not_speaking() {
        let signal = TtsSpeakingSignal::default();
        assert!(!signal.is_speaking());
    }

    #[test]
    fn speaking_signal_true_while_a_sentence_is_pending() {
        let signal = TtsSpeakingSignal::default();
        signal.on_sentence_started();
        assert!(signal.is_speaking());
        signal.on_sentence_finished();
        assert!(!signal.is_speaking());
    }

    #[test]
    fn speaking_signal_true_while_sink_is_active_even_with_no_pending_sentences() {
        // Models the case every `speak()` call has already returned, but
        // the player's sink is still playing the last sentence's audio.
        let signal = TtsSpeakingSignal::default();
        signal.set_sink_active(true);
        assert!(signal.is_speaking());
        signal.set_sink_active(false);
        assert!(!signal.is_speaking());
    }

    #[test]
    fn speaking_signal_true_if_either_condition_holds() {
        let signal = TtsSpeakingSignal::default();
        signal.on_sentence_started();
        signal.set_sink_active(true);
        assert!(signal.is_speaking());

        signal.on_sentence_finished();
        assert!(signal.is_speaking(), "sink still active, must stay true");

        signal.set_sink_active(false);
        assert!(!signal.is_speaking());
    }

    #[test]
    fn force_clear_resets_both_conditions() {
        let signal = TtsSpeakingSignal::default();
        signal.on_sentence_started();
        signal.on_sentence_started();
        signal.set_sink_active(true);
        signal.force_clear();
        assert!(!signal.is_speaking());
    }
}
