//! Streams raw linear16 PCM audio from Deepgram Flux onto the default
//! output device as it arrives, via `rodio`.
//!
//! `rodio::OutputStream` is not `Send`/`Sync` (it wraps a raw pointer
//! internally), so it cannot be held inside `AppState` or moved between
//! threads — attempting to store a `PlaybackQueue` containing one directly
//! made every Tauri command taking `State<'_, AppState>` fail to compile
//! (`*mut () cannot be shared between threads safely`). The fix: the
//! `OutputStream`/`Sink` are created and live entirely on one dedicated
//! thread (`spawn_player_thread`), never leaving it; every other part of
//! this app only ever holds a plain `mpsc::Sender`, which is `Send`.
//!
//! Unlike the old per-sentence-HTTP-request TTS providers this app used
//! before Flux, there is no chunk-reordering to do here: one answer speaks
//! through exactly one Flux WebSocket session (see `tts::deepgram_flux`),
//! and its audio frames arrive over that single connection already in
//! playback order — `rodio::Sink::append` playing buffers strictly in the
//! order they were appended is then exactly the order they need to play in,
//! with no `seq`/holding-buffer bookkeeping required.

use std::sync::mpsc;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait};
use rodio::{OutputStream, Sink};

use super::deepgram_flux::SAMPLE_RATE;
use super::TtsSpeakingSignal;

/// How often the player thread checks `sink.empty()` to keep
/// `TtsSpeakingSignal`'s sink-active flag current — both while otherwise
/// idle (no new commands arriving) AND after every command, so a sentence
/// that finishes playing mid-burst of chunks is noticed promptly rather than
/// only once the channel goes quiet. Short enough that the mic un-mutes
/// promptly after Veronica stops talking (see `voice_command::mod`'s
/// mic-assistant pump, which checks this signal before forwarding audio to
/// STT — this is what stops Veronica hearing and re-transcribing her own
/// speech), long enough to be a negligible amount of thread wake-ups. Safe
/// to have this flag lag reality by up to this interval: `TtsSpeakingSignal`
/// also tracks `pending_sentences` (see `tts::mod`) independently, which is
/// what actually keeps the mic muted through any gap where the sink is
/// briefly empty but more speech is still coming.
const IDLE_POLL_INTERVAL: Duration = Duration::from_millis(100);

/// A command sent to the dedicated playback thread.
enum PlayerCommand {
    /// One PCM chunk, already in playback order (see this module's doc).
    Chunk(Vec<u8>),
    Stop,
    /// Closes the current `OutputStream`/`Sink` and opens a fresh one on the
    /// same device — see `PlaybackHandle::reopen`'s doc for why this exists.
    Reopen,
}

/// A `Send`-safe handle to a playback thread that owns the actual
/// `rodio::OutputStream`/`Sink` internally. This is what `TtsSession` and
/// `AppState` hold — never the stream/sink themselves.
#[derive(Clone)]
pub struct PlaybackHandle {
    tx: mpsc::Sender<PlayerCommand>,
}

impl PlaybackHandle {
    /// Spawns the dedicated playback thread and opens the default audio
    /// output device on it. Blocks briefly on this call only to surface a
    /// device-open failure synchronously (no speakers, driver issue) rather
    /// than have it appear silently one PCM chunk later; the thread itself
    /// still does all playback work asynchronously from then on.
    ///
    /// `speaking` is set to `true` the moment real audio is appended to the
    /// sink and cleared once the sink genuinely finishes playing everything
    /// queued (polled — see `IDLE_POLL_INTERVAL`) — this is the ground-truth
    /// signal `voice_command::mod`'s mic-assistant pump checks before
    /// forwarding audio to STT, so Veronica's own speech (picked up
    /// acoustically by the mic, with no cable involved) doesn't get
    /// transcribed and answered as if the user said it.
    ///
    /// `device_name`: the friendly name of a specific output device to open
    /// instead of the system default (see `state::SelectedDevices`'s doc),
    /// matched against `cpal::Device::name()` — cpal has no by-ID lookup the
    /// way `wasapi::DeviceEnumerator::get_device` does for capture, so this
    /// enumerates `cpal::default_host().output_devices()` and matches by
    /// name, falling back to the default device if no device with that name
    /// is currently present (e.g. it was unplugged) rather than failing the
    /// whole session outright.
    pub fn start(speaking: TtsSpeakingSignal, device_name: Option<String>) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<PlayerCommand>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        std::thread::Builder::new()
            .name("tts-player".into())
            .spawn(move || {
                let opened = open_output_stream(device_name.as_deref());
                let (stream, sink) = match opened {
                    Ok(pair) => {
                        let _ = ready_tx.send(Ok(()));
                        pair
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(err));
                        return;
                    }
                };

                let mut player = Player::new(sink);
                // Owns the current stream so `Reopen` can drop and replace it
                // in place — `Player`/`Sink` don't need to know a swap ever
                // happened, since a fresh `Sink` is handed to them each time.
                let mut current_stream = stream;
                loop {
                    match rx.recv_timeout(IDLE_POLL_INTERVAL) {
                        Ok(PlayerCommand::Chunk(bytes)) => player.append(&bytes),
                        Ok(PlayerCommand::Stop) => player.handle_stop(),
                        Ok(PlayerCommand::Reopen) => {
                            match open_output_stream(device_name.as_deref()) {
                                Ok((new_stream, new_sink)) => {
                                    player = Player::new(new_sink);
                                    current_stream = new_stream;
                                }
                                Err(err) => {
                                    // Keep the old (possibly dead) stream
                                    // rather than tearing it down with
                                    // nothing to replace it — a turn that
                                    // fails to reopen still gets a best-effort
                                    // attempt through whatever was already
                                    // open, matching this module's existing
                                    // "never hard-fail a turn over audio
                                    // output" stance (see this fn's doc).
                                    log::warn!("tts::player: failed to reopen output stream, reusing previous one: {err}");
                                }
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                    // Refreshed after every command (not only on an idle
                    // timeout) so a sentence that finishes playing mid-burst
                    // of incoming chunks is reflected promptly — see
                    // IDLE_POLL_INTERVAL's doc for why staleness here is
                    // still safe either way.
                    speaking.set_sink_active(!player.sink_is_empty());
                }
                // Channel closed (handle dropped): let any already-queued
                // audio finish playing naturally rather than cutting it off
                // — `sink`/`stream` simply drop here once this returns. The
                // mic-mute signal is intentionally left as whatever it last
                // was; TtsSession::stop() (called before this handle would
                // ever be dropped without an explicit stop — see
                // veronica::ask_veronica) already clears it when a session
                // ends normally.
                drop(current_stream);
            })
            .map_err(|e| e.to_string())?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { tx }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err("audio player thread exited before reporting readiness".to_string()),
        }
    }

    /// Queues one PCM chunk, appended to the sink in the order this is
    /// called — see this module's doc for why no reordering is needed.
    pub fn enqueue(&self, bytes: Vec<u8>) {
        let _ = self.tx.send(PlayerCommand::Chunk(bytes));
    }

    /// Stops playback and clears anything queued — used when the user asks
    /// a new question while a previous answer is still being spoken, or
    /// disables voice output mid-answer.
    pub fn stop(&self) {
        let _ = self.tx.send(PlayerCommand::Stop);
    }

    /// Closes and reopens the output stream/sink on the same configured
    /// device — call once at the start of every turn (see
    /// `TtsSession::begin_turn`).
    ///
    /// `TtsSession`/`PlaybackHandle` are now persistent, cross-turn objects
    /// (opened once when the mic-assistant session activates, not per
    /// answer — see `tts::mod`'s doc), which means the underlying WASAPI
    /// stream can sit open with no audio flowing through it for the entire
    /// silent gap between turns. A Bluetooth output device (verified
    /// live-observed on a MITASHI 106 headset) drops an idle audio session
    /// under exactly that condition, and `cpal`/`rodio` 0.20 give the
    /// application no callback or health check for that — `rodio::Sink`'s
    /// `error_callback` is internal and unconditionally just logs via
    /// `eprintln!`/`tracing::error!` (see `rodio::stream`'s
    /// `new_output_stream_with_format`), so a dead stream otherwise stays
    /// silently dead for the rest of the app session with no way to detect
    /// it from here. The old, working pre-redesign code never held a stream
    /// open long enough between turns to hit this.
    ///
    /// Reopening is a local device operation (typically tens of
    /// milliseconds), not a network round trip, so doing it every turn does
    /// not reintroduce the per-turn latency the persistent-session redesign
    /// was meant to remove — only the Deepgram Flux *network* connection
    /// needs to stay persistent for that latency win, and it still does
    /// (this only touches the local playback stream).
    pub fn reopen(&self) {
        let _ = self.tx.send(PlayerCommand::Reopen);
    }
}

/// Opens `device_name` (matched against `cpal::Device::name()`) if given and
/// still present, otherwise the system default output device, and creates
/// its `Sink` — see `PlaybackHandle::start`'s doc.
fn open_output_stream(device_name: Option<&str>) -> Result<(OutputStream, Sink), String> {
    let (stream, handle) = match device_name {
        Some(name) => {
            let found = cpal::default_host()
                .output_devices()
                .ok()
                .and_then(|mut devices| devices.find(|d| d.name().map(|n| n == name).unwrap_or(false)));
            match found {
                Some(device) => OutputStream::try_from_device(&device)
                    .map_err(|e| format!("failed to open output device \"{name}\": {e}"))?,
                None => {
                    log::warn!("selected output device \"{name}\" not found, falling back to the system default");
                    OutputStream::try_default()
                        .map_err(|e| format!("failed to open default audio output device: {e}"))?
                }
            }
        }
        None => OutputStream::try_default()
            .map_err(|e| format!("failed to open default audio output device: {e}"))?,
    };
    let sink = Sink::try_new(&handle).map_err(|e| format!("failed to create audio sink: {e}"))?;
    Ok((stream, sink))
}

/// The operations `Player` needs from a sink: append a buffer of samples to
/// play next, stop/clear everything queued, and report whether everything
/// appended has finished playing. `rodio::Sink` is the real implementation;
/// tests substitute a `Vec`-recording fake so the decode/append logic can be
/// verified without a real audio output device, which unit tests can't rely
/// on having.
trait AudioSink {
    fn append_samples(&self, samples: Vec<i16>);
    fn stop_all(&self);
    fn is_empty(&self) -> bool;
}

impl AudioSink for Sink {
    fn append_samples(&self, samples: Vec<i16>) {
        self.append(rodio::buffer::SamplesBuffer::new(1, SAMPLE_RATE, samples));
    }

    fn stop_all(&self) {
        self.stop();
    }

    fn is_empty(&self) -> bool {
        self.empty()
    }
}

/// Decodes and appends PCM chunks to the sink, in the order `append` is
/// called — pulled out of the thread closure into its own type so this can
/// be unit-tested independently of spawning a real thread/audio device.
struct Player<S: AudioSink> {
    sink: S,
    decoder: PcmDecoder,
}

impl<S: AudioSink> Player<S> {
    fn new(sink: S) -> Self {
        Self { sink, decoder: PcmDecoder::new() }
    }

    fn append(&mut self, bytes: &[u8]) {
        let samples = self.decoder.decode(bytes);
        if !samples.is_empty() {
            log::debug!("tts::player: appending {} samples to sink", samples.len());
            self.sink.append_samples(samples);
        }
    }

    fn handle_stop(&mut self) {
        self.sink.stop_all();
    }

    /// Whether the sink has finished playing everything appended to it so
    /// far — this is the mic-mute signal's "is audio currently sounding"
    /// check.
    fn sink_is_empty(&self) -> bool {
        self.sink.is_empty()
    }
}

/// Decodes raw linear16 little-endian PCM bytes into `i16` samples,
/// buffering a possible odd trailing byte across calls (a Flux chunk
/// boundary is not guaranteed to land on a 2-byte sample boundary). One
/// decoder instance lives for the whole player thread, correctly carrying
/// leftover-byte state across every chunk of the one continuous Flux stream
/// it decodes.
struct PcmDecoder {
    leftover: Option<u8>,
}

impl PcmDecoder {
    fn new() -> Self {
        Self { leftover: None }
    }

    fn decode(&mut self, bytes: &[u8]) -> Vec<i16> {
        let mut samples = Vec::with_capacity(bytes.len() / 2 + 1);
        let mut iter = bytes.iter().copied();

        if let Some(low) = self.leftover.take() {
            if let Some(high) = iter.next() {
                samples.push(i16::from_le_bytes([low, high]));
            } else {
                // A single trailing byte with nothing to pair it with yet —
                // put it right back and wait for the next chunk.
                self.leftover = Some(low);
                return samples;
            }
        }

        loop {
            let Some(low) = iter.next() else { break };
            match iter.next() {
                Some(high) => samples.push(i16::from_le_bytes([low, high])),
                None => {
                    self.leftover = Some(low);
                    break;
                }
            }
        }

        samples
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn decodes_even_length_chunk_fully() {
        let mut d = PcmDecoder::new();
        // Two little-endian i16 samples: 1 and -1.
        let bytes = [1u8, 0, 0xFF, 0xFF];
        assert_eq!(d.decode(&bytes), vec![1i16, -1i16]);
    }

    #[test]
    fn buffers_odd_trailing_byte_across_calls() {
        let mut d = PcmDecoder::new();
        // First call: one full sample (1) plus one dangling low byte.
        let first = [1u8, 0, 0x34];
        assert_eq!(d.decode(&first), vec![1i16]);
        // Second call: the dangling byte pairs with this call's first byte.
        let second = [0x12u8, 5, 0];
        let decoded = d.decode(&second);
        assert_eq!(decoded[0], i16::from_le_bytes([0x34, 0x12]));
        assert_eq!(decoded[1], 5i16);
    }

    #[test]
    fn empty_input_produces_no_samples() {
        let mut d = PcmDecoder::new();
        assert_eq!(d.decode(&[]), Vec::<i16>::new());
    }

    /// Records every appended sample buffer, in the order `Player` appended
    /// them — this order is exactly what a listener would hear, so
    /// asserting on it is asserting on the actual user-facing behavior.
    #[derive(Default)]
    struct FakeSink {
        appended: RefCell<Vec<Vec<i16>>>,
        stopped: RefCell<bool>,
    }

    impl AudioSink for FakeSink {
        fn append_samples(&self, samples: Vec<i16>) {
            self.appended.borrow_mut().push(samples);
        }
        fn stop_all(&self) {
            *self.stopped.borrow_mut() = true;
        }
        fn is_empty(&self) -> bool {
            true
        }
    }

    /// One i16 sample as its 2-byte little-endian PCM encoding — lets tests
    /// build chunk bytes from plain sample values instead of hand-writing
    /// byte pairs everywhere.
    fn pcm(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    #[test]
    fn chunks_play_in_the_order_appended() {
        let mut player = Player::new(FakeSink::default());
        player.append(&pcm(&[1, 2]));
        player.append(&pcm(&[3, 4]));
        assert_eq!(player.sink.appended.borrow().as_slice(), &[vec![1i16, 2], vec![3i16, 4]]);
    }

    #[test]
    fn stop_stops_the_sink() {
        let mut player = Player::new(FakeSink::default());
        player.append(&pcm(&[1]));
        player.handle_stop();
        assert!(*player.sink.stopped.borrow());
    }
}
