//! Streams raw linear16 PCM audio from Deepgram onto the default output
//! device as it arrives, via `rodio`.
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
//! One playback thread lives for the whole answer (not one per sentence):
//! `rodio::Sink::append` plays buffers strictly in the order they were
//! appended, so consecutive sentences play back-to-back with no gap/overlap
//! ONLY IF their audio is appended in sentence order. Each sentence's
//! Deepgram request runs concurrently on its own thread (see `tts::mod`'s
//! `TtsSession`) for latency — sentence 2 starts synthesizing while sentence
//! 1 is still streaming — which means sentence 2's PCM chunks can genuinely
//! arrive at this player before sentence 1's have all arrived. This module
//! used to append every chunk in raw arrival order regardless of which
//! sentence it belonged to; two sentences streaming concurrently would have
//! their audio chopped up and interleaved, which is what sounded like
//! repeated/garbled speech. Fixed by tagging every chunk with a `seq`
//! (assigned once per `speak()` call, in call order) and holding back any
//! chunk whose sentence isn't next in line — see `Player::handle_chunk`.

use std::collections::HashMap;
use std::sync::mpsc;
use std::time::Duration;

use rodio::{OutputStream, Sink};

use super::deepgram::SAMPLE_RATE;
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
    /// One PCM chunk belonging to sentence number `seq` (assigned in
    /// `TtsSession::speak` call order). `is_last` marks the final chunk of
    /// that sentence, once its Deepgram request's response body is
    /// exhausted — needed so the player knows when it's safe to advance to
    /// the next sentence, rather than guessing from chunk size/timing.
    Chunk { seq: u64, bytes: Vec<u8>, is_last: bool },
    Stop,
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
    pub fn start(speaking: TtsSpeakingSignal) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<PlayerCommand>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();

        std::thread::Builder::new()
            .name("tts-player".into())
            .spawn(move || {
                let opened = OutputStream::try_default()
                    .map_err(|e| format!("failed to open default audio output device: {e}"))
                    .and_then(|(stream, handle)| {
                        Sink::try_new(&handle).map_err(|e| format!("failed to create audio sink: {e}")).map(|sink| (stream, sink))
                    });
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
                loop {
                    match rx.recv_timeout(IDLE_POLL_INTERVAL) {
                        Ok(PlayerCommand::Chunk { seq, bytes, is_last }) => {
                            player.handle_chunk(seq, bytes, is_last);
                        }
                        Ok(PlayerCommand::Stop) => {
                            player.handle_stop();
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
                drop(stream);
            })
            .map_err(|e| e.to_string())?;

        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self { tx }),
            Ok(Err(err)) => Err(err),
            Err(_) => Err("audio player thread exited before reporting readiness".to_string()),
        }
    }

    /// Queues one PCM chunk belonging to sentence `seq`, to be appended to
    /// the sink once every earlier-numbered sentence's audio has been fully
    /// appended (see this module's doc for why: preventing two concurrently
    /// synthesizing sentences' audio from interleaving). `is_last` marks the
    /// final chunk of that sentence.
    pub fn enqueue(&self, seq: u64, bytes: Vec<u8>, is_last: bool) {
        let _ = self.tx.send(PlayerCommand::Chunk { seq, bytes, is_last });
    }

    /// Stops playback and clears anything queued — used when the user asks
    /// a new question while a previous answer is still being spoken, or
    /// disables voice output mid-answer.
    pub fn stop(&self) {
        let _ = self.tx.send(PlayerCommand::Stop);
    }
}

/// A sentence's chunks that arrived before it was that sentence's turn to
/// play, plus whether the last of them has been seen — tracked explicitly
/// (not inferred from the chunk list) so a fully-buffered held sentence can
/// be recognized and flushed even though its `is_last` chunk arrived while
/// it was still waiting its turn.
#[derive(Default)]
struct HeldSentence {
    chunks: Vec<Vec<u8>>,
    complete: bool,
}

/// The operations `Player`'s ordering logic needs from a sink: append a
/// buffer of samples to play next, stop/clear everything queued, and report
/// whether everything appended has finished playing. `rodio::Sink` is the
/// real implementation; tests substitute a `Vec`-recording fake so the
/// ordering logic — the part a real bug lived in — can be verified without a
/// real audio output device, which unit tests can't rely on having.
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

/// All the ordering/decoding state for one player thread — pulled out of the
/// thread closure into its own type so the ordering logic (the part worth
/// getting right) can be reasoned about and unit-tested independently of
/// spawning a real thread/audio device.
struct Player<S: AudioSink> {
    sink: S,
    decoder: PcmDecoder,
    /// The sentence number allowed to have its audio appended to the sink
    /// right now. Every chunk with this `seq` is appended immediately, in
    /// arrival order, exactly as if there were only ever one sentence in
    /// flight — this is the single append path; nothing bypasses it.
    next_to_play: u64,
    /// Sentences that arrived before it was their turn, keyed by `seq`.
    /// Flushed (via the same single append path) as soon as `next_to_play`
    /// reaches them.
    held: HashMap<u64, HeldSentence>,
}

impl<S: AudioSink> Player<S> {
    fn new(sink: S) -> Self {
        Self { sink, decoder: PcmDecoder::new(), next_to_play: 0, held: HashMap::new() }
    }

    fn handle_chunk(&mut self, seq: u64, bytes: Vec<u8>, is_last: bool) {
        if seq < self.next_to_play {
            // Stray chunk for an already-finished/skipped sentence (e.g.
            // arrived just after a Stop reset bookkeeping) — nothing
            // meaningful to do with it.
            return;
        }
        if seq > self.next_to_play {
            // Not this sentence's turn yet — hold it, still in its own
            // arrival order, until every earlier sentence finishes.
            let entry = self.held.entry(seq).or_default();
            entry.chunks.push(bytes);
            if is_last {
                entry.complete = true;
            }
            return;
        }

        // seq == next_to_play: append now, in arrival order, exactly as the
        // pre-ordering version of this player always did for a single
        // sentence — this is what keeps latency low for the common case
        // (only one sentence in flight, or this one already at the front).
        self.append(&bytes);
        if is_last {
            self.advance();
        }
    }

    /// Called when the sentence that was `next_to_play` has sent its final
    /// chunk: moves to the next sentence number, then keeps flushing
    /// already-held sentences for as long as each next one in line has
    /// already been fully buffered (`complete`) — stopping at the first
    /// sentence still in flight, since its remaining chunks will arrive via
    /// `handle_chunk`'s fast path directly once it's genuinely its turn.
    fn advance(&mut self) {
        self.next_to_play += 1;
        while let Some(sentence) = self.held.get(&self.next_to_play) {
            if !sentence.complete {
                break;
            }
            let HeldSentence { chunks, .. } = self.held.remove(&self.next_to_play).unwrap();
            for chunk in &chunks {
                self.append(chunk);
            }
            self.next_to_play += 1;
        }
    }

    fn append(&mut self, bytes: &[u8]) {
        let samples = self.decoder.decode(bytes);
        if !samples.is_empty() {
            self.sink.append_samples(samples);
        }
    }

    fn handle_stop(&mut self) {
        self.sink.stop_all();
        self.held.clear();
        // Advance past everything: any seq already in flight is now stale.
        // TtsSession::stop() also flips its own `stopped` flag so those
        // threads stop sending new chunks, but resetting here too means a
        // late-arriving stray chunk for an old seq can't wedge playback
        // waiting on a sentence that will never send its is_last.
        self.next_to_play = u64::MAX;
    }

    /// Whether the sink has finished playing everything appended to it so
    /// far. Note this can be `true` even with sentences still `held` (their
    /// Deepgram request hasn't sent a chunk yet) — that's fine for the
    /// mic-mute signal's purposes: no audio is *currently sounding*, which
    /// is what matters, even though more is expected to arrive shortly.
    fn sink_is_empty(&self) -> bool {
        self.sink.is_empty()
    }
}

/// Decodes raw linear16 little-endian PCM bytes into `i16` samples,
/// buffering a possible odd trailing byte across calls (a Deepgram chunk
/// boundary is not guaranteed to land on a 2-byte sample boundary).
///
/// One decoder instance is shared across ALL sentences on the player thread
/// (not one per sentence) — safe specifically because `Player` only ever
/// feeds it chunks in strict sentence order (see `Player::append`'s single
/// call path): a leftover odd byte from the end of sentence N's PCM stream
/// is real trailing state that correctly carries into sentence N+1's first
/// chunk, exactly as it would if the whole answer had come from one
/// continuous stream.
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

    // -- Player ordering ---------------------------------------------------
    //
    // These are the regression tests for the actual bug this module fixed:
    // two sentences synthesizing concurrently, whose chunks arrive
    // interleaved rather than one sentence fully at a time, must still be
    // appended to the sink in sentence order — never interleaved — or
    // playback sounds like garbled/repeated speech.

    use std::cell::RefCell;

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
            // Not exercised by the ordering tests below (which assert on
            // `appended` directly, not on sink_is_empty()) — a real
            // "has queued audio finished playing" concept has no meaning
            // for a fake that just records appends instantly with no
            // playback timing, so always-empty is the honest answer.
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
    fn single_sentence_plays_chunks_in_arrival_order() {
        let mut player = Player::new(FakeSink::default());
        player.handle_chunk(0, pcm(&[1, 2]), false);
        player.handle_chunk(0, pcm(&[3, 4]), true);
        assert_eq!(player.sink.appended.borrow().as_slice(), &[vec![1i16, 2], vec![3i16, 4]]);
    }

    #[test]
    fn later_sentence_arriving_first_is_held_back_not_played_early() {
        let mut player = Player::new(FakeSink::default());
        // Sentence 1 (seq=1) finishes synthesizing and arrives completely
        // BEFORE sentence 0 (seq=0) has sent anything — this is the
        // realistic race (two concurrent Deepgram requests, no guarantee
        // which responds first).
        player.handle_chunk(1, pcm(&[100]), true);
        assert!(player.sink.appended.borrow().is_empty(), "seq=1 must not play before seq=0 has even started");

        player.handle_chunk(0, pcm(&[1]), false);
        player.handle_chunk(0, pcm(&[2]), true);
        // Once seq=0 finishes, the already-fully-buffered seq=1 should be
        // released immediately after it, in the correct order.
        assert_eq!(
            player.sink.appended.borrow().as_slice(),
            &[vec![1i16], vec![2i16], vec![100i16]],
            "seq=0 must play fully before seq=1, even though seq=1's audio arrived first"
        );
    }

    #[test]
    fn interleaved_arrival_across_two_sentences_still_plays_in_sentence_order() {
        // The exact failure mode this fixes: chunks from two sentences
        // arriving genuinely interleaved (as they would from two
        // concurrently-streaming HTTP responses), not one sentence then the
        // other.
        let mut player = Player::new(FakeSink::default());
        player.handle_chunk(0, pcm(&[1]), false); // seq0 chunk1
        player.handle_chunk(1, pcm(&[100]), false); // seq1 chunk1 (held)
        player.handle_chunk(0, pcm(&[2]), false); // seq0 chunk2
        player.handle_chunk(1, pcm(&[101]), true); // seq1 chunk2, seq1 done (held)
        player.handle_chunk(0, pcm(&[3]), true); // seq0 chunk3, seq0 done -> releases seq1

        assert_eq!(
            player.sink.appended.borrow().as_slice(),
            &[vec![1i16], vec![2i16], vec![3i16], vec![100i16], vec![101i16]],
            "must play as 'seq0 chunk1, chunk2, chunk3, then seq1 chunk1, chunk2' — \
             never interleaved as it arrived"
        );
    }

    #[test]
    fn three_sentences_where_the_middle_one_is_still_in_flight_stalls_correctly() {
        let mut player = Player::new(FakeSink::default());
        // seq=2 fully arrives first (held).
        player.handle_chunk(2, pcm(&[300]), true);
        // seq=0 finishes.
        player.handle_chunk(0, pcm(&[1]), true);
        // seq=1 hasn't sent anything yet — seq=2 must NOT be released even
        // though it's fully buffered, because seq=1 is still pending.
        assert_eq!(player.sink.appended.borrow().as_slice(), &[vec![1i16]]);

        // seq=1 finally arrives and finishes — now both seq=1 and the
        // already-held seq=2 should release, in order.
        player.handle_chunk(1, pcm(&[200]), true);
        assert_eq!(
            player.sink.appended.borrow().as_slice(),
            &[vec![1i16], vec![200i16], vec![300i16]]
        );
    }

    #[test]
    fn stop_discards_held_sentences_and_ignores_late_stray_chunks() {
        let mut player = Player::new(FakeSink::default());
        player.handle_chunk(1, pcm(&[100]), true); // held, waiting on seq=0
        player.handle_stop();
        assert!(*player.sink.stopped.borrow());

        // A stray chunk for the old seq=0 (in-flight Deepgram thread hadn't
        // yet seen the stopped flag) must not resurrect old playback.
        player.handle_chunk(0, pcm(&[1]), true);
        assert!(player.sink.appended.borrow().is_empty(), "stray post-stop chunk must be ignored, not played");
    }
}
