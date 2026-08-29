"""Voice-activity/endpoint-detection engine for Veronica, backed by
sherpa-onnx's OnlineRecognizer.

This process's own transcription is NOT the app's transcript text — Groq
Cloud's Whisper API (see src-tauri/src/stt/groq.rs) is the only source of
transcript text in Veronica. This process exists purely to decide *when* an
utterance is complete (silence-based endpoint detection) so the Rust side
knows which span of buffered audio to send to Groq; the "final" text this
script emits is a byproduct of running the recognizer, discarded by the
Rust reader thread rather than shown to the user.

Wire protocol (mirrors tts_sidecar/sidecar.py; see src-tauri/src/stt/sidecar.rs
and src-tauri/src/stt/events.rs for the Rust side of this contract):

  stdin  -> length-prefixed frames (4-byte little-endian u32 length + raw
            PCM16LE mono audio samples at STT_SAMPLE_RATE, default 16000).
            A zero-length frame is a flush marker: finalize whatever
            utterance is currently in progress right now (used when the
            user pauses/stops recording).
  stdout -> newline-delimited JSON, one object per line, flushed
            immediately after every write:
              {"type": "ready"}
              {"type": "partial", "text": "...", "source": "..."}
              {"type": "final", "text": "...", "source": "...",
               "start_time": <float seconds>, "end_time": <float seconds>}
              {"type": "error", "message": "..."}
            "partial" and "final.text" are this engine's own decode —
            ignored by the Rust side (see above) but still emitted, since
            get_result()/is_endpoint() require actually decoding audio
            regardless of whether the text is used.
  stderr -> free-form logs only (never parsed by Rust).

Invocation: `python sidecar.py SYSTEM_AUDIO` or `python sidecar.py MICROPHONE`
— the one positional arg is echoed back verbatim in every event's "source"
field, so it must match whatever Rust's `AudioSource` serializes as exactly
(see src-tauri/src/audio/mod.rs: SCREAMING_SNAKE_CASE, i.e. those two exact
strings).

Model: models/stt/nemo-fastconformer-80ms-int8/ by default — an English-only
NeMo FastConformer model, 1025-token vocab. Model choice matters much less
here than it would for a real transcription engine, since only its endpoint
timing is used; picked for being small/fast and already present rather than
for transcription accuracy. A multilingual model (Nemotron-3.5) was tried
here previously and reverted after it intermittently decoded live mic audio
into Devanagari/Hindi script with no way to pin its output language via this
sherpa-onnx version's Python binding — moot now for transcript quality (Groq
never sees that text), but still worth avoiding: language drift could in
principle also affect this model's own confidence/endpoint timing, which
this app does still depend on.

Threading: stdin is read on a dedicated background thread and handed to the
main thread via a queue, so a write to stdout (a partial/final event) is
never blocked behind a stdin read — mirroring the reader-thread/writer-thread
split already used on the Rust side (`SttSidecar::spawn`'s dedicated
stdout-reader thread) and in tts_sidecar/sidecar.py's design note.
"""

from __future__ import annotations

import json
import os
import queue
import struct
import sys
import threading
import time
import traceback
from pathlib import Path


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Configuration (all optional env vars, sensible defaults)
# --------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "nemo-fastconformer-80ms-int8"


def _resolve_model_dir() -> Path:
    configured = os.environ.get("STT_MODEL_DIR")
    if configured:
        return Path(configured)
    # repo_root/models/stt/<name> relative to this script's directory
    # (this script lives at repo_root/streaming_asr_sidecar/sidecar.py),
    # mirroring how the old frozen sidecar resolved its default model path
    # relative to __file__.
    return Path(__file__).resolve().parent.parent / "models" / "stt" / DEFAULT_MODEL_NAME


def _sample_rate() -> int:
    raw = os.environ.get("STT_SAMPLE_RATE", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return 16000


def _num_threads() -> int:
    raw = os.environ.get("STT_NUM_THREADS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return 4


def _end_silence_seconds() -> float:
    raw = os.environ.get("STT_END_SILENCE_MS", "").strip()
    if raw:
        try:
            return max(0.05, int(raw) / 1000.0)
        except ValueError:
            pass
    return 0.7  # 700ms default, within the 600-800ms range asked for


def _vad_gate_enabled() -> bool:
    return os.environ.get("STT_VAD_GATE_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------
# Optional VAD pre-gate. Off by default (see STT_VAD_GATE_ENABLED docstring
# in src-tauri/src/stt/sidecar.rs — the old sidecar's gate was found to drop
# 48-64% of real mic chunks). When disabled this is a true no-op pass-through,
# not just a lowered threshold, so there is no accuracy risk from leaving it
# on the default path.
# --------------------------------------------------------------------------


class EnergyVadGate:
    """A conservative RMS-energy gate, only ever consulted when explicitly
    enabled via STT_VAD_GATE_ENABLED=true. Deliberately simple (no ML VAD
    dependency) since it exists only as an opt-in escape hatch for a
    genuinely noisy room, not as the default speech/non-speech decision
    maker for this sidecar.
    """

    def __init__(self, threshold: float = 0.006) -> None:
        self.threshold = threshold

    def is_speech(self, samples) -> bool:
        import numpy as np

        if samples.size == 0:
            return False
        rms = float(np.sqrt(np.mean(np.square(samples))))
        return rms >= self.threshold


class PassThroughGate:
    """The default no-op gate: every chunk is treated as speech. Exists as
    its own class (rather than an `if enabled` branch scattered through the
    recognize loop) so "disabled" is structurally a true pass-through, not a
    threshold of zero that could still special-case an all-silence chunk.
    """

    def is_speech(self, samples) -> bool:  # noqa: ARG002 - intentionally ignores samples
        return True


# --------------------------------------------------------------------------
# Recognizer wrapper
# --------------------------------------------------------------------------


class SttRecognizer:
    """Wraps sherpa_onnx.OnlineRecognizer for the NeMo FastConformer
    streaming transducer model. `model_type=""` (sherpa-onnx's own
    auto-detection — this model does not need an explicit NeMo model_type
    string, verified empirically against this exact model directory) and
    `feature_dim=80` are required for this specific model family; a
    differently-shaped drop-in model would need these adjusted, which is why
    they're named constants here rather than buried in the constructor call.
    """

    MODEL_TYPE = ""
    FEATURE_DIM = 80

    def __init__(self, model_dir: Path, sample_rate: int, num_threads: int, end_silence_s: float) -> None:
        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.num_threads = num_threads
        self.end_silence_s = end_silence_s
        self.recognizer = None

    def load(self) -> None:
        import sherpa_onnx

        tokens = self.model_dir / "tokens.txt"
        encoder = self.model_dir / "encoder.int8.onnx"
        decoder = self.model_dir / "decoder.int8.onnx"
        joiner = self.model_dir / "joiner.int8.onnx"

        for path in (tokens, encoder, decoder, joiner):
            if not path.is_file():
                raise FileNotFoundError(f"missing model file: {path}")

        log(
            f"loading model from {self.model_dir} "
            f"(sample_rate={self.sample_rate}, num_threads={self.num_threads}, "
            f"end_silence={self.end_silence_s:.2f}s)"
        )

        # decoding_method is hardcoded to greedy_search, not exposed via an
        # env var: sherpa-onnx 1.13.6's NeMo/Nemotron transducer backend
        # (csrc/online-recognizer-transducer-nemo-impl.h) does not implement
        # modified_beam_search at all. Passing it doesn't raise a Python
        # exception — it hits a SHERPA_ONNX_LOGE(...) + exit() in the C++
        # layer, killing the whole process with exit code 127 and no
        # traceback (verified empirically against this exact model
        # directory). Not worth exposing as a switchable option even now
        # that this engine's own transcription text is discarded, since
        # decoding_method also affects is_ready()/is_endpoint() timing this
        # app does still depend on, and a wrong value crashes the process
        # outright rather than degrading gracefully.
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens),
            encoder=str(encoder),
            decoder=str(decoder),
            joiner=str(joiner),
            num_threads=self.num_threads,
            sample_rate=self.sample_rate,
            feature_dim=self.FEATURE_DIM,
            model_type=self.MODEL_TYPE,
            decoding_method="greedy_search",
            enable_endpoint_detection=True,
            # rule1: pure trailing silence with no speech decoded yet.
            # rule2: the tunable one — trailing silence after *some* speech
            # was decoded, wired to STT_END_SILENCE_MS as instructed.
            # rule3: effectively disabled (matches Meeting-AI's proven
            # streaming_asr_sidecar config, 300s). Cutting an utterance in
            # half mid-sentence because it ran past a timer is worse than a
            # long segment — silence-based endpointing (rule1/rule2) already
            # handles real utterance boundaries. The previous 20.0s value
            # force-finalized any answer/question running past 20s of
            # continuous speech regardless of whether the speaker paused,
            # which read as garbled/truncated transcription for anyone
            # talking for more than 20s straight.
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=self.end_silence_s,
            rule3_min_utterance_length=300.0,
        )
        log("model loaded")

    # Leading silence fed into every freshly-created/reset stream before any
    # real audio reaches it. The streaming encoder has no left context at the
    # start of a stream, so its first output frames are unreliable and the
    # first word of the first utterance was being dropped outright — measured
    # on 16kHz mono test audio, "A twelve B thirty four C nine" decoded as
    # "Twelve B thirty four C nine" with no priming, and correctly with as
    # little as 200ms of it. Later utterances in a session happened to be
    # fine only because the previous utterance's trailing silence acted as
    # accidental priming; this makes that context explicit and unconditional
    # rather than an artifact of what came before.
    #
    # 300ms (vs. the 200ms where the effect first disappeared) buys margin
    # without cost: it is silence, so it can never add words, and it is fed
    # before the utterance clock starts so it does not affect endpointing or
    # add any user-visible latency.
    PRIMING_SILENCE_S = 0.3

    def _prime(self, stream) -> None:
        """Feeds `PRIMING_SILENCE_S` of silence so the encoder has left
        context before the first real sample of an utterance arrives."""
        import numpy as np

        silence = np.zeros(int(self.sample_rate * self.PRIMING_SILENCE_S), dtype=np.float32)
        stream.accept_waveform(self.sample_rate, silence)
        while self.is_ready(stream):
            self.decode_stream(stream)

    def create_stream(self):
        assert self.recognizer is not None
        stream = self.recognizer.create_stream()
        self._prime(stream)
        return stream

    def accept_waveform(self, stream, samples) -> None:
        stream.accept_waveform(self.sample_rate, samples)

    def is_ready(self, stream) -> bool:
        assert self.recognizer is not None
        return self.recognizer.is_ready(stream)

    def decode_stream(self, stream) -> None:
        assert self.recognizer is not None
        self.recognizer.decode_stream(stream)

    def get_result(self, stream) -> str:
        assert self.recognizer is not None
        return self.recognizer.get_result(stream)

    def is_endpoint(self, stream) -> bool:
        assert self.recognizer is not None
        return self.recognizer.is_endpoint(stream)

    def reset(self, stream) -> None:
        assert self.recognizer is not None
        self.recognizer.reset(stream)
        # Re-prime: a reset stream is as cold as a brand new one, so without
        # this the first word after *every* endpoint would be at risk, not
        # just the first word of the session (see PRIMING_SILENCE_S).
        self._prime(stream)


# --------------------------------------------------------------------------
# stdin frame reading (background thread)
# --------------------------------------------------------------------------

FLUSH_MARKER = object()  # sentinel put on the queue for a zero-length frame
DISCARD_MARKER = object()  # sentinel for a one-byte frame — see below
EOF_MARKER = object()  # sentinel put on the queue when stdin closes


def read_frame(stream) -> bytes | None:
    header = stream.read(4)
    if len(header) < 4:
        return None
    (length,) = struct.unpack("<I", header)
    if length == 0:
        return b""
    payload = stream.read(length)
    if len(payload) < length:
        return None
    return payload


def stdin_reader_thread(out_queue: "queue.Queue") -> None:
    stdin = sys.stdin.buffer
    try:
        while True:
            frame = read_frame(stdin)
            if frame is None:
                break
            # Real audio frames are always PCM16LE mono — an even number of
            # bytes. A one-byte frame can never be valid audio, so it's an
            # unambiguous second marker alongside the existing zero-byte
            # flush marker, with no wire-format version bump needed.
            if len(frame) == 0:
                out_queue.put(FLUSH_MARKER)
            elif len(frame) == 1:
                out_queue.put(DISCARD_MARKER)
            else:
                out_queue.put(frame)
    except Exception:  # noqa: BLE001 - surfaced via EOF_MARKER, loop below logs it
        log(traceback.format_exc())
    finally:
        out_queue.put(EOF_MARKER)


# --------------------------------------------------------------------------
# Main recognition loop
# --------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        emit({"type": "error", "message": "usage: sidecar.py SYSTEM_AUDIO|MICROPHONE"})
        sys.exit(2)
    source = sys.argv[1]

    sample_rate = _sample_rate()
    num_threads = _num_threads()
    end_silence_s = _end_silence_seconds()
    vad_gate = EnergyVadGate() if _vad_gate_enabled() else PassThroughGate()

    model_dir = _resolve_model_dir()
    recognizer = SttRecognizer(model_dir, sample_rate, num_threads, end_silence_s)

    try:
        recognizer.load()
    except Exception as exc:  # noqa: BLE001 - must surface as a sidecar error line, not a crash
        emit({"type": "error", "message": f"failed to load STT model: {exc}"})
        log(traceback.format_exc())
        sys.exit(1)

    emit({"type": "ready"})

    frame_queue: "queue.Queue" = queue.Queue()
    reader = threading.Thread(target=stdin_reader_thread, args=(frame_queue,), daemon=True)
    reader.start()

    stream = recognizer.create_stream()
    last_partial_text = ""
    utterance_start: float | None = None
    has_pending_audio = False

    def finalize_current_utterance(end_time: float) -> None:
        nonlocal utterance_start, last_partial_text, has_pending_audio, stream
        # This engine's own decode is never shown to the user (see module
        # docstring) — Groq (on the Rust side) is the only real transcription
        # source; this local `text` is purely a leftover of the decode this
        # engine already had to run for endpoint detection. Every caller of
        # this function already only calls it when `has_pending_audio` is
        # true (a real, VAD-passed utterance happened) — gating again here on
        # `text` being non-empty is redundant AND wrong: this engine's own
        # decode can legitimately come back empty for audio a stronger model
        # (Groq) would still transcribe fine (e.g. quieter/narrowband mic
        # input, like Bluetooth headset audio), which silently dropped the
        # entire utterance — no "final" line ever reached the Rust side, so
        # Groq was never even called. Emitting unconditionally lets Groq be
        # the one to decide whether there's real speech in the buffered audio.
        start_time = utterance_start if utterance_start is not None else end_time
        emit(
            {
                "type": "final",
                "text": recognizer.get_result(stream).strip(),
                "source": source,
                "start_time": start_time,
                "end_time": end_time,
            }
        )
        # Fresh stream for the next utterance — resetting the same stream
        # object after an endpoint is the sherpa-onnx-recommended pattern
        # for streaming transducers with endpoint detection enabled.
        recognizer.reset(stream)
        last_partial_text = ""
        utterance_start = None
        has_pending_audio = False

    try:
        while True:
            item = frame_queue.get()

            if item is EOF_MARKER:
                # stdin closed: Rust side shut down. Finalize anything
                # in-flight so no trailing words are silently dropped, then
                # exit cleanly.
                if has_pending_audio:
                    finalize_current_utterance(time.monotonic())
                break

            if item is FLUSH_MARKER:
                # Explicit "finalize now" from the Rust side (user paused/
                # stopped). Feed a little trailing silence so the endpoint
                # detector's trailing-silence rules actually fire on this
                # exact frame, rather than only finalizing on the *next*
                # audio frame's worth of decoding.
                if has_pending_audio:
                    import numpy as np

                    silence = np.zeros(int(sample_rate * (end_silence_s + 0.05)), dtype=np.float32)
                    stream.accept_waveform(sample_rate, silence)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    finalize_current_utterance(time.monotonic())
                continue

            if item is DISCARD_MARKER:
                # "Throw away whatever's in-progress, do NOT transcribe it" —
                # distinct from FLUSH_MARKER, which always finalizes+emits.
                # Used by the Rust side when it decides an in-progress
                # fragment is too short to plausibly be real speech (e.g.
                # ambient noise sitting in the decoder right as the mic is
                # about to be muted for TTS) — resets decoder state exactly
                # like a normal finalize does, just without ever calling
                # Groq on it.
                if has_pending_audio:
                    recognizer.reset(stream)
                    last_partial_text = ""
                    utterance_start = None
                    has_pending_audio = False
                continue

            # Regular audio frame: raw PCM16LE mono bytes.
            import numpy as np

            pcm16 = np.frombuffer(item, dtype="<i2")
            samples = pcm16.astype(np.float32) / 32768.0

            if not vad_gate.is_speech(samples):
                continue

            if utterance_start is None:
                utterance_start = time.monotonic()
            has_pending_audio = True

            recognizer.accept_waveform(stream, samples)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)

            text = recognizer.get_result(stream).strip()
            if text and text != last_partial_text:
                last_partial_text = text
                emit({"type": "partial", "text": text, "source": source})

            if recognizer.is_endpoint(stream):
                finalize_current_utterance(time.monotonic())

    except Exception as exc:  # noqa: BLE001 - one bad chunk must not kill the sidecar silently
        emit({"type": "error", "message": str(exc)})
        log(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
