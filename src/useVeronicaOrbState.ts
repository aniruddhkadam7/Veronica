import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";
import { listen } from "@tauri-apps/api/event";
import type { OrbState } from "@/registry/lib/orb-state";
import { useSttSpeaking } from "./ui";
import type { AudioLevelEvent } from "./types";

/// Derives the orb's visual state from the REAL Veronica pipeline events —
/// never a fake/synthetic timer — so VeronicaWidget.tsx (the always-on
/// floating indicator) and VeronicaOverlay.tsx (the full conversation view)
/// always agree on what Veronica is actually doing, since both mount this
/// same hook rather than keeping separate, driftable copies of this logic.
///
/// Event -> state mapping (see src/types.ts's doc for exactly where each of
/// these is emitted on the Rust side):
///   "veronica:thinking-start"   -> thinking
///   "veronica:answer-delta"     -> clears thinking (an answer is arriving)
///   "veronica:action-start"     -> connecting (used as the "executing an
///                                  action" state — ParticlesOrb's orb-state
///                                  lib has no separate "executing" value,
///                                  and "connecting" is otherwise unused by
///                                  this app, so it's repurposed here rather
///                                  than forking the orb component)
///   "veronica:action-complete"  -> clears the executing flag
///   "veronica:interrupted"      -> immediately clears thinking/executing/
///                                  speaking (a recognized "stop"/"wait"/
///                                  "cancel" utterance — see
///                                  veronica::try_interrupt), so the orb
///                                  snaps back to listening right away
///                                  rather than waiting on whatever it
///                                  interrupted to unwind on its own.
///   "tts:speaking-changed"      -> speaking (real mute-signal transition,
///                                  not delta/complete timing)
///   "veronica:answer-complete"  -> clears thinking/speaking as a fallback
///                                  (e.g. an answer with tts disabled, or the
///                                  greeting path, never fires
///                                  tts:speaking-changed at all)
///   "veronica:error"            -> error, auto-clears after ERROR_DISPLAY_MS
///   STT audio (via useSttSpeaking, transcript:update-driven)
///                                -> listening, when nothing above is active
///
/// `audio:level` (real mic RMS, emitted by the mic-assistant pump — see
/// voice_command/mod.rs) and `tts:audio-level` (real Flux playback RMS, see
/// tts/mod.rs's `on_audio`) drive `levelRef`, a single ref merging whichever
/// of the two is relevant to the CURRENT orb state — mic level while
/// listening, TTS level while speaking, untouched (and left to decay via
/// ParticlesOrb's own smoothing) otherwise. Written directly in the event
/// listeners below via `.current =`, never through `useState`: these events
/// arrive at audio-chunk cadence (tens of times a second), and routing them
/// through React state would re-render this hook's subscribers that often.
/// `levelRef` is exactly the primitive `useOrbLevel`/`ParticlesOrb` already
/// expect (see `registry/lib/use-orb-level.ts`), so passing it straight
/// through as `levelRef` is the whole integration — no polling, no timers,
/// no synthetic animation standing in for real audio.
export interface VeronicaOrbStatus {
  orbState: OrbState;
  levelRef: RefObject<number>;
  lastError: string | null;
}

/// How long a real error stays visually flagged before the orb falls back to
/// whatever state the pipeline is actually in — long enough to register as
/// "something went wrong", short enough not to get stuck showing an error
/// for a single failed sentence/utterance while the rest of the session
/// keeps working fine.
const ERROR_DISPLAY_MS = 3500;

export function useVeronicaOrbState(): VeronicaOrbStatus {
  const [thinking, setThinking] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [lastError, setLastError] = useState<string | null>(null);
  const sttSpeaking = useSttSpeaking();
  const errorTimerRef = useRef<number | null>(null);
  // Mirrors `speaking` for the audio-level listeners below (registered once,
  // in the same effect, so their closures can't see later `useState`
  // updates) — see `levelRef`'s doc for why mic/TTS levels are gated by
  // "which state is active" rather than always both feeding the same ref.
  const speakingRef = useRef(false);
  // The single merged level ref passed through to ParticlesOrb — see the
  // `VeronicaOrbStatus.levelRef` doc above for why this is a ref, not state.
  const levelRef = useRef(0);

  useEffect(() => {
    const unlistenThinkingStart = listen("veronica:thinking-start", () => {
      setThinking(true);
    });
    const unlistenDelta = listen("veronica:answer-delta", () => {
      setThinking(false);
    });
    const unlistenComplete = listen("veronica:answer-complete", () => {
      setThinking(false);
      setSpeaking(false);
      speakingRef.current = false;
    });
    const unlistenActionStart = listen("veronica:action-start", () => {
      setThinking(false);
      setExecuting(true);
    });
    const unlistenActionComplete = listen("veronica:action-complete", () => {
      setExecuting(false);
    });
    // Fired by `try_interrupt` (veronica.rs) the instant a bare "stop"/
    // "wait"/"hold on"/"cancel" utterance is recognized — see requirement
    // 12: an interruption must switch the orb to listening IMMEDIATELY,
    // not wait for `tts:speaking-changed`/`answer-complete` to eventually
    // arrive from whatever was cancelled (those still fire too, but this
    // is the authoritative, synchronous-with-the-command signal).
    const unlistenInterrupted = listen("veronica:interrupted", () => {
      setThinking(false);
      setExecuting(false);
      setSpeaking(false);
      speakingRef.current = false;
    });
    const unlistenSpeakingChanged = listen<boolean>("tts:speaking-changed", (event) => {
      setSpeaking(event.payload);
      speakingRef.current = event.payload;
      // A turn boundary — the level from whichever source was just active
      // must not visibly linger as the fallback synthetic curve's starting
      // point when the orb state changes a moment later; ParticlesOrb's own
      // smoothing (`approach`, rate 7.7) handles the actual visual decay
      // whenever new values stop arriving, this just avoids a stale reading
      // from one source briefly appearing on the other's frames.
      levelRef.current = 0;
    });
    // Real mic RMS (mic-assistant pump, voice_command/mod.rs) — only feeds
    // the orb while listening is actually the live state; while speaking,
    // the mic is muted for STT anyway (see TtsSpeakingSignal) and its level
    // reflects Veronica's own voice picked up acoustically, not the user's.
    const unlistenLevel = listen<AudioLevelEvent>("audio:level", (event) => {
      if (event.payload.source === "MICROPHONE" && !speakingRef.current) {
        levelRef.current = event.payload.rms_level;
      }
    });
    // Real Flux playback RMS (tts/mod.rs's on_audio) — only feeds the orb
    // while actually speaking, so a late-arriving chunk right after
    // tts:speaking-changed(false) can't briefly re-animate an orb that has
    // already moved on to listening/idle.
    const unlistenTtsLevel = listen<number>("tts:audio-level", (event) => {
      if (speakingRef.current) levelRef.current = event.payload;
    });
    // Payload is now `{turnId, message}` (see veronica.rs's `TurnErrorPayload`)
    // rather than a bare string — turnId isn't needed here (any recent
    // error is worth a brief visual flag regardless of which turn it
    // belongs to), just the message text.
    const unlistenError = listen<{ turnId: string; message: string }>("veronica:error", (event) => {
      setLastError(event.payload.message);
      if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current);
      errorTimerRef.current = window.setTimeout(() => {
        errorTimerRef.current = null;
        setLastError(null);
      }, ERROR_DISPLAY_MS);
    });

    return () => {
      unlistenThinkingStart.then((f) => f());
      unlistenDelta.then((f) => f());
      unlistenComplete.then((f) => f());
      unlistenActionStart.then((f) => f());
      unlistenActionComplete.then((f) => f());
      unlistenInterrupted.then((f) => f());
      unlistenSpeakingChanged.then((f) => f());
      unlistenLevel.then((f) => f());
      unlistenTtsLevel.then((f) => f());
      unlistenError.then((f) => f());
      if (errorTimerRef.current) window.clearTimeout(errorTimerRef.current);
    };
  }, []);

  // Precedence: a real error always wins (most urgent), then an in-flight
  // action, then thinking, then speaking, then listening, then idle — each
  // of these is a real, currently-true pipeline condition, never more than
  // one meaningfully "correct" at a time in practice (e.g. thinking always
  // clears before speaking begins), but ordered defensively in case two
  // flip in the same tick.
  const orbState: OrbState = lastError
    ? "error"
    : executing
      ? "connecting"
      : thinking
        ? "thinking"
        : speaking
          ? "speaking"
          : sttSpeaking
            ? "listening"
            : "idle";

  return { orbState, levelRef, lastError };
}
