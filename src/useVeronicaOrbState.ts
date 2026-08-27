import { useEffect, useRef, useState } from "react";
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
/// `audio:level` (real mic RMS, now emitted by the mic-assistant pump too —
/// see voice_command/mod.rs) is exposed as `micLevel` for callers that want
/// to react to actual input amplitude beyond just the coarse listening
/// boolean; ParticlesOrb itself already reads `--orb-level` for its own
/// per-state energy curve, so most callers don't need to touch this
/// directly.
export interface VeronicaOrbStatus {
  orbState: OrbState;
  micLevel: number;
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
  const [micLevel, setMicLevel] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const sttSpeaking = useSttSpeaking();
  const errorTimerRef = useRef<number | null>(null);

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
    });
    const unlistenActionStart = listen("veronica:action-start", () => {
      setThinking(false);
      setExecuting(true);
    });
    const unlistenActionComplete = listen("veronica:action-complete", () => {
      setExecuting(false);
    });
    const unlistenSpeakingChanged = listen<boolean>("tts:speaking-changed", (event) => {
      setSpeaking(event.payload);
    });
    const unlistenLevel = listen<AudioLevelEvent>("audio:level", (event) => {
      if (event.payload.source === "MICROPHONE") setMicLevel(event.payload.rms_level);
    });
    const unlistenError = listen<string>("veronica:error", (event) => {
      setLastError(event.payload);
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
      unlistenSpeakingChanged.then((f) => f());
      unlistenLevel.then((f) => f());
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

  return { orbState, micLevel, lastError };
}
