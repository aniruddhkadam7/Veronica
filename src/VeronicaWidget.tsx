import { useCallback, useEffect, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { TranscriptSegment } from "./types";
import { ParticlesOrb } from "@/registry/orbe/particles-orb/particles-orb";
import {
  AUTO_AI_SILENCE_MS_COMPLETE,
  AUTO_AI_SILENCE_MS_INCOMPLETE,
  classifyQuestionCompleteness,
  joinSpeech,
} from "./questionCompleteness";
import { loadOverlaySettings } from "./overlaySettings";
import { loadLlmProvider } from "./llmProviderSetting";
import { useVeronicaOrbState } from "./useVeronicaOrbState";

/// The always-on floating indicator — a small, chrome-less, always-on-top
/// window (see veronica_widget.rs) docked to the bottom-right corner,
/// showing only the orb. The mic-assistant session is started by App.tsx's
/// handleActivate BEFORE this window is even shown, so by the time this
/// mounts, real audio is already flowing — this component only drives the
/// question pipeline and the orb's visual state, never session start/stop
/// (see App.tsx's Start/Open/Stop model).
///
/// The user can keep working normally on the desktop while this runs: no
/// window focus is stolen, there is no conversation UI here at all — just
/// the orb reacting to the REAL pipeline (see useVeronicaOrbState), and this
/// component's own real transcript-buffer -> silence-timeout -> ask_veronica
/// pipeline (mirroring VeronicaOverlay.tsx's, since there is no text field
/// here for the user to edit before sending). Voice output is forced on
/// regardless of the saved "Voice output" toggle: with no answer text
/// rendered anywhere in this window, TTS is the only way an answer is ever
/// perceptible here.
///
/// Clicking the orb does nothing (no expand-to-overlay) — "Open" in App.tsx
/// is the only way to reveal the full conversation view, and it does so
/// without touching this session at all (see App.tsx's handleOpenOverlay).
export function VeronicaWidget() {
  const { orbState, lastError } = useVeronicaOrbState();

  const busyRef = useRef(false);
  const committedRef = useRef("");
  const autoAiTimerRef = useRef<number | null>(null);
  const askAIRef = useRef<() => void>(() => {});

  const askAI = useCallback(async () => {
    const trimmed = committedRef.current.trim();
    if (!trimmed || busyRef.current) return;

    committedRef.current = "";
    busyRef.current = true;

    try {
      const settings = loadOverlaySettings();
      const chosenProvider = loadLlmProvider();
      const llmProvider =
        chosenProvider === "anthropic" || chosenProvider === "openai" || chosenProvider === "gemini"
          ? chosenProvider
          : null;
      // No on-screen conversation to fall back to here — voice is the only
      // channel, so this always asks for speech regardless of the saved
      // toggle (see this module's doc comment).
      await invoke<string>("ask_veronica", {
        question: trimmed,
        history: [],
        options: {
          answerLength: settings.answerLength,
          responseStyle: settings.responseStyle,
          humanization: settings.humanization,
          llmProvider,
          ttsEnabled: true,
        },
      });
    } catch {
      // Real failures already surface via useVeronicaOrbState's
      // "veronica:error" listener (Groq/Deepgram/sidecar failures); an
      // ask_veronica command rejection itself (e.g. no LLM key configured)
      // has no dedicated banner in this window, same as before — the orb
      // simply returns to idle/listening.
    } finally {
      busyRef.current = false;
    }
  }, []);

  useEffect(() => {
    askAIRef.current = askAI;
  }, [askAI]);

  useEffect(() => {
    const clearAutoAiTimer = () => {
      if (autoAiTimerRef.current) {
        window.clearTimeout(autoAiTimerRef.current);
        autoAiTimerRef.current = null;
      }
    };

    const armAutoAiTimer = () => {
      clearAutoAiTimer();
      if (busyRef.current) return;
      const completeness = classifyQuestionCompleteness(committedRef.current);
      const delay =
        completeness === "complete" ? AUTO_AI_SILENCE_MS_COMPLETE : AUTO_AI_SILENCE_MS_INCOMPLETE;
      autoAiTimerRef.current = window.setTimeout(() => {
        autoAiTimerRef.current = null;
        askAIRef.current();
      }, delay);
    };

    const unlistenTranscript = listen<TranscriptSegment>("transcript:update", (event) => {
      const segment = event.payload;
      // Mic only — the widget has no system-audio toggle to opt into the
      // other-speaker feed (see the full overlay's toggleSystemAudio).
      if (segment.source !== "MICROPHONE") return;
      if (segment.final_text) {
        committedRef.current = joinSpeech(committedRef.current, segment.final_text);
        armAutoAiTimer();
      } else if (segment.partial_text) {
        clearAutoAiTimer();
      }
    });

    return () => {
      unlistenTranscript.then((f) => f());
      clearAutoAiTimer();
    };
  }, []);

  return (
    <div
      className="veronica-widget-root"
      title={lastError ?? "Veronica is listening"}
      aria-label="Veronica"
    >
      <ParticlesOrb state={orbState} size={120} speed={1} colorFrom="#f0abfc" colorTo="#818cf8" />
    </div>
  );
}
