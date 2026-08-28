import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ParticlesOrb } from "@/registry/orbe/particles-orb/particles-orb";
import { loadOverlaySettings } from "./overlaySettings";
import { useVeronicaOrbState } from "./useVeronicaOrbState";
import { useAutoAsk } from "./useAutoAsk";
import { loadWidgetSettings, type WidgetSettings } from "./widgetSettings";

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
/// the orb reacting to the REAL pipeline (see useVeronicaOrbState) and
/// `useAutoAsk` driving the same event-driven "finalized speech -> send"
/// pipeline VeronicaOverlay.tsx uses (no fixed post-transcription wait —
/// see questionCompleteness.ts). Voice output is forced on regardless of
/// the saved "Voice output" toggle: with no answer text rendered anywhere
/// in this window, TTS is the only way an answer is ever perceptible here.
///
/// Clicking the orb does nothing (no expand-to-overlay) — "Open" in App.tsx
/// is the only way to reveal the full conversation view, and it does so
/// without touching this session at all (see App.tsx's handleOpenOverlay).
export function VeronicaWidget() {
  const { orbState, lastError } = useVeronicaOrbState();
  const [widgetSettings, setWidgetSettings] = useState<WidgetSettings>(() => loadWidgetSettings());

  // show_widget (veronica_widget.rs) always docks the OS window at its own
  // fixed default side on every show/re-show — it has no access to
  // localStorage, so it can't know the user's saved orb size. Without this,
  // reopening the widget after picking a bigger orb in Settings left the
  // window at the old (smaller) size while the orb rendered at the new,
  // bigger one — clipping its particles against the window's hard edge,
  // which read as a visible square around the orb.
  //
  // Re-running this on every orbSize change (not just on mount) is what
  // makes this self-healing rather than a one-shot fix: the Settings panel
  // (WidgetPanel.tsx) already calls resize_veronica_widget itself when the
  // slider moves, but that call only reaches this window if it's open and
  // listening for "veronica:widget-settings-changed" *at that moment* — a
  // widget opened later, or one whose settings-changed listener raced the
  // resize call, would otherwise be stuck at a stale window size until the
  // user happened to touch the slider again. A duplicate resize call here is
  // harmless (idempotent — same size in, same dock position out), so there's
  // no reason not to keep this in sync on every settings update too.
  useEffect(() => {
    invoke("resize_veronica_widget", { orbSize: widgetSettings.orbSize }).catch(() => {});
  }, [widgetSettings.orbSize]);

  // The Settings panel lives in the full overlay window (a separate
  // webview), so it can't update this component's state directly — it saves
  // to localStorage (see widgetSettings.ts) and emits this event; re-reading
  // from storage here (rather than trusting the event payload) keeps this
  // window's view of the settings and the persisted value from ever
  // diverging, matching how useVeronicaOrbState treats the Rust pipeline as
  // the source of truth rather than passed-along event payloads.
  useEffect(() => {
    const unlisten = listen("veronica:widget-settings-changed", () => {
      setWidgetSettings(loadWidgetSettings());
    });
    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  const answerOptions = useCallback(() => {
    const settings = loadOverlaySettings();
    return {
      answerLength: settings.answerLength,
      responseStyle: settings.responseStyle,
      humanization: settings.humanization,
      // No on-screen conversation to fall back to here — voice is the only
      // channel, so this always asks for speech regardless of the saved
      // toggle (see this module's doc comment).
      ttsEnabled: true,
    };
  }, []);

  // Mic only — the widget has no system-audio toggle to opt into the
  // other-speaker feed (see the full overlay's toggleSystemAudio).
  const acceptSource = useCallback((source: string) => source === "MICROPHONE", []);

  // Real failures already surface via useVeronicaOrbState's "veronica:error"
  // listener (Groq/Deepgram/sidecar/ask_veronica failures); the orb simply
  // returns to idle/listening on error, same as before this hook existed.
  useAutoAsk({ acceptSource, answerOptions });

  return (
    <div
      className="veronica-widget-root"
      title={lastError ?? "Veronica is listening"}
      aria-label="Veronica"
      data-tauri-drag-region={widgetSettings.dragEnabled ? "deep" : undefined}
    >
      <ParticlesOrb
        state={orbState}
        size={widgetSettings.orbSize}
        speed={widgetSettings.speed}
        intensity={widgetSettings.intensity}
        colorFrom={widgetSettings.colorFrom}
        colorTo={widgetSettings.colorTo}
      />
    </div>
  );
}
