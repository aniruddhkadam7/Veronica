import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { ConfirmationRequestedEvent } from "./types";

/// Mirrors the backend's pending-confirmation state for the visual dialog —
/// the voice path resolves the SAME backend state independently (the user
/// just answering yes/no out loud on the next turn), so this hook only
/// needs to track "is one currently pending" and offer a button-click way to
/// resolve it; it never itself decides whether a reply counts as yes/no
/// (see `confirmation::classify_reply` on the Rust side for that logic).
export interface PendingConfirmation {
  turnId: string;
  summary: string;
  detail: string;
  risk: "sensitive" | "destructive";
}

export interface UseConfirmationResult {
  pending: PendingConfirmation | null;
  respond: (approved: boolean) => Promise<void>;
}

/// Subscribes to `veronica:confirmation-requested` and exposes a `respond`
/// callback that calls the `respond_to_confirmation` command directly —
/// unlike the voice path, a button click is unambiguous, so this skips text
/// classification entirely (see veronica.rs's `respond_to_confirmation`).
///
/// The dialog also clears itself if a NEWER, unrelated turn's terminal event
/// arrives while this one is still unanswered — mirrors the backend's own
/// "stale pending confirmation, drop it" rule (see `run_turn`'s top) so the
/// visual dialog can never outlive the backend state it's supposed to
/// reflect. Consumed identically by VeronicaOverlay.tsx and
/// VeronicaWidget.tsx, matching the pattern `useVeronicaOrbState` already
/// establishes for pipeline-driven state shared between both windows.
export function useConfirmation(): UseConfirmationResult {
  const [pending, setPending] = useState<PendingConfirmation | null>(null);
  // Mirrors `pending` for the answer-complete listener below, which closes
  // over stale state otherwise (subscribed once, on mount).
  const pendingRef = useRef<PendingConfirmation | null>(null);

  useEffect(() => {
    const unlistenRequested = listen<ConfirmationRequestedEvent>("veronica:confirmation-requested", (event) => {
      const next: PendingConfirmation = {
        turnId: event.payload.turnId,
        summary: event.payload.summary,
        detail: event.payload.detail,
        risk: event.payload.risk,
      };
      pendingRef.current = next;
      setPending(next);
    });

    // A newer turn's terminal event (answer-complete for a DIFFERENT
    // turnId, or an interruption) means the user moved on to something else
    // instead of answering yes/no — drop the dialog rather than leaving it
    // stuck on screen forever, mirroring the backend's own "unrelated
    // utterance drops the stale pending confirmation" rule.
    const unlistenComplete = listen<{ turnId: string }>("veronica:answer-complete", (event) => {
      const current = pendingRef.current;
      if (current && event.payload.turnId !== current.turnId) {
        pendingRef.current = null;
        setPending(null);
      }
    });
    const unlistenInterrupted = listen("veronica:interrupted", () => {
      if (pendingRef.current) {
        pendingRef.current = null;
        setPending(null);
      }
    });

    return () => {
      unlistenRequested.then((f) => f());
      unlistenComplete.then((f) => f());
      unlistenInterrupted.then((f) => f());
    };
  }, []);

  const respond = useCallback(async (approved: boolean) => {
    const current = pendingRef.current;
    if (!current) return;
    // Clear immediately (optimistic) — the backend's own answer-complete
    // event for this turnId will follow and render the actual result as a
    // normal assistant message; the dialog itself doesn't wait for that.
    pendingRef.current = null;
    setPending(null);
    try {
      await invoke("respond_to_confirmation", { turnId: current.turnId, approved });
    } catch {
      // Best-effort — if the confirmation already expired (a race with a
      // newer turn), there's nothing left to show the user beyond the
      // dialog already being gone.
    }
  }, []);

  return { pending, respond };
}
