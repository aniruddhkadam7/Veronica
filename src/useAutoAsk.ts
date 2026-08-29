import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { TranscriptSegment } from "./types";
import { joinSpeech, SAFETY_NET_MS } from "./questionCompleteness";
import { classifyTurnAction } from "./turnHeuristics";
import { loadLlmProvider } from "./llmProviderSetting";

/// Shared "listen for finalized speech, decide when it's a complete
/// question, send it" logic — used by both VeronicaWidget.tsx (the
/// always-on floating orb) and VeronicaOverlay.tsx (the full conversation
/// view), which previously each hand-rolled a near-identical copy of this
/// with a fixed 900ms/2200ms post-transcription wait applied to every
/// single question. That's gone: a `Final` transcript segment is sent as
/// soon as it reads as a complete question — see `questionCompleteness.ts`'s
/// doc for why a short bounded safety net (not a blanket timer) is still
/// used for genuinely incomplete-looking text.
///
/// Firing `ask_veronica` again while a previous call for this session is
/// still in flight is deliberately allowed, not guarded against: the
/// backend's `AppState::begin_turn` cancels whatever the previous turn was
/// still doing the moment a new one starts (see `veronica::ask_veronica`
/// and `voice_command::mod`'s barge-in path), so a fast follow-up
/// correctly supersedes a stale in-flight answer rather than being dropped
/// or queued behind it.
/// Why a turn was finalized — passed through to the backend as
/// `AskOptions.finalizeReason` purely for debugging (requirement 10); never
/// used for a routing/behavior decision on either side.
export type FinalizeReason = "classifier_complete" | "safety_net_elapsed" | "manual_submit";

export interface AskAnswerOptions {
  answerLength: string;
  responseStyle: string;
  humanization: string;
  ttsEnabled: boolean;
}

export interface UseAutoAskConfig {
  /// Which transcript sources feed the buffer — re-read on every segment
  /// (not captured once) so a toggle flipped mid-session (e.g. the
  /// overlay's system-audio switch) takes effect immediately.
  acceptSource: (source: TranscriptSegment["source"]) => boolean;
  answerOptions: () => AskAnswerOptions;
  /// Called synchronously, right before `ask_veronica` is invoked, with the
  /// exact question text AND the turn_id this call was sent with — e.g.
  /// VeronicaWidget uses `turnId` to correlate its own (invisible, but
  /// still real) conversation-history bookkeeping to the right question,
  /// the same turn_id `ask_veronica`'s events carry — never by "whichever
  /// question was asked most recently," which breaks the moment two turns
  /// can be in flight close together (see VeronicaOverlay.tsx's
  /// `applyTurnEvent` doc for the full story on why that assumption fails).
  onAskStart?: (question: string, turnId: string) => void;
  onAskSettled?: (question: string, turnId: string, error: unknown) => void;
  /// Called on every buffer change (a partial arriving, a Final joined in,
  /// the buffer cleared/sent/suppressed) with the current live-preview
  /// text. Callers with a visible composer (VeronicaOverlay.tsx) use this
  /// to drive it; callers with no text UI (VeronicaWidget.tsx) omit it.
  onBufferChange?: (text: string) => void;
  /// Whether a Sensitive/Destructive confirmation is currently pending —
  /// read fresh on every segment, like `acceptSource`, never captured once.
  /// Defaults to `() => false`. See `classifyTurnAction`'s doc for why this
  /// skips filler suppression entirely while true.
  hasPendingConfirmation?: () => boolean;
}

export interface UseAutoAskResult {
  /// The buffered-but-not-yet-sent text — live preview only (e.g. showing
  /// it in a question box); the hook itself doesn't need callers to read
  /// this for correctness.
  committedText: string;
  /// Sends `text` immediately, bypassing the completeness gate — e.g. a
  /// manual "Ask" button click on already-typed/edited text.
  askNow: (text: string) => Promise<void>;
  /// Imperatively overwrites the live buffer — e.g. the overlay's composer
  /// textarea being edited by hand.
  setBuffer: (text: string) => void;
  /// Clears the buffer and any pending safety timer without sending — e.g.
  /// after an interrupt, or on session reset.
  clearBuffer: () => void;
}

export function useAutoAsk(config: UseAutoAskConfig): UseAutoAskResult {
  const configRef = useRef(config);
  configRef.current = config;

  const committedRef = useRef("");
  const [committedText, setCommittedText] = useState("");
  const safetyTimerRef = useRef<number | null>(null);
  // Guards against the exact same finalized transcript triggering
  // ask_veronica twice (requirement 9) — mirrors VeronicaOverlay.tsx's
  // identical dedup guard on the same underlying risk (a re-delivered Final
  // segment around a mute/barge-in boundary).
  const lastDispatchedRef = useRef<{ text: string; at: number } | null>(null);
  const DEDUPE_WINDOW_MS = 4000;
  // Real STT event cadence, not a synthetic poll — updated only when an
  // actual partial_text event arrives. See turnHeuristics.ts's
  // LIVE_ARRIVAL_GRACE_MS for how this gates finalization.
  const lastPartialAtRef = useRef<number | null>(null);
  // Updated by the tts:speaking-changed listener below — feeds
  // classifyTurnAction's closed-form-response self-hearing guard.
  const ttsStoppedAtRef = useRef<number | null>(null);

  const setCommitted = useCallback((value: string) => {
    committedRef.current = value;
    setCommittedText(value);
    configRef.current.onBufferChange?.(value);
  }, []);

  const setBuffer = useCallback(
    (text: string) => {
      setCommitted(text);
    },
    [setCommitted],
  );

  const clearSafetyTimer = useCallback(() => {
    if (safetyTimerRef.current) {
      window.clearTimeout(safetyTimerRef.current);
      safetyTimerRef.current = null;
    }
  }, []);

  const clearBuffer = useCallback(() => {
    clearSafetyTimer();
    setCommitted("");
  }, [clearSafetyTimer, setCommitted]);

  const dispatch = useCallback(async (text: string, finalizeReason: FinalizeReason) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const now = Date.now();
    if (lastDispatchedRef.current && lastDispatchedRef.current.text === trimmed && now - lastDispatchedRef.current.at < DEDUPE_WINDOW_MS) {
      return;
    }
    lastDispatchedRef.current = { text: trimmed, at: now };
    const cfg = configRef.current;
    const turnId = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    cfg.onAskStart?.(trimmed, turnId);
    try {
      const chosenProvider = loadLlmProvider();
      const llmProvider =
        chosenProvider === "anthropic" || chosenProvider === "openai" || chosenProvider === "gemini" ? chosenProvider : null;
      const options = cfg.answerOptions();
      // No `history` param: ask_veronica derives conversational context
      // from the shared backend conversation store (see conversation.rs's
      // `completed_history`) rather than a caller-supplied list — a
      // follow-up asked here now correctly sees turns from the overlay too.
      await invoke<string>("ask_veronica", {
        question: trimmed,
        turnId,
        options: {
          answerLength: options.answerLength,
          responseStyle: options.responseStyle,
          humanization: options.humanization,
          llmProvider,
          ttsEnabled: options.ttsEnabled,
          finalizeReason,
        },
      });
      cfg.onAskSettled?.(trimmed, turnId, null);
    } catch (e) {
      cfg.onAskSettled?.(trimmed, turnId, e);
    }
  }, []);

  const dispatchRef = useRef(dispatch);
  dispatchRef.current = dispatch;

  useEffect(() => {
    // `hasPriorBufferedText` must reflect the buffer's state BEFORE the
    // just-arrived Final was joined in — this is what lets classifyTurnAction
    // tell an isolated bare filler ("um" with nothing else buffered) apart
    // from a short word continuing an unfinished request ("...called"
    // [pause] "Documents").
    const maybeSend = (hadPriorBufferedText: boolean) => {
      clearSafetyTimer();
      const text = committedRef.current;
      if (!text.trim()) return;
      const action = classifyTurnAction(text, {
        hasPendingConfirmation: configRef.current.hasPendingConfirmation?.() ?? false,
        hasPriorBufferedText: hadPriorBufferedText,
        lastPartialArrivedMsAgo: lastPartialAtRef.current === null ? null : Date.now() - lastPartialAtRef.current,
        msSinceTtsStoppedSpeaking: ttsStoppedAtRef.current === null ? null : Date.now() - ttsStoppedAtRef.current,
      });
      if (action === "suppress") {
        setCommitted("");
        return;
      }
      if (action === "send") {
        setCommitted("");
        dispatchRef.current(text, "classifier_complete");
        return;
      }
      // "hold" — reads as an incomplete fragment ("...and", a dangling
      // comma, still-arriving speech) — held so it can merge with whatever
      // the next Final adds, rather than sending a fragment straight to the
      // fast router (a bare fragment could match a real, possibly
      // irreversible, command out of context). The safety net only fires if
      // nothing more arrives.
      safetyTimerRef.current = window.setTimeout(() => {
        safetyTimerRef.current = null;
        const pending = committedRef.current;
        setCommitted("");
        dispatchRef.current(pending, "safety_net_elapsed");
      }, SAFETY_NET_MS);
    };

    const unlistenTranscript = listen<TranscriptSegment>("transcript:update", (event) => {
      const segment = event.payload;
      if (!configRef.current.acceptSource(segment.source)) return;
      if (segment.final_text) {
        // Interruption check FIRST, before this ever joins the buffer or
        // becomes a real ask_veronica call (requirement 6) — "stop"/"wait"/
        // "hold on"/"cancel" is a control signal, never a normal question.
        // `try_interrupt` both decides AND (if true) stops TTS/cancels the
        // in-flight turn in one atomic backend call. Only the newly-added
        // fragment is checked, matching VeronicaOverlay.tsx's identical
        // guard on this exact backend command.
        const finalText = segment.final_text;
        const hadPriorBufferedText = committedRef.current.trim().length > 0;
        invoke<boolean>("try_interrupt", { text: finalText }).then((wasInterrupt) => {
          if (wasInterrupt) {
            clearSafetyTimer();
            return;
          }
          setCommitted(joinSpeech(committedRef.current, finalText));
          maybeSend(hadPriorBufferedText);
        });
      } else if (segment.partial_text) {
        // Still speaking — a real, live signal that finalization must wait
        // (both via the safety timer being cleared, and via
        // lastPartialAtRef for classifyTurnAction's live-arrival check).
        lastPartialAtRef.current = Date.now();
        clearSafetyTimer();
      }
    });

    const unlistenTtsSpeaking = listen<boolean>("tts:speaking-changed", (event) => {
      if (!event.payload) ttsStoppedAtRef.current = Date.now();
    });

    return () => {
      unlistenTranscript.then((f) => f());
      unlistenTtsSpeaking.then((f) => f());
      clearSafetyTimer();
    };
  }, [clearSafetyTimer, setCommitted]);

  return { committedText, askNow: (text: string) => dispatchRef.current(text, "manual_submit"), setBuffer, clearBuffer };
}
