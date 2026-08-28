import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { TranscriptSegment } from "./types";
import { classifyQuestionCompleteness, joinSpeech, SAFETY_NET_MS } from "./questionCompleteness";
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
export interface AskAnswerOptions {
  answerLength: string;
  responseStyle: string;
  humanization: string;
  ttsEnabled: boolean;
}

export interface PriorTurn {
  question: string;
  answer: string;
}

export interface UseAutoAskConfig {
  /// Which transcript sources feed the buffer — re-read on every segment
  /// (not captured once) so a toggle flipped mid-session (e.g. the
  /// overlay's system-audio switch) takes effect immediately.
  acceptSource: (source: TranscriptSegment["source"]) => boolean;
  answerOptions: () => AskAnswerOptions;
  getHistory?: () => PriorTurn[];
  /// Called synchronously, right before `ask_veronica` is invoked, with the
  /// exact question text being sent — e.g. VeronicaOverlay uses this to
  /// push a new pending turn into its conversation list.
  onAskStart?: (question: string) => void;
  onAskSettled?: (question: string, error: unknown) => void;
}

export interface UseAutoAskResult {
  /// The buffered-but-not-yet-sent text — live preview only (e.g. showing
  /// it in a question box); the hook itself doesn't need callers to read
  /// this for correctness.
  committedText: string;
  /// Sends `text` immediately, bypassing the completeness gate — e.g. a
  /// manual "Ask" button click on already-typed/edited text.
  askNow: (text: string) => Promise<void>;
}

export function useAutoAsk(config: UseAutoAskConfig): UseAutoAskResult {
  const configRef = useRef(config);
  configRef.current = config;

  const committedRef = useRef("");
  const [committedText, setCommittedText] = useState("");
  const safetyTimerRef = useRef<number | null>(null);

  const setCommitted = useCallback((value: string) => {
    committedRef.current = value;
    setCommittedText(value);
  }, []);

  const clearSafetyTimer = useCallback(() => {
    if (safetyTimerRef.current) {
      window.clearTimeout(safetyTimerRef.current);
      safetyTimerRef.current = null;
    }
  }, []);

  const dispatch = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    const cfg = configRef.current;
    cfg.onAskStart?.(trimmed);
    try {
      const chosenProvider = loadLlmProvider();
      const llmProvider =
        chosenProvider === "anthropic" || chosenProvider === "openai" || chosenProvider === "gemini" ? chosenProvider : null;
      const options = cfg.answerOptions();
      await invoke<string>("ask_veronica", {
        question: trimmed,
        history: cfg.getHistory?.() ?? [],
        options: {
          answerLength: options.answerLength,
          responseStyle: options.responseStyle,
          humanization: options.humanization,
          llmProvider,
          ttsEnabled: options.ttsEnabled,
        },
      });
      cfg.onAskSettled?.(trimmed, null);
    } catch (e) {
      cfg.onAskSettled?.(trimmed, e);
    }
  }, []);

  const dispatchRef = useRef(dispatch);
  dispatchRef.current = dispatch;

  useEffect(() => {
    const maybeSend = () => {
      clearSafetyTimer();
      const text = committedRef.current;
      if (!text.trim()) return;
      if (classifyQuestionCompleteness(text) === "complete") {
        setCommitted("");
        dispatchRef.current(text);
        return;
      }
      // Reads as an incomplete fragment ("...and", a dangling comma) —
      // held so it can merge with whatever the next Final adds, rather
      // than sending a fragment straight to the fast router (a bare
      // fragment could match a real, possibly irreversible, command out of
      // context). The safety net only fires if nothing more arrives.
      safetyTimerRef.current = window.setTimeout(() => {
        safetyTimerRef.current = null;
        const pending = committedRef.current;
        setCommitted("");
        dispatchRef.current(pending);
      }, SAFETY_NET_MS);
    };

    const unlistenTranscript = listen<TranscriptSegment>("transcript:update", (event) => {
      const segment = event.payload;
      if (!configRef.current.acceptSource(segment.source)) return;
      if (segment.final_text) {
        setCommitted(joinSpeech(committedRef.current, segment.final_text));
        maybeSend();
      } else if (segment.partial_text) {
        // Still speaking — the safety net must not fire mid-utterance.
        clearSafetyTimer();
      }
    });

    return () => {
      unlistenTranscript.then((f) => f());
      clearSafetyTimer();
    };
  }, [clearSafetyTimer, setCommitted]);

  return { committedText, askNow: (text: string) => dispatchRef.current(text) };
}
