import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";
import ReactMarkdown from "react-markdown";
import type { TranscriptSegment } from "./types";
import {
  loadOverlaySettings,
  saveOverlaySettings,
  SIZE_FRACTIONS,
  type AnswerLength,
  type OverlaySettings,
  type ResponseStyle,
} from "./overlaySettings";
import { OverlaySettingsPanel } from "./OverlaySettingsPanel";
import {
  AUTO_AI_SILENCE_MS_COMPLETE,
  AUTO_AI_SILENCE_MS_INCOMPLETE,
  classifyQuestionCompleteness,
  joinSpeech,
} from "./questionCompleteness";
import { AudioLevelBars, useSttSpeaking } from "./ui";
import { loadLlmProvider } from "./llmProviderSetting";

interface OverlayCaptureStatus {
  excluded: boolean;
}

/// One exchange in the running conversation.
///
/// `answer` is empty while the reply is still streaming; `pending` marks the
/// turn currently being answered so the UI can show a thinking state without a
/// separate top-level status variable.
interface Turn {
  id: string;
  question: string;
  answer: string;
  pending: boolean;
  failed?: boolean;
}

const overlayWindow = getCurrentWebviewWindow();

/// Cap on how tall the question box can grow before it scrolls internally
/// instead of continuing to shrink the conversation area above it.
const MAX_QUESTION_INPUT_PX = 160;

/// Range for the header's opacity slider. The floor matches
/// `MIN_USABLE_OPACITY` in overlaySettings.ts — below that the overlay is hard
/// to find on a busy desktop, and a user who cannot see it cannot drag the
/// slider back up.
const OPACITY_MIN = 0.15;
const OPACITY_MAX = 1;

// Auto AI silence-window constants, the completeness classifier, and
// joinSpeech() now live in questionCompleteness.ts, shared with Custom
// Agents' overlay so both Auto AI implementations behave identically.

export function VeronicaOverlay() {
  // The whole conversation, oldest first. Nothing is ever removed or replaced
  // — a new question appends, and streaming deltas only ever mutate the last
  // turn's answer.
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [captureExcluded, setCaptureExcluded] = useState<boolean | null>(null);
  // Whether audio capture is confirmed actually running, so the header can
  // say "Listening" truthfully instead of unconditionally the moment the
  // overlay opens. `start_system_audio_capture` now only resolves once
  // WASAPI + STT are both confirmed ready (or rejects with a real error) —
  // see commands.rs's start_capture_inner — so this reflects real state,
  // not an assumption. "capture already running" (a second, harmless
  // invocation racing an already-successful start elsewhere) is treated as
  // active too, not an error.
  const [captureActive, setCaptureActive] = useState(false);
  // System audio (the other speaker/app sound) is opt-in — off by default,
  // toggled on only when explicitly wanted (see the header toggle and
  // Ctrl+Shift+V). Mic listening itself is always on for the whole session
  // (started by App.tsx's handleStart before this window ever shows) and
  // has no separate on/off state here.
  const [systemAudioActive, setSystemAudioActive] = useState(false);
  const systemAudioActiveRef = useRef(false);
  // True from the moment a question is sent (askAI) until its
  // answer starts streaming back — the "Thinking" segment of Veronica's
  // Idle -> Listening -> Thinking -> Speaking cycle. `busy` alone can't
  // distinguish this from "Speaking" (the delta/complete phase), so this
  // flips off on the first answer-delta or on answer-complete/failure.
  const [veronicaThinking, setVeronicaThinking] = useState(false);
  // True only while STT is actively producing transcript output right now —
  // gates the listening animation so it doesn't run through silence just
  // because capture is technically still open.
  const sttSpeaking = useSttSpeaking();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [confirmingClose, setConfirmingClose] = useState(false);
  // Shows the opacity percentage in the header for a moment after adjusting.
  const [opacityHint, setOpacityHint] = useState(false);
  const [settings, setSettings] = useState<OverlaySettings>(() => loadOverlaySettings());

  const questionRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Everything said since the last "Ask AI", as finalized by STT. This is
  // the buffer that makes the field behave like continuous dictation across
  // pauses.
  //
  // The STT layer finalizes an utterance whenever it hears trailing silence,
  // and then starts a *fresh* segment for whatever is said next. That is
  // correct behaviour for a transcript, but the overlay must not treat it as
  // "the question ended" — a speaker pausing mid-sentence would otherwise have
  // everything before the pause replaced by the few words after it. So finals
  // accumulate here and are only cleared by an explicit user action.
  const committedRef = useRef("");
  const sessionStartedAtRef = useRef(Date.now());
  // Mirrors `busy` for the event listeners, which close over stale state
  // otherwise. Answer deltas must keep landing regardless of re-renders.
  const busyRef = useRef(false);
  const hintTimerRef = useRef<number | null>(null);
  // Debounce timer that fires askAI() once speech has stopped for
  // AUTO_AI_SILENCE_MS — always active now (there is no manual toggle;
  // talking to Veronica and pausing always sends). Lives in a ref (not
  // state) since it's set/cleared from the transcript listener, which must
  // not re-subscribe on every keystroke-equivalent state change.
  const autoAiTimerRef = useRef<number | null>(null);
  // askAI changes identity on every render; mirrored into a ref so the
  // transcript listener (subscribed once, on mount) always calls the latest
  // version instead of one captured at effect-setup time.
  const askAIRef = useRef<() => void>(() => {});

  useEffect(
    () => () => {
      if (hintTimerRef.current) window.clearTimeout(hintTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    saveOverlaySettings(settings);
  }, [settings]);

  // Live transcript feed: reuse the exact same "transcript:update" event the
  // main window listens to (see App.tsx) — both windows share the same Rust
  // backend process/state, so no separate capture pipeline is needed here.
  //
  // This behaves like continuous dictation. There is deliberately no question
  // detection, no sentence-end detection, and nothing that clears the field on
  // silence: finalized utterances are appended to a running buffer, and a
  // silence-based auto-send (below) fires once the user stops talking — the
  // whole point of talking to Veronica. How long that pause needs to be is
  // adaptive: classifyQuestionCompleteness() judges from the buffered text's
  // own wording whether it reads as a finished question or a trailing
  // clause, so a genuine pause mid-question ("I used this because...") gets
  // more room to continue before it's sent, while a question that already
  // reads as complete fires quickly.
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
      // The user's own voice (microphone) always feeds the question box —
      // mic listening is always on for the whole session. System audio (the
      // other speaker/app sound) only does while the header's toggle is on
      // — see toggleSystemAudio — so it stays silent by default.
      if (segment.source === "SYSTEM_AUDIO" && !systemAudioActiveRef.current) return;
      if (segment.source !== "SYSTEM_AUDIO" && segment.source !== "MICROPHONE") return;
      // While an answer is streaming, keep buffering finals silently: the
      // user may already be asking the next question. They land in the
      // field, not in the answer that's still arriving.
      if (segment.final_text) {
        committedRef.current = joinSpeech(committedRef.current, segment.final_text);
        setQuestion(committedRef.current);
        armAutoAiTimer();
      } else if (segment.partial_text) {
        setQuestion(joinSpeech(committedRef.current, segment.partial_text));
        // Still speaking — any timer counting down from an earlier final must
        // not fire mid-sentence.
        clearAutoAiTimer();
      }
    });

    // Deltas always belong to the last turn — it is the only one that can be
    // pending, because Ask AI is disabled until the previous answer completes.
    const unlistenDelta = listen<string>("veronica:answer-delta", (event) => {
      setVeronicaThinking(false);
      setTurns((prev) => {
        if (!prev.length) return prev;
        const last = prev[prev.length - 1];
        if (!last.pending) return prev;
        return [...prev.slice(0, -1), { ...last, answer: last.answer + event.payload }];
      });
    });

    const unlistenComplete = listen<string>("veronica:answer-complete", (event) => {
      setVeronicaThinking(false);
      setTurns((prev) => {
        if (!prev.length) return prev;
        const last = prev[prev.length - 1];
        if (!last.pending) return prev;
        // Prefer the completed answer from the event: it is authoritative,
        // where the accumulated deltas could have dropped one.
        return [
          ...prev.slice(0, -1),
          { ...last, answer: event.payload || last.answer, pending: false },
        ];
      });
      busyRef.current = false;
      setBusy(false);
    });

    return () => {
      unlistenTranscript.then((f) => f());
      unlistenDelta.then((f) => f());
      unlistenComplete.then((f) => f());
      clearAutoAiTimer();
    };
  }, []);

  useEffect(() => {
    invoke<OverlayCaptureStatus>("show_interview_overlay")
      .then((status) => setCaptureExcluded(status.excluded))
      .catch((e) => setError(String(e)));

    // A brand-new overlay window is created at the Rust-side default size,
    // which does not know about a returning user's saved Small/Medium/Large
    // choice. Apply it here so that choice takes effect immediately rather
    // than only after the user next opens Settings and touches the size
    // control themselves.
    invoke("resize_interview_overlay", { fraction: SIZE_FRACTIONS[settings.size] }).catch(() => {
      // Best-effort — the window simply stays at its default size.
    });

    // App.tsx's handleStart already started mic assistant before showing
    // this overlay, so the live transcript this overlay listens to already
    // exists by the time this mounts. Calling start_mic_assistant here too
    // is just a readiness check — "mic assistant already running" is the
    // expected/harmless outcome; anything else (e.g. no microphone device)
    // must surface, or the overlay would sit on "Starting…" forever with no
    // explanation.
    invoke("start_mic_assistant")
      .then(() => setCaptureActive(true))
      .catch((e) => {
        if (String(e) !== "mic assistant already running") {
          setError(`Could not start listening: ${String(e)}`);
        } else {
          setCaptureActive(true);
        }
      });
  }, []);

  // The overlay window is *reused* across sessions, not recreated (see
  // overlay_window.rs's show_overlay_window) — its WebView2 process/DOM
  // stays alive while hidden between "Close" and the next "Start". Without
  // this, reopening for a brand-new session would still show the previous
  // session's whole conversation the instant the window reappears, since
  // none of this component's state resets on its own. The Rust side emits
  // this right before re-showing an existing window.
  useEffect(() => {
    const unlistenReset = listen("overlay:reset-session", () => {
      setTurns([]);
      setQuestion("");
      setBusy(false);
      setError(null);
      setConfirmingClose(false);
      setSettingsOpen(false);
      setVeronicaThinking(false);
      committedRef.current = "";
      busyRef.current = false;
      sessionStartedAtRef.current = Date.now();
    });
    return () => {
      unlistenReset.then((f) => f());
    };
  }, []);

  // Follow the conversation as it grows, the way a chat window does.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [turns, question]);

  const askAI = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || busyRef.current) return;

    setError(null);
    // The question moves into the conversation and the buffer resets, so
    // anything said from here starts the next question instead of being
    // appended onto one already sent.
    const turn: Turn = {
      id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      question: trimmed,
      answer: "",
      pending: true,
    };
    setTurns((prev) => [...prev, turn]);
    setQuestion("");
    committedRef.current = "";
    busyRef.current = true;
    setBusy(true);
    setVeronicaThinking(true);

    // Prior turns only — the one just pushed is the current question, and
    // completed turns are the only ones worth replaying.
    const history = turns
      .filter((t) => !t.pending && t.answer.trim() && !t.failed)
      .map((t) => ({ question: t.question, answer: t.answer }));

    try {
      const chosenProvider = loadLlmProvider();
      const llmProvider =
        chosenProvider === "anthropic" || chosenProvider === "openai" || chosenProvider === "gemini"
          ? chosenProvider
          : null;
      // ask_veronica either answers directly or, when the request is
      // asking it to do something (open an app/file/folder/URL, check
      // simple system info), runs that action server-side and returns its
      // result instead — see veronica::ask_veronica and
      // personal/prompts/veronica.rs's ACTION-TAKING section. There is no
      // separate "is this a command" step here: every question goes
      // through the same call, and the backend decides.
      await invoke<string>("ask_veronica", {
        question: trimmed,
        history,
        options: {
          answerLength: settings.answerLength,
          responseStyle: settings.responseStyle,
          humanization: settings.humanization,
          llmProvider,
        },
      });
    } catch (e) {
      setError(String(e));
      // Keep the question visible in the conversation and mark it failed,
      // rather than silently dropping what was asked.
      setTurns((prev) =>
        prev.map((t) => (t.id === turn.id ? { ...t, pending: false, failed: true } : t)),
      );
      busyRef.current = false;
      setBusy(false);
      setVeronicaThinking(false);
    }
  }, [question, turns, settings]);

  useEffect(() => {
    askAIRef.current = askAI;
  }, [askAI]);

  // Hides the overlay. Turns still pending/failed are simply dropped —
  // there is no history feature to archive them to.
  const closeOverlay = useCallback(async () => {
    setConfirmingClose(false);

    if (autoAiTimerRef.current) {
      window.clearTimeout(autoAiTimerRef.current);
      autoAiTimerRef.current = null;
    }
    committedRef.current = "";
    sessionStartedAtRef.current = Date.now();
    setTurns([]);
    setQuestion("");
    setError(null);
    busyRef.current = false;
    setBusy(false);
    setVeronicaThinking(false);

    // The overlay window is reused (not destroyed) between sessions, so a
    // capture session left running would otherwise keep transcribing into
    // the next session's freshly-cleared question box.
    invoke("stop_mic_assistant").catch(() => {});
    if (systemAudioActiveRef.current) {
      systemAudioActiveRef.current = false;
      setSystemAudioActive(false);
      invoke("stop_audio_capture").catch(() => {});
    }

    invoke("hide_interview_overlay").catch(() => overlayWindow.hide());
  }, []);

  // Entry point for both the ✕ button and Escape: conversations with at
  // least one exchange ask for confirmation first (closing mid-conversation
  // is easy to trigger by accident), while an empty session just closes
  // immediately — there's nothing to lose by skipping the prompt.
  const requestClose = useCallback(() => {
    const hasConversation = turns.some((t) => !t.pending && t.answer.trim());
    if (hasConversation) {
      setConfirmingClose(true);
    } else {
      closeOverlay();
    }
  }, [turns, closeOverlay]);

  // Header toggle for system audio (the other speaker/app sound) — off by
  // default; click to start listening to it too, click again to stop.
  // Independent of mic listening, which is always on for the whole session.
  const toggleSystemAudio = useCallback(() => {
    if (systemAudioActiveRef.current) {
      systemAudioActiveRef.current = false;
      setSystemAudioActive(false);
      invoke("stop_audio_capture").catch((e) => setError(String(e)));
      return;
    }
    systemAudioActiveRef.current = true;
    setSystemAudioActive(true);
    setError(null);
    invoke("start_system_audio_capture").catch((e) => {
      if (String(e) === "capture already running") return;
      systemAudioActiveRef.current = false;
      setSystemAudioActive(false);
      setError(String(e));
    });
  }, []);
  const toggleSystemAudioRef = useRef(toggleSystemAudio);
  useEffect(() => {
    toggleSystemAudioRef.current = toggleSystemAudio;
  }, [toggleSystemAudio]);

  // Veronica's global shortcut (Ctrl+Shift+V, registered Rust-side in
  // lib.rs) toggles system audio listening — forwarded as an event rather
  // than calling start/stop_system_audio_capture directly from Rust so this
  // component's React state (which the header button rendering depends on)
  // stays the single source of truth.
  useEffect(() => {
    const unlisten = listen("veronica:toggle-shortcut", () => {
      toggleSystemAudioRef.current();
    });
    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  // Grows the textarea to fit its content, up to MAX_QUESTION_INPUT_PX — past
  // that it scrolls internally instead of continuing to eat the conversation
  // area above it. Plain CSS can't do this (a textarea's height doesn't track
  // its own content), so height is measured and set imperatively whenever the
  // text changes, including from STT — a long dictated question needs this
  // exactly as much as one the user typed.
  const autoGrow = useCallback(() => {
    const el = questionRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_QUESTION_INPUT_PX)}px`;
  }, []);

  const handleQuestionChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      // Adopt the user's text as the new buffer. Without this, a later
      // utterance would append onto the pre-edit text and silently undo their
      // correction.
      committedRef.current = e.target.value;
      setQuestion(e.target.value);
    },
    [],
  );

  // Runs after every render where `question` changed — covers typing, STT
  // partials/finals, and the reset to "" after Ask AI — rather than being
  // wired into each of those call sites individually.
  useEffect(() => {
    autoGrow();
  }, [question, autoGrow]);

  const handleQuestionKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        askAI();
      }
      // Shift+Enter falls through to the textarea's default behavior and
      // inserts a newline.
    },
    [askAI],
  );

  const handleOpacityChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const next = Math.min(OPACITY_MAX, Math.max(OPACITY_MIN, Number(e.target.value)));
    setSettings((prev) => ({ ...prev, opacity: next }));
    // Show the percentage while dragging, then get out of the way.
    setOpacityHint(true);
    if (hintTimerRef.current) window.clearTimeout(hintTimerRef.current);
    hintTimerRef.current = window.setTimeout(() => setOpacityHint(false), 1000);
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (confirmingClose) {
        if (e.key === "Escape") setConfirmingClose(false);
        return;
      }
      if (settingsOpen) {
        if (e.key === "Escape") setSettingsOpen(false);
        return;
      }
      // The textarea has its own Enter/Shift+Enter handling above; this
      // top-level listener only needs to cover ESC (hide) globally and ENTER
      // when focus is somewhere else in the overlay.
      const activeIsTextarea = document.activeElement === questionRef.current;
      if (e.key === "Enter" && !activeIsTextarea) {
        e.preventDefault();
        askAI();
      } else if (e.key === "Escape") {
        requestClose();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [askAI, settingsOpen, confirmingClose, requestClose]);

  const overlayStyle: React.CSSProperties = {
    fontSize: `${settings.fontSize}px`,
    // Applied to the panel background rather than the whole element, so text
    // stays fully legible while the window itself reads as translucent.
    ["--overlay-alpha" as string]: settings.opacity,
  };

  const hasConversation = turns.length > 0;
  const opacityPercent = Math.round(settings.opacity * 100);

  // Veronica's Idle -> Listening -> Thinking -> Speaking cycle, derived from
  // state that already exists rather than tracked separately (so it can
  // never drift out of sync with what's actually happening):
  //   - Thinking: a question has been sent and no answer text has arrived
  //     yet (veronicaThinking, cleared on the first delta).
  //   - Speaking: the answer is actively streaming in (busy, past Thinking).
  //   - Listening: not currently busy, and STT is hearing speech right now
  //     (sttSpeaking) — silence while listening is still technically on
  //     reads as Idle.
  //   - Idle: silent and not busy.
  const veronicaState: "idle" | "listening" | "thinking" | "speaking" = veronicaThinking
    ? "thinking"
    : busy
      ? "speaking"
      : sttSpeaking
        ? "listening"
        : "idle";

  return (
    <div
      className={`overlay-root density-${settings.density} size-${settings.size}`}
      style={overlayStyle}
    >
      {/* "deep" so the whole bar drags — including the empty space and the
          status dot — rather than only direct clicks on this exact element,
          which is what a bare attribute means. Safe for the controls on the
          right: Tauri's drag script treats BUTTON/INPUT as clickable and
          blocks dragging for them (tauri/src/window/scripts/drag.js). */}
      <div
        className="overlay-header"
        data-tauri-drag-region={settings.dragEnabled ? "deep" : undefined}
      >
        <AudioLevelBars active={sttSpeaking && !busy} />
        <div className="overlay-title">
          {/* While adjusting, the title doubles as the readout — the change
              itself can be hard to judge against a dark desktop, and this
              avoids spending permanent header space on a number. Before
              capture is confirmed active, this says "Starting…" rather than
              claiming "Listening" while audio/STT may still be initializing
              (or may have failed — see the error banner below). */}
          {opacityHint
            ? `Opacity ${opacityPercent}%`
            : veronicaState === "thinking"
              ? "Veronica — Thinking…"
              : veronicaState === "speaking"
                ? "Veronica — Speaking…"
                : veronicaState === "listening"
                  ? "Veronica — Listening"
                  : captureActive
                    ? "Veronica — Idle"
                    : "Starting…"}
        </div>

        <div className="overlay-header-actions">
          {/* System audio toggle: off by default. Click to also listen to
              the other speaker/app sound (feeds the same question box mic
              speech does) and click again (or Ctrl+Shift+V) to stop — see
              toggleSystemAudio and the transcript:update listener. Mic
              listening itself has no toggle here — it's always on for the
              whole session, started before this overlay ever shows. */}
          <button
            type="button"
            className={`overlay-icon-button overlay-voice-command-button${systemAudioActive ? " active" : ""}`}
            onClick={toggleSystemAudio}
            title={
              systemAudioActive
                ? "Listening to system audio too — click or Ctrl+Shift+V to stop"
                : "Click or Ctrl+Shift+V to also listen to system audio"
            }
          >
            🔊
          </button>

          {/* Safe inside the header's drag region without any extra handling:
              Tauri's drag script treats INPUT as clickable and blocks dragging
              for it (see tauri/src/window/scripts/drag.js). Do NOT wrap this in
              a <label> or give it a data-tauri-drag-region attribute — either
              would hand the gesture back to the window and make the bar
              undraggable. */}
          <input
            type="range"
            className="overlay-opacity-slider"
            min={OPACITY_MIN}
            max={OPACITY_MAX}
            step={0.01}
            value={settings.opacity}
            onChange={handleOpacityChange}
            aria-label="Overlay opacity"
            title={`Opacity — ${opacityPercent}%`}
          />

          <button
            className="overlay-icon-button"
            onClick={() => setSettingsOpen((v) => !v)}
            title="Settings"
          >
            ⚙
          </button>
          <button className="overlay-icon-button close" onClick={requestClose} title="Close (Esc)">
            ✕
          </button>
        </div>
      </div>

      {/* Screen-capture exclusion (SetWindowDisplayAffinity/
          WDA_EXCLUDEFROMCAPTURE) can fail — older Windows builds don't
          support it, and the OS call can simply error. When that happens
          this window IS visible to screen share/recording, which defeats
          the whole point of a private overlay, so this can't be a detail
          tucked away in Settings — it has to stay on screen for the entire
          session, not just flash once. Rendered outside the
          settings/confirm branches below so it persists across all of the
          overlay's views. */}
      {captureExcluded === false && (
        <p className="overlay-capture-warning">
          ⚠ Screen capture protection unavailable — this window may be visible if you share your screen
        </p>
      )}

      {confirmingClose ? (
        <div className="overlay-confirm-close">
          <p className="overlay-confirm-close-title">End this conversation?</p>
          <p className="overlay-confirm-close-body">Closing will clear the current conversation.</p>
          <div className="overlay-confirm-close-actions">
            <button className="overlay-text-button" onClick={() => setConfirmingClose(false)}>
              Keep going
            </button>
            <button className="overlay-text-button primary" onClick={closeOverlay}>
              Close
            </button>
          </div>
        </div>
      ) : settingsOpen ? (
        <OverlaySettingsPanel
          settings={settings}
          onChange={setSettings}
          onClose={() => setSettingsOpen(false)}
          captureExcluded={captureExcluded}
        />
      ) : (
        <>
          <div className="overlay-chat" ref={scrollRef}>
            {!hasConversation && !question && (
              <p className="overlay-empty">Waiting for you to speak…</p>
            )}

            {turns.map((turn) => (
              <div key={turn.id} className="chat-turn">
                <div className="chat-message interviewer">
                  <span className="chat-role">You</span>
                  <p className="chat-text">{turn.question}</p>
                </div>
                <div className={`chat-message assistant${turn.failed ? " failed" : ""}`}>
                  <span className="chat-role">Veronica</span>
                  {turn.answer ? (
                    <div className="chat-text chat-markdown">
                      <ReactMarkdown>{turn.answer}</ReactMarkdown>
                    </div>
                  ) : turn.failed ? (
                    <p className="chat-text muted">Couldn't get an answer.</p>
                  ) : (
                    <p className="chat-text muted thinking">Thinking…</p>
                  )}
                </div>
              </div>
            ))}
          </div>

          <div className="overlay-compose">
            <textarea
              ref={questionRef}
              className="overlay-question-input"
              value={question}
              onChange={handleQuestionChange}
              onKeyDown={handleQuestionKeyDown}
              placeholder={hasConversation ? "Ask Veronica…" : "Ask Veronica, or say it out loud…"}
              rows={1}
            />
            <button
              className="overlay-ask-button"
              onClick={askAI}
              disabled={!question.trim() || busy}
              title="Ask (Enter)"
            >
              Ask
            </button>
          </div>

          {error && <p className="overlay-error">{error}</p>}
        </>
      )}
    </div>
  );
}

export type { AnswerLength, ResponseStyle };
