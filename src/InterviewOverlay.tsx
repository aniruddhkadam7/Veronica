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
import { MeetingSummaryView, type MeetingSummary } from "./MeetingSummary";

interface OverlayCaptureStatus {
  excluded: boolean;
}

/// Veronica's two modes — mirrors the Rust `state::Mode` enum's wire shape
/// (`"INTERVIEW" | "MEETING"`, same as `App.tsx`'s `Mode` type). Read once on
/// mount via `get_mode` and again on every `overlay:reset-session` (the
/// overlay window is reused, not recreated, between sessions — see
/// closeOverlay below), and updated locally + pushed to the backend via
/// `set_mode` whenever a voice command switches it (tryModeSwitchCommand).
type Mode = "INTERVIEW" | "MEETING";

/// One line of Meeting mode's live dual-speaker transcript panel — ported
/// from the retired MeetingOverlay.tsx verbatim.
interface TranscriptEntry {
  id: string;
  speaker: "Others" | "Me";
  text: string;
}

interface ActiveMeetingInfo {
  meetingTitle: string;
  participants: string;
}

/// Reads the meeting title/participants App.tsx stashed in localStorage
/// right before opening the overlay (see startMeeting in App.tsx) — ported
/// from the retired MeetingOverlay.tsx verbatim.
function loadActiveMeeting(): ActiveMeetingInfo {
  try {
    const raw = window.localStorage.getItem("meeting-mode:active-meeting");
    if (!raw) return { meetingTitle: "", participants: "" };
    const parsed = JSON.parse(raw);
    return {
      meetingTitle: parsed.meetingTitle || "",
      participants: parsed.participants || "",
    };
  } catch {
    return { meetingTitle: "", participants: "" };
  }
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

export function InterviewOverlay() {
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
  // invocation racing an already-successful start from InterviewSetup) is
  // treated as active too, not an error.
  const [captureActive, setCaptureActive] = useState(false);
  // Toggled by the header mic button: while true, the user's own microphone
  // is transcribed into the question box exactly like interviewer (system
  // audio) speech, and auto-asks once they stop talking regardless of the
  // separate Auto AI toggle — see start_mic_assistant/stop_mic_assistant and
  // the transcript:update listener below. Mirrored into a ref because that
  // listener is subscribed once on mount and must see the live value.
  const [micAssistantActive, setMicAssistantActive] = useState(false);
  const micAssistantActiveRef = useRef(false);
  // True from the moment a mic-assistant question is sent (askAI) until its
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

  // Veronica's active mode — see the `Mode` type comment above. Mirrored into
  // a ref for the same reason micAssistantActiveRef exists: the
  // transcript:update listener (subscribed once, on mount) must always see
  // the live value, not the one captured when it was set up.
  const [mode, setModeState] = useState<Mode>("INTERVIEW");
  const modeRef = useRef<Mode>("INTERVIEW");
  const setMode = useCallback((next: Mode) => {
    modeRef.current = next;
    setModeState(next);
    invoke("set_mode", { mode: next }).catch(() => {});
  }, []);

  // Meeting mode's live dual-speaker transcript panel + end-of-meeting
  // summary — ported from the retired MeetingOverlay.tsx. `summary`/
  // `endMeeting` currently have no button wired to them (matching
  // MeetingOverlay's own state before the merge — see the "temporarily
  // removed" note near requestClose's JSX below), kept rather than deleted
  // so restoring the End Meeting button is a one-line change.
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [summary, setSummary] = useState<MeetingSummary | null>(null);
  const meetingInfoRef = useRef<ActiveMeetingInfo>(loadActiveMeeting());
  const rawTurnsRef = useRef<{ speaker: "ME" | "OTHER"; text: string }[]>([]);

  const questionRef = useRef<HTMLTextAreaElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Everything the interviewer has said since the last "Ask AI", as finalized
  // by STT. This is the buffer that makes the field behave like continuous
  // dictation across pauses.
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
  // Auto AI: debounce timer that fires askAI() once the interviewer has
  // stopped talking for AUTO_AI_SILENCE_MS. Lives in a ref (not state) since
  // it's set/cleared from the transcript listener, which must not re-subscribe
  // on every keystroke-equivalent state change.
  const autoAiTimerRef = useRef<number | null>(null);
  // askAI/settings change identity on every render; mirrored into refs so the
  // transcript listener (subscribed once, on mount) always calls the latest
  // version instead of one captured at effect-setup time.
  const askAIRef = useRef<() => void>(() => {});
  const autoAIEnabledRef = useRef(false);

  useEffect(
    () => () => {
      if (hintTimerRef.current) window.clearTimeout(hintTimerRef.current);
    },
    [],
  );

  useEffect(() => {
    saveOverlaySettings(settings);
  }, [settings]);

  useEffect(() => {
    autoAIEnabledRef.current = settings.autoAI;
    // Turning Auto AI off mid-pause must cancel any timer already counting
    // down — otherwise a question could still auto-send just after the user
    // switched it off.
    if (!settings.autoAI && autoAiTimerRef.current) {
      window.clearTimeout(autoAiTimerRef.current);
      autoAiTimerRef.current = null;
    }
  }, [settings.autoAI]);

  // Live transcript feed: reuse the exact same "transcript:update" event the
  // main window listens to (see App.tsx) — both windows share the same Rust
  // backend process/state, so no separate capture pipeline is needed here.
  //
  // This behaves like continuous dictation. There is deliberately no question
  // detection, no sentence-end detection, and nothing that clears the field on
  // silence: finalized utterances are appended to a running buffer and only an
  // explicit "Ask AI" (or the user editing/clearing it) ends the question.
  //
  // Auto AI (when enabled in settings) layers a silence-based heuristic on
  // top of this same buffer rather than replacing it: any new partial/final
  // text always resets the debounce timer below, so a mid-question pause
  // never gets treated as "done" — only a real gap in speech does, the same
  // signal a human listener would use. How long that gap needs to be is
  // itself adaptive: classifyQuestionCompleteness() judges from the buffered
  // text's own wording whether it reads as a finished question or a
  // trailing clause, so a genuine pause mid-question ("I used this
  // because...") gets more room to continue before Auto AI gives up and
  // sends it, while a question that already reads as complete fires quickly.
  useEffect(() => {
    const clearAutoAiTimer = () => {
      if (autoAiTimerRef.current) {
        window.clearTimeout(autoAiTimerRef.current);
        autoAiTimerRef.current = null;
      }
    };

    const armAutoAiTimer = () => {
      clearAutoAiTimer();
      // Mic assistant mode always auto-asks on the user's own speech — the
      // whole point of talking to it — regardless of the separate Auto AI
      // toggle, which otherwise only governs interviewer (system audio) speech.
      if ((!autoAIEnabledRef.current && !micAssistantActiveRef.current) || busyRef.current) return;
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

      // Meeting mode: every finalized segment (either speaker) is a live
      // transcript line, not dictation into the question box — ported from
      // MeetingOverlay.tsx verbatim. Mode switch commands ("enter interview
      // mode" etc.) are still recognized from the user's own mic speech even
      // here, so they work regardless of which mode they're spoken from.
      if (modeRef.current === "MEETING") {
        if (segment.final_text) {
          if (
            segment.source === "MICROPHONE" &&
            (tryModeSwitchCommandRef.current(segment.final_text) ||
              tryVeronicaActionRef.current(segment.final_text))
          ) {
            return;
          }
          const speaker = segment.source === "SYSTEM_AUDIO" ? "Others" : "Me";
          setTranscript((prev) => [
            ...prev.slice(-49),
            { id: segment.id, speaker, text: segment.final_text as string },
          ]);
          rawTurnsRef.current.push({
            speaker: speaker === "Others" ? "OTHER" : "ME",
            text: segment.final_text,
          });
        }
        return;
      }

      // Interview mode (unchanged): interviewer speech (system audio) always
      // feeds the question box. The user's own voice (microphone) only does
      // while mic assistant mode is toggled on — see startMicAssistant/
      // stopMicAssistant — so plain background mic pickup can never silently
      // populate the field.
      if (segment.source === "MICROPHONE" && !micAssistantActiveRef.current) return;
      if (segment.source !== "SYSTEM_AUDIO" && segment.source !== "MICROPHONE") return;
      // While an answer is streaming, keep buffering finals silently: the
      // interviewer may already be asking the next question. They land in the
      // field, not in the answer that's still arriving.
      if (segment.final_text) {
        if (
          segment.source === "MICROPHONE" &&
          (tryModeSwitchCommandRef.current(segment.final_text) ||
            tryLaunchAppCommandRef.current(segment.final_text) ||
            tryVeronicaActionRef.current(segment.final_text))
        ) {
          return;
        }
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

    // App.tsx calls set_mode right before showing this overlay (see
    // startInterview/startMeeting), so the backend's current mode is
    // authoritative here — this window is a separate React tree with no
    // direct access to App.tsx's own `mode` state.
    invoke<Mode>("get_mode")
      .then((m) => {
        modeRef.current = m;
        setModeState(m);
        if (m === "MEETING") meetingInfoRef.current = loadActiveMeeting();
      })
      .catch(() => {});

    // A brand-new overlay window is created at the Rust-side default size
    // (interview_mode::window::DEFAULT_SIZE_FRACTION), which does not know
    // about a returning user's saved Small/Medium/Large choice. Apply it here
    // so that choice takes effect immediately rather than only after the user
    // next opens Settings and touches the size control themselves.
    invoke("resize_interview_overlay", { fraction: SIZE_FRACTIONS[settings.size] }).catch(() => {
      // Best-effort — the window simply stays at its default size.
    });

    // Interview Mode can now be opened without having started a recording
    // first (see App.tsx), but the live transcript this overlay listens to
    // only exists once WASAPI+STT capture is actually running. Start it here
    // if it isn't already active so "Listening" is never just a silent
    // no-op; "capture already running" from a prior Start Recording click is
    // the one expected/harmless failure — anything else (e.g. no audio
    // device) must surface, or the overlay would sit on "Waiting for the
    // interviewer to speak…" forever with no explanation.
    invoke("start_system_audio_capture")
      .then(() => setCaptureActive(true))
      .catch((e) => {
        if (String(e) !== "capture already running") {
          setError(`Could not start audio capture: ${String(e)}`);
        } else {
          // A session was already started elsewhere (e.g. InterviewSetup's
          // own call, which this effect always races against) — that
          // session is genuinely active, so this is not a failure.
          setCaptureActive(true);
        }
      });
  }, []);

  // The overlay window is *reused* across interviews, not recreated (see
  // overlay_window.rs's show_overlay_window) — its WebView2 process/DOM
  // stays alive while hidden between "Close" and the next "Start". Without
  // this, reopening for a brand-new interview would still show the
  // previous session's whole conversation the instant the window
  // reappears, since none of this component's state resets on its own.
  // The Rust side emits this right before re-showing an existing window.
  useEffect(() => {
    const unlistenReset = listen("overlay:reset-session", () => {
      setTurns([]);
      setQuestion("");
      setBusy(false);
      setError(null);
      setConfirmingClose(false);
      setSettingsOpen(false);
      setVeronicaThinking(false);
      setTranscript([]);
      setSummary(null);
      committedRef.current = "";
      busyRef.current = false;
      rawTurnsRef.current = [];
      sessionStartedAtRef.current = Date.now();
      invoke<Mode>("get_mode")
        .then((m) => {
          modeRef.current = m;
          setModeState(m);
          if (m === "MEETING") meetingInfoRef.current = loadActiveMeeting();
        })
        .catch(() => {});
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
    if (micAssistantActiveRef.current) setVeronicaThinking(true);

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
      const currentMode = modeRef.current;
      await invoke<string>("ask_veronica", {
        question: trimmed,
        mode: currentMode,
        history,
        // ask_veronica picks whichever of these two matches `mode` and
        // ignores the other — see veronica::ask_veronica.
        interviewOptions:
          currentMode === "INTERVIEW"
            ? {
                answerLength: settings.answerLength,
                responseStyle: settings.responseStyle,
                role: settings.role.trim() || null,
                jobDescription: settings.jobDescription.trim() || null,
                englishLevel: settings.englishLevel,
                humanization: settings.humanization,
                llmProvider,
              }
            : null,
        meetingOptions:
          currentMode === "MEETING"
            ? {
                answerLength: settings.answerLength,
                responseStyle: settings.responseStyle,
                humanization: settings.humanization,
                meetingTitle: meetingInfoRef.current.meetingTitle || null,
                participants: meetingInfoRef.current.participants || null,
                llmProvider,
              }
            : null,
      });
    } catch (e) {
      setError(String(e));
      // Keep the question visible in the conversation and mark it failed,
      // rather than silently dropping what the interviewer asked.
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

  // Archives the Q&A turns so far as one history entry, then hides the
  // overlay. Turns still pending/failed are skipped — only completed
  // exchanges are worth keeping. An empty result (nothing ever asked) is a
  // no-op on the Rust side, so opening and closing without using it never
  // clutters history.
  //
  // The overlay window itself is never destroyed between interviews — Rust's
  // show_overlay_window reuses the existing webview (see interview_mode/
  // window.rs) purely for speed, so this component never remounts and its
  // React state would otherwise survive from one interview into the next.
  // Clearing turns/question/committedRef/sessionStartedAtRef here, right
  // after archiving, is what makes the *next* "Start Interview" actually
  // start blank instead of resuming the just-archived conversation.
  // Set false the instant the user closes the overlay — endMeeting's async
  // continuation checks this before touching any state, so a slow/hanging
  // backend summarize call still in flight when the user closes mid-request
  // can't call setSummary/setError into a window that's no longer the
  // user's current session. Ported from MeetingOverlay.tsx verbatim.
  const sessionActiveRef = useRef(true);

  const closeOverlay = useCallback(async () => {
    sessionActiveRef.current = false;
    setConfirmingClose(false);

    // Interview mode archives the Q&A turns as one history entry on close —
    // unchanged from before the merge. Meeting mode never archived on plain
    // close either (only the currently-unreachable endMeeting/"End Meeting"
    // flow does, via archive_meeting, which needs a summary this path
    // doesn't produce) — ported from MeetingOverlay.tsx's closeOverlay
    // verbatim, not a new restriction introduced by this merge.
    if (modeRef.current === "INTERVIEW") {
      const completed = turns
        .filter((t) => !t.pending && !t.failed && t.answer.trim())
        .map((t) => ({ question: t.question, answer: t.answer }));
      try {
        await invoke("archive_interview_session", {
          startedAtMs: sessionStartedAtRef.current,
          role: settings.role.trim() || null,
          company: settings.company.trim() || null,
          turns: completed,
        });
      } catch (e) {
        console.error("Failed to archive interview session:", e);
      }
    }

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
    setTranscript([]);
    setSummary(null);
    rawTurnsRef.current = [];

    // The overlay window is reused (not destroyed) between interviews, so a
    // mic assistant session left running would otherwise keep transcribing
    // into the next interview's freshly-cleared question box.
    if (micAssistantActiveRef.current) {
      micAssistantActiveRef.current = false;
      setMicAssistantActive(false);
      invoke("stop_mic_assistant").catch(() => {});
    }

    invoke("hide_interview_overlay").catch(() => overlayWindow.hide());
  }, [turns, settings.role, settings.company]);

  // Entry point for both the ✕ button and Escape: conversations with at least
  // one exchange ask for confirmation first (closing mid-interview is easy to
  // trigger by accident), while an empty session just closes immediately —
  // there's nothing to lose by skipping the prompt. Meeting mode also treats
  // any live transcript as "a conversation" (its questions may be typed only
  // rarely — the transcript itself is the substance worth confirming about),
  // matching MeetingOverlay.tsx's requestClose.
  const requestClose = useCallback(() => {
    const hasConversation =
      turns.some((t) => !t.pending && t.answer.trim()) ||
      (modeRef.current === "MEETING" && (transcript.length > 0 || rawTurnsRef.current.length > 0));
    if (hasConversation && !summary) {
      setConfirmingClose(true);
    } else {
      closeOverlay();
    }
  }, [turns, transcript, summary, closeOverlay]);

  // Meeting mode's end-of-meeting summary flow — ported verbatim from the
  // retired MeetingOverlay.tsx, where it was already disconnected from the
  // UI ("temporarily removed... under investigation" per that file's
  // comment). No button here calls this; kept rather than deleted so
  // restoring it is a one-line change, matching pre-merge behavior exactly.
  const endMeeting = useCallback(async () => {
    sessionActiveRef.current = true;
    setError(null);
    try {
      await invoke("stop_audio_capture").catch(() => {});
      const info = meetingInfoRef.current;
      const result = await invoke<MeetingSummary>("end_meeting", {
        turns: rawTurnsRef.current,
        meetingTitle: info.meetingTitle || null,
        participants: info.participants || null,
      });
      if (!sessionActiveRef.current) return;
      setSummary(result);
      try {
        await invoke("archive_meeting", {
          startedAtMs: sessionStartedAtRef.current,
          meetingTitle: info.meetingTitle || null,
          participants: info.participants || null,
          turns: rawTurnsRef.current,
          summary: result,
        });
      } catch (e) {
        console.error("Failed to archive meeting:", e);
      }
    } catch (e) {
      if (sessionActiveRef.current) setError(String(e));
    }
  }, []);
  // Unread while unreachable (see the comment above) — this no-op reference
  // just satisfies noUnusedLocals in the meantime.
  void endMeeting;

  // Recognizes an "open <name>" spoken while mic assistant mode is on and
  // launches the matching allowlisted app — the one voice-triggered action
  // that reaches outside the app, and only ever runs a path the user
  // themselves typed into Settings ahead of time (see OverlaySettingsPanel's
  // Voice Commands section and voice_command::launch_app), never anything
  // derived from the recognized text itself. Checked against each finalized
  // segment before it's folded into the question buffer, so a successful
  // match doesn't also leave "open notepad" sitting in the question box.
  const tryLaunchAppCommand = useCallback(
    (finalText: string): boolean => {
      const match = /^\s*open (.+?)\s*$/i.exec(finalText);
      if (!match) return false;
      const requestedName = match[1].trim().toLowerCase();
      const app = settings.voiceLaunchApps.find(
        (a) => a.name.trim().toLowerCase() === requestedName,
      );
      if (!app) return false;
      invoke("launch_app", { path: app.path }).catch((e) => setError(String(e)));
      return true;
    },
    [settings.voiceLaunchApps],
  );
  const tryLaunchAppCommandRef = useRef(tryLaunchAppCommand);
  useEffect(() => {
    tryLaunchAppCommandRef.current = tryLaunchAppCommand;
  }, [tryLaunchAppCommand]);

  // Recognizes Veronica's spoken mode-switch commands while mic assistant is
  // on, checked before tryLaunchAppCommand at the same interception point (a
  // finalized mic segment, before it's folded into the question buffer /
  // meeting transcript) so none of these three phrases ever land as
  // dictated text. "exit mode" closes the overlay outright (going through
  // requestClose, so a mid-conversation close still asks for confirmation
  // exactly like the ✕ button/Esc) rather than falling back to a mode —
  // confirmed with the user this means "leave Veronica," not "return to
  // Interview mode."
  const tryModeSwitchCommand = useCallback(
    (finalText: string): boolean => {
      const trimmed = finalText.trim().toLowerCase();
      if (/^enter interview mode$/.test(trimmed)) {
        if (modeRef.current !== "INTERVIEW") setMode("INTERVIEW");
        return true;
      }
      if (/^enter meeting mode$/.test(trimmed)) {
        if (modeRef.current !== "MEETING") setMode("MEETING");
        return true;
      }
      if (/^exit mode$/.test(trimmed)) {
        requestClose();
        return true;
      }
      return false;
    },
    [setMode, requestClose],
  );
  const tryModeSwitchCommandRef = useRef(tryModeSwitchCommand);
  useEffect(() => {
    tryModeSwitchCommandRef.current = tryModeSwitchCommand;
  }, [tryModeSwitchCommand]);

  // Veronica's action-taking system: a REQUIRED wake phrase ("Veronica, " /
  // "Hey Veronica, "), not a verb heuristic — anything without it is always
  // plain dictation, exactly as before this feature existed, so an
  // interview/meeting answer that happens to start with "so what's
  // important here…" can never be misread as a command. Only the text
  // after the wake phrase is sent to the backend for LLM intent
  // classification (personal/prompts/intent.rs) — see
  // actions::run_veronica_action, which enforces the actual safety gate;
  // this recognizer only decides WHETHER to ask, never what's allowed.
  const tryVeronicaAction = useCallback(
    (finalText: string): boolean => {
      const match = /^\s*(?:hey\s+)?veronica[,:]?\s+(.+)$/i.exec(finalText);
      if (!match) return false;
      const utterance = match[1].trim();
      if (!utterance) return false;

      const turn: Turn = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        question: finalText.trim(),
        answer: "",
        pending: true,
      };
      setTurns((prev) => [...prev, turn]);
      setVeronicaThinking(true);

      invoke<string>("run_veronica_action", { utterance })
        .then((result) => {
          setTurns((prev) =>
            prev.map((t) => (t.id === turn.id ? { ...t, answer: result, pending: false } : t)),
          );
        })
        .catch((e) => {
          setTurns((prev) =>
            prev.map((t) => (t.id === turn.id ? { ...t, pending: false, failed: true } : t)),
          );
          setError(String(e));
        })
        .finally(() => {
          setVeronicaThinking(false);
        });

      return true;
    },
    [],
  );
  const tryVeronicaActionRef = useRef(tryVeronicaAction);
  useEffect(() => {
    tryVeronicaActionRef.current = tryVeronicaAction;
  }, [tryVeronicaAction]);

  const toggleMicAssistant = useCallback(() => {
    if (micAssistantActiveRef.current) {
      micAssistantActiveRef.current = false;
      setMicAssistantActive(false);
      setVeronicaThinking(false);
      invoke("stop_mic_assistant").catch((e) => setError(String(e)));
      return;
    }
    micAssistantActiveRef.current = true;
    setMicAssistantActive(true);
    setError(null);
    invoke("start_mic_assistant").catch((e) => {
      micAssistantActiveRef.current = false;
      setMicAssistantActive(false);
      setError(String(e));
    });
  }, []);
  const toggleMicAssistantRef = useRef(toggleMicAssistant);
  useEffect(() => {
    toggleMicAssistantRef.current = toggleMicAssistant;
  }, [toggleMicAssistant]);

  // Veronica's global shortcut (Ctrl+Shift+V, registered Rust-side in
  // lib.rs) toggles the same mic-assistant session the 🎤 button does —
  // forwarded as an event rather than calling start/stop_mic_assistant
  // directly from Rust so this component's React state (which the header
  // and button rendering depend on) stays the single source of truth.
  useEffect(() => {
    const unlisten = listen("veronica:toggle-shortcut", () => {
      toggleMicAssistantRef.current();
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
  //   - Thinking: a mic-assistant question has been sent and no answer text
  //     has arrived yet (veronicaThinking, cleared on the first delta).
  //   - Speaking: the answer is actively streaming in (busy, past Thinking).
  //   - Listening: mic assistant is on, not currently busy, and STT is
  //     hearing speech right now (sttSpeaking) — silence while the mic is
  //     merely open still reads as Idle, matching the header's existing
  //     "Listening" vs "Starting…" distinction below.
  //   - Idle: mic assistant is off, or on but silent and not busy.
  const veronicaState: "idle" | "listening" | "thinking" | "speaking" = !micAssistantActive
    ? "idle"
    : veronicaThinking
      ? "thinking"
      : busy
        ? "speaking"
        : sttSpeaking
          ? "listening"
          : "idle";

  // Meeting mode's end-of-meeting summary screen, reached only through
  // endMeeting (currently unreachable — see its comment above). Ported from
  // MeetingOverlay.tsx verbatim, including this early return in place of the
  // normal chat/compose view.
  if (summary) {
    return (
      <div
        className={`overlay-root density-${settings.density} size-${settings.size}`}
        style={overlayStyle}
      >
        <div
          className="overlay-header"
          data-tauri-drag-region={settings.dragEnabled ? "deep" : undefined}
        >
          <div className="overlay-title">Meeting Ended</div>
          <div className="overlay-header-actions">
            <button className="overlay-icon-button close" onClick={closeOverlay} title="Close">
              ✕
            </button>
          </div>
        </div>
        <div className="overlay-chat" ref={scrollRef}>
          <MeetingSummaryView summary={summary} />
        </div>
      </div>
    );
  }

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
            : micAssistantActive
              ? veronicaState === "thinking"
                ? "Veronica — Thinking…"
                : veronicaState === "speaking"
                  ? "Veronica — Speaking…"
                  : veronicaState === "listening"
                    ? "Veronica — Listening"
                    : "Veronica — Idle"
              : busy
                ? "Answering…"
                : mode === "MEETING"
                  ? "Veronica — Meeting"
                  : captureActive
                    ? "Veronica — Interview"
                    : "Starting…"}
        </div>

        <div className="overlay-header-actions">
          {/* A <button>, not a checkbox+<label>: Tauri's drag script already
              treats BUTTON as clickable and blocks dragging for it (see the
              comment on the opacity slider below), and a bare button avoids
              the <label> pitfall that comment warns about — no risk of the
              click handing the drag gesture back to the window. */}
          {/* Meeting mode's question box is manually typed, not dictated, so
              Auto AI (which auto-sends on the interviewer's dictated silence)
              has no equivalent there — disabled for visual parity only,
              matching MeetingOverlay.tsx's toggle before the merge. */}
          <button
            type="button"
            className={`overlay-auto-ai-toggle${settings.autoAI && mode === "INTERVIEW" ? " on" : ""}`}
            onClick={() => setSettings((prev) => ({ ...prev, autoAI: !prev.autoAI }))}
            disabled={mode === "MEETING"}
            title={
              mode === "MEETING"
                ? "Auto AI is not available in Meeting Mode — questions here are typed, not dictated"
                : settings.autoAI
                  ? "Auto AI is on — questions send automatically when the interviewer stops talking"
                  : "Auto AI is off — use Ask AI or Enter to send"
            }
          >
            <span className="overlay-auto-ai-dot" />
            Auto AI
          </button>

          {/* Mic assistant toggle: click to start transcribing the user's
              own microphone into the question box (auto-asking once they
              stop talking, regardless of the separate Auto AI toggle above)
              and click again to stop — see start_mic_assistant/
              stop_mic_assistant and the transcript:update listener. Saying
              "open <name>" for an app configured in Settings launches it
              instead of being added to the question. */}
          <button
            type="button"
            className={`overlay-icon-button overlay-voice-command-button${micAssistantActive ? " active" : ""} veronica-${veronicaState}`}
            onClick={toggleMicAssistant}
            title={
              micAssistantActive
                ? `Veronica is on (${veronicaState}) — click, press Esc, or Ctrl+Shift+V to stop`
                : "Click or press Ctrl+Shift+V to talk to Veronica"
            }
          >
            🎤
            {micAssistantActive && <span className="veronica-state-dot" />}
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
          tucked away in Settings (see OverlaySettingsPanel's "Screen Capture
          Protection" row) — it has to stay on screen for the entire
          session, not just flash once, since the interviewer's screen share
          might start at any point after this. Rendered outside the
          settings/confirm branches below so it persists across all of the
          overlay's views. */}
      {captureExcluded === false && (
        <p className="overlay-capture-warning">
          ⚠ Screen capture protection unavailable — this window may be visible if you share your screen
        </p>
      )}

      {confirmingClose ? (
        <div className="overlay-confirm-close">
          <p className="overlay-confirm-close-title">
            {mode === "MEETING" ? "Is the meeting over?" : "Is the interview over?"}
          </p>
          <p className="overlay-confirm-close-body">
            {mode === "MEETING"
              ? "Closing without ending the meeting will lose the summary."
              : "Closing will save this conversation to your interview history."}
          </p>
          <div className="overlay-confirm-close-actions">
            <button className="overlay-text-button" onClick={() => setConfirmingClose(false)}>
              Keep going
            </button>
            <button className="overlay-text-button primary" onClick={closeOverlay}>
              {mode === "MEETING" ? "Close anyway" : "Yes, end interview"}
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
            {mode === "MEETING" && !hasConversation && transcript.length === 0 && (
              <p className="overlay-empty">Waiting for the meeting to begin…</p>
            )}
            {mode === "INTERVIEW" && !hasConversation && !question && (
              <p className="overlay-empty">Waiting for the interviewer to speak…</p>
            )}

            {/* Meeting mode's live dual-speaker transcript — ported from
                MeetingOverlay.tsx verbatim. Interview mode has no equivalent:
                its dictation lands directly in the question box below
                instead of a separate running log. */}
            {mode === "MEETING" &&
              transcript.map((entry) => (
                <div className="chat-message interviewer" key={entry.id}>
                  <span className="chat-role">{entry.speaker}</span>
                  <p className="chat-text">{entry.text}</p>
                </div>
              ))}

            {turns.map((turn) => (
              <div key={turn.id} className="chat-turn">
                <div className="chat-message interviewer">
                  <span className="chat-role">{mode === "MEETING" ? "You asked" : "Interviewer"}</span>
                  <p className="chat-text">{turn.question}</p>
                </div>
                <div className={`chat-message assistant${turn.failed ? " failed" : ""}`}>
                  <span className="chat-role">Answer</span>
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
              placeholder={
                mode === "MEETING"
                  ? "Ask a question…"
                  : hasConversation
                    ? "Next question…"
                    : "Waiting for the interviewer to speak…"
              }
              rows={1}
            />
            <button
              className="overlay-ask-button"
              onClick={askAI}
              disabled={!question.trim() || busy}
              title={mode === "MEETING" ? "Ask (Enter)" : "Ask AI (Enter)"}
            >
              {mode === "MEETING" ? "Ask" : "Ask AI"}
            </button>
          </div>

          {error && <p className="overlay-error">{error}</p>}
        </>
      )}
    </div>
  );
}

export type { AnswerLength, ResponseStyle };
