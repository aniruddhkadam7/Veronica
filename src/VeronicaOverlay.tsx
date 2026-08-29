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
import { classifyQuestionCompleteness, joinSpeech, SAFETY_NET_MS } from "./questionCompleteness";
import { AudioLevelBars, useSttSpeaking } from "./ui";
import { ParticlesOrb } from "@/registry/orbe/particles-orb/particles-orb";
import { loadLlmProvider } from "./llmProviderSetting";
import { useVeronicaOrbState } from "./useVeronicaOrbState";

interface OverlayCaptureStatus {
  excluded: boolean;
}

/// Every status a turn can ever be in, end to end. `thinking`/`streaming`
/// are the only NON-terminal ones — once a turn reaches `complete`,
/// `interrupted`, or `error` it is done forever: `applyTurnEvent` below
/// refuses to modify a turn already in one of those three states, no matter
/// what event arrives for it afterwards (rule: "a completed turn cannot be
/// modified by a later turn").
type TurnStatus = "thinking" | "streaming" | "complete" | "interrupted" | "error";

const TERMINAL_STATUSES: ReadonlySet<TurnStatus> = new Set(["complete", "interrupted", "error"]);

/// One exchange in the running conversation. `id` IS the turn_id sent to
/// `ask_veronica` and echoed back on every `veronica:*` event for this turn
/// (see veronica.rs's `TurnIdPayload`/`AnswerDeltaPayload`/etc.) — the
/// single correlation key used everywhere, rather than a separate
/// client-only id plus positional ("last in the array") lookups. That
/// positional pattern was the actual root cause of turns getting a
/// permanently stuck "Thinking…" placeholder while a LATER turn received
/// the answer meant for an EARLIER one: this file used to update
/// `prev[prev.length - 1]` on every `answer-delta`/`answer-complete`,
/// which is only correct if it's structurally impossible for more than one
/// turn to be non-terminal at once — an assumption a rapid follow-up, a VAD
/// utterance split, or any other race could violate.
///
/// "Thinking…" is rendered from `status === "thinking"` (see the JSX
/// below) — it is never stored as literal text in `answer`, so it can never
/// end up as a permanent, un-replaceable message the way literal text
/// would.
interface Turn {
  id: string;
  question: string;
  answer: string;
  status: TurnStatus;
  createdAt: number;
  completedAt?: number;
}

/// Generates a fresh turn id — `crypto.randomUUID()` is available in the
/// WebView2 runtime this app ships on; falls back to a timestamp+random
/// string for any environment where it somehow isn't (matches the shape the
/// old client-only turn id used).
function newTurnId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

/// Wire shape of `get_conversation_history`'s result — see
/// conversation.rs's `ConversationTurn`. Field names and `status` string
/// values are deliberately identical to this file's own `Turn`/`TurnStatus`
/// so hydrating from the shared backend store and building a turn live from
/// events produce the exact same shape.
interface ConversationTurnDto {
  id: string;
  question: string;
  answer: string;
  status: TurnStatus;
  createdAtMs: number;
  completedAtMs: number | null;
}

function fromDto(dto: ConversationTurnDto): Turn {
  return {
    id: dto.id,
    question: dto.question,
    answer: dto.answer,
    status: dto.status,
    createdAt: dto.createdAtMs,
    completedAt: dto.completedAtMs ?? undefined,
  };
}

/// Applies one backend event to `turns`, but ONLY to the turn whose id
/// matches `turnId` AND only while that turn is still non-terminal — this
/// is the one place that enforces "never modify a completed turn" and "a
/// stale/superseded turn's late event can never corrupt whichever turn is
/// actually current," regardless of event arrival order or timing.
function applyTurnEvent(turns: Turn[], turnId: string, updater: (turn: Turn) => Turn): Turn[] {
  let changed = false;
  const next = turns.map((t) => {
    if (t.id !== turnId || TERMINAL_STATUSES.has(t.status)) return t;
    changed = true;
    return updater(t);
  });
  return changed ? next : turns;
}

/// Merges a freshly-fetched backend snapshot into whatever local `turns`
/// this window already has, keyed by turn_id — needed because hydration is
/// an async round trip: a turn can be created/updated by a LIVE event
/// arriving in between the fetch starting and its response landing. For any
/// id present in both: the backend snapshot wins for a terminal turn (it is
/// authoritative and can only be more complete than a local guess), while a
/// LOCAL turn that is still non-terminal (still thinking/streaming) wins
/// over a snapshot taken before that turn finished, so an in-progress
/// stream is never rolled back to a stale, less-complete snapshot value.
/// Turns present only locally (started after the snapshot was taken) are
/// kept; turns present only in the snapshot (from the widget, or an earlier
/// session before this window existed) are added. Result stays sorted
/// oldest-first by `createdAt`, matching both sources' own ordering.
function mergeHydratedTurns(local: Turn[], snapshot: Turn[]): Turn[] {
  const byId = new Map<string, Turn>();
  for (const turn of snapshot) byId.set(turn.id, turn);
  for (const turn of local) {
    const fromSnapshot = byId.get(turn.id);
    if (!fromSnapshot || !TERMINAL_STATUSES.has(fromSnapshot.status)) {
      byId.set(turn.id, turn);
    }
  }
  return Array.from(byId.values()).sort((a, b) => a.createdAt - b.createdAt);
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

/// How long the "just summoned via hotkey" wake-up glow stays elevated
/// (brighter/faster orb pulse, see `justWokeUp`) after `veronica:auto-opened`
/// fires, before falling back to whatever `veronicaState` would show anyway.
/// Long enough to read as a deliberate greeting beat, short enough not to
/// linger once the user has clearly moved on to typing/talking.
const WAKE_UP_GLOW_MS = 2500;

/// Window within which an identical submitted question is treated as a
/// duplicate rather than a genuine repeat — see `askAI`'s dedup guard
/// (requirement 9). Long enough to absorb a re-delivered Final segment
/// around a mute/barge-in boundary, short enough that actually repeating
/// yourself a few seconds later ("did you get that?") still goes through.
const DEDUPE_WINDOW_MS = 4000;

// The completeness classifier, joinSpeech(), and the bounded safety-net
// constant live in questionCompleteness.ts, shared with Custom Agents'
// overlay so both Auto AI implementations behave identically.

export function VeronicaOverlay() {
  // The whole conversation, oldest first — a local mirror of the ONE shared
  // backend conversation store (`AppState.conversation`, see conversation.rs),
  // not this window's own source of truth. On mount/show this is hydrated
  // from `get_conversation_history` (see the mount effect and the
  // `overlay:reset-session` handler below) rather than starting empty, so
  // anything said/answered through the floating widget while this window
  // was closed is already here the instant it opens. Live turns still
  // arrive the same way as before (`veronica:answer-delta`/`-complete`
  // events, applied via `applyTurnEvent`) — hydration and live events both
  // write into this same array, keyed by the same turn_id, so a turn
  // started just before hydration finishes can't be duplicated (see
  // `mergeHydratedTurns` below).
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
  // Mirrors App.tsx's header mute toggle (the same `set_mic_muted`/
  // `get_mic_muted` backend flag, see voice_command::mod) so muting is
  // available from the overlay too, not just the compact toolbar. Initialized
  // from the backend on mount rather than assumed `false`, since the overlay
  // can reopen on top of a session that was muted from the other window.
  const [micMuted, setMicMuted] = useState(false);
  // True only while STT is actively producing transcript output right now —
  // gates the listening animation so it doesn't run through silence just
  // because capture is technically still open.
  const sttSpeaking = useSttSpeaking();
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [confirmingClose, setConfirmingClose] = useState(false);
  // Shows the opacity percentage in the header for a moment after adjusting.
  const [opacityHint, setOpacityHint] = useState(false);
  const [settings, setSettings] = useState<OverlaySettings>(() => loadOverlaySettings());
  // True for a brief window right after Veronica is summoned via the global
  // hotkey/tray from a fully-closed state — drives the orb's elevated
  // "waking up" glow independent of veronicaState (which would otherwise
  // just say "idle" the instant the greeting line finishes speaking).
  const [justWokeUp, setJustWokeUp] = useState(false);
  const wakeUpTimerRef = useRef<number | null>(null);

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
  // Mirrors `busy` for the event listeners, which close over stale state
  // otherwise. Answer deltas must keep landing regardless of re-renders.
  const busyRef = useRef(false);
  const hintTimerRef = useRef<number | null>(null);
  // Safety-net timer: only armed when the buffered text still looks
  // incomplete after a Final (see the transcript listener below) — a
  // complete-looking Final sends immediately, with no timer at all. Lives
  // in a ref (not state) since it's set/cleared from the transcript
  // listener, which must not re-subscribe on every keystroke-equivalent
  // state change.
  const autoAiTimerRef = useRef<number | null>(null);
  // askAI changes identity on every render; mirrored into a ref so the
  // transcript listener (subscribed once, on mount) always calls the latest
  // version instead of one captured at effect-setup time.
  const askAIRef = useRef<() => void>(() => {});
  // The exact text of the most recent turn actually SUBMITTED to
  // ask_veronica, plus when — guards against the same finalized transcript
  // triggering a second request (requirement 9). The local VAD/STT sidecar
  // can occasionally re-emit the same Final segment (e.g. a flush around a
  // mute/barge-in boundary), and this must never turn into two identical
  // conversation turns. A short time window (not "forever") so a genuine
  // repeat later in the conversation ("play it again", asked twice minutes
  // apart) is never silently swallowed.
  const lastSubmittedRef = useRef<{ text: string; at: number } | null>(null);
  // The turn_id of whichever turn was started MOST RECENTLY — used only to
  // decide whether an incoming event should update `busy`/`error` UI state
  // (a stale, superseded turn's own completion must not stomp on whatever
  // the newer active turn is doing). Turn CONTENT updates themselves never
  // consult this — they're correlated purely by matching the event's own
  // turnId against a turn in `turns`, via `applyTurnEvent`.
  const activeTurnIdRef = useRef<string | null>(null);

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
  // This behaves like continuous dictation. There is deliberately no
  // sentence-end detection beyond `classifyQuestionCompleteness`, and
  // nothing that clears the field on silence: a `Final` transcript segment
  // is the real turn-completion signal (the local VAD already decided the
  // utterance ended — see stt/sidecar.rs), so a complete-looking Final is
  // sent the instant it arrives, with no added wait. Only a Final that
  // still looks incomplete (trails off on "and", a dangling comma) arms a
  // short bounded safety-net timer instead of sending immediately — sending
  // a bare fragment straight to the fast router risks it matching a real,
  // possibly irreversible, command out of context; the safety net only
  // fires if the speaker trails off and nothing more arrives to merge with.
  useEffect(() => {
    const clearAutoAiTimer = () => {
      if (autoAiTimerRef.current) {
        window.clearTimeout(autoAiTimerRef.current);
        autoAiTimerRef.current = null;
      }
    };

    const armAutoAiTimer = () => {
      clearAutoAiTimer();
      // No `busyRef` gate here anymore: a new complete-looking utterance
      // starts its own turn immediately, even while a previous one is still
      // THINKING/streaming/speaking — exactly like a real conversation,
      // where you can start talking again before the other person finishes.
      // The backend (`AppState::begin_turn`) cancels whatever the previous
      // turn was still doing the moment this new one starts, and every
      // event this new turn's `ask_veronica` call emits carries ITS OWN
      // turn_id — so the two can never cross streams no matter how they
      // overlap in time (see `applyTurnEvent`).
      if (classifyQuestionCompleteness(committedRef.current) === "complete") {
        askAIRef.current();
        return;
      }
      autoAiTimerRef.current = window.setTimeout(() => {
        autoAiTimerRef.current = null;
        askAIRef.current();
      }, SAFETY_NET_MS);
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
        console.debug("[TRANSCRIPT_FINAL]", segment.final_text);
        const candidate = joinSpeech(committedRef.current, segment.final_text);
        // Interruption check FIRST, before this ever becomes a buffered
        // question or a Turn (requirement 6): "stop"/"wait"/"hold on"/
        // "cancel" must never appear as a "YOU: stop" message and must
        // never produce a visible assistant reply. `try_interrupt` is a
        // single atomic backend call — it both decides AND (if true) stops
        // TTS/cancels the in-flight turn, so there's no separate detect-
        // then-act race. Only the newly-added fragment is checked (not the
        // whole buffer) so an interruption said cleanly on its own always
        // matches, even if something was left over in the composer.
        invoke<boolean>("try_interrupt", { text: segment.final_text }).then((wasInterrupt) => {
          if (wasInterrupt) {
            // Clear the composer too — an interruption is a control signal,
            // never conversation content left sitting in the input.
            committedRef.current = "";
            setQuestion("");
            clearAutoAiTimer();
            return;
          }
          committedRef.current = candidate;
          setQuestion(committedRef.current);
          armAutoAiTimer();
        });
      } else if (segment.partial_text) {
        console.debug("[TRANSCRIPT_INTERIM]", segment.partial_text);
        setQuestion(joinSpeech(committedRef.current, segment.partial_text));
        // Still speaking — any timer counting down from an earlier final must
        // not fire mid-sentence.
        clearAutoAiTimer();
      }
    });

    // Correlated by turn_id (see the `Turn`/`applyTurnEvent` doc above), NOT
    // by array position — this is the actual fix for the live-observed bug
    // where one turn's real answer landed on a DIFFERENT (later) turn's
    // message bubble while the first stayed stuck on "Thinking…" forever.
    // `applyTurnEvent` also refuses to touch an already-terminal turn, so a
    // late/stale delta for a turn that already completed (or was
    // interrupted) is a safe no-op instead of corrupting it.
    const unlistenDelta = listen<{ turnId: string; delta: string }>("veronica:answer-delta", (event) => {
      const { turnId, delta } = event.payload;
      console.debug("[ASSISTANT_DELTA]", turnId, delta);
      setTurns((prev) => applyTurnEvent(prev, turnId, (t) => ({ ...t, status: "streaming", answer: t.answer + delta })));
      if (turnId === activeTurnIdRef.current) setBusy(true);
    });

    const unlistenComplete = listen<{ turnId: string; answer: string; cancelled: boolean }>("veronica:answer-complete", (event) => {
      const { turnId, answer, cancelled } = event.payload;
      setTurns((prev) =>
        applyTurnEvent(prev, turnId, (t) => ({
          ...t,
          // Prefer the completed answer from the event where it's non-empty
          // — it's authoritative, where accumulated deltas could have
          // dropped one; an empty+cancelled completion means this turn was
          // superseded before producing anything worth keeping.
          answer: answer || t.answer,
          status: cancelled ? "interrupted" : answer || t.answer ? "complete" : "error",
          completedAt: Date.now(),
        })),
      );
      // Only the ACTIVE turn's completion should clear the busy/composing
      // state — a stale turn finishing (interrupted by a newer one) must
      // not stomp on whatever the newer turn is currently doing.
      if (turnId === activeTurnIdRef.current) {
        busyRef.current = false;
        setBusy(false);
      }
    });

    // Fired by `try_interrupt` (see veronica.rs) the instant a bare "stop"/
    // "wait"/"hold on"/"cancel" utterance is recognized — distinct from
    // `answer-complete`'s `cancelled: true` (which fires for the turn that
    // WAS speaking and gets no visible "(interrupted)" text either, per
    // requirement 6 — see the `applyTurnEvent`'s "interrupted" branch
    // rendering nothing below). This listener exists so busy/composing
    // state clears immediately even if no turn was actually active (e.g.
    // the user said "stop" while Veronica was merely listening, which is a
    // harmless no-op interruption rather than an error).
    const unlistenInterrupted = listen("veronica:interrupted", () => {
      busyRef.current = false;
      setBusy(false);
    });

    return () => {
      unlistenTranscript.then((f) => f());
      unlistenDelta.then((f) => f());
      unlistenComplete.then((f) => f());
      unlistenInterrupted.then((f) => f());
      clearAutoAiTimer();
    };
  }, []);

  // Fired by the Rust side (veronica_window::wake_veronica) only when the
  // global hotkey/tray brought Veronica back from a fully-closed-to-tray
  // state — i.e. what the user experiences as "the app was closed and I hit
  // the shortcut". A plain toggle-open while already in use never fires
  // this, so the greeting doesn't repeat on every open/close.
  useEffect(() => {
    const unlisten = listen("veronica:auto-opened", () => {
      setJustWokeUp(true);
      invoke<string>("speak_greeting").catch(() => {
        // Best-effort — TTS may be unavailable (no API key/network); the
        // orb's own wake-up glow still carries the moment silently.
      });
      if (wakeUpTimerRef.current) window.clearTimeout(wakeUpTimerRef.current);
      wakeUpTimerRef.current = window.setTimeout(() => {
        wakeUpTimerRef.current = null;
        setJustWokeUp(false);
      }, WAKE_UP_GLOW_MS);
    });
    return () => {
      unlisten.then((f) => f());
      if (wakeUpTimerRef.current) window.clearTimeout(wakeUpTimerRef.current);
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

    invoke<boolean>("get_mic_muted")
      .then(setMicMuted)
      .catch(() => {});

    // Hydrates from the ONE shared conversation store (see conversation.rs)
    // instead of starting empty — this is what makes "Open shows everything
    // said through the widget so far" true. Merged (not replaced) via
    // `mergeHydratedTurns` in case a live turn was already created locally
    // between this window mounting and this fetch resolving.
    invoke<ConversationTurnDto[]>("get_conversation_history")
      .then((dtos) => setTurns((prev) => mergeHydratedTurns(prev, dtos.map(fromDto))))
      .catch(() => {
        // Best-effort — a fresh/empty conversation on failure is the
        // pre-existing behavior, not a regression.
      });
  }, []);

  // The overlay window is *reused* across sessions, not recreated (see
  // overlay_window.rs's show_overlay_window) — its WebView2 process/DOM
  // stays alive while hidden between "Close" and the next "Start". This
  // resets only this window's own transient UI state (an open Settings
  // panel, a pending close-confirmation, a stale composer draft) — NOT the
  // conversation itself, which is re-hydrated from the shared backend store
  // exactly like the mount effect above, so reopening the overlay continues
  // the SAME live conversation rather than starting a new one (requirement:
  // "do not create a new/reset conversation when opening the overlay"). The
  // Rust side emits this right before re-showing an existing window.
  useEffect(() => {
    const unlistenReset = listen("overlay:reset-session", () => {
      setQuestion("");
      setError(null);
      setConfirmingClose(false);
      setSettingsOpen(false);
      committedRef.current = "";
      lastSubmittedRef.current = null;
      invoke<boolean>("get_mic_muted")
        .then(setMicMuted)
        .catch(() => {});
      invoke<ConversationTurnDto[]>("get_conversation_history")
        .then((dtos) => setTurns((prev) => mergeHydratedTurns(prev, dtos.map(fromDto))))
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
    if (!trimmed) return;

    // Deduplication (requirement 9): the exact same finalized transcript
    // must never trigger two ask_veronica calls. Guards the case a Final
    // segment is effectively re-delivered (e.g. a sidecar flush around a
    // mute/barge-in boundary re-sends what was already sent) rather than
    // representing a genuine new utterance. Scoped to a short window, not
    // forever — asking the identical question again later in the
    // conversation is a normal, real thing to do.
    const now = Date.now();
    if (lastSubmittedRef.current && lastSubmittedRef.current.text === trimmed && now - lastSubmittedRef.current.at < DEDUPE_WINDOW_MS) {
      setQuestion("");
      committedRef.current = "";
      return;
    }
    lastSubmittedRef.current = { text: trimmed, at: now };

    setError(null);
    // This turn's id is generated HERE, once, and used as both the React
    // key and the id ask_veronica's events are keyed by — a single
    // identity end to end, not a client-only id plus positional lookups.
    // No `busyRef` check: a new turn is allowed to start even while a
    // previous one is still active (THINKING/streaming/speaking) — the
    // backend cancels the previous turn's generation the instant this one
    // starts (`AppState::begin_turn`), and `applyTurnEvent`'s turn_id
    // correlation means the two can never cross wires no matter how they
    // overlap. This IS what makes "start talking again before Veronica
    // finishes" (barge-in / a fast follow-up) work naturally instead of
    // being silently dropped.
    const turnId = newTurnId();
    console.debug("[TURN_SUBMITTED]", turnId, trimmed);
    activeTurnIdRef.current = turnId;
    const turn: Turn = { id: turnId, question: trimmed, answer: "", status: "thinking", createdAt: Date.now() };
    setTurns((prev) => [...prev, turn]);
    setQuestion("");
    committedRef.current = "";
    busyRef.current = true;
    setBusy(true);

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
      // through the same call, and the backend decides. Turn finalization
      // (success, interrupted, or error) always happens via the
      // `veronica:answer-complete` event listener above, which the backend
      // guarantees fires on every path — this call's own rejection below is
      // only used to surface a top-level error banner, not to mutate turn
      // state a second time.
      //
      // No `history` param: ask_veronica derives conversational context
      // from the shared backend conversation store itself (see
      // conversation.rs's `completed_history`), not from whatever this one
      // window happens to have in its own local `turns` — a follow-up asked
      // here now correctly sees turns that happened through the widget too.
      await invoke<string>("ask_veronica", {
        question: trimmed,
        turnId,
        options: {
          answerLength: settings.answerLength,
          responseStyle: settings.responseStyle,
          humanization: settings.humanization,
          llmProvider,
          ttsEnabled: settings.voiceOutputEnabled,
        },
      });
    } catch (e) {
      // "cancelled" means a newer turn superseded this one before it
      // finished — expected/normal (see veronica.rs), not a real failure,
      // so no error banner for it.
      if (String(e) !== "cancelled") {
        setError(String(e));
      }
      if (turnId === activeTurnIdRef.current) {
        busyRef.current = false;
        setBusy(false);
      }
    }
  }, [question, turns, settings]);

  useEffect(() => {
    askAIRef.current = askAI;
  }, [askAI]);

  // Hides the overlay window ONLY — the conversation and the mic-assistant
  // session both keep running exactly as they were (App.tsx's Stop button
  // is the only thing that ends the session; see its `handleStop` doc).
  // Requirement: "closing the overlay must not stop or reset the
  // conversation/session." `turns` is deliberately left as-is (not cleared)
  // — it's a live mirror of the shared backend conversation, which the
  // widget keeps growing while this window is hidden; the next
  // `overlay:reset-session` (fired on reshow) re-hydrates it anyway, so
  // there is nothing to proactively clear here even for tidiness.
  const closeOverlay = useCallback(async () => {
    setConfirmingClose(false);

    if (autoAiTimerRef.current) {
      window.clearTimeout(autoAiTimerRef.current);
      autoAiTimerRef.current = null;
    }
    setError(null);

    // System audio (the other speaker/app sound) is this window's own
    // opt-in toggle, off by default and not part of the shared
    // conversation session — stopping it on close (and re-enabling it
    // fresh next time, rather than having it silently keep running
    // detached from any visible toggle) matches its existing "off by
    // default" behavior. Mic listening itself is untouched — it belongs to
    // the session, not this window.
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
  // immediately — there's nothing to lose by skipping the prompt. Since
  // closing no longer ends the session (see closeOverlay's doc), the
  // confirmation copy itself now says "hide"/"keep listening" rather than
  // implying the conversation would be lost — see the confirm-close JSX.
  const requestClose = useCallback(() => {
    const hasConversation = turns.some((t) => t.status === "complete" && t.answer.trim());
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
  // Same optimistic toggle as App.tsx's header mute button — flips local
  // state immediately and fires the shared `set_mic_muted` command; the mic
  // itself and the session stay running either way (see voice_command::mod's
  // pump loop, which just withholds audio from STT while muted).
  const toggleMicMuted = useCallback(() => {
    setMicMuted((cur) => {
      const next = !cur;
      invoke("set_mic_muted", { muted: next }).catch((e) => setError(String(e)));
      return next;
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

  // Real pipeline-driven orb state (thinking/executing/speaking/error/
  // listening), shared with VeronicaWidget.tsx via useVeronicaOrbState so
  // both surfaces always agree — see that hook's doc for exactly which
  // Rust-emitted event drives each state. `veronicaThinking`/`busy` above
  // remain purely for this window's own turn/pending bookkeeping (deciding
  // when a new question can be sent, which turn to append streamed text
  // to) — orthogonal to what the orb visually shows.
  const { orbState: pipelineOrbState, levelRef, lastError: pipelineError } = useVeronicaOrbState();
  // While the post-hotkey wake-up glow is active and nothing else is
  // already happening, show the orb as "speaking" (its brightest, fastest
  // state) — it's the moment Veronica's greeting line is actually playing,
  // so the visual should read as alive, not idle. The greeting's TTS never
  // fires "tts:speaking-changed" (that event is emitted by the mic-assistant
  // pump's mute-gate check, not by speak_greeting directly), so without this
  // override the orb would sit on idle throughout the whole greeting.
  const orbState = justWokeUp && pipelineOrbState === "idle" ? "speaking" : pipelineOrbState;
  // Kept as its own label-only condition (not derived from orbState, which
  // is now "connecting" for an in-progress action, not "thinking") so the
  // header text below still reads sensibly for both cases. useVeronicaOrbState
  // never actually returns "disabled" (not in its own state machine), but
  // TypeScript widens to OrbState's full union — handled defensively anyway.
  const veronicaState: "idle" | "listening" | "thinking" | "speaking" =
    orbState === "connecting" ? "thinking" : orbState === "thinking" || orbState === "speaking" || orbState === "listening" ? orbState : "idle";

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
        <ParticlesOrb state={orbState} size={20} speed={1} colorFrom="#f0abfc" colorTo="#818cf8" levelRef={levelRef} />
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
            : pipelineOrbState === "error"
              ? "Veronica — Error"
              : pipelineOrbState === "connecting"
                ? "Veronica — Working…"
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
          {/* Mic mute toggle — mirrors App.tsx's compact-toolbar mute button
              (same backend flag). Mutes/unmutes the user's own mic without
              ending the session, unlike Close. */}
          <button
            type="button"
            className={`overlay-icon-button overlay-mic-mute-button${micMuted ? " active" : ""}`}
            onClick={toggleMicMuted}
            title={micMuted ? "Unmute Veronica's mic" : "Mute Veronica's mic"}
            aria-pressed={micMuted}
          >
            {micMuted ? "🔇" : "🎙"}
          </button>

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
          <p className="overlay-confirm-close-title">Hide this window?</p>
          <p className="overlay-confirm-close-body">
            Veronica keeps listening and the conversation continues — you can reopen it anytime from "Open".
          </p>
          <div className="overlay-confirm-close-actions">
            <button className="overlay-text-button" onClick={() => setConfirmingClose(false)}>
              Keep going
            </button>
            <button className="overlay-text-button primary" onClick={closeOverlay}>
              Hide
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

            {turns
              // A turn superseded by a newer one (barge-in, or a fast
              // follow-up) before producing any real text is dropped from
              // the visible conversation entirely — requirement 6 forbids a
              // fake "(interrupted)" assistant reply, and a user-turn with
              // no matching answer at all reads as a broken exchange, not a
              // real one. `applyTurnEvent` still updates this turn's status
              // in state (so its final event is handled exactly once and
              // busy/orb bookkeeping stays correct) — it simply never
              // reaches the screen.
              .filter((turn) => !(turn.status === "interrupted" && !turn.answer.trim()))
              .map((turn) => (
                <div key={turn.id} className="chat-turn">
                  <div className="chat-message interviewer">
                    <span className="chat-role">You</span>
                    <p className="chat-text">{turn.question}</p>
                  </div>
                  <div className={`chat-message assistant${turn.status === "error" ? " failed" : ""}`}>
                    <span className="chat-role">Veronica</span>
                    {/* "Thinking…"/streaming text is rendered purely from
                        `status`/`answer` here — it is never written into
                        `turn.answer` itself, so a turn that's still in
                        progress can never leave a literal "Thinking…" string
                        behind as if it were a real, permanent message. */}
                    {turn.answer ? (
                      <div className="chat-text chat-markdown">
                        <ReactMarkdown>{turn.answer}</ReactMarkdown>
                      </div>
                    ) : turn.status === "error" ? (
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
          {/* Real background-thread pipeline failures (STT sidecar crash,
              a Groq/Deepgram call failing for one utterance/sentence) that
              have no invoke().catch() to surface through — see
              useVeronicaOrbState's doc. Shown separately from `error` since
              the two can be true at once (e.g. an in-flight ask_veronica
              failing AND a background TTS sentence failing moments apart). */}
          {pipelineError && <p className="overlay-error">{pipelineError}</p>}
        </>
      )}
    </div>
  );
}

export type { AnswerLength, ResponseStyle };
