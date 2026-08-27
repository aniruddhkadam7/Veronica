import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { listen } from "@tauri-apps/api/event";
import "./App.css";
import smallbirdLogo from "./assets/smallbird-logo.png";
import {
  DocumentContext,
  emptyDocumentContextState,
  uploadDocumentContext,
  type DocumentContextState,
} from "./DocumentContext";
import { Account } from "./Account";
import { SettingsPopover } from "./SettingsPopover";
import { LowEndHardwareBanner } from "./LowEndHardwareBanner";
import { HeaderDropdown } from "./HeaderDropdown";
import { answerStyleToOverlayFields, loadOverlaySettings, saveOverlaySettings } from "./overlaySettings";
import { Button, Spinner } from "./ui";
import {
  IconAccount,
  IconAttachment,
  IconChevronDown,
  IconClose,
  IconMaximize,
  IconMinimize,
  IconRestore,
  IconSettings,
} from "./Icons";
import { IconAnthropic, IconDeepSeek, IconGemini, IconOpenAI } from "./ProviderIcons";
import { INTERVIEW_CONTEXT_SECTIONS, MEETING_CONTEXT_SECTIONS } from "./headerPopups";
import { hasStoredLlmProvider, loadLlmProvider, saveLlmProvider, type LlmProvider } from "./llmProviderSetting";
import { getPersonalApiKey } from "./personalApiKeys";

export type Mode = "INTERVIEW" | "MEETING";
type SessionState = "IDLE" | "STARTING" | "LISTENING";
type Popover = "MODE" | "MODEL" | "CONTEXT" | "SETTINGS" | "ACCOUNT" | null;

const MODE_LABELS: Record<Mode, string> = {
  INTERVIEW: "Interview",
  MEETING: "Meeting",
};

// Only "openai", "anthropic", and "gemini" have a real backend
// implementation (see apps/backend/app/services/llm/__init__.py::
// get_llm_provider) — "deepseek" is listed so the dropdown shows all four,
// but stays disabled until a provider is actually built for it.
const LLM_PROVIDERS: { value: LlmProvider; label: string; icon: typeof IconOpenAI; available: boolean }[] = [
  { value: "anthropic", label: "Anthropic", icon: IconAnthropic, available: true },
  { value: "openai", label: "OpenAI", icon: IconOpenAI, available: true },
  { value: "gemini", label: "Gemini", icon: IconGemini, available: true },
  { value: "deepseek", label: "DeepSeek", icon: IconDeepSeek, available: false },
];

function App() {
  const [mode, setMode] = useState<Mode>("INTERVIEW");
  const [llmProvider, setLlmProvider] = useState<LlmProvider>(() => loadLlmProvider());

  // First launch (no provider explicitly chosen yet): auto-select whichever
  // provider actually has an API key saved, instead of silently defaulting
  // to Anthropic and leaving the user to guess why requests fail. Checked in
  // LLM_PROVIDERS order so the first configured, available provider wins.
  useEffect(() => {
    if (hasStoredLlmProvider()) return;
    let cancelled = false;
    (async () => {
      for (const { value, available } of LLM_PROVIDERS) {
        if (!available || value === "deepseek") continue;
        const key = await getPersonalApiKey(value).catch(() => null);
        if (cancelled) return;
        if (key) {
          setLlmProvider(value);
          saveLlmProvider(value);
          return;
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  const [sessionState, setSessionState] = useState<SessionState>("IDLE");
  const [openPopover, setOpenPopover] = useState<Popover>(null);
  const [error, setError] = useState<string | null>(null);
  const [isMaximized, setIsMaximized] = useState(false);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  const [interviewContext, setInterviewContext] = useState<DocumentContextState>(() =>
    emptyDocumentContextState(INTERVIEW_CONTEXT_SECTIONS),
  );
  const [meetingContext, setMeetingContext] = useState<DocumentContextState>(() =>
    emptyDocumentContextState(MEETING_CONTEXT_SECTIONS),
  );
  const [meetingTitle, setMeetingTitle] = useState("");
  const [meetingParticipants, setMeetingParticipants] = useState("");

  // Prewarm STT the moment the app launches — the mic model loads while the
  // user is still picking a mode/attaching context, so Start doesn't eat the
  // full load time.
  const prewarmRef = useRef<Promise<void> | null>(null);
  useEffect(() => {
    const promise = invoke<void>("start_system_audio_capture").catch((e) => {
      if (String(e) !== "capture already running") throw e;
    });
    prewarmRef.current = promise;
    promise.catch(() => {});
  }, []);

  // History/status content only has room to render once the user has
  // natively maximized/resized the window larger than its compact toolbar
  // footprint — there is no in-app expand control anymore, so this tracks
  // the OS window state directly instead of app state.
  useEffect(() => {
    const appWindow = getCurrentWindow();
    appWindow.isMaximized().then(setIsMaximized).catch(() => {});
    const unlisten = appWindow.onResized(() => {
      appWindow.isMaximized().then(setIsMaximized).catch(() => {});
    });
    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  // The main window is `transparent: true` (tauri.conf.json) so the extra
  // height it grows into for an open header dropdown is genuinely
  // see-through rather than a visible grey panel — see App.css's
  // `.compact-transparent` rule. That transparency only makes sense in the
  // compact toolbar state (maximized view wants its normal opaque
  // background), so this marks/unmarks html+body the same way main.tsx
  // marks overlay windows, rather than via CSS :has().
  useEffect(() => {
    document.documentElement.classList.toggle("compact-transparent", !isMaximized);
    document.body.classList.toggle("compact-transparent", !isMaximized);
  }, [isMaximized]);

  // The overlay window (Interview/Meeting) and this main window are separate
  // webviews with no shared React state — ending a session from inside the
  // overlay (its own ✕ / "Yes, end interview" / Escape) has no way to reset
  // this window's Start/Stop button back to idle on its own. The Rust side
  // emits this event whenever an overlay closes, from whichever side
  // triggered it, so this listener is what keeps the two in sync instead of
  // requiring a redundant click on Stop here after already ending it there.
  useEffect(() => {
    const unlisten = listen("interview-mode:overlay-closed", () => {
      setSessionState("IDLE");
      setHistoryRefreshKey((k) => k + 1);
    });
    return () => {
      unlisten.then((f) => f());
    };
  }, []);

  // HeaderDropdown itself measures its real rendered height and resizes the
  // main window to match exactly on mount, then shrinks it back to 56px on
  // unmount (see HeaderDropdown.tsx) — so toggling here is just React
  // state; no resize call needed at this layer.
  const togglePopover = useCallback((p: Exclude<Popover, null>) => {
    setOpenPopover((cur) => (cur === p ? null : p));
  }, []);

  const closePopover = useCallback(() => setOpenPopover(null), []);

  const startInterview = useCallback(async () => {
    const overlaySettings = loadOverlaySettings();
    saveOverlaySettings({ ...overlaySettings, ...answerStyleToOverlayFields(overlaySettings.answerStyle) });

    uploadDocumentContext(INTERVIEW_CONTEXT_SECTIONS, interviewContext);

    // Opt-in, signed-in-only: resolves to { rejection: null } when the user
    // isn't signed in, or when something unrelated to entitlement went wrong
    // (credential-store hiccup, network blip reaching the backend) — local
    // recording has no dependency on this feature and must never be blocked
    // by it. Only a genuine backend entitlement rejection (no remaining
    // minutes, concurrent session limit, etc.) sets `rejection`, which we
    // propagate so Start actually stops — see start_backend_session's
    // Rust-side doc (BackendSessionResult) for the full contract.
    const sessionResult = await invoke<{ rejection: string | null }>("start_backend_session", {
      sttMode: "local",
    });
    if (sessionResult.rejection) {
      throw new Error(sessionResult.rejection);
    }

    await (prewarmRef.current ??
      invoke("start_system_audio_capture").catch((e) => {
        if (String(e) !== "capture already running") throw e;
      }));
    await invoke("clear_transcript").catch(() => {});
    // Veronica has one overlay window for both modes now — set_mode tells
    // it (and ask_veronica) which behavior to use before it's shown; see
    // veronica::set_mode and InterviewOverlay.tsx's `mode` state.
    await invoke("set_mode", { mode: "INTERVIEW" }).catch(() => {});
    await invoke("show_interview_overlay");
  }, [interviewContext]);

  const startMeeting = useCallback(async () => {
    window.localStorage.setItem(
      "meeting-mode:active-meeting",
      JSON.stringify({ meetingTitle: meetingTitle.trim(), participants: meetingParticipants.trim() }),
    );

    uploadDocumentContext(MEETING_CONTEXT_SECTIONS, meetingContext);

    await invoke("clear_meeting_session").catch(() => {});
    await (prewarmRef.current ??
      invoke("start_system_audio_capture").catch((e) => {
        if (String(e) !== "capture already running") throw e;
      }));
    await invoke("set_mode", { mode: "MEETING" }).catch(() => {});
    await invoke("show_interview_overlay");
  }, [meetingContext, meetingTitle, meetingParticipants]);

  const handleStart = useCallback(async () => {
    setError(null);
    setSessionState("STARTING");
    closePopover();
    try {
      if (mode === "INTERVIEW") {
        await startInterview();
      } else {
        await startMeeting();
      }
      setSessionState("LISTENING");
    } catch (e) {
      setError(String(e));
      setSessionState("IDLE");
    }
  }, [mode, startInterview, startMeeting, closePopover]);

  const handleStop = useCallback(async () => {
    try {
      await invoke("hide_interview_overlay");
    } catch (e) {
      setError(String(e));
    } finally {
      setSessionState("IDLE");
      setHistoryRefreshKey((k) => k + 1);
      if (mode === "INTERVIEW") {
        // Fire-and-forget: finalizes the backend session (if one was
        // started) so its minutes get decremented. Never blocks Stop and
        // never surfaces an error to the user — a network blip here must
        // not cost the user the interview they just finished. No-ops
        // silently when the user isn't signed in (see start_backend_session).
        invoke("end_backend_session").catch(() => {});
      }
    }
  }, [mode]);

  const startLabel =
    sessionState === "STARTING" ? "Starting…" : sessionState === "LISTENING" ? "Listening" : "Start";

  const handleMinimize = useCallback(() => {
    getCurrentWindow().minimize();
  }, []);
  const handleToggleMaximize = useCallback(() => {
    getCurrentWindow().toggleMaximize();
  }, []);
  const handleClose = useCallback(() => {
    getCurrentWindow().close();
  }, []);

  return (
    <main className={`app-shell ${isMaximized ? "" : "app-shell-compact"}`}>
      <div className="title-bar" data-tauri-drag-region>
        <div className="window-controls">
          <button className="window-control-btn" onClick={handleMinimize} aria-label="Minimize" title="Minimize">
            <IconMinimize />
          </button>
          <button
            className="window-control-btn"
            onClick={handleToggleMaximize}
            aria-label={isMaximized ? "Restore" : "Maximize"}
            title={isMaximized ? "Restore" : "Maximize"}
          >
            {isMaximized ? <IconRestore /> : <IconMaximize />}
          </button>
          <button className="window-control-btn close" onClick={handleClose} aria-label="Close" title="Close">
            <IconClose />
          </button>
        </div>
      </div>

      <header className="compact-header">
        <div className="compact-header-brand">
          <img className="header-logo" src={smallbirdLogo} alt="Smallbird" />
          <span className="header-product">SmallBird</span>
        </div>

        <div className="compact-header-controls">
          <div className="dropdown-anchor">
            <button
              className="compact-header-btn"
              onClick={() => togglePopover("MODE")}
              disabled={sessionState !== "IDLE"}
              aria-haspopup="menu"
              aria-expanded={openPopover === "MODE"}
            >
              <span>Mode: {MODE_LABELS[mode]}</span>
              <IconChevronDown />
            </button>
            {openPopover === "MODE" && (
              <HeaderDropdown onClose={closePopover} className="header-dropdown-menu header-dropdown-mode">
                <div role="menu">
                  {(["INTERVIEW", "MEETING"] as Mode[]).map((m) => (
                    <button
                      key={m}
                      role="menuitemradio"
                      aria-checked={mode === m}
                      className={`dropdown-item${mode === m ? " active" : ""}`}
                      onClick={() => {
                        setMode(m);
                        closePopover();
                      }}
                    >
                      {MODE_LABELS[m]}
                    </button>
                  ))}
                </div>
              </HeaderDropdown>
            )}
          </div>

          <div className="dropdown-anchor">
            <button
              className="compact-header-btn"
              onClick={() => togglePopover("MODEL")}
              disabled={sessionState !== "IDLE"}
              aria-haspopup="menu"
              aria-expanded={openPopover === "MODEL"}
              title="Which AI model answers your questions"
            >
              {(() => {
                const current = LLM_PROVIDERS.find((p) => p.value === llmProvider)!;
                const CurrentIcon = current.icon;
                return (
                  <>
                    <CurrentIcon size={14} />
                    <span>{current.label}</span>
                  </>
                );
              })()}
              <IconChevronDown />
            </button>
            {openPopover === "MODEL" && (
              <HeaderDropdown onClose={closePopover} className="header-dropdown-menu header-dropdown-model">
                <div role="menu">
                  {LLM_PROVIDERS.map((p) => (
                    <button
                      key={p.value}
                      role="menuitemradio"
                      aria-checked={llmProvider === p.value}
                      disabled={!p.available}
                      className={`dropdown-item dropdown-item-icon${llmProvider === p.value ? " active" : ""}`}
                      title={p.available ? undefined : "Coming soon"}
                      onClick={() => {
                        if (!p.available) return;
                        setLlmProvider(p.value);
                        saveLlmProvider(p.value);
                        closePopover();
                      }}
                    >
                      <p.icon size={16} />
                      <span>{p.label}</span>
                      {!p.available && <span className="dropdown-item-badge">Soon</span>}
                    </button>
                  ))}
                </div>
              </HeaderDropdown>
            )}
          </div>

          <div className="dropdown-anchor">
            <button
              className="compact-header-btn icon"
              onClick={() => togglePopover("CONTEXT")}
              title="Context"
              aria-haspopup="dialog"
              aria-expanded={openPopover === "CONTEXT"}
            >
              <IconAttachment />
              <span>Context</span>
            </button>
            {openPopover === "CONTEXT" && (
              <HeaderDropdown onClose={closePopover} className="header-dropdown-panel header-dropdown-wide">
                <div className="popover-overlay">
                  <div className="popover" role="dialog" aria-label="Context">
                    <div className="popover-header">
                      <span className="setup-section-label">Attach context</span>
                      <button className="modal-close-btn" onClick={closePopover} aria-label="Close">
                        ✕
                      </button>
                    </div>
                    <div className="popover-body">
                      {mode === "MEETING" && (
                        <div className="setup-identity">
                          <div className="setup-identity-field">
                            <label htmlFor="meeting-title">Meeting Title</label>
                            <input
                              id="meeting-title"
                              className="setup-input"
                              value={meetingTitle}
                              onChange={(e) => setMeetingTitle(e.target.value)}
                              placeholder="e.g. Q3 Roadmap Review"
                            />
                          </div>
                          <div className="setup-identity-field">
                            <label htmlFor="meeting-participants">Participants</label>
                            <input
                              id="meeting-participants"
                              className="setup-input"
                              value={meetingParticipants}
                              onChange={(e) => setMeetingParticipants(e.target.value)}
                              placeholder="e.g. Alex, Priya, Sam"
                            />
                          </div>
                        </div>
                      )}
                      <DocumentContext
                        sections={mode === "INTERVIEW" ? INTERVIEW_CONTEXT_SECTIONS : MEETING_CONTEXT_SECTIONS}
                        state={mode === "INTERVIEW" ? interviewContext : meetingContext}
                        onChange={mode === "INTERVIEW" ? setInterviewContext : setMeetingContext}
                      />
                    </div>
                  </div>
                </div>
              </HeaderDropdown>
            )}
          </div>

          {sessionState === "LISTENING" ? (
            <Button variant="danger" onClick={handleStop}>
              Stop
            </Button>
          ) : (
            <Button variant="primary" onClick={handleStart} disabled={sessionState === "STARTING"}>
              {/* The STT model load behind Start can take a couple of
                  seconds (cold-starting the sidecar process + loading the
                  ONNX model) — a static label made that wait look frozen.
                  This spinner is the only signal the user has that anything
                  is happening until the overlay itself opens. */}
              {sessionState === "STARTING" ? (
                <span className="btn-spinner">
                  <Spinner />
                  {startLabel}
                </span>
              ) : (
                startLabel
              )}
            </Button>
          )}

          <div className="dropdown-anchor">
            <button
              className="compact-header-btn icon"
              onClick={() => togglePopover("SETTINGS")}
              title="Settings"
              aria-label="Settings"
              aria-haspopup="dialog"
              aria-expanded={openPopover === "SETTINGS"}
            >
              <IconSettings />
            </button>
            {openPopover === "SETTINGS" && (
              <HeaderDropdown onClose={closePopover} className="header-dropdown-panel header-dropdown-settings">
                <SettingsPopover
                  onClose={closePopover}
                  mode={mode}
                  historyRefreshKey={historyRefreshKey}
                  onApiKeySaved={(provider) => {
                    setLlmProvider(provider);
                    saveLlmProvider(provider);
                  }}
                />
              </HeaderDropdown>
            )}
          </div>

          <div className="dropdown-anchor">
            <button
              className="compact-header-btn icon"
              onClick={() => togglePopover("ACCOUNT")}
              title="Account"
              aria-label="Account"
              aria-haspopup="dialog"
              aria-expanded={openPopover === "ACCOUNT"}
            >
              <IconAccount />
            </button>
            {openPopover === "ACCOUNT" && (
              <HeaderDropdown onClose={closePopover} className="header-dropdown-panel">
                <Account onClose={closePopover} />
              </HeaderDropdown>
            )}
          </div>
        </div>
      </header>

      {error && <p className="error compact-header-error">{error}</p>}

      {isMaximized && (
        <div className="expanded-content">
          <LowEndHardwareBanner />
        </div>
      )}
    </main>
  );
}

export default App;
