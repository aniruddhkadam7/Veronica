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
import { Button, Spinner } from "./ui";
import { ParticlesOrb } from "@/registry/orbe/particles-orb/particles-orb";
import {
  IconAccount,
  IconAttachment,
  IconChevronDown,
  IconClose,
  IconMaximize,
  IconMic,
  IconMicOff,
  IconMinimize,
  IconRestore,
  IconSettings,
} from "./Icons";
import { IconAnthropic, IconDeepSeek, IconGemini, IconOpenAI } from "./ProviderIcons";
import { VERONICA_CONTEXT_SECTIONS } from "./headerPopups";
import { hasStoredLlmProvider, loadLlmProvider, saveLlmProvider, type LlmProvider } from "./llmProviderSetting";
import { getPersonalApiKey } from "./personalApiKeys";

// "ACTIVE" replaces the old "LISTENING": Veronica is listening via the
// floating widget, whether or not the full conversation overlay ("Open") is
// currently shown on top of it — see the Start/Open/Stop rework below.
type SessionState = "IDLE" | "STARTING" | "ACTIVE";
type Popover = "MODEL" | "CONTEXT" | "SETTINGS" | "ACCOUNT" | null;

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
  const [micMuted, setMicMuted] = useState(false);
  const [openPopover, setOpenPopover] = useState<Popover>(null);
  const [error, setError] = useState<string | null>(null);
  const [isMaximized, setIsMaximized] = useState(false);

  const [documentContext, setDocumentContext] = useState<DocumentContextState>(() =>
    emptyDocumentContextState(VERONICA_CONTEXT_SECTIONS),
  );

  // Prewarm the STT sidecar the moment the app launches — the model loads
  // while the user is still attaching context, so Start doesn't eat the
  // full load time. Uses a throwaway mic-assistant start/stop cycle purely
  // to force the STT model to load early; actual listening only begins
  // when the user presses Start (see handleStart), not here.
  const prewarmRef = useRef<Promise<void> | null>(null);
  useEffect(() => {
    const promise = invoke<void>("start_mic_assistant")
      .then(() => invoke<void>("stop_mic_assistant").catch(() => {}))
      .catch((e) => {
        if (String(e) !== "mic assistant already running") throw e;
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

  // The overlay window and this main window are separate webviews with no
  // shared React state. Closing the overlay (its own ✕ / Escape) now only
  // means "stop showing the conversation view" — Veronica keeps
  // listening/working via the floating widget underneath, matching the
  // "closing the overlay returns to the small floating widget; Veronica
  // keeps working/listening normally" requirement — it must NOT end the
  // session the way it used to when Stop and "close the overlay" were the
  // same action. So this only toggles the overlay-visible flag, never
  // sessionState.
  const [overlayVisible, setOverlayVisible] = useState(false);
  useEffect(() => {
    const unlisten = listen("interview-mode:overlay-closed", () => {
      setOverlayVisible(false);
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

  // Activates Veronica as an always-on floating widget (bottom-right orb,
  // no conversation UI) — NOT the full overlay. This is what the orb
  // toolbar button and "Start" both trigger; "Open" (below) is a separate,
  // non-destructive action that only shows/hides the conversation view on
  // top of this already-running session.
  const handleActivate = useCallback(async () => {
    setError(null);
    setSessionState("STARTING");
    setMicMuted(false);
    closePopover();
    try {
      uploadDocumentContext(VERONICA_CONTEXT_SECTIONS, documentContext);

      // Opt-in, signed-in-only: resolves to { rejection: null } when the
      // user isn't signed in, or when something unrelated to entitlement
      // went wrong (credential-store hiccup, network blip reaching the
      // backend) — local recording has no dependency on this feature and
      // must never be blocked by it. Only a genuine backend entitlement
      // rejection (no remaining minutes, concurrent session limit, etc.)
      // sets `rejection`, which we propagate so activation actually stops —
      // see start_backend_session's Rust-side doc (BackendSessionResult)
      // for the full contract.
      const sessionResult = await invoke<{ rejection: string | null }>("start_backend_session", {
        sttMode: "local",
      });
      if (sessionResult.rejection) {
        throw new Error(sessionResult.rejection);
      }

      // Wait for the prewarm cycle (start+stop, purely to force the STT
      // model to load early) to finish before actually starting to listen,
      // so this doesn't race the prewarm's own start_mic_assistant/
      // stop_mic_assistant calls.
      await (prewarmRef.current ?? Promise.resolve());
      await invoke("clear_transcript").catch(() => {});
      await invoke("start_mic_assistant").catch((e) => {
        if (String(e) !== "mic assistant already running") throw e;
      });
      await invoke("show_veronica_widget");
      setSessionState("ACTIVE");
    } catch (e) {
      setError(String(e));
      setSessionState("IDLE");
    }
  }, [documentContext, closePopover]);

  // Fully deactivates: stops the mic, stops Veronica mid-answer if she's
  // still speaking, hides both the widget and the overlay (if it happened to
  // be open), and ends the backend session. This is the ONLY thing that ends
  // the session — closing the overlay (see `interview-mode:overlay-closed`
  // above) no longer does.
  const handleStop = useCallback(async () => {
    invoke("hide_veronica_widget").catch(() => {});
    if (overlayVisible) {
      invoke("hide_interview_overlay").catch(() => {});
    }
    setOverlayVisible(false);
    setSessionState("IDLE");
    setMicMuted(false);
    invoke("stop_mic_assistant").catch(() => {});
    // Cancels any in-flight turn and stops TTS playback immediately — see
    // veronica::stop_speaking — so Stop actually silences Veronica instead
    // of only ending the mic while a reply keeps playing out.
    invoke("stop_speaking").catch(() => {});
    // Ends the conversation itself — the ONE shared history both the
    // widget and overlay read/write (see conversation.rs's doc). This is
    // the only place the conversation is ever cleared: opening/closing
    // either window must never reset it, only explicitly ending the
    // session via Stop.
    invoke("reset_conversation").catch(() => {});
    // Fire-and-forget: finalizes the backend session (if one was started)
    // so its minutes get decremented. Never blocks Stop and never surfaces
    // an error to the user — a network blip here must not cost the user
    // the conversation they just finished. No-ops silently when the user
    // isn't signed in (see start_backend_session).
    invoke("end_backend_session").catch(() => {});
  }, [overlayVisible]);

  // Mutes/unmutes the mic without ending the session — unlike Stop, this
  // leaves the backend session, widget, and overlay (if open) untouched;
  // it only tells the mic-assistant pump loop to withhold audio from STT
  // (see voice_command::mod's set_mic_muted). Optimistic local toggle since
  // the backend call can't meaningfully fail once a session is active.
  const handleToggleMute = useCallback(() => {
    setMicMuted((cur) => {
      const next = !cur;
      invoke("set_mic_muted", { muted: next }).catch((e) => setError(String(e)));
      return next;
    });
  }, []);

  // "Open": reveals the full conversation overlay on top of the already-
  // running widget session — never starts/restarts anything, since
  // start_mic_assistant was already called by handleActivate and the
  // overlay's own mount effect tolerates "already running" as success (see
  // VeronicaOverlay.tsx). This is what makes "the conversation continues
  // without interruption when opening/closing the overlay" true: nothing
  // about the mic/STT session is touched by opening or closing the view.
  const handleOpenOverlay = useCallback(async () => {
    try {
      await invoke("show_interview_overlay");
      setOverlayVisible(true);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  const activateLabel =
    sessionState === "STARTING" ? "Starting…" : sessionState === "ACTIVE" ? "Open" : "Start";

  // Clicking the orb button itself toggles activation — matches "clicking
  // the widget should place/activate it there"/"deactivate" from the spec,
  // independent of the separate Open/Stop buttons that appear once active.
  const handleOrbClick = useCallback(() => {
    if (sessionState === "ACTIVE") {
      handleStop();
    } else if (sessionState === "IDLE") {
      handleActivate();
    }
  }, [sessionState, handleStop, handleActivate]);

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
                      <DocumentContext
                        sections={VERONICA_CONTEXT_SECTIONS}
                        state={documentContext}
                        onChange={setDocumentContext}
                      />
                    </div>
                  </div>
                </div>
              </HeaderDropdown>
            )}
          </div>

          <button
            type="button"
            className={`compact-header-btn icon veronica-widget-launch-btn${sessionState === "ACTIVE" ? " active" : ""}`}
            onClick={handleOrbClick}
            disabled={sessionState === "STARTING"}
            title={
              sessionState === "ACTIVE"
                ? "Deactivate Veronica"
                : "Activate Veronica as a floating indicator (bottom-right, no chat window)"
            }
            aria-label={sessionState === "ACTIVE" ? "Deactivate Veronica" : "Activate Veronica"}
            aria-pressed={sessionState === "ACTIVE"}
          >
            <ParticlesOrb state={sessionState === "ACTIVE" ? "listening" : "idle"} size={20} speed={1} colorFrom="#f0abfc" colorTo="#818cf8" />
          </button>

          {sessionState === "ACTIVE" && (
            <button
              type="button"
              className={`compact-header-btn icon${micMuted ? " active" : ""}`}
              onClick={handleToggleMute}
              title={micMuted ? "Unmute Veronica's mic" : "Mute Veronica's mic"}
              aria-label={micMuted ? "Unmute microphone" : "Mute microphone"}
              aria-pressed={micMuted}
            >
              {micMuted ? <IconMicOff /> : <IconMic />}
            </button>
          )}

          {sessionState === "ACTIVE" && (
            <Button variant="danger" onClick={handleStop}>
              Stop
            </Button>
          )}

          <Button
            variant="primary"
            onClick={sessionState === "ACTIVE" ? handleOpenOverlay : handleActivate}
            disabled={sessionState === "STARTING"}
          >
            {/* The STT model load behind activation can take a couple of
                seconds (cold-starting the sidecar process + loading the
                ONNX model) — a static label made that wait look frozen.
                This spinner is the only signal the user has that anything
                is happening until the widget itself appears. Once active,
                this button's only job is opening the conversation view —
                it never restarts anything (see handleOpenOverlay). */}
            {sessionState === "STARTING" ? (
              <span className="btn-spinner">
                <Spinner />
                {activateLabel}
              </span>
            ) : (
              activateLabel
            )}
          </Button>

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
