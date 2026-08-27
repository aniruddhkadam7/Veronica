import { invoke } from "@tauri-apps/api/core";
import { useState } from "react";
import {
  DEFAULT_OVERLAY_SETTINGS,
  SIZE_FRACTIONS,
  type AnswerLength,
  type OverlayDensity,
  type OverlaySettings,
  type OverlaySize,
  type ResponseStyle,
} from "./overlaySettings";
import { Select } from "./ui";

interface Props {
  settings: OverlaySettings;
  onChange: (settings: OverlaySettings) => void;
  onClose: () => void;
  captureExcluded: boolean | null;
}

/// Veronica's Settings panel — everything the overlay needs beyond the
/// question/answer flow, without becoming the large Record/Prepare
/// dashboard. Renders inline in the same small overlay window (swapped in for
/// the question/answer body), not a separate window.
export function OverlaySettingsPanel({ settings, onChange, onClose, captureExcluded }: Props) {
  const [alwaysOnTop, setAlwaysOnTop] = useState(true);

  const set = <K extends keyof OverlaySettings>(key: K, value: OverlaySettings[K]) => {
    onChange({ ...settings, [key]: value });
  };

  const changeSize = (size: OverlaySize) => {
    set("size", size);
    invoke("resize_interview_overlay", { fraction: SIZE_FRACTIONS[size] }).catch(() => {
      // Best-effort — the CSS size-* class still applies even if the actual
      // OS window resize fails, so the panel remains usable either way.
    });
  };

  const toggleAlwaysOnTop = () => {
    const next = !alwaysOnTop;
    setAlwaysOnTop(next);
    invoke("set_overlay_always_on_top", { enabled: next }).catch(() => {
      // Best-effort — the window remains on top by default even if this call
      // fails, so there is nothing actionable to show the user here.
    });
  };

  const resetDefaults = () => onChange({ ...DEFAULT_OVERLAY_SETTINGS });

  return (
    <div className="overlay-body overlay-settings">
      <div className="overlay-settings-section">
        <span className="overlay-settings-label">Appearance</span>

        <label className="overlay-settings-row">
          <span>Opacity</span>
          <input
            type="range"
            min={0.15}
            max={1}
            step={0.02}
            value={settings.opacity}
            onChange={(e) => set("opacity", Number(e.target.value))}
          />
        </label>

        <div className="overlay-settings-row">
          <span>Overlay size</span>
          <Select
            className="select-overlay"
            value={settings.size}
            onChange={changeSize}
            aria-label="Overlay size"
            options={[
              { value: "small", label: "Small" },
              { value: "medium", label: "Medium" },
              { value: "large", label: "Large" },
            ]}
          />
        </div>

        <label className="overlay-settings-row">
          <span>Font size</span>
          <input
            type="range"
            min={12}
            max={20}
            step={1}
            value={settings.fontSize}
            onChange={(e) => set("fontSize", Number(e.target.value))}
          />
          <span className="overlay-settings-value">{settings.fontSize}px</span>
        </label>

        <div className="overlay-settings-row">
          <span>Text density</span>
          <Select
            className="select-overlay"
            value={settings.density}
            onChange={(v: OverlayDensity) => set("density", v)}
            aria-label="Text density"
            options={[
              { value: "compact", label: "Compact" },
              { value: "comfortable", label: "Comfortable" },
            ]}
          />
        </div>

        <label className="overlay-settings-row overlay-settings-checkbox">
          <input
            type="checkbox"
            checked={settings.dragEnabled}
            onChange={(e) => set("dragEnabled", e.target.checked)}
          />
          <span>Enable drag-to-move on header</span>
        </label>

        <label className="overlay-settings-row overlay-settings-checkbox">
          <input type="checkbox" checked={alwaysOnTop} onChange={toggleAlwaysOnTop} />
          <span>Always on top</span>
        </label>

        <label className="overlay-settings-row overlay-settings-checkbox">
          <input
            type="checkbox"
            checked={settings.voiceOutputEnabled}
            onChange={(e) => set("voiceOutputEnabled", e.target.checked)}
          />
          <span>Speak answers aloud (Deepgram)</span>
        </label>
      </div>

      <div className="overlay-settings-section">
        <span className="overlay-settings-label">Answer</span>

        <div className="overlay-settings-row">
          <span>Answer length</span>
          <Select
            className="select-overlay"
            value={settings.answerLength}
            onChange={(v: AnswerLength) => set("answerLength", v)}
            aria-label="Answer length"
            options={[
              { value: "brief", label: "Brief (1-3 sentences)" },
              { value: "default", label: "Default (~50-120 words)" },
              { value: "detailed", label: "Detailed" },
            ]}
          />
        </div>

        <div className="overlay-settings-row">
          <span>Response style</span>
          <Select
            className="select-overlay"
            value={settings.responseStyle}
            onChange={(v: ResponseStyle) => set("responseStyle", v)}
            aria-label="Response style"
            options={[
              { value: "natural", label: "Natural" },
              { value: "technical", label: "Technical" },
              { value: "concise", label: "Concise" },
            ]}
          />
        </div>
      </div>

      <div className="overlay-settings-section">
        <span className="overlay-settings-label">Speech Recognition</span>

        <label className="overlay-settings-row">
          <span>STT sensitivity</span>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={settings.sttSensitivity}
            onChange={(e) => set("sttSensitivity", Number(e.target.value))}
          />
        </label>
        <p className="overlay-settings-note">
          Higher sensitivity finalizes questions sooner after a pause; lower waits longer.
        </p>
      </div>

      <div className="overlay-settings-section">
        <span className="overlay-settings-label">Screen Capture Protection</span>
        <p>{captureExcluded ? "● Enabled" : "⚠ Unavailable"}</p>
      </div>

      <div className="overlay-settings-section">
        <span className="overlay-settings-label">Talking to Veronica</span>
        <p className="overlay-settings-note">
          Just talk — your voice is transcribed into the question box and asked automatically
          once you stop talking, no button needed. Use the 🔊 header toggle if you also want
          Veronica listening to system audio (the other speaker/app sound). Ask it to open an
          app, file, folder, or URL and it'll just do it — dangerous or destructive requests are
          always refused.
        </p>
      </div>

      <div className="overlay-settings-section">
        <span className="overlay-settings-label">Hotkeys</span>
        <p>ENTER — Ask AI</p>
        <p>SHIFT+ENTER — new line while editing</p>
        <p>ESC — hide overlay</p>
      </div>

      <div className="overlay-settings-actions">
        <button className="overlay-button" onClick={resetDefaults}>
          Reset to defaults
        </button>
        <button className="overlay-button primary" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
