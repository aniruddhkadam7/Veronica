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
  type VoiceLaunchApp,
} from "./overlaySettings";
import { Select } from "./ui";

interface Props {
  settings: OverlaySettings;
  onChange: (settings: OverlaySettings) => void;
  onClose: () => void;
  captureExcluded: boolean | null;
}

/// Interview Mode's Settings panel — everything the overlay needs beyond the
/// question/answer flow, without becoming the large Record/Prepare
/// dashboard. Renders inline in the same small overlay window (swapped in for
/// the question/answer body), not a separate window.
export function OverlaySettingsPanel({ settings, onChange, onClose, captureExcluded }: Props) {
  const [alwaysOnTop, setAlwaysOnTop] = useState(true);
  const [newAppName, setNewAppName] = useState("");
  const [newAppPath, setNewAppPath] = useState("");

  const set = <K extends keyof OverlaySettings>(key: K, value: OverlaySettings[K]) => {
    onChange({ ...settings, [key]: value });
  };

  const addVoiceApp = () => {
    const name = newAppName.trim();
    const path = newAppPath.trim();
    if (!name || !path) return;
    const withoutExisting = settings.voiceLaunchApps.filter(
      (a) => a.name.toLowerCase() !== name.toLowerCase(),
    );
    set("voiceLaunchApps", [...withoutExisting, { name, path }]);
    setNewAppName("");
    setNewAppPath("");
  };

  const removeVoiceApp = (name: string) => {
    set(
      "voiceLaunchApps",
      settings.voiceLaunchApps.filter((a) => a.name !== name),
    );
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
      </div>

      <div className="overlay-settings-section">
        <span className="overlay-settings-label">Answer</span>

        <label className="overlay-settings-row overlay-settings-checkbox">
          <input
            type="checkbox"
            checked={settings.autoAI}
            onChange={(e) => set("autoAI", e.target.checked)}
          />
          <span>Auto AI — send a question automatically once you stop talking</span>
        </label>

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

        <label className="overlay-settings-row overlay-settings-column">
          <span>Role (optional)</span>
          <input
            type="text"
            value={settings.role}
            onChange={(e) => set("role", e.target.value)}
            placeholder="e.g. Senior Backend Engineer"
          />
        </label>

        <label className="overlay-settings-row overlay-settings-column">
          <span>Job description (optional)</span>
          <textarea
            value={settings.jobDescription}
            onChange={(e) => set("jobDescription", e.target.value)}
            placeholder="Paste relevant job description context…"
            rows={3}
          />
        </label>
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
        <span className="overlay-settings-label">Voice Commands</span>
        <p className="overlay-settings-note">
          Click the mic button to talk: your voice is transcribed into the question box and
          asked automatically once you stop talking. Say "open &lt;name&gt;" for an app listed
          below to launch it instead.
        </p>

        {settings.voiceLaunchApps.length > 0 && (
          <ul className="overlay-voice-app-list">
            {settings.voiceLaunchApps.map((app: VoiceLaunchApp) => (
              <li key={app.name} className="overlay-settings-row">
                <span>
                  "open {app.name}" → {app.path}
                </span>
                <button
                  type="button"
                  className="overlay-text-button"
                  onClick={() => removeVoiceApp(app.name)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}

        <div className="overlay-settings-row overlay-settings-column">
          <span>Add an app you can open by voice</span>
          <input
            type="text"
            value={newAppName}
            onChange={(e) => setNewAppName(e.target.value)}
            placeholder="Name you'll say, e.g. notepad"
          />
          <input
            type="text"
            value={newAppPath}
            onChange={(e) => setNewAppPath(e.target.value)}
            placeholder="Command to run, e.g. notepad.exe"
          />
          <button
            type="button"
            className="overlay-button"
            onClick={addVoiceApp}
            disabled={!newAppName.trim() || !newAppPath.trim()}
          >
            Add app
          </button>
        </div>
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
