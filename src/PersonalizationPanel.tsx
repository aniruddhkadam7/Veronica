import { useState } from "react";
import {
  loadOverlaySettings,
  saveOverlaySettings,
  type AnswerLength,
  type Humanization,
  type ResponseStyle,
} from "./overlaySettings";
import { Select } from "./ui";

/// Personalization settings: how AI-generated answers should read (tone,
/// length, humanization), shared by Interview and Meeting Mode. Lives in
/// OverlaySettings/localStorage — the same store Veronica's one overlay
/// Settings panel (OverlaySettingsPanel) reads and writes — so a change made
/// here from the main window's Settings applies to the very next question
/// asked, in whichever mode is running, without a separate personalization
/// store to keep in sync.
///
/// englishLevel is intentionally left out — it has no Meeting Mode analogue
/// server-side (apps/backend/app/schemas/meeting.py has no such field), so
/// it stays Interview-only rather than showing here as a control that does
/// nothing in one of the two modes this panel covers. humanization DOES now
/// apply to both (see prompt_builder_meeting.py's _HUMANIZATION_INSTRUCTIONS
/// and meeting_mode/commands.rs's MeetingAskOptions).
export function PersonalizationPanel() {
  const [settings, setSettings] = useState(() => loadOverlaySettings());

  const set = <K extends keyof typeof settings>(key: K, value: (typeof settings)[K]) => {
    setSettings((prev) => {
      const next = { ...prev, [key]: value };
      saveOverlaySettings(next);
      return next;
    });
  };

  return (
    <div className="personalization-panel">
      <div className="personalization-row">
        <label htmlFor="personalization-answer-length">Answer length</label>
        <Select
          id="personalization-answer-length"
          className="setup-select"
          value={settings.answerLength}
          onChange={(v: AnswerLength) => set("answerLength", v)}
          options={[
            { value: "brief", label: "Brief" },
            { value: "default", label: "Default" },
            { value: "detailed", label: "Detailed" },
          ]}
        />
      </div>

      <div className="personalization-row">
        <label htmlFor="personalization-response-style">Response tone</label>
        <Select
          id="personalization-response-style"
          className="setup-select"
          value={settings.responseStyle}
          onChange={(v: ResponseStyle) => set("responseStyle", v)}
          options={[
            { value: "natural", label: "Natural" },
            { value: "technical", label: "Technical" },
            { value: "concise", label: "Concise" },
          ]}
        />
      </div>

      <div className="personalization-row">
        <label htmlFor="personalization-humanization">Humanize</label>
        <Select
          id="personalization-humanization"
          className="setup-select"
          value={settings.humanization}
          onChange={(v: Humanization) => set("humanization", v)}
          options={[
            { value: "natural", label: "Natural" },
            { value: "conversational", label: "Conversational" },
            { value: "formal", label: "Formal" },
          ]}
        />
      </div>
    </div>
  );
}
