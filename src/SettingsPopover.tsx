import { useState } from "react";
import { PerformancePanel } from "./PerformancePanel";
import { PersonalizationPanel } from "./PersonalizationPanel";
import { AttachedDocumentsPanel } from "./AttachedDocumentsPanel";
import { AudioSettingsPanel } from "./AudioSettingsPanel";
import { PersonalApiKeySettings } from "./PersonalApiKeySettings";
import { WidgetPanel } from "./WidgetPanel";
import type { PersonalLlmProvider } from "./personalApiKeys";

type SettingsSection =
  | "PERSONALIZATION"
  | "WIDGET"
  | "AUDIO"
  | "PERFORMANCE"
  | "CONTEXT"
  | "API_KEYS"
  | "PRIVACY"
  | "ABOUT";

const SECTIONS: { key: SettingsSection; label: string }[] = [
  { key: "PERSONALIZATION", label: "Personalization" },
  { key: "WIDGET", label: "Widget" },
  { key: "AUDIO", label: "Audio" },
  { key: "PERFORMANCE", label: "Performance" },
  { key: "CONTEXT", label: "Context / Documents" },
  { key: "API_KEYS", label: "API Keys" },
  { key: "PRIVACY", label: "Privacy" },
  { key: "ABOUT", label: "About / Version" },
];

interface Props {
  onClose: () => void;
  onApiKeySaved?: (provider: PersonalLlmProvider) => void;
}

/// Settings popover reached from the header's gear icon. Every section
/// (including Performance) renders inline in settings-popover-content
/// alongside the nav rail — Performance used to swap out the whole popover
/// for its own separate shell, which hid the nav entirely with no way back
/// to another section except closing and reopening Settings. Only sections
/// with real content are listed here — General/Audio/Notifications were
/// placeholder stubs and have been dropped.
export function SettingsPopover({ onClose, onApiKeySaved }: Props) {
  const [section, setSection] = useState<SettingsSection>("PERSONALIZATION");

  return (
    <div className="popover-overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="popover" role="dialog" aria-modal="true" aria-label="Settings">
        <div className="popover-header">
          <span className="setup-section-label">Settings</span>
          <button className="modal-close-btn" onClick={onClose} title="Close" aria-label="Close">
            ✕
          </button>
        </div>

        <div className="popover-body settings-popover-body">
          <div className="workspace-nav settings-popover-nav">
            {SECTIONS.map((s) => (
              <button
                key={s.key}
                className={["workspace-nav-item", section === s.key ? "active" : ""].filter(Boolean).join(" ")}
                onClick={() => setSection(s.key)}
              >
                {s.label}
              </button>
            ))}
          </div>

          <div className="settings-popover-content">
            {section === "PERSONALIZATION" && <PersonalizationPanel />}
            {section === "WIDGET" && <WidgetPanel />}
            {section === "AUDIO" && <AudioSettingsPanel />}
            {section === "PERFORMANCE" && <PerformancePanel />}
            {section === "CONTEXT" && <AttachedDocumentsPanel />}
            {section === "API_KEYS" && <PersonalApiKeySettings onKeySaved={onApiKeySaved} />}
            {section === "PRIVACY" && (
              <p className="setup-hint">
                Smallbird runs speech-to-text and document search locally on this device, and this
                personal build talks directly to your own configured AI provider. Nothing is sent to
                Smallbird Cloud or any other backend.
              </p>
            )}
            {section === "ABOUT" && (
              <p className="setup-hint">
                Smallbird — version 0.1.0
                <br />
                Personal Build — uses your own API key, no Smallbird Cloud.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
