import { useCallback, useEffect, useState } from "react";
import { Button } from "./ui";
import {
  clearPersonalApiKey,
  getPersonalApiKey,
  setPersonalApiKey,
  type PersonalLlmProvider,
} from "./personalApiKeys";

const PROVIDERS: { key: PersonalLlmProvider; label: string }[] = [
  { key: "openai", label: "OpenAI" },
  { key: "anthropic", label: "Anthropic" },
  { key: "gemini", label: "Gemini" },
];

/// Personal-build-only panel: lets the user store their own API key per
/// provider, using the same secure storage mechanism (Windows Credential
/// Manager via the `keyring` crate) already used for the Supabase session in
/// Account.tsx's Rust counterpart. Which provider is actually *used* for a
/// request stays governed by the existing header dropdown / llmProviderSetting.ts
/// — this panel only manages the keys, not the selection.
export function PersonalApiKeySettings({ onKeySaved }: { onKeySaved?: (provider: PersonalLlmProvider) => void }) {
  const [keys, setKeys] = useState<Record<PersonalLlmProvider, string>>({ openai: "", anthropic: "", gemini: "" });
  const [saved, setSaved] = useState<Record<PersonalLlmProvider, boolean>>({ openai: false, anthropic: false, gemini: false });
  const [busy, setBusy] = useState<PersonalLlmProvider | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    PROVIDERS.forEach(({ key }) => {
      getPersonalApiKey(key)
        .then((value) => {
          setKeys((prev) => ({ ...prev, [key]: value ?? "" }));
          setSaved((prev) => ({ ...prev, [key]: !!value }));
        })
        .catch(() => {});
    });
  }, []);

  const handleSave = useCallback(async (provider: PersonalLlmProvider) => {
    setBusy(provider);
    setStatus(null);
    try {
      await setPersonalApiKey(provider, keys[provider]);
      setSaved((prev) => ({ ...prev, [provider]: true }));
      setStatus(`${provider} key saved.`);
      onKeySaved?.(provider);
    } catch (err) {
      setStatus(`Failed to save ${provider} key: ${String(err)}`);
    } finally {
      setBusy(null);
    }
  }, [keys]);

  const handleClear = useCallback(async (provider: PersonalLlmProvider) => {
    setBusy(provider);
    setStatus(null);
    try {
      await clearPersonalApiKey(provider);
      setKeys((prev) => ({ ...prev, [provider]: "" }));
      setSaved((prev) => ({ ...prev, [provider]: false }));
      setStatus(`${provider} key cleared.`);
    } catch (err) {
      setStatus(`Failed to clear ${provider} key: ${String(err)}`);
    } finally {
      setBusy(null);
    }
  }, []);

  return (
    <div>
      <p className="setup-hint">
        This is a personal build — it uses your own API key directly, with no Smallbird Cloud account and
        no subscription. Keys are stored securely in Windows Credential Manager on this device only, and are
        never sent anywhere except directly to the provider you choose above in the model picker.
      </p>

      {PROVIDERS.map(({ key, label }) => (
        <div key={key} className="setup-identity-field" style={{ marginTop: "12px" }}>
          <label className="setup-section-label">{label}</label>
          <input
            className="setup-input"
            type="password"
            aria-label={`${label} API key`}
            placeholder={saved[key] ? "Key saved — enter a new key to replace it" : `${label} API key`}
            value={keys[key]}
            onChange={(e) => setKeys((prev) => ({ ...prev, [key]: e.target.value }))}
          />
          <div style={{ display: "flex", gap: "8px", marginTop: "6px" }}>
            <Button variant="primary" onClick={() => handleSave(key)} disabled={busy === key || !keys[key]}>
              {busy === key ? "Working…" : "Save"}
            </Button>
            <Button variant="ghost" onClick={() => handleClear(key)} disabled={busy === key || !saved[key]}>
              Clear
            </Button>
          </div>
        </div>
      ))}

      {status && <p className="setup-hint">{status}</p>}
    </div>
  );
}
