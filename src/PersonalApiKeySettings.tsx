import { useCallback, useEffect, useState } from "react";
import { Button } from "./ui";
import {
  clearPersonalApiKey,
  getPersonalApiKey,
  setPersonalApiKey,
  type PersonalLlmProvider,
  type PersonalServiceProvider,
} from "./personalApiKeys";

type ProviderKey = PersonalLlmProvider | PersonalServiceProvider;

const LLM_PROVIDERS: { key: PersonalLlmProvider; label: string }[] = [
  { key: "openai", label: "OpenAI" },
  { key: "anthropic", label: "Anthropic" },
  { key: "gemini", label: "Gemini" },
];

// Not LLM providers (see personalApiKeys.ts) — shown in their own section
// below so it's clear these configure voice input/output, not which model
// answers a question.
const SERVICE_PROVIDERS: { key: PersonalServiceProvider; label: string; hint: string }[] = [
  { key: "groq", label: "Groq (Speech-to-Text)", hint: "Get one at console.groq.com" },
  { key: "deepgram", label: "Deepgram (Text-to-Speech)", hint: "Get one at console.deepgram.com" },
];

const ALL_PROVIDERS: { key: ProviderKey; label: string }[] = [...LLM_PROVIDERS, ...SERVICE_PROVIDERS];

/// Personal-build-only panel: lets the user store their own API key per
/// provider, using the same secure storage mechanism (Windows Credential
/// Manager via the `keyring` crate) already used for the Supabase session in
/// Account.tsx's Rust counterpart. Which LLM provider is actually *used* for
/// a request stays governed by the existing header dropdown /
/// llmProviderSetting.ts — this panel only manages keys, not the LLM
/// selection. Groq/Deepgram have no such selector (each is the only STT/TTS
/// provider — see src-tauri/src/stt/groq.rs and src-tauri/src/tts/deepgram_flux.rs)
/// so their key is simply configured or not.
const LLM_PROVIDER_KEYS: readonly PersonalLlmProvider[] = ["openai", "anthropic", "gemini"];
const isLlmProvider = (key: ProviderKey): key is PersonalLlmProvider =>
  (LLM_PROVIDER_KEYS as readonly string[]).includes(key);

export function PersonalApiKeySettings({ onKeySaved }: { onKeySaved?: (provider: PersonalLlmProvider) => void }) {
  const [keys, setKeys] = useState<Record<ProviderKey, string>>({
    openai: "", anthropic: "", gemini: "", groq: "", deepgram: "",
  });
  const [saved, setSaved] = useState<Record<ProviderKey, boolean>>({
    openai: false, anthropic: false, gemini: false, groq: false, deepgram: false,
  });
  const [busy, setBusy] = useState<ProviderKey | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    ALL_PROVIDERS.forEach(({ key }) => {
      getPersonalApiKey(key)
        .then((value) => {
          setKeys((prev) => ({ ...prev, [key]: value ?? "" }));
          setSaved((prev) => ({ ...prev, [key]: !!value }));
        })
        .catch(() => {});
    });
  }, []);

  const handleSave = useCallback(async (provider: ProviderKey) => {
    setBusy(provider);
    setStatus(null);
    try {
      await setPersonalApiKey(provider, keys[provider]);
      setSaved((prev) => ({ ...prev, [provider]: true }));
      setStatus(`${provider} key saved.`);
      // Only an LLM provider key save should switch the active model
      // picker — Groq/Deepgram have no such selector, and firing this for
      // them would set llmProviderSetting.ts to an invalid "groq"/"deepgram"
      // value (see App.tsx's onApiKeySaved, which does exactly that).
      if (isLlmProvider(provider)) {
        onKeySaved?.(provider);
      }
    } catch (err) {
      setStatus(`Failed to save ${provider} key: ${String(err)}`);
    } finally {
      setBusy(null);
    }
  }, [keys]);

  const handleClear = useCallback(async (provider: ProviderKey) => {
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

  const renderField = (key: ProviderKey, label: string, hint?: string) => (
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
      {hint && <p className="setup-hint" style={{ marginTop: "4px" }}>{hint}</p>}
      <div style={{ display: "flex", gap: "8px", marginTop: "6px" }}>
        <Button variant="primary" onClick={() => handleSave(key)} disabled={busy === key || !keys[key]}>
          {busy === key ? "Working…" : "Save"}
        </Button>
        <Button variant="ghost" onClick={() => handleClear(key)} disabled={busy === key || !saved[key]}>
          Clear
        </Button>
      </div>
    </div>
  );

  return (
    <div>
      <p className="setup-hint">
        This is a personal build — it uses your own API key directly, with no Smallbird Cloud account and
        no subscription. Keys are stored securely in Windows Credential Manager on this device only, and are
        never sent anywhere except directly to the provider you choose above in the model picker.
      </p>

      {LLM_PROVIDERS.map(({ key, label }) => renderField(key, label))}

      <p className="setup-hint" style={{ marginTop: "20px", fontWeight: 600 }}>
        Voice (speech-to-text / text-to-speech)
      </p>
      <p className="setup-hint">
        Groq transcribes your voice input; Deepgram speaks answers aloud when the overlay's
        "Speak answers aloud" toggle is on. Both are optional — voice input/output simply
        won't work until each is configured, everything else in the app is unaffected.
      </p>
      {SERVICE_PROVIDERS.map(({ key, label, hint }) => renderField(key, label, hint))}

      {status && <p className="setup-hint">{status}</p>}
    </div>
  );
}
