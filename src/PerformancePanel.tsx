import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

interface HardwareProfile {
  cpuPhysicalCores: number;
  cpuLogicalCores: number;
  cpuBrand: string;
  totalRamMb: number;
  availableRamMb: number;
  gpuVendor: string | null;
  gpuName: string | null;
  gpuVramMb: number | null;
  storageIsSsd: boolean | null;
  windowsVersion: string;
}

type HardwareTier = "entry" | "standard" | "performance" | "highPerformance";
type PerformanceMode = "adaptive" | "maximumPerformance" | "batterySaver";

interface PerformanceConfig {
  sttNumThreads: number;
  ragTopK: number;
  ragMaxContextChars: number;
  ragSimilarityThreshold: number;
  ragRetrievalTimeoutMs: number;
  ragEmbedBatchSize: number;
  ragTorchThreads: number;
}

interface PerformanceModeInfo {
  mode: PerformanceMode;
  detectedTier: HardwareTier;
  effectiveConfig: PerformanceConfig;
  hardwareRefreshed: boolean;
  isBelowRecommended: boolean;
}

const TIER_LABELS: Record<HardwareTier, string> = {
  entry: "Entry",
  standard: "Standard",
  performance: "Performance",
  highPerformance: "High Performance",
};

const MODE_OPTIONS: { value: PerformanceMode; label: string; description: string }[] = [
  {
    value: "adaptive",
    label: "Adaptive",
    description: "Automatically tuned for your hardware. Recommended.",
  },
  {
    value: "maximumPerformance",
    label: "Maximum Performance",
    description: "Best accuracy and context. Uses more CPU and memory.",
  },
  {
    value: "batterySaver",
    label: "Battery Saver",
    description: "Lighter on CPU and memory — smoother on modest hardware.",
  },
];

function formatGpu(profile: HardwareProfile): string {
  if (!profile.gpuName) {
    return "No dedicated graphics card detected — that's fine, Smallbird doesn't require one.";
  }
  const vram = profile.gpuVramMb ? `, ${Math.round(profile.gpuVramMb / 1024)}GB` : "";
  return `${profile.gpuName}${vram}`;
}

function formatStorage(profile: HardwareProfile): string | null {
  if (profile.storageIsSsd === null) return null;
  return profile.storageIsSsd ? "SSD" : "HDD";
}

/// Smallbird Performance panel: shows the machine's detected hardware and lets
/// the user pick Adaptive (default)/Maximum Performance/Battery Saver.
/// Renders as plain content — no popover shell of its own — since it's
/// shown inline inside SettingsPopover.tsx's settings-popover-content
/// alongside the section nav (Sessions/Personalization/etc.), the same way
/// every other section renders. It used to render its own full
/// .popover-overlay/.popover with its own header/close button, which meant
/// opening it replaced the entire Settings popover — including the nav
/// rail — so there was no way back to another section short of closing and
/// reopening Settings; that's what looked like the nav "hiding" itself.
export function PerformancePanel() {
  const [profile, setProfile] = useState<HardwareProfile | null>(null);
  const [modeInfo, setModeInfo] = useState<PerformanceModeInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(true);

  useEffect(() => {
    invoke<HardwareProfile>("get_hardware_profile").then(setProfile).catch((err) => setError(String(err)));
    invoke<PerformanceModeInfo>("get_performance_mode").then(setModeInfo).catch((err) => setError(String(err)));
  }, []);

  const handleSelectMode = useCallback(async (mode: PerformanceMode) => {
    setBusy(true);
    setError(null);
    try {
      await invoke("set_performance_mode", { mode });
      const info = await invoke<PerformanceModeInfo>("get_performance_mode");
      setModeInfo(info);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <>
      {error && <p className="error">{error}</p>}

      {profile && (
        <>
          <p className="setup-hint">Hardware detected on this device:</p>
          <p>
            {profile.cpuLogicalCores} CPU cores
            {profile.cpuBrand ? ` (${profile.cpuBrand})` : ""}
          </p>
          <p>{Math.round(profile.totalRamMb / 1024)} GB memory</p>
          <p>{formatGpu(profile)}</p>
          {formatStorage(profile) && <p className="setup-hint">Storage: {formatStorage(profile)}</p>}
        </>
      )}

      {modeInfo && (
        <p className="setup-hint">
          Hardware tier: <strong>{TIER_LABELS[modeInfo.detectedTier]}</strong>
        </p>
      )}

      {modeInfo?.hardwareRefreshed && (
        <p className="setup-hint">
          Hardware profile refreshed because this looks like different hardware than last time — performance
          mode was reset to Adaptive.
        </p>
      )}

      {modeInfo?.isBelowRecommended && (
        <p className="setup-hint">
          Your PC can run Smallbird, but your hardware is below our recommended configuration. Real-time
          transcription may be slower.
        </p>
      )}

      <div className="agent-mode-toggle">
        {MODE_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            className={`agent-mode-tab${modeInfo?.mode === opt.value ? " active" : ""}`}
            onClick={() => handleSelectMode(opt.value)}
            disabled={busy}
          >
            {opt.label}
          </button>
        ))}
      </div>
      {busy && (
        <p className="setup-hint">
          Applying — this restarts the local document search service and can take up to a minute...
        </p>
      )}
      {!busy && modeInfo && (
        <p className="setup-hint">{MODE_OPTIONS.find((o) => o.value === modeInfo.mode)?.description}</p>
      )}

      {modeInfo && (
        <>
          <button
            className="link-button"
            style={{ color: "var(--text-muted)" }}
            onClick={() => setShowAdvanced((v) => !v)}
          >
            {showAdvanced ? "Hide" : "Show"} advanced details
          </button>
          {showAdvanced && (
            <ul className="setup-hint">
              <li>STT threads: {modeInfo.effectiveConfig.sttNumThreads}</li>
              <li>RAG top-K: {modeInfo.effectiveConfig.ragTopK}</li>
              <li>RAG context budget: {modeInfo.effectiveConfig.ragMaxContextChars} chars</li>
              <li>RAG retrieval timeout: {modeInfo.effectiveConfig.ragRetrievalTimeoutMs}ms</li>
            </ul>
          )}
        </>
      )}
    </>
  );
}
