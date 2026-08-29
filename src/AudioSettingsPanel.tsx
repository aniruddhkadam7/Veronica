import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Select, type SelectOption } from "./ui";

interface AudioDeviceInfo {
  id: string;
  name: string;
  isDefault: boolean;
  direction: "input" | "output";
}

interface SelectedDevicesInfo {
  inputId: string | null;
  outputId: string | null;
}

const SYSTEM_DEFAULT = "";

function deviceLabel(d: AudioDeviceInfo): string {
  return d.isDefault ? `${d.name} (default)` : d.name;
}

/// Lets the user pick which physical microphone Veronica listens through and
/// which speaker/output device her voice plays out of, instead of always
/// using whatever Windows currently has set as the default device. Selecting
/// "System default" (the initial state) restores the previous always-default
/// behavior. A change here only takes effect the next time the relevant
/// session starts (mic assistant / dictation for input, the next TTS answer
/// for output) — see selected_devices in src-tauri/src/state.rs — so this
/// panel does not attempt to hot-swap an already-open device.
export function AudioSettingsPanel() {
  const [inputDevices, setInputDevices] = useState<AudioDeviceInfo[]>([]);
  const [outputDevices, setOutputDevices] = useState<AudioDeviceInfo[]>([]);
  const [selected, setSelected] = useState<SelectedDevicesInfo>({ inputId: null, outputId: null });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refreshDevices = useCallback(async () => {
    setError(null);
    try {
      const [inputs, outputs, current] = await Promise.all([
        invoke<AudioDeviceInfo[]>("list_input_devices"),
        invoke<AudioDeviceInfo[]>("list_output_devices"),
        invoke<SelectedDevicesInfo>("get_selected_devices"),
      ]);
      setInputDevices(inputs);
      setOutputDevices(outputs);
      setSelected(current);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshDevices();
  }, [refreshDevices]);

  const handleInputChange = useCallback(async (value: string) => {
    const deviceId = value === SYSTEM_DEFAULT ? null : value;
    setSelected((cur) => ({ ...cur, inputId: deviceId }));
    try {
      await invoke("set_input_device", { deviceId });
    } catch (err) {
      setError(String(err));
    }
  }, []);

  const handleOutputChange = useCallback(async (value: string) => {
    // Output selection is matched by device name (see
    // src-tauri/src/tts/player.rs), not a WASAPI ID like input — the
    // option value below is the device's name, not its id.
    const deviceId = value === SYSTEM_DEFAULT ? null : value;
    setSelected((cur) => ({ ...cur, outputId: deviceId }));
    try {
      await invoke("set_output_device", { deviceId });
    } catch (err) {
      setError(String(err));
    }
  }, []);

  if (loading) {
    return <p className="setup-hint">Loading audio devices…</p>;
  }

  const inputOptions: SelectOption<string>[] = [
    { value: SYSTEM_DEFAULT, label: "System default" },
    ...inputDevices.map((d) => ({ value: d.id, label: deviceLabel(d) })),
  ];
  const outputOptions: SelectOption<string>[] = [
    { value: SYSTEM_DEFAULT, label: "System default" },
    ...outputDevices.map((d) => ({ value: d.name, label: deviceLabel(d) })),
  ];

  return (
    <div className="personalization-panel">
      {error && <p className="error">{error}</p>}

      <div className="personalization-row">
        <label htmlFor="audio-input-device">Microphone</label>
        <Select
          id="audio-input-device"
          className="setup-select"
          value={selected.inputId ?? SYSTEM_DEFAULT}
          options={inputOptions}
          onChange={handleInputChange}
        />
      </div>
      <p className="setup-hint">Which microphone Veronica listens through when she's activated.</p>

      <div className="personalization-row">
        <label htmlFor="audio-output-device">Speaker</label>
        <Select
          id="audio-output-device"
          className="setup-select"
          value={selected.outputId ?? SYSTEM_DEFAULT}
          options={outputOptions}
          onChange={handleOutputChange}
        />
      </div>
      <p className="setup-hint">Which speaker or headset Veronica's voice plays out of.</p>

      <p className="setup-hint">
        Changes apply the next time listening starts or Veronica speaks again — an already-active
        session keeps using the device it started with.
      </p>

      <button className="link-button" style={{ color: "var(--text-muted)" }} onClick={refreshDevices}>
        Refresh device list
      </button>
    </div>
  );
}
