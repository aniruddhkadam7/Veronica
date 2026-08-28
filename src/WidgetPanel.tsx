import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { emit } from "@tauri-apps/api/event";
import {
  DEFAULT_WIDGET_SETTINGS,
  WIDGET_MOTION_MAX,
  WIDGET_MOTION_MIN,
  WIDGET_ORB_SIZE_MAX,
  WIDGET_ORB_SIZE_MIN,
  loadWidgetSettings,
  saveWidgetSettings,
  type WidgetSettings,
} from "./widgetSettings";
import { Button } from "./ui";

/// Customization for the floating widget orb (VeronicaWidget.tsx) — color,
/// size, drag, and motion tuning (speed/intensity). Lives in its own
/// localStorage store (widgetSettings.ts), separate from OverlaySettings,
/// since the widget is a different window with no other settings surface of
/// its own. Every change here is applied live via the
/// "veronica:widget-settings-changed" broadcast the widget window listens
/// for, so there's no separate "Apply" step.
export function WidgetPanel() {
  const [settings, setSettingsState] = useState<WidgetSettings>(() => loadWidgetSettings());

  const set = <K extends keyof WidgetSettings>(key: K, value: WidgetSettings[K]) => {
    const next = { ...settings, [key]: value };
    setSettingsState(next);
    saveWidgetSettings(next);
    emit("veronica:widget-settings-changed").catch(() => {
      // Best-effort — if the widget isn't open yet, it picks up the new
      // value from localStorage the next time it mounts regardless.
    });
  };

  const changeOrbSize = (orbSize: number) => {
    set("orbSize", orbSize);
    invoke("resize_veronica_widget", { orbSize }).catch(() => {
      // Best-effort — nothing to resize if the widget window isn't open yet.
    });
  };

  const resetDefaults = () => {
    setSettingsState({ ...DEFAULT_WIDGET_SETTINGS });
    saveWidgetSettings(DEFAULT_WIDGET_SETTINGS);
    invoke("resize_veronica_widget", { orbSize: DEFAULT_WIDGET_SETTINGS.orbSize }).catch(() => {});
    emit("veronica:widget-settings-changed").catch(() => {});
  };

  return (
    <div className="personalization-panel">
      <div className="personalization-row">
        <label htmlFor="widget-color-from">Orb color (start)</label>
        <input
          id="widget-color-from"
          type="color"
          className="widget-color-input"
          value={settings.colorFrom}
          onChange={(e) => set("colorFrom", e.target.value)}
        />
      </div>

      <div className="personalization-row">
        <label htmlFor="widget-color-to">Orb color (end)</label>
        <input
          id="widget-color-to"
          type="color"
          className="widget-color-input"
          value={settings.colorTo}
          onChange={(e) => set("colorTo", e.target.value)}
        />
      </div>

      <div className="personalization-row">
        <label htmlFor="widget-orb-size">Orb size</label>
        <input
          id="widget-orb-size"
          type="range"
          min={WIDGET_ORB_SIZE_MIN}
          max={WIDGET_ORB_SIZE_MAX}
          step={4}
          value={settings.orbSize}
          onChange={(e) => changeOrbSize(Number(e.target.value))}
        />
        <span className="widget-panel-value">{settings.orbSize}px</span>
      </div>

      <div className="personalization-row">
        <label htmlFor="widget-speed">Animation speed</label>
        <input
          id="widget-speed"
          type="range"
          min={WIDGET_MOTION_MIN}
          max={WIDGET_MOTION_MAX}
          step={0.1}
          value={settings.speed}
          onChange={(e) => set("speed", Number(e.target.value))}
        />
        <span className="widget-panel-value">{settings.speed.toFixed(1)}x</span>
      </div>

      <div className="personalization-row">
        <label htmlFor="widget-intensity">Motion intensity</label>
        <input
          id="widget-intensity"
          type="range"
          min={WIDGET_MOTION_MIN}
          max={WIDGET_MOTION_MAX}
          step={0.1}
          value={settings.intensity}
          onChange={(e) => set("intensity", Number(e.target.value))}
        />
        <span className="widget-panel-value">{settings.intensity.toFixed(1)}x</span>
      </div>
      <p className="setup-hint">
        Speed controls how fast the orb's breathing/thinking/listening motion cycles; intensity
        controls how big that motion is (bigger ripples and pulses), independent of speed.
      </p>

      <div className="personalization-row">
        <label htmlFor="widget-draggable">Draggable</label>
        <input
          id="widget-draggable"
          type="checkbox"
          checked={settings.dragEnabled}
          onChange={(e) => set("dragEnabled", e.target.checked)}
        />
      </div>
      <p className="setup-hint">
        When enabled, click and drag the floating orb anywhere on screen to reposition it.
      </p>

      <Button variant="ghost" onClick={resetDefaults}>
        Reset widget to defaults
      </Button>
    </div>
  );
}
