// Veronica floating widget's own appearance settings — separate from
// overlaySettings.ts (which governs the full chat overlay window) since the
// widget is a different window with a different, much smaller surface (just
// the orb, no chat UI). Persisted to localStorage the same way, so both
// settings modules can be read/written independently without either one's
// version-migration logic touching the other's stored key.

export interface WidgetSettings {
  colorFrom: string; // hex, orb gradient start
  colorTo: string; // hex, orb gradient end
  orbSize: number; // px, 60 - 220 — passed straight to ParticlesOrb's `size`
  dragEnabled: boolean; // whether the whole widget window can be dragged
  // Motion tuning, passed straight to ParticlesOrb's `speed`/`intensity` —
  // `speed` controls how fast each state's animation cycles, `intensity`
  // controls how far the particles actually move (bigger ripples/pulses),
  // independently. Both apply uniformly across idle/listening/thinking/
  // speaking rather than as separate per-state knobs, since a single pair of
  // sliders is what "make the breathing/thinking motion bigger or calmer"
  // actually needs — per-state sliders would be four times the UI for
  // control most users won't reach for.
  speed: number; // 0.4 - 2.5
  intensity: number; // 0.4 - 2.5
}

export const DEFAULT_WIDGET_SETTINGS: WidgetSettings = {
  colorFrom: "#f0abfc",
  colorTo: "#818cf8",
  orbSize: 160,
  dragEnabled: true,
  speed: 1,
  intensity: 1,
};

export const WIDGET_ORB_SIZE_MIN = 60;
export const WIDGET_ORB_SIZE_MAX = 220;
export const WIDGET_MOTION_MIN = 0.4;
export const WIDGET_MOTION_MAX = 2.5;

const STORAGE_KEY = "veronica:widget-settings";

export function loadWidgetSettings(): WidgetSettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_WIDGET_SETTINGS };
    const parsed = JSON.parse(raw);
    const merged: WidgetSettings = { ...DEFAULT_WIDGET_SETTINGS, ...parsed };
    merged.orbSize = Math.min(WIDGET_ORB_SIZE_MAX, Math.max(WIDGET_ORB_SIZE_MIN, merged.orbSize));
    merged.speed = Math.min(WIDGET_MOTION_MAX, Math.max(WIDGET_MOTION_MIN, merged.speed));
    merged.intensity = Math.min(WIDGET_MOTION_MAX, Math.max(WIDGET_MOTION_MIN, merged.intensity));
    return merged;
  } catch {
    return { ...DEFAULT_WIDGET_SETTINGS };
  }
}

export function saveWidgetSettings(settings: WidgetSettings): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // localStorage unavailable — settings simply won't persist; not worth
    // surfacing an error for, matching overlaySettings.ts's same choice.
  }
}
