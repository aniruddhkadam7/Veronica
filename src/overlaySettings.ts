// Interview Mode overlay settings — persisted to localStorage so they
// survive between sessions without needing the backend or the large
// Record/Prepare dashboard. Kept in its own module (rather than inline in
// InterviewOverlay.tsx) so the shape is easy to share with the settings
// panel component and to unit-test in isolation.

export type AnswerLength = "brief" | "default" | "detailed";
export type ResponseStyle = "natural" | "technical" | "concise";
export type OverlayDensity = "compact" | "comfortable";
export type OverlaySize = "small" | "medium" | "large";
export type EnglishLevel = "simple" | "professional" | "advanced";
// The setup page's "Answer style" control — a distinct axis from
// ResponseStyle (which stays "natural" | "technical" | "concise" and is only
// ever set from the overlay's own Settings panel). This one maps onto
// ResponseStyle + AnswerLength together when Start Interview is pressed; see
// answerStyleToOverlayFields() below.
export type AnswerStyle = "natural" | "concise" | "detailed";
export type Humanization = "natural" | "conversational" | "formal";

/// One voice-launchable app: `name` is what you say ("open <name>"), `path`
/// is the exact command spawned (an exe on PATH, or a full path) — never
/// derived from the spoken command itself, only from what was typed here.
/// See voice_command::launch_app.
export interface VoiceLaunchApp {
  name: string;
  path: string;
}

export interface OverlaySettings {
  opacity: number; // 0.15 - 1.0 — alpha of the overlay's panel background
  size: OverlaySize;
  fontSize: number; // px, 12 - 20
  density: OverlayDensity;
  dragEnabled: boolean;
  sttSensitivity: number; // 0 - 100, informational — mirrors STT_END_SILENCE_MS intent
  answerLength: AnswerLength;
  responseStyle: ResponseStyle;
  role: string;
  company: string;
  jobDescription: string;
  englishLevel: EnglishLevel;
  answerStyle: AnswerStyle;
  humanization: Humanization;
  // When on, a completed interviewer question is sent to Ask AI automatically
  // a short debounce after the interviewer stops talking — see the
  // transcript:update listener in InterviewOverlay.tsx. Off by default so
  // existing behavior (manual Ask AI / Enter) is unchanged until opted in.
  autoAI: boolean;
  // Apps the push-to-talk voice command "open <name>" is allowed to launch —
  // configured here, not spoken, so a misheard command can never run
  // anything the user didn't explicitly allowlist. See VoiceLaunchApp.
  voiceLaunchApps: VoiceLaunchApp[];
}

export const DEFAULT_OVERLAY_SETTINGS: OverlaySettings = {
  opacity: 0.8,
  size: "large",
  fontSize: 14,
  density: "comfortable",
  dragEnabled: true,
  sttSensitivity: 50,
  answerLength: "default",
  responseStyle: "natural",
  role: "",
  company: "",
  jobDescription: "",
  englishLevel: "simple",
  answerStyle: "natural",
  humanization: "natural",
  autoAI: false,
  voiceLaunchApps: [],
};

/// Maps the setup page's single "Answer style" control onto the two
/// lower-level fields the backend already understands, so no new backend
/// axis is needed for it (see apps/backend/app/services/ask_service.py).
export function answerStyleToOverlayFields(style: AnswerStyle): {
  responseStyle: ResponseStyle;
  answerLength: AnswerLength;
} {
  switch (style) {
    case "concise":
      return { responseStyle: "concise", answerLength: "brief" };
    case "detailed":
      return { responseStyle: "natural", answerLength: "detailed" };
    case "natural":
    default:
      return { responseStyle: "natural", answerLength: "default" };
  }
}

/// Fraction of the primary monitor's shorter dimension the overlay window
/// occupies for each Small/Medium/Large choice. Shared between the settings
/// panel (which resizes on change) and the overlay's mount effect (which
/// applies the user's saved choice to a freshly created window) so the two
/// can never drift apart. Must track `DEFAULT_SIZE_FRACTION` in
/// `interview_mode/window.rs` for the "large" entry, since that Rust constant
/// is what a brand-new window uses before either of these ever runs.
export const SIZE_FRACTIONS: Record<OverlaySize, number> = {
  small: 0.45,
  medium: 0.6,
  large: 0.75,
};

const STORAGE_KEY = "interview-mode:overlay-settings";

// Bumped when a stored value would otherwise keep an old default alive.
//
// v2: the opacity slider used to bottom out at 0.4 and default to 0.82, back
// when the overlay also ran a backdrop blur and could never be properly
// see-through anyway.
// v3: default opacity moved to 0.8 and default size to "large" — a user who
// never touched either setting should land on the new defaults, not be stuck
// on whatever v1/v2 happened to write to disk.
// v4: added englishLevel/answerStyle/humanization (New Interview setup
// page's Answer Style controls) — purely additive, no migration needed since
// loadOverlaySettings() already merges stored settings over the defaults.
const SETTINGS_VERSION = 4;
const MIN_USABLE_OPACITY = 0.15;

export function loadOverlaySettings(): OverlaySettings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { ...DEFAULT_OVERLAY_SETTINGS };
    const parsed = JSON.parse(raw);
    // Merge over the defaults rather than trusting the stored shape outright,
    // so a settings schema added in a later version doesn't leave new fields
    // undefined for a user who saved settings under an older version.
    const merged: OverlaySettings = { ...DEFAULT_OVERLAY_SETTINGS, ...parsed };
    const storedVersion = parsed?.version ?? 1;

    if (storedVersion < 2) {
      merged.opacity = DEFAULT_OVERLAY_SETTINGS.opacity;
    }
    if (storedVersion < 3) {
      // Only move settings the user never touched — someone who deliberately
      // picked "small" or dragged opacity to 40% should keep that choice, not
      // get silently overridden by a change to what the defaults are.
      if (parsed?.opacity === 0.55) merged.opacity = DEFAULT_OVERLAY_SETTINGS.opacity;
      if (parsed?.size === "medium" || parsed?.size === undefined) {
        merged.size = DEFAULT_OVERLAY_SETTINGS.size;
      }
    }
    // Clamp regardless of provenance — a hand-edited or corrupted value of 0
    // would render an invisible overlay the user cannot find to fix.
    merged.opacity = Math.min(1, Math.max(MIN_USABLE_OPACITY, merged.opacity));

    return merged;
  } catch {
    return { ...DEFAULT_OVERLAY_SETTINGS };
  }
}

export function saveOverlaySettings(settings: OverlaySettings): void {
  try {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ ...settings, version: SETTINGS_VERSION }),
    );
  } catch {
    // localStorage unavailable (e.g. disabled) — settings simply won't
    // persist across restarts; not worth surfacing an error for.
  }
}
