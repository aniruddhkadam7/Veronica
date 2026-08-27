// Minimal monochrome line icons for the compact header — plain currentColor
// strokes so they inherit whatever text color the surrounding button uses
// (including hover/disabled states), rather than baked-in emoji glyphs that
// render inconsistently across platforms and read as decorative rather than
// as standard desktop-software UI chrome.

interface IconProps {
  size?: number;
  className?: string;
}

const BASE_PROPS = {
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function IconAttachment({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <path d="M16.5 6.5 9 14a3 3 0 0 0 4.24 4.24l7.07-7.07a5 5 0 0 0-7.07-7.07L6 11.34a7 7 0 0 0 9.9 9.9" />
    </svg>
  );
}

// Feather Icons' "settings" glyph (MIT licensed, feathericons.com) — a real
// authored gear outline rather than a freehand approximation, so the tooth
// spacing and proportions are actually correct.
export function IconSettings({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z" />
    </svg>
  );
}

export function IconAccount({ size = 16, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <circle cx="12" cy="8" r="3.5" />
      <path d="M4.5 20c1.2-3.6 4.2-5.5 7.5-5.5s6.3 1.9 7.5 5.5" />
    </svg>
  );
}

export function IconChevronDown({ size = 14, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

export function IconClose({ size = 14, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <path d="M6 6l12 12M18 6 6 18" />
    </svg>
  );
}

export function IconMinimize({ size = 14, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <path d="M5 12h14" />
    </svg>
  );
}

export function IconMaximize({ size = 14, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <rect x="5.5" y="5.5" width="13" height="13" rx="1.5" />
    </svg>
  );
}

export function IconRestore({ size = 14, className }: IconProps) {
  return (
    <svg width={size} height={size} className={className} {...BASE_PROPS} aria-hidden="true">
      <rect x="7.5" y="4" width="12.5" height="12.5" rx="1.5" />
      <path d="M16 8H5.5A1.5 1.5 0 0 0 4 9.5V20a1.5 1.5 0 0 0 1.5 1.5H16a1.5 1.5 0 0 0 1.5-1.5V9.5A1.5 1.5 0 0 0 16 8Z" />
    </svg>
  );
}
