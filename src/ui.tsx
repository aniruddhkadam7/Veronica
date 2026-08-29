import { useEffect, useLayoutEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";
import { listen } from "@tauri-apps/api/event";
import type { TranscriptSegment } from "./types";

type ButtonVariant = "primary" | "secondary" | "danger" | "ghost";
type ButtonSize = "md" | "sm";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

export function Button({ variant = "secondary", size = "md", className, ...rest }: ButtonProps) {
  const classes = ["btn", variant, size === "sm" ? "sm" : "", className]
    .filter(Boolean)
    .join(" ");
  return <button className={classes} {...rest} />;
}

interface CardProps {
  title?: ReactNode;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function Card({ title, actions, className, children }: CardProps) {
  const classes = ["card", className].filter(Boolean).join(" ");
  return (
    <section className={classes}>
      {(title || actions) && (
        <div className="card-header">
          {title && <h2 className="card-title">{title}</h2>}
          {actions && <div className="card-actions">{actions}</div>}
        </div>
      )}
      {children}
    </section>
  );
}

type Tone = "neutral" | "success" | "warning" | "danger" | "accent";

interface StatusDotProps {
  tone: Tone;
  pulse?: boolean;
}

export function StatusDot({ tone, pulse }: StatusDotProps) {
  const classes = ["status-dot", tone, pulse ? "pulse" : ""].filter(Boolean).join(" ");
  return <span className={classes} />;
}

interface BadgeProps {
  tone?: Tone;
  children: ReactNode;
}

export function Badge({ tone = "neutral", children }: BadgeProps) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

interface EmptyStateProps {
  icon?: ReactNode;
  title: string;
  description?: string;
}

export function EmptyState({ icon, title, description }: EmptyStateProps) {
  return (
    <div className="empty-state">
      {icon && <div className="empty-state-icon">{icon}</div>}
      <p className="empty-state-title">{title}</p>
      {description && <p className="empty-state-description">{description}</p>}
    </div>
  );
}

interface SkeletonProps {
  width?: string;
  height?: string;
}

export function Skeleton({ width = "100%", height = "1em" }: SkeletonProps) {
  return <span className="skeleton" style={{ width, height }} />;
}

interface SpinnerProps {
  size?: number;
  className?: string;
}

/// A small rotating ring, for inline use next to a "Starting…"/"Loading…"
/// label — e.g. inside a disabled button while an async action is in flight.
export function Spinner({ size = 14, className }: SpinnerProps) {
  const classes = ["spinner", className].filter(Boolean).join(" ");
  return (
    <span
      className={classes}
      style={{ width: size, height: size }}
      role="status"
      aria-label="Loading"
    />
  );
}

interface AudioLevelBarsProps {
  active: boolean;
  className?: string;
}

/// Three-bar "listening" indicator next to the overlay's status text. Always
/// bounces at a fixed fast/high animation while `active` — it's a liveness
/// cue for "STT is running", not a real volume meter, so it stays equally
/// visible through quiet stretches of the *current* utterance instead of
/// going flat between words. Callers gate `active` on `useSttSpeaking()` so it
/// stops moving once STT itself has gone idle (no speech to transcribe),
/// rather than running continuously for the whole session.
export function AudioLevelBars({ active, className }: AudioLevelBarsProps) {
  if (!active) return null;
  const classes = ["audio-level-bars", className].filter(Boolean).join(" ");
  return (
    <span className={classes} role="img" aria-label="Listening">
      <span className="audio-level-bar" />
      <span className="audio-level-bar" />
      <span className="audio-level-bar" />
    </span>
  );
}

/// How long to keep the listening animation running after the last
/// transcript event before treating STT as idle again. Long enough to
/// bridge the gap between a partial and its following final for the same
/// utterance, short enough that the animation visibly stops during real
/// silence rather than appearing to run all the time.
const STT_IDLE_TIMEOUT_MS = 1200;

/// True only while STT is actively producing transcript output (partial or
/// final segments), going false again after `STT_IDLE_TIMEOUT_MS` of no new
/// segments. This is what should gate the `AudioLevelBars` animation — capture
/// being started is not the same as speech currently being transcribed.
export function useSttSpeaking(): boolean {
  const [speaking, setSpeaking] = useState(false);
  const idleTimerRef = useRef<number | null>(null);

  useEffect(() => {
    const unlisten = listen<TranscriptSegment>("transcript:update", (event) => {
      const segment = event.payload;
      if (!segment.partial_text && !segment.final_text) return;
      setSpeaking(true);
      if (idleTimerRef.current) window.clearTimeout(idleTimerRef.current);
      idleTimerRef.current = window.setTimeout(() => {
        idleTimerRef.current = null;
        setSpeaking(false);
      }, STT_IDLE_TIMEOUT_MS);
    });
    return () => {
      unlisten.then((fn) => fn());
      if (idleTimerRef.current) window.clearTimeout(idleTimerRef.current);
    };
  }, []);

  return speaking;
}


export interface SelectOption<T extends string> {
  value: T;
  label: ReactNode;
}

interface SelectProps<T extends string> {
  value: T;
  options: SelectOption<T>[];
  onChange: (value: T) => void;
  className?: string;
  id?: string;
  disabled?: boolean;
  "aria-label"?: string;
  /// Replaces an option's default `{opt.label}` row content when provided —
  /// e.g. the Settings -> Audio voice picker uses this to add a per-voice
  /// preview button next to the label. Receives the option and must still
  /// render something that fills the row; the row's own click-to-select and
  /// keyboard/highlight behavior is unaffected either way; a nested
  /// interactive element (like a preview button) should stop propagation
  /// itself if it must not also trigger selection.
  renderOption?: (opt: SelectOption<T>) => ReactNode;
}

/// Custom dropdown that replaces bare `<select>` elements app-wide, so every
/// dropdown shares one look-and-feel (trigger button + floating list of
/// `.dropdown-item`s) instead of the OS's native select styling, which can't
/// be themed to match the rest of the UI (dark overlay panels especially —
/// native select popups render with the OS's light/dark theme, not ours).
/// Callers pass a `className` (e.g. "select-overlay") to pick up per-surface
/// theming; base look lives in the shared `.select-*` rules in App.css.
export function Select<T extends string>({
  value,
  options,
  onChange,
  className,
  id,
  disabled,
  "aria-label": ariaLabel,
  renderOption,
}: SelectProps<T>) {
  const [open, setOpen] = useState(false);
  const [highlighted, setHighlighted] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const selectedIndex = Math.max(0, options.findIndex((o) => o.value === value));
  const selected = options[selectedIndex];

  useLayoutEffect(() => {
    if (open) setHighlighted(selectedIndex);
  }, [open, selectedIndex]);

  useEffect(() => {
    if (!open) return;
    const handlePointerDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", handlePointerDown, true);
    return () => document.removeEventListener("mousedown", handlePointerDown, true);
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const el = listRef.current?.children[highlighted] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest" });
  }, [open, highlighted]);

  const commit = (index: number) => {
    const opt = options[index];
    if (!opt) return;
    onChange(opt.value);
    setOpen(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    switch (e.key) {
      case "Enter":
      case " ":
        e.preventDefault();
        if (open) commit(highlighted);
        else setOpen(true);
        break;
      case "Escape":
        if (open) {
          e.preventDefault();
          setOpen(false);
        }
        break;
      case "ArrowDown":
        e.preventDefault();
        if (!open) setOpen(true);
        else setHighlighted((i) => Math.min(options.length - 1, i + 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        if (!open) setOpen(true);
        else setHighlighted((i) => Math.max(0, i - 1));
        break;
      case "Tab":
        setOpen(false);
        break;
    }
  };

  const classes = ["select", className].filter(Boolean).join(" ");

  return (
    <div className={classes} ref={rootRef}>
      <button
        type="button"
        id={id}
        className="select-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={handleKeyDown}
      >
        <span className="select-trigger-label">{selected?.label ?? value}</span>
        <span className="select-trigger-caret" aria-hidden="true">▾</span>
      </button>
      {open && (
        <div className="select-menu" role="listbox" ref={listRef}>
          {options.map((opt, i) => (
            <button
              key={opt.value}
              type="button"
              role="option"
              aria-selected={opt.value === value}
              className={[
                "dropdown-item",
                opt.value === value ? "active" : "",
                i === highlighted ? "highlighted" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              onMouseEnter={() => setHighlighted(i)}
              onClick={() => commit(i)}
            >
              {renderOption ? renderOption(opt) : opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
