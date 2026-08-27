import { useLayoutEffect, useRef } from "react";

// `.header-dropdown`'s `top` is set relative to `.dropdown-anchor`
// (`position: relative`, wrapping just the trigger button), which itself
// already sits inside the header row — not relative to the window's own
// top edge. So this is only the 56px header's own height, not
// header+title-bar; the 32px custom title bar (App.css's .title-bar)
// above the header is already "baked in" to the anchor's own position and
// must not be added again here.
const HEADER_HEIGHT = 56;
// Small gap so the dropdown reads as anchored to the header rather than
// touching it edge-to-edge.
const GAP_ABOVE = 2;

/// Anchored dropdown/popover shell for the compact header's Mode/Context/
/// Settings/Account buttons. Purely an absolutely positioned overlay (see
/// `.header-dropdown` in App.css: `position: absolute`, anchored to
/// `.dropdown-anchor`) — it renders into space the main window already has
/// (the window is a fixed 760x720 from launch; see main_window.rs), so
/// opening/closing one of these never resizes, repositions, or reflows the
/// OS window or any sibling element. An earlier version measured its own
/// height and asked the Rust side to grow the real window to fit, which
/// reliably read as the whole window jittering on every click; that's gone
/// now in favor of this being a plain floating layer, like any in-page
/// dropdown.
export function HeaderDropdown({
  onClose,
  children,
  className,
}: {
  onClose: () => void;
  children: React.ReactNode;
  className?: string;
}) {
  const outerRef = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const handlePointerDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (outerRef.current && outerRef.current.contains(target)) return;
      // A click on the anchor button that owns this dropdown is handled by
      // that button's own onClick (which toggles the popover shut) — if
      // this capture-phase listener also called onClose() here, mousedown
      // would close it a beat before the click's toggle logic reopens it,
      // so the second tap on the button re-opened the dropdown instead of
      // closing it.
      if (outerRef.current?.parentElement?.contains(target)) return;
      onClose();
    };
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    // Capture phase so this still sees the click even if a header button's
    // own onClick (e.g. re-opening a different popover) stops propagation.
    document.addEventListener("mousedown", handlePointerDown, true);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handlePointerDown, true);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [onClose]);

  return (
    <div
      ref={outerRef}
      className={["header-dropdown", className].filter(Boolean).join(" ")}
      style={{ top: HEADER_HEIGHT + GAP_ABOVE }}
    >
      {children}
    </div>
  );
}
