import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Button } from "./ui";

const DISMISSED_KEY = "smallbird-low-end-hardware-banner-dismissed";

interface PerformanceModeInfo {
  isBelowRecommended: boolean;
}

/// One-time, non-blocking notice for machines that scored into the lowest
/// hardware tier (Entry) — mirrors UpdateBanner's shape (session-level
/// invoke check, small dismissible banner), but the dismissal persists
/// across launches via localStorage: unlike an update, "this hardware is
/// below recommended" doesn't change from one launch to the next, so
/// re-showing it every session would just be repetitive nagging on a
/// machine the user already knows is modest. Smallbird always starts and
/// runs regardless of this notice — see hardware::manager's module doc —
/// this is purely informational.
export function LowEndHardwareBanner() {
  const [belowRecommended, setBelowRecommended] = useState(false);
  const [dismissed, setDismissed] = useState(() => localStorage.getItem(DISMISSED_KEY) === "1");

  useEffect(() => {
    invoke<PerformanceModeInfo>("get_performance_mode")
      .then((info) => setBelowRecommended(info.isBelowRecommended))
      .catch(() => {
        // Hardware detection is best-effort by design (see
        // hardware::profile) — if the command itself is unavailable for
        // some reason, just stay silent rather than surfacing an error for
        // a purely informational notice.
      });
  }, []);

  const dismiss = () => {
    localStorage.setItem(DISMISSED_KEY, "1");
    setDismissed(true);
  };

  if (!belowRecommended || dismissed) return null;

  return (
    <div className="update-banner">
      <span>Your PC can run Smallbird, but your hardware is below our recommended configuration. Real-time transcription may be slower.</span>
      <div className="update-banner-actions">
        <Button variant="ghost" size="sm" onClick={dismiss}>
          Got it
        </Button>
      </div>
    </div>
  );
}
