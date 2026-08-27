import { Button } from "./ui";

/// Personal build: there is no Smallbird Cloud account, sign-in, or sync —
/// this panel exists only to explain that and point at the API Keys
/// settings, matching the shape of the original Account panel this was
/// split from.
export function Account({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="popover-overlay"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="popover" role="dialog" aria-modal="true" aria-label="Account">
        <div className="popover-header">
          <span className="setup-section-label">Account</span>
          <button className="modal-close-btn" onClick={onClose} title="Close" aria-label="Close">
            ✕
          </button>
        </div>

        <div className="popover-body">
          <p className="setup-hint">
            This is a personal build — it uses your own API key directly (see Settings → API Keys) and
            works fully offline. No sign-in or subscription needed.
          </p>

          <div className="popover-footer">
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
