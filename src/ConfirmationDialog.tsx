import type { PendingConfirmation } from "./useConfirmation";

interface Props {
  pending: PendingConfirmation;
  onRespond: (approved: boolean) => void;
}

/// Visual fallback for a pending Sensitive/Destructive confirmation — the
/// primary path is voice (Veronica speaks `detail` and the user answers
/// yes/no on the next turn), this is the on-screen backup for a silent
/// environment or a user who'd rather click. Styled by `risk` (destructive
/// gets the stronger warning color) so the two tiers read differently at a
/// glance. Renders inline in the same small overlay window, matching
/// OverlaySettingsPanel's pattern rather than a separate window/modal.
export function ConfirmationDialog({ pending, onRespond }: Props) {
  return (
    <div className={`overlay-confirm-dialog overlay-confirm-dialog-${pending.risk}`} role="alertdialog" aria-live="assertive">
      <p className="overlay-confirm-dialog-title">{pending.summary}</p>
      <p className="overlay-confirm-dialog-body">{pending.detail}</p>
      <div className="overlay-confirm-dialog-actions">
        <button className="overlay-text-button" onClick={() => onRespond(false)}>
          No
        </button>
        <button
          className={`overlay-text-button primary${pending.risk === "destructive" ? " destructive" : ""}`}
          onClick={() => onRespond(true)}
        >
          Yes
        </button>
      </div>
    </div>
  );
}
