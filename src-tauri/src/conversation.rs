//! The one shared conversation history — the single source of truth both
//! the floating widget (`veronica_widget`) and the full overlay
//! (`veronica_window`) read from and write into. They are separate
//! WebView2 windows with no shared JS state (see `overlay_window`'s doc),
//! so "the widget and overlay show the same live conversation" can only be
//! true if the conversation itself lives here, in the one Rust process
//! both windows talk to — never duplicated as parallel React state in each
//! window.
//!
//! Turn-id-keyed and append-only for the lifetime of one session (cleared
//! only by `ConversationStore::reset`, called when the user explicitly
//! deactivates Veronica via Stop — see `veronica_widget::hide_veronica_widget`
//! doc). Closing/hiding the overlay never calls `reset`: the conversation
//! keeps living and growing in the backend regardless of which window (if
//! any) is currently showing it.
//!
//! In-memory only, matching every other piece of session state in this app
//! (`AppState.transcript`, `AppState.working_state`) — no new persistence
//! layer, per the requirement not to add unnecessary storage. An app
//! restart loses the conversation exactly like it already loses the
//! transcript and working state, which is the existing, established
//! behavior this deliberately does not change.

use serde::Serialize;

/// Mirrors the frontend's `TurnStatus` (see VeronicaOverlay.tsx) exactly,
/// so a hydrated turn renders identically to one built live from events —
/// the overlay's `applyTurnEvent` machinery doesn't need to know whether a
/// turn arrived via hydration or a live stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum TurnStatus {
    Thinking,
    Streaming,
    Complete,
    Interrupted,
    Error,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConversationTurn {
    pub id: String,
    pub question: String,
    pub answer: String,
    pub status: TurnStatus,
    pub created_at_ms: u64,
    pub completed_at_ms: Option<u64>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// The shared store itself — held behind one `Mutex` in `AppState`, exactly
/// like `AppState.transcript`. Every mutating method is idempotent/keyed by
/// `turn_id`, so whichever window's `ask_veronica` call originated a turn,
/// and however many times its events replay, the stored turn is never
/// duplicated — the requirement to "preserve existing turn IDs and prevent
/// duplicates" holds structurally, not by caller discipline.
#[derive(Debug, Default)]
pub struct ConversationStore {
    turns: Vec<ConversationTurn>,
}

impl ConversationStore {
    /// Adds a new user turn if `turn_id` isn't already present — a no-op,
    /// not a duplicate, if it is (e.g. a retried/duplicate `ask_veronica`
    /// invocation for a turn id already recorded). Called once, right when
    /// a turn is created, before any answer content exists for it.
    pub fn create_turn(&mut self, turn_id: &str, question: &str) {
        if self.turns.iter().any(|t| t.id == turn_id) {
            return;
        }
        self.turns.push(ConversationTurn {
            id: turn_id.to_string(),
            question: question.to_string(),
            answer: String::new(),
            status: TurnStatus::Thinking,
            created_at_ms: now_ms(),
            completed_at_ms: None,
        });
    }

    /// Appends one streamed delta to the named turn's answer — a no-op if
    /// the turn is unknown (defensive; should not happen since
    /// `create_turn` always runs first) or already terminal (mirrors the
    /// frontend's `applyTurnEvent`: a completed/interrupted/errored turn
    /// can never be modified by a later event for the same id).
    pub fn append_delta(&mut self, turn_id: &str, delta: &str) {
        if let Some(turn) = self.turn_mut_if_active(turn_id) {
            turn.status = TurnStatus::Streaming;
            turn.answer.push_str(delta);
        }
    }

    /// Finalizes a turn: `answer`, when non-empty, is authoritative (mirrors
    /// the frontend's own "prefer the completed answer where non-empty"
    /// rule for the exact same reason — accumulated deltas could have
    /// dropped one). `cancelled` distinguishes an interrupted/superseded
    /// turn from one that legitimately finished (possibly with an empty
    /// answer, which is treated as an error).
    pub fn complete_turn(&mut self, turn_id: &str, answer: &str, cancelled: bool) {
        if let Some(turn) = self.turn_mut_if_active(turn_id) {
            if !answer.is_empty() {
                turn.answer = answer.to_string();
            }
            turn.status = if cancelled {
                TurnStatus::Interrupted
            } else if !turn.answer.trim().is_empty() {
                TurnStatus::Complete
            } else {
                TurnStatus::Error
            };
            turn.completed_at_ms = Some(now_ms());
        }
    }

    fn turn_mut_if_active(&mut self, turn_id: &str) -> Option<&mut ConversationTurn> {
        self.turns.iter_mut().find(|t| t.id == turn_id && !is_terminal(t.status))
    }

    /// The full conversation so far, oldest first — exactly what a
    /// newly-opened overlay hydrates from instead of starting empty. Turns
    /// interrupted with no real answer are dropped, matching the overlay's
    /// own render-time filter (see VeronicaOverlay.tsx): a superseded turn
    /// that never produced content isn't a real exchange worth showing.
    pub fn snapshot(&self) -> Vec<ConversationTurn> {
        self.turns.iter().filter(|t| !(t.status == TurnStatus::Interrupted && t.answer.trim().is_empty())).cloned().collect()
    }

    /// Completed (question, answer) pairs, oldest first — what `ask_veronica`
    /// sends the LLM as conversational context for follow-ups ("what about
    /// that?"). Derived from this SAME shared store rather than trusted from
    /// whichever window's own client-side history the caller happened to
    /// pass, so a follow-up asked through the widget correctly resolves
    /// against something said through the overlay a moment earlier, and vice
    /// versa — the two windows have no other state in common to get this
    /// right from on their own.
    pub fn completed_history(&self) -> Vec<(String, String)> {
        self.turns
            .iter()
            .filter(|t| t.status == TurnStatus::Complete && !t.answer.trim().is_empty())
            .map(|t| (t.question.clone(), t.answer.clone()))
            .collect()
    }

    /// Clears the whole conversation — called only when the user explicitly
    /// ends the session (Stop), never by opening/closing the overlay. See
    /// this module's doc.
    pub fn reset(&mut self) {
        self.turns.clear();
    }
}

fn is_terminal(status: TurnStatus) -> bool {
    matches!(status, TurnStatus::Complete | TurnStatus::Interrupted | TurnStatus::Error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_turn_then_snapshot_shows_it_thinking() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "hi");
        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].question, "hi");
        assert_eq!(snap[0].status, TurnStatus::Thinking);
    }

    #[test]
    fn duplicate_create_turn_is_a_no_op() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "hi");
        store.create_turn("t1", "hi again — should not replace");
        assert_eq!(store.snapshot().len(), 1);
        assert_eq!(store.snapshot()[0].question, "hi");
    }

    #[test]
    fn deltas_accumulate_and_mark_streaming() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "hi");
        store.append_delta("t1", "Hello");
        store.append_delta("t1", " there");
        let snap = store.snapshot();
        assert_eq!(snap[0].answer, "Hello there");
        assert_eq!(snap[0].status, TurnStatus::Streaming);
    }

    #[test]
    fn complete_turn_prefers_authoritative_answer_when_non_empty() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "hi");
        store.append_delta("t1", "partial");
        store.complete_turn("t1", "Hello there.", false);
        let snap = store.snapshot();
        assert_eq!(snap[0].answer, "Hello there.");
        assert_eq!(snap[0].status, TurnStatus::Complete);
        assert!(snap[0].completed_at_ms.is_some());
    }

    #[test]
    fn complete_turn_with_empty_answer_and_not_cancelled_is_an_error() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "hi");
        store.complete_turn("t1", "", false);
        assert_eq!(store.snapshot()[0].status, TurnStatus::Error);
    }

    #[test]
    fn cancelled_turn_with_no_real_answer_is_dropped_from_the_snapshot() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "stale question");
        store.complete_turn("t1", "", true);
        assert!(store.snapshot().is_empty());
    }

    #[test]
    fn cancelled_turn_that_already_streamed_real_text_is_kept() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "long question");
        store.append_delta("t1", "partial but real answer");
        store.complete_turn("t1", "", true);
        let snap = store.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].status, TurnStatus::Interrupted);
        assert_eq!(snap[0].answer, "partial but real answer");
    }

    #[test]
    fn a_terminal_turn_can_never_be_modified_by_a_later_event() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "hi");
        store.complete_turn("t1", "done", false);
        store.append_delta("t1", " more text that must be ignored");
        assert_eq!(store.snapshot()[0].answer, "done");
    }

    #[test]
    fn two_turns_stay_independent_and_chronological() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "first question");
        store.complete_turn("t1", "first answer", false);
        store.create_turn("t2", "second question");
        store.complete_turn("t2", "second answer", false);
        let snap = store.snapshot();
        assert_eq!(snap.len(), 2);
        assert_eq!(snap[0].question, "first question");
        assert_eq!(snap[1].question, "second question");
    }

    #[test]
    fn completed_history_only_includes_finished_non_empty_turns() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "first question");
        store.complete_turn("t1", "first answer", false);
        store.create_turn("t2", "still thinking question");
        store.create_turn("t3", "errored question");
        store.complete_turn("t3", "", false);
        let history = store.completed_history();
        assert_eq!(history, vec![("first question".to_string(), "first answer".to_string())]);
    }

    #[test]
    fn reset_clears_everything() {
        let mut store = ConversationStore::default();
        store.create_turn("t1", "hi");
        store.complete_turn("t1", "done", false);
        store.reset();
        assert!(store.snapshot().is_empty());
    }
}
