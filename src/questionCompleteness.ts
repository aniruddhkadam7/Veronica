// Shared "is this buffered question finished, or still being spoken" text
// heuristic, used by both Interview Mode's Auto AI and Custom Agents' Auto AI
// so the two behave identically and never drift apart. No LLM call — purely
// a cheap surface-text classifier.
//
// This used to gate a fixed post-transcription wait (900ms if the text
// looked complete, 2200ms if it looked like it trailed off) applied after
// EVERY question before ever calling ask_veronica. That's gone: a `Final`
// transcript segment IS the real turn-completion signal (the local VAD
// already decided the utterance ended — see stt/sidecar.rs), so a
// "complete"-looking Final is sent immediately, with no added wait at all.
// `classifyQuestionCompleteness` is now used as a content gate, not a timer
// gate: an "incomplete"-looking Final (trails off on "and", a dangling
// comma, mid-clause) is held and merged with whatever the next Final adds,
// since sending a bare fragment straight to the fast router risks it
// matching a real (and for something like a window-close, irreversible)
// command out of context. `SAFETY_NET_MS` below is only a fallback for the
// case the speaker trails off and never continues — not the primary
// mechanism, and far shorter than the old worst case.

/// Only used if the buffered text still looks "incomplete" with nothing
/// more arriving — e.g. the speaker trailed off mid-thought and moved on.
/// Sends whatever's buffered rather than holding it forever. Not consulted
/// at all for text that already looks complete, which sends with zero
/// added delay.
export const SAFETY_NET_MS = 1500;

/// Words that, at the very end of the buffered question, mean the sentence is
/// still being assembled: conjunctions, prepositions, and articles that
/// almost never end a real spoken question. Checked against the last word
/// only, case-insensitively — this is a cheap surface heuristic, not grammar
/// parsing.
const TRAILING_INCOMPLETE_WORDS = new Set([
  "and",
  "or",
  "but",
  "because",
  "so",
  "with",
  "for",
  "like",
  "if",
  "that",
  "which",
  "to",
  "of",
  "in",
  "on",
  "the",
  "a",
  "an",
  "is",
  "are",
  "was",
  "were",
]);

const TOPIC_SHIFT_OPENERS = ["so", "okay so", "now", "alright"];

/// Classifies whether the buffered question reads as finished or as still
/// being assembled, purely from surface text. Biased towards "complete": the
/// cost of guessing wrong is a slightly early send, not a wrong answer.
export function classifyQuestionCompleteness(text: string): "complete" | "incomplete" {
  const trimmed = text.trim();
  if (!trimmed) return "incomplete";

  if (trimmed.endsWith(",")) return "incomplete";

  const words = trimmed.replace(/[.?!]+$/, "").split(/\s+/);
  const lastWord = words[words.length - 1]?.toLowerCase().replace(/[^a-z']/g, "");
  if (lastWord && TRAILING_INCOMPLETE_WORDS.has(lastWord)) return "incomplete";

  const lowered = trimmed.toLowerCase();
  const startsWithShift = TOPIC_SHIFT_OPENERS.some(
    (opener) => lowered === opener || lowered.startsWith(`${opener} `) || lowered.startsWith(`${opener},`),
  );
  if (startsWithShift && words.length <= 3) return "incomplete";

  return "complete";
}

/// Appends one chunk of speech to another with exactly one separating space.
export function joinSpeech(existing: string, next: string): string {
  const left = existing.trimEnd();
  const right = next.trim();
  if (!left) return right;
  if (!right) return left;
  return `${left} ${right}`;
}
