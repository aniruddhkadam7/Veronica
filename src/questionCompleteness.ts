// Shared "is this buffered question finished, or still being spoken" text
// heuristic, used by both Interview Mode's Auto AI and Custom Agents' Auto AI
// so the two behave identically and never drift apart. No LLM call — purely
// a cheap surface-text classifier.

/// How long the speaker must stay silent (no new partial/final STT text)
/// before Auto AI treats the buffered question as complete and sends it, once
/// the text itself looks finished.
export const AUTO_AI_SILENCE_MS_COMPLETE = 900;

/// Silence window used instead of AUTO_AI_SILENCE_MS_COMPLETE when the
/// buffered text looks like it trails off mid-clause (ends on "and", a
/// comma, etc.).
export const AUTO_AI_SILENCE_MS_INCOMPLETE = 2200;

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
