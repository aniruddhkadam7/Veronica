import { classifyQuestionCompleteness } from "./questionCompleteness";

/// Composes `questionCompleteness.ts`'s single-purpose "is this text
/// finished" classifier with the other signals a real turn-boundary
/// decision needs: is the whole buffered utterance nothing but filler, is
/// speech still actively arriving right now, and (for the narrow TTS-tail
/// self-hearing case) did this text arrive suspiciously soon after
/// Veronica's own voice stopped. Kept in its own module rather than growing
/// questionCompleteness.ts in place — these are different concerns
/// (whole-utterance classification vs. continuation-vs-fresh-utterance vs.
/// temporal gating) that both `useAutoAsk` and its consumers need composed
/// together, and questionCompleteness.ts's own logic stays independently
/// diffable/tested.

/// Words that, alone or repeated, mean the speaker hasn't said anything
/// worth acting on yet — not a trailing-word check like
/// `TRAILING_INCOMPLETE_WORDS` (which only looks at the LAST word of a
/// longer utterance): this is a WHOLE-UTTERANCE check. "yeah I think so"
/// must NOT classify as filler-only (real content is present); only an
/// utterance made ENTIRELY of these words does.
const FILLER_WORDS = new Set([
  "um",
  "umm",
  "uh",
  "uhh",
  "erm",
  "hmm",
  "hm",
  "okay",
  "ok",
  "yeah",
  "yep",
  "yup",
  "right",
  "mhm",
  "mm",
]);

/// True only if EVERY word in `text` (after stripping surrounding
/// punctuation) is a filler word — a bare "um", "okay", or "um, yeah", but
/// never a real request that merely starts or ends with one.
export function isFillerOnly(text: string): boolean {
  const trimmed = text.trim();
  if (!trimmed) return true;
  const words = trimmed
    .replace(/[.?!,]+$/, "")
    .split(/\s+/)
    .map((w) => w.toLowerCase().replace(/[^a-z']/g, ""))
    .filter((w) => w.length > 0);
  if (words.length === 0) return true;
  return words.every((w) => FILLER_WORDS.has(w));
}

/// A closed list of verbs that structurally need an object/target — "Can you
/// open" or "please create" with nothing after them isn't a finished
/// request, but no SINGLE trailing word signals that (the word-set approach
/// in questionCompleteness.ts only ever looks at the last word).
const OBJECT_NEEDING_VERBS = new Set(["perform", "do", "make", "create", "write", "open", "get", "find", "give", "take", "need", "want", "use"]);

/// Bare quantifier phrases that almost always introduce a noun that hasn't
/// arrived yet — "some"/"any" alone are already caught by
/// `TRAILING_INCOMPLETE_WORDS`'s last-word check, but multi-word forms like
/// "a few"/"a couple of" are not.
const TRAILING_QUANTIFIER_PHRASES = [/\ba\s+few$/, /\ba\s+couple(\s+of)?$/, /\bany$/, /\bsome$/];

/// Catches multi-word dangling constructions the single-trailing-word
/// approach can't see as a unit — supplements, does not replace,
/// `classifyQuestionCompleteness`.
export function isDanglingConstruction(text: string): boolean {
  const trimmed = text
    .trim()
    .toLowerCase()
    .replace(/[.?!,]+$/, "");
  if (!trimmed) return false;
  if (TRAILING_QUANTIFIER_PHRASES.some((re) => re.test(trimmed))) return true;
  const words = trimmed.split(/\s+/);
  const lastWord = words[words.length - 1]?.replace(/[^a-z']/g, "");
  if (lastWord && OBJECT_NEEDING_VERBS.has(lastWord)) return true;
  return false;
}

/// A small, fixed set of short, closed-form conversational responses —
/// exact whole-utterance match only (same discipline as
/// `interrupt.rs`'s `INTERRUPT_PHRASES`), never a substring/contains check,
/// so a real request that happens to start with one of these ("Thanks, can
/// you also open Chrome") is never caught by this.
const CLOSED_FORM_RESPONSES = new Set(["thank you", "thanks", "you're welcome", "youre welcome", "no problem", "anytime", "sure thing", "of course"]);

export function isBareClosedFormResponse(text: string): boolean {
  const trimmed = text
    .trim()
    .toLowerCase()
    .replace(/[.?!,]+$/, "");
  return CLOSED_FORM_RESPONSES.has(trimmed);
}

/// How long a `partial_text` event's arrival is treated as "speech is still
/// actively coming in" — sourced from real STT event cadence (see
/// `classifyTurnAction`'s `lastPartialArrivedMsAgo`), never a blind sleep.
/// Must be comfortably longer than typical inter-partial latency during
/// continuous speech and comfortably shorter than a natural mid-sentence
/// pause the speaker intends to resume from. Tune empirically against real
/// STT_END_SILENCE_MS behavior — this is a reasoned starting value, not a
/// guessed-correct final one.
export const LIVE_ARRIVAL_GRACE_MS = 700;

/// How soon after Veronica's own TTS audio stops a closed-form response
/// (see `isBareClosedFormResponse`) is treated as a probable mis-hearing of
/// her own trailing audio rather than a real reply — deliberately short so
/// a genuine "thanks!" said well into a later part of the conversation is
/// never suppressed.
export const CLOSED_FORM_GRACE_MS = 2000;

export type TurnAction = "suppress" | "hold" | "send";

export interface ClassifyTurnActionOptions {
  /// Whether a Sensitive/Destructive confirmation is currently pending —
  /// see `useConfirmation`. When true, filler/closed-form suppression is
  /// skipped entirely so a bare "okay"/"yes" reaches the confirmation-reply
  /// path instead of being silently dropped.
  hasPendingConfirmation: boolean;
  /// Whether there was already non-empty buffered text BEFORE this Final
  /// was joined in — distinguishes an isolated bare filler ("um" with
  /// nothing else buffered) from a short word continuing an unfinished
  /// request ("...called" [pause] "Documents").
  hasPriorBufferedText: boolean;
  /// Milliseconds since the last `partial_text` event arrived, or `null` if
  /// none has arrived yet this buffer. A real, live signal from the actual
  /// STT pipeline — never a synthetic timer.
  lastPartialArrivedMsAgo: number | null;
  /// Milliseconds since Veronica's own TTS last stopped speaking, or `null`
  /// if it hasn't spoken yet this session / isn't tracked.
  msSinceTtsStoppedSpeaking: number | null;
}

/// The single decision point for "what should happen with the buffered
/// text now that a new Final has been joined into it" — composes the
/// completeness classifier with filler/closed-form suppression and the
/// live-arrival override. Called by the shared turn-submission hook
/// (`useAutoAsk`) in place of a bare `classifyQuestionCompleteness(...) ===
/// "complete"` check.
export function classifyTurnAction(bufferedText: string, opts: ClassifyTurnActionOptions): TurnAction {
  if (opts.hasPendingConfirmation) {
    // Even a bare "okay"/"yes" must reach the confirmation-reply path — see
    // confirmation::classify_reply on the backend, which is what actually
    // interprets it. This classifier's only job here is "don't suppress it
    // as filler."
    return "send";
  }

  if (!opts.hasPriorBufferedText) {
    if (isFillerOnly(bufferedText)) return "suppress";
    if (
      isBareClosedFormResponse(bufferedText) &&
      opts.msSinceTtsStoppedSpeaking !== null &&
      opts.msSinceTtsStoppedSpeaking < CLOSED_FORM_GRACE_MS
    ) {
      return "suppress";
    }
  }

  if (opts.lastPartialArrivedMsAgo !== null && opts.lastPartialArrivedMsAgo < LIVE_ARRIVAL_GRACE_MS) {
    return "hold";
  }

  if (classifyQuestionCompleteness(bufferedText) === "incomplete" || isDanglingConstruction(bufferedText)) {
    return "hold";
  }

  return "send";
}
