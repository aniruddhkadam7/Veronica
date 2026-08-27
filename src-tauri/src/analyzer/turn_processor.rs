//! Deterministic question/answer extraction from a finalized `InterviewSession`.
//!
//! Heuristic (no LLM, no semantic understanding — source, timestamps, transcript
//! order, and simple text patterns only, per spec section 3):
//!
//! 1. Walk finalized segments in transcript order.
//! 2. A `SYSTEM_AUDIO` segment (the interviewer's side, since system audio
//!    captures the meeting/call output — the other participant) is treated as a
//!    candidate "question" if its text ends with `?`, OR starts with a common
//!    interrogative/question-request word (see `looks_like_question`) — this
//!    catches interviewer prompts that PocketSphinx-transcribed without a
//!    trailing question mark, which is common since PocketSphinx does not
//!    reliably produce punctuation.
//! 3. Everything from a `MICROPHONE` segment up to (but not including) the next
//!    question-like `SYSTEM_AUDIO` segment is concatenated as that question's
//!    answer.
//! 4. Consecutive `SYSTEM_AUDIO` segments that don't individually look like a
//!    question are merged into the *next* question-like segment's text, so a
//!    question split across multiple STT finalization boundaries (a very
//!    common PocketSphinx behavior — see docs/progress.md Step 4) is treated as
//!    one question rather than several fragments.
//!
//! This is intentionally conservative: it is fine for it to miss some
//! implicit/unmarked questions (they simply won't be analyzed), but it should
//! not misfire on ordinary interviewer commentary. Manual verification against
//! a real interview transcript is tracked in docs/progress.md Step 10.

use crate::transcript::InterviewSession;

#[derive(Debug, Clone, serde::Serialize)]
pub struct QuestionAnswerPair {
    pub question_id: String,
    pub question: String,
    pub candidate_answer: String,
    /// Relative "MM:SS" or "HH:MM:SS" timestamp of the question, matching the
    /// backend's expected format (see backend::types::WireTranscriptSegment).
    pub timestamp: String,
}

const QUESTION_STARTERS: &[&str] = &[
    "what", "why", "how", "when", "where", "who", "which", "can you", "could you",
    "would you", "do you", "did you", "have you", "is there", "are there", "tell me",
    "walk me through", "explain", "describe",
];

fn looks_like_question(text: &str) -> bool {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return false;
    }
    if trimmed.ends_with('?') {
        return true;
    }
    let lower = trimmed.to_lowercase();
    QUESTION_STARTERS.iter().any(|starter| lower.starts_with(starter))
}

fn format_relative_timestamp(session_started_at_ms: u64, segment_timestamp_ms: u64) -> String {
    let relative_ms = segment_timestamp_ms.saturating_sub(session_started_at_ms);
    let total_secs = relative_ms / 1000;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes:02}:{seconds:02}")
    }
}

/// Extracts question/answer pairs from a finalized interview session. Only
/// considers `final_text` (never `partial_text`) — this always runs after
/// recording has stopped, on the complete finalized transcript.
pub fn extract_question_answers(session: &InterviewSession) -> Vec<QuestionAnswerPair> {
    use crate::audio::AudioSource;

    let mut pairs = Vec::new();
    let mut pending_question: Option<(String, u64)> = None; // (text, timestamp_ms)
    let mut pending_answer_parts: Vec<String> = Vec::new();
    let mut question_counter = 0usize;

    let finalize_pending = |pending_question: &mut Option<(String, u64)>,
                            pending_answer_parts: &mut Vec<String>,
                            question_counter: &mut usize,
                            pairs: &mut Vec<QuestionAnswerPair>,
                            session: &InterviewSession| {
        if let Some((question_text, question_ts)) = pending_question.take() {
            *question_counter += 1;
            pairs.push(QuestionAnswerPair {
                question_id: format!("q{question_counter}"),
                question: question_text,
                candidate_answer: pending_answer_parts.join(" "),
                timestamp: format_relative_timestamp(session.started_at_ms, question_ts),
            });
        }
        pending_answer_parts.clear();
    };

    for segment in &session.segments {
        let Some(text) = segment.final_text.as_deref() else {
            continue;
        };
        let text = text.trim();
        if text.is_empty() {
            continue;
        }

        match segment.source {
            AudioSource::SystemAudio => {
                // A pending question with no answer started yet, arriving
                // within a short gap, is very likely the same spoken question
                // split across two STT finalization boundaries (common
                // PocketSphinx behavior) — merge it in regardless of whether
                // this fragment alone also looks like a question, rather than
                // starting a second question. Only treat it as a genuinely
                // new question once an answer has begun, or after a longer
                // gap suggesting a real pause between separate questions.
                const CONTINUATION_GAP_MS: u64 = 4_000;
                let is_likely_continuation = pending_answer_parts.is_empty()
                    && pending_question
                        .as_ref()
                        .map(|(_, prev_ts)| segment.timestamp.saturating_sub(*prev_ts) <= CONTINUATION_GAP_MS)
                        .unwrap_or(false);

                if is_likely_continuation {
                    if let Some((existing_text, prev_ts)) = pending_question.as_mut() {
                        existing_text.push(' ');
                        existing_text.push_str(text);
                        *prev_ts = segment.timestamp;
                    }
                } else if looks_like_question(text) {
                    // A new question starts: close out whatever question/answer
                    // was pending, then start a fresh one.
                    finalize_pending(
                        &mut pending_question,
                        &mut pending_answer_parts,
                        &mut question_counter,
                        &mut pairs,
                        session,
                    );
                    pending_question = Some((text.to_string(), segment.timestamp));
                }
                // Non-question interviewer speech with no pending question, or
                // stray non-question audio after an answer has already
                // started, is ignored (e.g. small talk, back-channel remarks).
            }
            AudioSource::Microphone => {
                if pending_question.is_some() {
                    pending_answer_parts.push(text.to_string());
                }
                // Microphone speech before any question has been identified is
                // not attributed to a question (e.g. candidate's opening
                // remarks before the interviewer asks anything).
            }
        }
    }

    finalize_pending(
        &mut pending_question,
        &mut pending_answer_parts,
        &mut question_counter,
        &mut pairs,
        session,
    );

    pairs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio::AudioSource;
    use crate::transcript::{InterviewSession, TranscriptSegment};

    fn segment(source: AudioSource, text: &str, timestamp_ms: u64) -> TranscriptSegment {
        TranscriptSegment {
            id: format!("seg-{timestamp_ms}"),
            timestamp: timestamp_ms,
            source,
            partial_text: None,
            final_text: Some(text.to_string()),
            start_time: None,
            end_time: None,
        }
    }

    fn session_with(segments: Vec<TranscriptSegment>) -> InterviewSession {
        let mut session = InterviewSession::new();
        session.started_at_ms = 0;
        session.segments = segments;
        session
    }

    #[test]
    fn extracts_single_question_answer_pair() {
        let session = session_with(vec![
            segment(AudioSource::SystemAudio, "Can you tell me about yourself?", 1000),
            segment(AudioSource::Microphone, "Sure, I'm an engineer.", 2000),
        ]);
        let pairs = extract_question_answers(&session);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].question, "Can you tell me about yourself?");
        assert_eq!(pairs[0].candidate_answer, "Sure, I'm an engineer.");
        assert_eq!(pairs[0].question_id, "q1");
    }

    #[test]
    fn extracts_multiple_question_answer_pairs_in_order() {
        let session = session_with(vec![
            segment(AudioSource::SystemAudio, "What is RAG?", 1000),
            segment(AudioSource::Microphone, "Retrieval augmented generation.", 2000),
            segment(AudioSource::SystemAudio, "Why did you choose it?", 3000),
            segment(AudioSource::Microphone, "For grounding answers.", 4000),
        ]);
        let pairs = extract_question_answers(&session);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].question, "What is RAG?");
        assert_eq!(pairs[1].question, "Why did you choose it?");
        assert_eq!(pairs[1].question_id, "q2");
    }

    #[test]
    fn concatenates_multiple_microphone_segments_into_one_answer() {
        let session = session_with(vec![
            segment(AudioSource::SystemAudio, "Explain your project.", 1000),
            segment(AudioSource::Microphone, "I built a RAG system.", 2000),
            segment(AudioSource::Microphone, "It used FastAPI and a vector store.", 3000),
        ]);
        let pairs = extract_question_answers(&session);
        assert_eq!(pairs.len(), 1);
        assert_eq!(
            pairs[0].candidate_answer,
            "I built a RAG system. It used FastAPI and a vector store."
        );
    }

    #[test]
    fn detects_question_without_trailing_question_mark() {
        // PocketSphinx frequently produces no punctuation at all.
        let session = session_with(vec![
            segment(AudioSource::SystemAudio, "how did you handle false positives", 1000),
            segment(AudioSource::Microphone, "we tuned the threshold", 2000),
        ]);
        let pairs = extract_question_answers(&session);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].question, "how did you handle false positives");
    }

    #[test]
    fn ignores_non_question_interviewer_smalltalk_before_first_question() {
        let session = session_with(vec![
            segment(AudioSource::SystemAudio, "thanks for joining today", 500),
            segment(AudioSource::SystemAudio, "What is your experience with Python?", 1000),
            segment(AudioSource::Microphone, "Five years.", 2000),
        ]);
        let pairs = extract_question_answers(&session);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].question, "What is your experience with Python?");
    }

    #[test]
    fn merges_split_question_fragments_before_answer_starts() {
        // Common PocketSphinx behavior: one spoken question becomes two
        // finalized STT segments because of an endpointing gap.
        let session = session_with(vec![
            segment(AudioSource::SystemAudio, "Can you explain the", 1000),
            segment(AudioSource::SystemAudio, "architecture you used?", 1500),
            segment(AudioSource::Microphone, "It was a microservices design.", 2000),
        ]);
        let pairs = extract_question_answers(&session);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].question, "Can you explain the architecture you used?");
    }

    #[test]
    fn empty_transcript_produces_no_pairs() {
        let session = session_with(vec![]);
        let pairs = extract_question_answers(&session);
        assert!(pairs.is_empty());
    }

    #[test]
    fn question_with_no_answer_still_produces_a_pair_with_empty_answer() {
        let session = session_with(vec![segment(
            AudioSource::SystemAudio,
            "What is your greatest weakness?",
            1000,
        )]);
        let pairs = extract_question_answers(&session);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].candidate_answer, "");
    }

    #[test]
    fn microphone_only_transcript_produces_no_pairs() {
        let session = session_with(vec![segment(AudioSource::Microphone, "Just talking to myself.", 1000)]);
        let pairs = extract_question_answers(&session);
        assert!(pairs.is_empty());
    }

    #[test]
    fn timestamp_is_relative_to_session_start() {
        let mut session = session_with(vec![segment(AudioSource::SystemAudio, "What's your name?", 65_000)]);
        session.started_at_ms = 5_000;
        let pairs = extract_question_answers(&session);
        // (65000 - 5000) ms = 60s = 00:01:00
        assert_eq!(pairs[0].timestamp, "01:00");
    }
}
