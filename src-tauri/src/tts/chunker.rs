//! Splits a stream of LLM answer deltas into sentence-sized chunks to speak.
//!
//! The LLM streams text token-by-token; sending each token to TTS
//! individually would mean dozens of tiny, choppy requests per answer.
//! Waiting for the whole answer would mean no speech starts until the LLM
//! is completely done, defeating the point of streaming. This buffers
//! deltas and releases one chunk per completed sentence, so the first
//! sentence can be sent to Deepgram (and start playing) while the LLM is
//! still generating the rest of the answer.

pub struct SentenceChunker {
    buffer: String,
    /// Whether any chunk has been released yet. While `false`, a comma (or
    /// other soft-pause punctuation) also counts as a release boundary —
    /// see `push` — so the very first sound starts as soon as there's a
    /// natural-sounding clause to speak, not only once a whole sentence
    /// (which can be a long time for the first sentence of a long answer)
    /// has arrived. Every later sentence still waits for real sentence
    /// punctuation, since splitting every sentence at commas would sound
    /// choppy rather than merely reducing time-to-first-sound.
    released_any: bool,
}

impl SentenceChunker {
    pub fn new() -> Self {
        Self { buffer: String::new(), released_any: false }
    }

    /// Feeds one delta from the LLM stream. Returns zero or more chunks that
    /// are now ready to speak (a single delta can complete more than one
    /// buffered chunk at once, e.g. "Yes. No.").
    pub fn push(&mut self, delta: &str) -> Vec<String> {
        self.buffer.push_str(delta);
        let mut out = Vec::new();

        loop {
            let boundary = if self.released_any {
                find_sentence_boundary(&self.buffer)
            } else {
                // Before the first release: a comma/soft-pause boundary is
                // also acceptable, whichever comes first — see the
                // `released_any` field doc for why this only applies once.
                find_soft_boundary(&self.buffer).or_else(|| find_sentence_boundary(&self.buffer))
            };
            let Some(boundary) = boundary else {
                break;
            };
            let sentence: String = self.buffer.drain(..boundary).collect();
            let trimmed = sentence.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
                self.released_any = true;
            }
        }

        out
    }

    /// Call once the LLM stream has ended: returns whatever trailing text
    /// never hit a sentence boundary (e.g. an answer with no closing
    /// punctuation), or `None` if the buffer is empty/whitespace-only.
    pub fn finish(&mut self) -> Option<String> {
        let trimmed = self.buffer.trim().to_string();
        self.buffer.clear();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        }
    }
}

impl Default for SentenceChunker {
    fn default() -> Self {
        Self::new()
    }
}

/// Finds the end (exclusive byte index) of the first complete sentence in
/// `buffer`, or `None` if no sentence-ending punctuation has arrived yet.
///
/// A `.`/`!`/`?` counts as a boundary when followed by whitespace, OR when
/// it's the very last byte currently buffered (nothing has arrived after it
/// yet — a fresh LLM delta might complete "3.14" into a non-boundary next,
/// but until that happens there's nothing to lose by treating it as one: if
/// more digits/text do arrive immediately after in the next delta, this
/// function runs again on the new, longer buffer and correctly does NOT
/// treat that same `.` as a boundary the second time, since it's no longer
/// the last byte and isn't followed by whitespace either). What's NOT a
/// boundary is punctuation immediately followed by more non-whitespace text
/// within the SAME buffer ("3.14", "Mr. Smith") — that's the unambiguous
/// case, and it's cheap to wait one more delta for the ambiguous
/// end-of-buffer case rather than delay every real sentence until the
/// answer's very next token arrives.
fn find_sentence_boundary(buffer: &str) -> Option<usize> {
    let bytes = buffer.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if matches!(b, b'.' | b'!' | b'?') {
            // Absorb a run of closing punctuation/quotes ("?!", ".\"") so
            // the boundary lands after all of it, not mid-punctuation.
            let mut end = i + 1;
            while end < bytes.len() && matches!(bytes[end], b'.' | b'!' | b'?' | b'"' | b'\'' | b')') {
                end += 1;
            }
            if end == bytes.len() || bytes[end].is_ascii_whitespace() {
                return Some(end);
            }
        }
        if b == b'\n' && i > 0 {
            // A newline (e.g. a list item or paragraph break) always ends
            // whatever's buffered, even with no terminal punctuation.
            return Some(i + 1);
        }
    }
    None
}

/// Like `find_sentence_boundary` but also accepts a comma or semicolon
/// followed by whitespace — used only for the first chunk of an answer (see
/// `SentenceChunker::released_any`) to cut time-to-first-sound on a long
/// opening sentence, at the cost of that first spoken clause sounding
/// slightly more fragmented than a full sentence would.
///
/// Requires at least `MIN_SOFT_CHUNK_CHARS` before considering a comma a
/// boundary, so "Well, " or "So, " doesn't get sent to Deepgram as its own
/// one-word utterance — that would be slower overall (fixed per-request
/// latency paid twice) and sound worse, not better.
fn find_soft_boundary(buffer: &str) -> Option<usize> {
    const MIN_SOFT_CHUNK_CHARS: usize = 20;
    let bytes = buffer.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if i < MIN_SOFT_CHUNK_CHARS {
            continue;
        }
        if matches!(b, b',' | b';') && bytes.get(i + 1).is_some_and(|c| c.is_ascii_whitespace()) {
            return Some(i + 2);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flushes_on_sentence_boundary() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("Hello there. "), vec!["Hello there."]);
    }

    #[test]
    fn holds_incomplete_sentence() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("Hello there"), Vec::<String>::new());
    }

    #[test]
    fn flushes_incrementally_across_pushes() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("Hello"), Vec::<String>::new());
        assert_eq!(c.push(" there."), vec!["Hello there."]);
    }

    #[test]
    fn splits_multiple_sentences_in_one_delta() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("Yes. No. Maybe. "), vec!["Yes.", "No.", "Maybe."]);
    }

    #[test]
    fn keeps_trailing_quote_with_sentence() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("She said \"hi.\" "), vec!["She said \"hi.\""]);
    }

    #[test]
    fn decimal_point_is_not_a_sentence_boundary() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.push("The value is 3.14 exactly."), vec!["The value is 3.14 exactly."]);
    }

    #[test]
    fn long_run_on_text_flushes_without_punctuation() {
        let mut c = SentenceChunker::new();
        let chunks = c.push("first line\nsecond line continues");
        assert_eq!(chunks, vec!["first line"]);
    }

    #[test]
    fn first_chunk_of_an_answer_can_release_on_a_comma_for_lower_latency() {
        let mut c = SentenceChunker::new();
        // Long enough to clear MIN_SOFT_CHUNK_CHARS before the comma.
        let chunks = c.push("Well, given everything you've described, it depends on the situation.");
        assert_eq!(
            chunks,
            vec!["Well, given everything you've described,", "it depends on the situation."],
            "the first clause should release early at the comma; the rest waits for the sentence"
        );
    }

    #[test]
    fn short_leading_comma_does_not_release_a_tiny_fragment() {
        let mut c = SentenceChunker::new();
        // "Well," alone is well under MIN_SOFT_CHUNK_CHARS — must not be
        // released as its own one-word utterance.
        assert_eq!(c.push("Well, "), Vec::<String>::new());
    }

    #[test]
    fn comma_boundary_only_applies_before_the_first_release() {
        let mut c = SentenceChunker::new();
        // First sentence releases (on its period, since no comma appears
        // before it), which sets released_any — the second sentence's
        // internal comma must NOT trigger an early split after that.
        let first = c.push("This is the opening sentence with no comma. ");
        assert_eq!(first, vec!["This is the opening sentence with no comma."]);

        let second = c.push("Second, with a comma in it, ends here. ");
        assert_eq!(
            second,
            vec!["Second, with a comma in it, ends here."],
            "once released_any is true, only real sentence punctuation should split"
        );
    }

    #[test]
    fn finish_on_empty_buffer_returns_none() {
        let mut c = SentenceChunker::new();
        assert_eq!(c.finish(), None);
    }

    #[test]
    fn finish_returns_trailing_incomplete_text() {
        let mut c = SentenceChunker::new();
        c.push("no closing punctuation here");
        assert_eq!(c.finish(), Some("no closing punctuation here".to_string()));
    }

    #[test]
    fn finish_clears_the_buffer() {
        let mut c = SentenceChunker::new();
        c.push("leftover");
        c.finish();
        assert_eq!(c.finish(), None);
    }
}
