//! RetrievalPlanner: turns one interview question into a filtered set of
//! retrieved RAG chunks suitable for sending to the analysis backend.
//!
//! Per spec sections 4-5:
//!
//!     Question -> local embedding -> vector search -> top K chunks
//!         -> remove duplicates -> apply similarity threshold
//!         -> limit total context size -> final relevant context
//!
//! This runs once per question (not once for the whole interview) — each
//! question gets its own targeted retrieval rather than one query built from
//! the entire transcript.

use super::client::RagClient;
use super::types::SearchResultItem;

pub const DEFAULT_TOP_K: u32 = 5;
pub const DEFAULT_SIMILARITY_THRESHOLD: f64 = 0.3;
pub const DEFAULT_MAX_CONTEXT_CHARS: usize = 4_000;
/// Must stay <= the backend's `MAX_CHUNK_TEXT_LENGTH`
/// (apps/backend/app/schemas/interview.py) — a single retrieved chunk that
/// exceeds this gets truncated locally before being sent, since the RAG
/// service's chunker targets ~650 tokens per chunk but does not itself
/// enforce a hard character cap, and a real document occasionally produces
/// one larger chunk (e.g. a heading section with little internal structure to
/// split on). Caught during Step 10 manual end-to-end testing: an oversized
/// chunk reached the backend and was rejected with a 422 validation error.
const MAX_CHUNK_CHARS: usize = 3_900;

pub struct RetrievalPlanner {
    client: RagClient,
    top_k: u32,
    similarity_threshold: f64,
    max_context_chars: usize,
    timeout: Option<std::time::Duration>,
    agent_id: Option<String>,
}

impl RetrievalPlanner {
    pub fn new() -> Self {
        Self {
            client: RagClient::new(),
            top_k: DEFAULT_TOP_K,
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
            max_context_chars: DEFAULT_MAX_CONTEXT_CHARS,
            timeout: None,
            agent_id: None,
        }
    }

    pub fn with_config(mut self, top_k: u32, similarity_threshold: f64, max_context_chars: usize) -> Self {
        self.top_k = top_k;
        self.similarity_threshold = similarity_threshold;
        self.max_context_chars = max_context_chars;
        self
    }

    /// Caps how long retrieval may take before it is abandoned and the caller
    /// proceeds with no retrieved context. Only used by Interview Mode, where
    /// the retrieval sits in front of a live answer; the offline analysis
    /// pipeline keeps the client's normal (generous) budget.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Scopes retrieval to one Custom Agent's own knowledge base instead of
    /// the original global/unscoped one. Existing callers (Interview/Sales/
    /// Consulting Mode) never call this and keep searching the global KB.
    pub fn with_agent_scope(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Retrieves and filters context for a single question. Returns an empty
    /// vec (not an error) if the RAG service is unavailable or the knowledge
    /// base has no relevant results — the analysis pipeline proceeds without
    /// retrieved context in that case rather than failing the whole request.
    pub async fn plan_for_question(&self, question: &str) -> Vec<SearchResultItem> {
        let response = match self
            .client
            .search_with_timeout(question, self.top_k, self.agent_id.as_deref(), self.timeout)
            .await
        {
            Ok(response) => response,
            Err(err) => {
                log::warn!("retrieval failed for question, proceeding without context: {err}");
                return Vec::new();
            }
        };

        self.filter_and_limit(response.results)
    }

    fn filter_and_limit(&self, results: Vec<SearchResultItem>) -> Vec<SearchResultItem> {
        let mut seen_texts = std::collections::HashSet::new();
        let mut kept = Vec::new();
        let mut total_chars = 0usize;

        for mut result in results {
            if result.score < self.similarity_threshold {
                continue;
            }
            truncate_chunk_text(&mut result.text, MAX_CHUNK_CHARS);
            // Dedup on exact chunk text — the same chunk can occasionally be
            // returned more than once from overlapping queries in future
            // multi-query planning; cheap to guard against now.
            if !seen_texts.insert(result.text.clone()) {
                continue;
            }
            if total_chars + result.text.len() > self.max_context_chars {
                if kept.is_empty() {
                    // Always keep at least the single best result even if it
                    // alone exceeds the total budget, rather than sending
                    // nothing — it has already been truncated to
                    // MAX_CHUNK_CHARS above, so it still respects the
                    // backend's hard per-chunk validation limit.
                    kept.push(result);
                }
                break;
            }
            total_chars += result.text.len();
            kept.push(result);
        }

        kept
    }
}

impl Default for RetrievalPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Truncates `text` in place to at most `max_chars` *characters* (not bytes —
/// this must stay UTF-8-char-boundary-safe, since chunk text can contain
/// multi-byte characters). No-op if already within the limit.
fn truncate_chunk_text(text: &mut String, max_chars: usize) {
    if text.chars().count() <= max_chars {
        return;
    }
    let truncated: String = text.chars().take(max_chars).collect();
    *text = truncated;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::SearchResultMetadata;

    fn result(text: &str, score: f64) -> SearchResultItem {
        SearchResultItem {
            chunk_id: format!("chunk-{text}"),
            score,
            text: text.to_string(),
            metadata: SearchResultMetadata {
                filename: "test.pdf".to_string(),
                document_type: "PROJECT".to_string(),
            },
        }
    }

    fn planner() -> RetrievalPlanner {
        RetrievalPlanner::new()
    }

    #[test]
    fn filters_out_results_below_similarity_threshold() {
        let p = planner();
        let results = vec![result("relevant", 0.8), result("irrelevant", 0.1)];
        let kept = p.filter_and_limit(results);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].text, "relevant");
    }

    #[test]
    fn removes_duplicate_chunk_text() {
        let p = planner();
        let results = vec![result("same text", 0.8), result("same text", 0.75)];
        let kept = p.filter_and_limit(results);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn keeps_at_least_one_result_even_if_it_exceeds_the_budget() {
        let p = RetrievalPlanner::new().with_config(5, 0.3, 10);
        let huge_text = "x".repeat(1000);
        let results = vec![result(&huge_text, 0.9)];
        let kept = p.filter_and_limit(results);
        assert_eq!(kept.len(), 1);
    }

    #[test]
    fn truncates_a_chunk_larger_than_the_backend_hard_limit() {
        // Regression test: an oversized chunk from a real document previously
        // reached the backend untouched and was rejected with a 422
        // (question_answers.0.retrieved_context.0.text: String should have at
        // most 4000 characters) — found during Step 10 manual end-to-end
        // testing. Every chunk sent to the backend must respect its
        // MAX_CHUNK_TEXT_LENGTH regardless of the total context budget.
        let p = planner(); // default max_context_chars = 4000
        let oversized_text = "x".repeat(6_000);
        let results = vec![result(&oversized_text, 0.9)];
        let kept = p.filter_and_limit(results);
        assert_eq!(kept.len(), 1);
        assert!(kept[0].text.chars().count() <= MAX_CHUNK_CHARS);
    }

    #[test]
    fn truncation_is_utf8_char_boundary_safe() {
        let mut text = "é".repeat(5_000); // multi-byte chars throughout
        truncate_chunk_text(&mut text, MAX_CHUNK_CHARS);
        assert_eq!(text.chars().count(), MAX_CHUNK_CHARS);
    }

    #[test]
    fn truncation_is_a_noop_when_already_within_limit() {
        let mut text = "short text".to_string();
        truncate_chunk_text(&mut text, MAX_CHUNK_CHARS);
        assert_eq!(text, "short text");
    }

    #[test]
    fn stops_adding_results_once_context_budget_is_reached() {
        let p = RetrievalPlanner::new().with_config(5, 0.0, 30);
        let results = vec![
            result(&"a".repeat(15), 0.9),
            result(&"b".repeat(15), 0.8),
            result(&"c".repeat(15), 0.7),
        ];
        let kept = p.filter_and_limit(results);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn empty_results_returns_empty() {
        let p = planner();
        let kept = p.filter_and_limit(vec![]);
        assert!(kept.is_empty());
    }

    #[test]
    fn all_below_threshold_returns_empty() {
        let p = planner();
        let results = vec![result("a", 0.1), result("b", 0.05)];
        let kept = p.filter_and_limit(results);
        assert!(kept.is_empty());
    }
}
