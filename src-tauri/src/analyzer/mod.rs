//! InterviewAnalyzer
//! ├── TurnProcessor            (groups finalized transcript segments into turns)
//! ├── QuestionAnswerExtractor  (derives Question/Answer pairs from turns)
//! └── (RetrievalPlanner lives in `crate::rag` — see `rag::plan_retrieval_queries`)
//!
//! Runs entirely after recording has stopped, as a read-only derivation over the
//! already-finalized `InterviewSession` — it never mutates `TranscriptManager` or
//! any segment, and nothing here runs during recording or calls an LLM. This is
//! deterministic post-processing only (spec section 3: "Do not use an LLM for
//! basic segmentation if a deterministic approach is sufficient").

mod turn_processor;

pub use turn_processor::{extract_question_answers, QuestionAnswerPair};
