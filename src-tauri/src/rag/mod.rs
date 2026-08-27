//! Local RAG service process management + HTTP client.
//!
//! The RAG engine (`packages/rag`) is a separate local-only Python/FastAPI
//! process, managed here as a child process the same way the PocketSphinx STT
//! sidecar is managed in `crate::stt`. Unlike the STT sidecar, this one speaks
//! plain request/response HTTP (bound to 127.0.0.1 only) rather than a
//! stdin/stdout streaming protocol, since document upload and search are
//! naturally request/response operations, not a continuous stream.
//!
//! This module has no code path that sends documents, chunks, or embeddings to
//! any remote server — every request it makes goes to the local RAG service's
//! 127.0.0.1 port, and that service in turn never calls out anywhere either.

mod client;
mod process;
mod retrieval_planner;
pub mod types;

pub use client::RagClient;
pub use process::{wait_until_healthy_default, EmbedProcessConfig, RagServiceHandle};
pub use retrieval_planner::RetrievalPlanner;
pub use types::{DocumentMetadata, DocumentTextResponse, KnowledgeBaseStatus, SearchResponse, SearchResultItem};
