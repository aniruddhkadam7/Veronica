use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DocumentMetadata {
    pub document_id: String,
    pub filename: String,
    pub document_type: String,
    pub file_size: u64,
    pub content_hash: String,
    pub created_at: f64,
    pub updated_at: f64,
    pub status: String,
    pub chunk_count: u32,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DocumentListResponse {
    pub documents: Vec<DocumentMetadata>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DocumentTextResponse {
    pub document_id: String,
    pub text: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct KnowledgeBaseStatus {
    pub document_count: u32,
    pub chunk_count: u32,
    pub status: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchResultMetadata {
    pub filename: String,
    pub document_type: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchResultItem {
    pub chunk_id: String,
    pub score: f64,
    pub text: String,
    pub metadata: SearchResultMetadata,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub latency_ms: f64,
}
