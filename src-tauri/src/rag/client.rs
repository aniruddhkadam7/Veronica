use std::time::Duration;

use super::process::RagServiceHandle;
use super::types::{
    DocumentListResponse, DocumentMetadata, DocumentTextResponse, KnowledgeBaseStatus,
    SearchResponse,
};

pub struct RagClient {
    http: reqwest::Client,
    base_url: String,
}

impl RagClient {
    pub fn new() -> Self {
        crate::tls_init::ensure_installed();
        let mut headers = reqwest::header::HeaderMap::new();
        if let Ok(value) =
            reqwest::header::HeaderValue::from_str(&format!("Bearer {}", RagServiceHandle::auth_token()))
        {
            headers.insert(reqwest::header::AUTHORIZATION, value);
        }
        Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(120)) // document processing can be slow
                .default_headers(headers)
                .build()
                .unwrap_or_default(),
            base_url: RagServiceHandle::base_url(),
        }
    }

    pub async fn health_check(&self) -> Result<(), String> {
        let url = format!("{}/health", self.base_url);
        let response = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(3))
            .send()
            .await
            .map_err(|e| format!("RAG service unreachable: {e}"))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("RAG service health check returned {}", response.status()))
        }
    }

    /// `agent_id: None` uploads into the original global/unscoped knowledge
    /// base (Interview/Sales/Consulting Mode's shared KB, unchanged
    /// behavior). `Some(id)` scopes the document to that Custom Agent's own
    /// workspace — see `packages/rag/app/vector_store.py`'s agent_id column.
    pub async fn upload_document(
        &self,
        filename: String,
        bytes: Vec<u8>,
        document_type: &str,
        agent_id: Option<&str>,
    ) -> Result<DocumentMetadata, String> {
        let part = reqwest::multipart::Part::bytes(bytes).file_name(filename);
        let mut form = reqwest::multipart::Form::new()
            .part("file", part)
            .text("document_type", document_type.to_string());
        if let Some(agent_id) = agent_id {
            form = form.text("agent_id", agent_id.to_string());
        }

        let url = format!("{}/documents/upload", self.base_url);
        let response = self
            .http
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;

        parse_response_or_error(response).await
    }

    pub async fn list_documents(&self, agent_id: Option<&str>) -> Result<Vec<DocumentMetadata>, String> {
        let mut url = format!("{}/documents", self.base_url);
        if let Some(agent_id) = agent_id {
            url = format!("{url}?agent_id={}", urlencode(agent_id));
        }
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;
        let parsed: DocumentListResponse = parse_response_or_error(response).await?;
        Ok(parsed.documents)
    }

    /// Fetches the cleaned, extracted plain text for a document already
    /// uploaded via `upload_document`, once its pipeline has produced it.
    /// Returns `Ok(None)` while extraction is still in progress (or the
    /// document doesn't exist) rather than an error — the New Interview
    /// setup page polls this after a PDF/DOCX upload to feed the real
    /// extracted text into its auto-analysis summary, and "not ready yet" is
    /// a completely normal, expected state to poll through.
    pub async fn get_document_text(&self, document_id: &str) -> Result<Option<String>, String> {
        let url = format!("{}/documents/{document_id}/text", self.base_url);
        let response = self
            .http
            .get(&url)
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        let parsed: DocumentTextResponse = parse_response_or_error(response).await?;
        Ok(Some(parsed.text))
    }

    pub async fn delete_document(&self, document_id: &str) -> Result<(), String> {
        let url = format!("{}/documents/{document_id}", self.base_url);
        let response = self
            .http
            .delete(&url)
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("failed to delete document: {}", response.status()))
        }
    }

    pub async fn clear_knowledge_base(&self, agent_id: Option<&str>) -> Result<(), String> {
        let mut url = format!("{}/knowledge-base/clear", self.base_url);
        if let Some(agent_id) = agent_id {
            url = format!("{url}?agent_id={}", urlencode(agent_id));
        }
        let response = self
            .http
            .post(&url)
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("failed to clear knowledge base: {}", response.status()))
        }
    }

    pub async fn knowledge_base_status(&self, agent_id: Option<&str>) -> Result<KnowledgeBaseStatus, String> {
        let mut url = format!("{}/knowledge-base/status", self.base_url);
        if let Some(agent_id) = agent_id {
            url = format!("{url}?agent_id={}", urlencode(agent_id));
        }
        let response = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;
        parse_response_or_error(response).await
    }

    /// STT/RAG scheduling coordination (Phase B — see
    /// packages/rag/app/throttle.py's module doc). `active=true` tells the
    /// RAG service's embedding step to yield while STT is running on
    /// low-end hardware; must be refreshed periodically while STT stays
    /// active (see `hardware::stt_rag_coordination`) or it auto-expires on
    /// the RAG side. Never affects `/search` — only the embedding step
    /// inside document upload/indexing checks this.
    pub async fn set_throttle(&self, active: bool) -> Result<(), String> {
        let url = format!("{}/internal/throttle", self.base_url);
        let response = self
            .http
            .post(&url)
            .timeout(Duration::from_secs(3))
            .json(&serde_json::json!({ "active": active }))
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;
        if response.status().is_success() {
            Ok(())
        } else {
            Err(format!("failed to set RAG throttle: {}", response.status()))
        }
    }

    pub async fn search(&self, query: &str, top_k: u32, agent_id: Option<&str>) -> Result<SearchResponse, String> {
        self.search_with_timeout(query, top_k, agent_id, None).await
    }

    /// `timeout` overrides the client-wide 120s budget for this one request.
    /// Interview Mode passes a short one: retrieval there sits directly in
    /// front of a live answer, and context is optional, so a slow or
    /// not-yet-started RAG service must cost a fraction of a second and then
    /// be abandoned — never hold up the answer. `agent_id: None` searches the
    /// original global/unscoped knowledge base; `Some(id)` scopes the search
    /// to that Custom Agent's own uploaded documents only.
    pub async fn search_with_timeout(
        &self,
        query: &str,
        top_k: u32,
        agent_id: Option<&str>,
        timeout: Option<Duration>,
    ) -> Result<SearchResponse, String> {
        let url = format!("{}/search", self.base_url);
        let mut request = self.http.post(&url).json(&serde_json::json!({
            "query": query,
            "top_k": top_k,
            "agent_id": agent_id,
        }));
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let response = request
            .send()
            .await
            .map_err(|e| format!("failed to reach RAG service: {e}"))?;
        parse_response_or_error(response).await
    }
}

/// Minimal query-string percent-encoding for the one class of characters an
/// agent id (`agent-<hex>-<hex>`, see agents::new_id) could ever contain that
/// needs escaping. Not a general-purpose encoder — avoids pulling in a URL
/// crate for query params this narrow.
fn urlencode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for byte in value.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{:02X}", byte)),
        }
    }
    out
}

impl Default for RagClient {
    fn default() -> Self {
        Self::new()
    }
}

async fn parse_response_or_error<T: serde::de::DeserializeOwned>(
    response: reqwest::Response,
) -> Result<T, String> {
    let status = response.status();
    if status.is_success() {
        response
            .json::<T>()
            .await
            .map_err(|e| format!("RAG service returned an unexpected response: {e}"))
    } else {
        let body_text = response.text().await.unwrap_or_default();
        Err(format!("RAG service returned {status}: {body_text}"))
    }
}
