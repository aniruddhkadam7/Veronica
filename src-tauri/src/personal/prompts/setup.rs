//! Ports `apps/backend/app/services/setup_analysis_service.py` — one-shot
//! resume/job-description extraction for the New Interview setup page.

use crate::backend::{SetupAnalysisRequest, SetupAnalysisResponse};
use super::ChatMessage;

pub const SYSTEM_PROMPT: &str = "You extract a thorough, structured summary from a candidate's resume and/or a job description, for an interview-prep tool. You are not writing anything for the interview itself — only summarizing what was given, but you should be THOROUGH: pull out every distinct responsibility, skill, technology, and qualification actually stated in the text, not just the first few. Read the whole document before answering.

Respond with ONLY a single JSON object, no prose before or after, matching exactly this shape:

{
  \"job_title\": string or null,
  \"company\": string or null,
  \"seniority\": string or null (e.g. \"Junior\", \"Mid-level\", \"Senior\", \"Lead/Staff\" — inferred from years-of-experience requirements, title, or scope of responsibility if stated),
  \"employment_type\": string or null (e.g. \"Full-time\", \"Contract\", \"Remote\", \"Hybrid\" — only if the text actually says so),
  \"key_responsibilities\": string[] (up to 12, one item per distinct duty or responsibility actually listed — do not merge multiple duties into one item, do not stop at 3-4 if more are stated),
  \"required_skills\": string[] (up to 12, every distinct skill or qualification named, including soft skills and years-of-experience requirements if stated as their own bullet),
  \"technologies\": string[] (up to 12, every specific tool, language, framework, platform, or product name mentioned — list names only, no descriptions),
  \"focus_areas\": string[] (up to 8, short phrases describing what an interviewer for this role would likely probe, derived from the responsibilities and requirements above),
  \"candidate_highlights\": string[] (up to 10, short phrases about the candidate's own relevant experience/projects/achievements from the resume — omit entirely if no resume text was given)
}

Only use information actually present in the text given to you — extract comprehensively, but never invent employers, titles, metrics, or requirements that are not in the text. If only a resume is given, leave job_title/company/seniority/employment_type/required_skills/focus_areas empty or null unless they are inferable from the resume itself. If only a job description is given, leave candidate_highlights as an empty list.";

pub const TEMPERATURE: f32 = 0.1;
pub const MAX_TOKENS: u32 = 1100;

fn build_user_prompt(request: &SetupAnalysisRequest) -> String {
    let mut parts = Vec::new();
    if let Some(resume) = &request.resume_text {
        let trimmed = resume.trim();
        if !trimmed.is_empty() {
            parts.push(format!("RESUME:\n{trimmed}"));
        }
    }
    if let Some(jd) = &request.job_description_text {
        let trimmed = jd.trim();
        if !trimmed.is_empty() {
            parts.push(format!("JOB DESCRIPTION:\n{trimmed}"));
        }
    }
    parts.push("Return the JSON object now.".to_string());
    parts.join("\n\n---\n\n")
}

/// `None` when both resume and job description are absent/blank — the
/// caller must not make an LLM call at all in that case (matches the
/// Python short-circuit).
pub fn build_messages(request: &SetupAnalysisRequest) -> Option<Vec<ChatMessage>> {
    let has_resume = request.resume_text.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
    let has_jd = request.job_description_text.as_deref().map(|s| !s.trim().is_empty()).unwrap_or(false);
    if !has_resume && !has_jd {
        return None;
    }
    Some(vec![ChatMessage::system(SYSTEM_PROMPT), ChatMessage::user(build_user_prompt(request))])
}

/// Extracts a JSON object using a greedy first-`{`-to-last-`}` scan
/// (matches Python's `_extract_json`'s `re.search(r"\{.*\}", ..., re.DOTALL)`
/// — same greedy-to-the-end behavior as `prompts::extract_json_object`, kept
/// as a separate function so this module doesn't take on a dependency on
/// analysis/meeting/notes' shared extractor for what is, in the Python
/// source, a genuinely separate (if equivalent) implementation).
fn extract_json(text: &str) -> Result<String, String> {
    let trimmed = text.trim();
    let start = trimmed.find('{').ok_or("no JSON object found")?;
    let end = trimmed.rfind('}').ok_or("no JSON object found")?;
    if end < start {
        return Err("no JSON object found".to_string());
    }
    Ok(trimmed[start..=end].to_string())
}

/// Any failure (parse, validation, or provider-level) returns an empty
/// response — this is the most defensive/fail-silent service in the
/// codebase; it never surfaces an error message to the client (the schema
/// has no `message` field at all).
pub fn parse_response(raw: &str) -> SetupAnalysisResponse {
    extract_json(raw)
        .and_then(|json_str| serde_json::from_str(&json_str).map_err(|e| e.to_string()))
        .unwrap_or_default()
}

pub fn empty_response() -> SetupAnalysisResponse {
    SetupAnalysisResponse::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_call_when_both_inputs_blank() {
        let request = SetupAnalysisRequest { resume_text: Some("   ".to_string()), job_description_text: None };
        assert!(build_messages(&request).is_none());
    }

    #[test]
    fn builds_messages_when_resume_present() {
        let request = SetupAnalysisRequest { resume_text: Some("Senior engineer, 5 years Rust".to_string()), job_description_text: None };
        let messages = build_messages(&request).unwrap();
        assert!(messages[1].content.contains("RESUME:"));
        assert!(!messages[1].content.contains("JOB DESCRIPTION:"));
    }

    #[test]
    fn parse_response_returns_empty_on_malformed_json() {
        let response = parse_response("not json");
        assert_eq!(response, SetupAnalysisResponse::default());
    }

    #[test]
    fn parse_response_extracts_valid_fields() {
        let raw = r#"{"job_title": "Engineer", "required_skills": ["Rust", "Python"]}"#;
        let response = parse_response(raw);
        assert_eq!(response.job_title, Some("Engineer".to_string()));
        assert_eq!(response.required_skills, vec!["Rust".to_string(), "Python".to_string()]);
    }
}
