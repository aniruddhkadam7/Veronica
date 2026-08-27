//! Ports `apps/backend/app/services/analysis_service.py` +
//! `prompt_builder.py` — the two-stage interview analysis pipeline:
//! Stage 1 scores each question individually, Stage 2 asks the model for a
//! fresh qualitative overall assessment from the Stage-1 summaries (NOT a
//! mathematical average — averaging only happens in the fallback path when
//! Stage 2 itself fails to parse).

use crate::backend::{
    InterviewAnalysisRequest, OverallInterviewAnalysis, QuestionAnalysis, RetrievedSourceRef, WireQuestionAnswer,
};
use super::{extract_json_object, ChatMessage};

pub const SYSTEM_INSTRUCTIONS: &str = "You are an expert technical interviewer and interview coach.

Analyze the candidate's completed interview using only the candidate context and retrieved knowledge provided.

Do not invent experience.

Do not attribute technologies, projects, employers, responsibilities, or achievements to the candidate unless supported by the supplied context.

Distinguish between:
- What the candidate actually said
- What is supported by their documents
- What is missing
- What could be improved

Evaluate answers based on technical correctness, practical depth, clarity, structure, relevance, and communication.

When suggesting an improved answer, preserve the candidate's actual experience and do not fabricate accomplishments.

If information needed to verify a claim is missing from the supplied context, say so explicitly using the phrase \"Not established from the supplied candidate context.\" Do not guess.

If the candidate's answer contradicts the supplied candidate context (for example, claiming a technology or deployment method not found in their resume or project documents), flag this directly in your assessment rather than silently accepting or silently correcting it.

You must respond with valid JSON only, matching exactly the schema described in the user message. Do not include any text outside the JSON object.";

pub const RUBRIC: &str = "Score each dimension from 0-100 using this rubric:

Technical Knowledge (0-100): depth and correctness of technical explanation.
Communication (0-100): clarity, structure, and how well the answer would land in a real interview.
Practical Experience (0-100): concrete, specific, hands-on detail vs. generic/theoretical answers.
Clarity (0-100): how easy the answer is to follow.
Confidence (0-100): how directly and assuredly the candidate answered.

You must briefly explain the reasoning behind any score above 85 or below 40 in the relevant text field (assessment, or summary for overall scores). Avoid arbitrary scoring — every score should be traceable to something specific in the answer or context.";

pub const DEFAULT_DISCLAIMER: &str = "These scores and this feedback are AI-generated estimates based only on the transcript and documents you provided. They are not an objective measurement of your skills or interview performance.";

fn context_header(role: &Option<String>, company: &Option<String>, job_description: &Option<String>) -> String {
    let mut parts = Vec::new();
    if let Some(r) = role {
        parts.push(format!("ROLE\n{r}"));
    }
    if let Some(c) = company {
        parts.push(format!("COMPANY\n{c}"));
    }
    if let Some(jd) = job_description {
        parts.push(format!("JOB DESCRIPTION\n{jd}"));
    }
    if parts.is_empty() {
        "(No role/company/job description provided.)".to_string()
    } else {
        parts.join("\n\n")
    }
}

fn format_retrieved_context(qa: &WireQuestionAnswer) -> String {
    if qa.retrieved_context.is_empty() {
        return "(No relevant context was retrieved from the candidate's documents for this question.)".to_string();
    }
    qa.retrieved_context
        .iter()
        .map(|chunk| format!("[Source: {} ({}), relevance {:.2}]\n{}", chunk.source_filename, chunk.document_type, chunk.score, chunk.text))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Returns (system_prompt, user_prompt) for Stage 1 (per-question analysis).
pub fn build_question_prompt(
    qa: &WireQuestionAnswer,
    role: &Option<String>,
    company: &Option<String>,
    job_description: &Option<String>,
) -> (String, String) {
    let header = context_header(role, company, job_description);
    let answer = if qa.candidate_answer.is_empty() { "(No answer was captured for this question.)" } else { &qa.candidate_answer };

    let user_prompt = format!(
        "{header}\n\nRELEVANT RETRIEVED KNOWLEDGE\n{}\n\nINTERVIEW QUESTION\n{}\n\nCANDIDATE ANSWER\n{answer}\n\n{RUBRIC}\n\nRespond with a single JSON object matching exactly this schema:\n{{\n  \"assessment\": \"string — what the candidate actually said, evaluated against the rubric, referencing the retrieved knowledge where relevant\",\n  \"score\": 0,\n  \"strengths\": [\"string\", \"...\"],\n  \"issues\": [\"string\", \"...\"],\n  \"improved_answer\": \"string — first-person, conversational, interview-ready, grounded only in the retrieved knowledge and the candidate's actual answer; do not fabricate accomplishments\"\n}}\n\nThe field \"question_id\" is not part of your response — it is added separately.",
        format_retrieved_context(qa),
        qa.question,
    );

    (SYSTEM_INSTRUCTIONS.to_string(), user_prompt)
}

/// Returns (system_prompt, user_prompt) for Stage 2 (aggregate overall
/// analysis), built from the already-structured per-question analyses.
pub fn build_overall_prompt(
    question_analyses: &[QuestionAnalysis],
    role: &Option<String>,
    company: &Option<String>,
    job_description: &Option<String>,
) -> (String, String) {
    let header = context_header(role, company, job_description);

    let successful: Vec<&QuestionAnalysis> = question_analyses.iter().filter(|q| !q.failed).collect();
    let mut questions_block = successful
        .iter()
        .enumerate()
        .map(|(i, qa)| {
            let strengths = if qa.strengths.is_empty() { "(none noted)".to_string() } else { qa.strengths.join(", ") };
            let issues = if qa.issues.is_empty() { "(none noted)".to_string() } else { qa.issues.join(", ") };
            format!("Q{}: {}\nScore: {}\nAssessment: {}\nStrengths: {strengths}\nIssues: {issues}", i + 1, qa.question, qa.score, qa.assessment)
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    if questions_block.is_empty() {
        questions_block = "(No question-level analyses completed successfully.)".to_string();
    }

    let user_prompt = format!(
        "{header}\n\nPER-QUESTION ANALYSIS RESULTS\n{questions_block}\n\n{RUBRIC}\n\nBased on the per-question analyses above, produce an overall assessment of the candidate's interview performance. Respond with a single JSON object matching exactly this schema:\n{{\n  \"overall_score\": 0,\n  \"technical_score\": 0,\n  \"communication_score\": 0,\n  \"practical_experience_score\": 0,\n  \"confidence_score\": 0,\n  \"summary\": \"string — 2-4 sentence overall summary\",\n  \"strengths\": [\"string\", \"...\"],\n  \"weaknesses\": [\"string\", \"...\"],\n  \"recommendations\": [\"string\", \"...\"]\n}}"
    );

    (SYSTEM_INSTRUCTIONS.to_string(), user_prompt)
}

pub fn question_messages(system_prompt: &str, user_prompt: &str) -> Vec<ChatMessage> {
    vec![ChatMessage::system(system_prompt.to_string()), ChatMessage::user(user_prompt.to_string())]
}

/// Parses a Stage-1 raw model response into a `QuestionAnalysis`. On
/// malformed JSON/missing-field errors, returns a `failed=true` record with
/// `retrieved_sources` still populated (matches the Python `except
/// (ValidationError, ValueError, KeyError, TypeError, JSONDecodeError)`
/// branch) — this function does not distinguish that from a provider-level
/// failure (network/auth/rate-limit); callers construct the
/// `retrieved_sources`-empty failed variant themselves for that case, since
/// this function only ever runs after a successful provider call.
pub fn parse_question_analysis(raw: &str, qa: &WireQuestionAnswer) -> QuestionAnalysis {
    let retrieved_sources: Vec<RetrievedSourceRef> = qa
        .retrieved_context
        .iter()
        .map(|c| RetrievedSourceRef { filename: c.source_filename.clone(), document_type: c.document_type.clone(), score: c.score, text: c.text.clone() })
        .collect();

    let parse_result: Result<serde_json::Value, String> = extract_json_object(raw).and_then(|json_str| serde_json::from_str(&json_str).map_err(|e| e.to_string()));

    match parse_result {
        Ok(parsed) => {
            let assessment = parsed.get("assessment").and_then(|v| v.as_str());
            let score = parsed.get("score").and_then(|v| v.as_i64());
            match (assessment, score) {
                (Some(assessment), Some(score)) => QuestionAnalysis {
                    question_id: qa.question_id.clone(),
                    question: qa.question.clone(),
                    candidate_answer: qa.candidate_answer.clone(),
                    assessment: assessment.to_string(),
                    score: score as i32,
                    strengths: parsed.get("strengths").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default(),
                    issues: parsed.get("issues").and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default(),
                    improved_answer: parsed.get("improved_answer").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    retrieved_sources,
                    failed: false,
                    error_message: None,
                },
                _ => failed_question_analysis(qa, retrieved_sources, "response JSON was missing required fields (assessment/score)".to_string()),
            }
        }
        Err(err) => failed_question_analysis(qa, retrieved_sources, err),
    }
}

fn failed_question_analysis(qa: &WireQuestionAnswer, retrieved_sources: Vec<RetrievedSourceRef>, error_message: String) -> QuestionAnalysis {
    QuestionAnalysis {
        question_id: qa.question_id.clone(),
        question: qa.question.clone(),
        candidate_answer: qa.candidate_answer.clone(),
        assessment: String::new(),
        score: 0,
        strengths: vec![],
        issues: vec![],
        improved_answer: String::new(),
        retrieved_sources,
        failed: true,
        error_message: Some(error_message),
    }
}

/// A provider-level failure (network/auth/rate-limit, not a parse error) —
/// matches the Python `except Exception` branch, which leaves
/// `retrieved_sources` empty (bug-for-bug parity with the reference).
pub fn provider_failure_question_analysis(qa: &WireQuestionAnswer, error_message: String) -> QuestionAnalysis {
    QuestionAnalysis {
        question_id: qa.question_id.clone(),
        question: qa.question.clone(),
        candidate_answer: qa.candidate_answer.clone(),
        assessment: String::new(),
        score: 0,
        strengths: vec![],
        issues: vec![],
        improved_answer: String::new(),
        retrieved_sources: vec![],
        failed: true,
        error_message: Some(format!("LLM request failed: {error_message}")),
    }
}

fn no_questions_result(session_id: &str) -> OverallInterviewAnalysis {
    OverallInterviewAnalysis {
        session_id: session_id.to_string(),
        status: "completed".to_string(),
        overall_score: 0,
        technical_score: 0,
        communication_score: 0,
        practical_experience_score: 0,
        confidence_score: 0,
        summary: "No interview questions were identified in the transcript, so no analysis could be generated.".to_string(),
        strengths: vec![],
        weaknesses: vec![],
        recommendations: vec![],
        questions: vec![],
        disclaimer: DEFAULT_DISCLAIMER.to_string(),
        message: "No question/answer pairs were extracted from the transcript.".to_string(),
    }
}

/// Assembles the final `OverallInterviewAnalysis` from a Stage-2 raw model
/// response. Returns `Err` (not a fallback result) when parsing/validation
/// fails — the caller is expected to fall back to `average_score_fallback`
/// in that case, exactly mirroring Python's try/except split between
/// `_finalize`'s primary path and its `except Exception` branch.
pub fn parse_overall_result(raw: &str, question_analyses: Vec<QuestionAnalysis>, session_id: &str) -> Result<OverallInterviewAnalysis, String> {
    let json_str = extract_json_object(raw)?;
    let parsed: serde_json::Value = serde_json::from_str(&json_str).map_err(|e| e.to_string())?;

    let get_int = |key: &str| -> Result<i32, String> {
        parsed.get(key).and_then(|v| v.as_i64()).map(|v| v as i32).ok_or_else(|| format!("missing or invalid field: {key}"))
    };
    let get_strs = |key: &str| -> Vec<String> {
        parsed.get(key).and_then(|v| v.as_array()).map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect()).unwrap_or_default()
    };

    let status = if question_analyses.iter().any(|q| q.failed) { "partial" } else { "completed" };

    Ok(OverallInterviewAnalysis {
        session_id: session_id.to_string(),
        status: status.to_string(),
        overall_score: get_int("overall_score")?,
        technical_score: get_int("technical_score")?,
        communication_score: get_int("communication_score")?,
        practical_experience_score: get_int("practical_experience_score")?,
        confidence_score: get_int("confidence_score")?,
        summary: parsed.get("summary").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        strengths: get_strs("strengths"),
        weaknesses: get_strs("weaknesses"),
        recommendations: get_strs("recommendations"),
        questions: question_analyses,
        disclaimer: DEFAULT_DISCLAIMER.to_string(),
        message: String::new(),
    })
}

/// The fallback path when Stage 2 fails entirely (provider error or
/// unparseable response) — the ONLY place an actual mathematical average of
/// per-question scores is computed; every dimension gets the same value.
pub fn average_score_fallback(question_analyses: Vec<QuestionAnalysis>, session_id: &str, error_message: &str) -> OverallInterviewAnalysis {
    let successful: Vec<&QuestionAnalysis> = question_analyses.iter().filter(|q| !q.failed).collect();
    let avg_score = if successful.is_empty() { 0 } else { (successful.iter().map(|q| q.score as i64).sum::<i64>() / successful.len() as i64) as i32 };

    OverallInterviewAnalysis {
        session_id: session_id.to_string(),
        status: "partial".to_string(),
        overall_score: avg_score,
        technical_score: avg_score,
        communication_score: avg_score,
        practical_experience_score: avg_score,
        confidence_score: avg_score,
        summary: String::new(),
        strengths: vec![],
        weaknesses: vec![],
        recommendations: vec![],
        questions: question_analyses,
        disclaimer: DEFAULT_DISCLAIMER.to_string(),
        message: format!("Overall analysis could not be generated ({error_message}); showing per-question results with an approximate average score."),
    }
}

/// Builds the "no questions at all" short-circuit result — never calls
/// Stage 2.
pub fn finalize_empty(session_id: &str) -> OverallInterviewAnalysis {
    no_questions_result(session_id)
}

pub const MAX_ANALYSIS_TEMPERATURE: f32 = 0.2;

/// Wraps `request.question_answers`, applying the same truncation the
/// backend applies (`MAX_QUESTIONS_TO_ANALYZE`, default 50).
pub fn questions_to_analyze(request: &InterviewAnalysisRequest, max_questions: usize) -> &[WireQuestionAnswer] {
    let len = request.question_answers.len().min(max_questions);
    &request.question_answers[..len]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{RetrievedSourceRef as RSR, WireRetrievedChunk};

    fn sample_qa() -> WireQuestionAnswer {
        WireQuestionAnswer {
            question_id: "q1".to_string(),
            question: "What is RAG?".to_string(),
            candidate_answer: "Retrieval augmented generation".to_string(),
            timestamp: "00:00:10".to_string(),
            retrieved_context: vec![WireRetrievedChunk { text: "RAG combines retrieval with generation".to_string(), source_filename: "resume.pdf".to_string(), document_type: "resume".to_string(), score: 0.8 }],
        }
    }

    #[test]
    fn question_prompt_includes_retrieved_context_with_source_info() {
        let (_sys, user) = build_question_prompt(&sample_qa(), &None, &None, &None);
        assert!(user.contains("resume.pdf"));
        assert!(user.contains("0.80"));
        assert!(user.contains("RAG combines retrieval with generation"));
    }

    #[test]
    fn question_prompt_omits_header_when_no_role_info() {
        let (_sys, user) = build_question_prompt(&sample_qa(), &None, &None, &None);
        assert!(user.contains("(No role/company/job description provided.)"));
    }

    #[test]
    fn parses_valid_question_analysis_json() {
        let raw = r#"{"assessment": "Good answer", "score": 80, "strengths": ["clear"], "issues": [], "improved_answer": "Better answer"}"#;
        let result = parse_question_analysis(raw, &sample_qa());
        assert!(!result.failed);
        assert_eq!(result.score, 80);
        assert_eq!(result.assessment, "Good answer");
        assert_eq!(result.retrieved_sources.len(), 1);
    }

    #[test]
    fn marks_failed_on_malformed_json_but_keeps_retrieved_sources() {
        let result = parse_question_analysis("not json at all", &sample_qa());
        assert!(result.failed);
        assert_eq!(result.score, 0);
        assert_eq!(result.retrieved_sources.len(), 1); // still populated, per the ValueError branch
    }

    #[test]
    fn provider_failure_leaves_retrieved_sources_empty() {
        let result = provider_failure_question_analysis(&sample_qa(), "timeout".to_string());
        assert!(result.failed);
        assert!(result.retrieved_sources.is_empty());
        assert!(result.error_message.unwrap().contains("timeout"));
    }

    fn sample_question_analysis(score: i32, failed: bool) -> QuestionAnalysis {
        QuestionAnalysis {
            question_id: "q1".to_string(),
            question: "Q".to_string(),
            candidate_answer: "A".to_string(),
            assessment: "assessment".to_string(),
            score,
            strengths: vec![],
            issues: vec![],
            improved_answer: String::new(),
            retrieved_sources: vec![RSR { filename: "f".to_string(), document_type: "t".to_string(), score: 0.5, text: "x".to_string() }],
            failed,
            error_message: None,
        }
    }

    #[test]
    fn overall_prompt_excludes_failed_questions() {
        let analyses = vec![sample_question_analysis(80, false), sample_question_analysis(0, true)];
        let (_sys, user) = build_overall_prompt(&analyses, &None, &None, &None);
        assert!(user.contains("Q1:"));
        assert!(!user.contains("Q2:"));
    }

    #[test]
    fn average_fallback_uses_only_successful_scores() {
        let analyses = vec![sample_question_analysis(80, false), sample_question_analysis(60, false), sample_question_analysis(0, true)];
        let result = average_score_fallback(analyses, "s1", "parse error");
        assert_eq!(result.overall_score, 70); // (80+60)/2, failed one excluded
        assert_eq!(result.technical_score, 70); // same value across every dimension
        assert_eq!(result.status, "partial");
    }

    #[test]
    fn average_fallback_handles_all_failed() {
        let analyses = vec![sample_question_analysis(0, true)];
        let result = average_score_fallback(analyses, "s1", "parse error");
        assert_eq!(result.overall_score, 0);
    }

    #[test]
    fn parse_overall_result_status_is_partial_when_any_question_failed() {
        let raw = r#"{"overall_score": 70, "technical_score": 70, "communication_score": 70, "practical_experience_score": 70, "confidence_score": 70, "summary": "ok", "strengths": [], "weaknesses": [], "recommendations": []}"#;
        let analyses = vec![sample_question_analysis(80, false), sample_question_analysis(0, true)];
        let result = parse_overall_result(raw, analyses, "s1").unwrap();
        assert_eq!(result.status, "partial");
    }

    #[test]
    fn parse_overall_result_errors_on_missing_field() {
        let raw = r#"{"overall_score": 70}"#;
        let result = parse_overall_result(raw, vec![], "s1");
        assert!(result.is_err());
    }

    #[test]
    fn finalize_empty_never_calls_stage_two() {
        let result = finalize_empty("s1");
        assert_eq!(result.status, "completed");
        assert!(result.questions.is_empty());
    }
}
