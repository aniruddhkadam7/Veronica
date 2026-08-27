//! Veronica: the one assistant behind the one overlay. `ask_veronica` is the
//! single entry point the overlay calls regardless of which mode is active —
//! it just dispatches to whichever existing mode's request-building/prompt
//! path already existed (`interview_mode`'s `AskRequest`/`ask::build_messages`
//! or `meeting_mode`'s `MeetingAskRequest`/`meeting::build_ask_messages`),
//! reusing both verbatim rather than merging their prompt logic. `set_mode`/
//! `get_mode` track which mode is active in `AppState` so a voice command
//! spoken inside the overlay window (a separate React tree from the main
//! window) can switch modes without a round trip through the main window.

use tauri::{AppHandle, Emitter, State};

use crate::backend::{AskRequest, AskRetrievedChunk, ConversationTurn, MeetingAskRequest, MeetingConversationTurn, MeetingRetrievedChunk};
use crate::interview_mode::commands::AskOptions;
use crate::meeting_mode::commands::MeetingAskOptions;
use crate::rag::{RagClient, RetrievalPlanner};
use crate::state::{AppState, Mode};

/// One prior exchange in this Veronica session, as sent by the overlay —
/// shared by both modes (each mode's backend request type wants its own
/// conversation-turn struct, so this is converted into whichever one
/// `ask_veronica` needs based on `mode`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PriorTurn {
    pub question: String,
    pub answer: String,
}

/// Mirrors `interview_mode::commands::MAX_HISTORY_TURNS` /
/// `meeting_mode::commands::MAX_HISTORY_TURNS` — both were already the same
/// value (6), kept as one constant now that history-trimming is shared.
const MAX_HISTORY_TURNS: usize = 6;

fn trim_interview_history(turns: Vec<PriorTurn>) -> Vec<ConversationTurn> {
    let mut history: Vec<ConversationTurn> = turns
        .into_iter()
        .filter(|t| !t.question.trim().is_empty() && !t.answer.trim().is_empty())
        .map(|t| ConversationTurn { question: t.question.trim().to_string(), answer: t.answer.trim().to_string() })
        .collect();
    if history.len() > MAX_HISTORY_TURNS {
        history.drain(..history.len() - MAX_HISTORY_TURNS);
    }
    history
}

fn trim_meeting_history(turns: Vec<PriorTurn>) -> Vec<MeetingConversationTurn> {
    let mut history: Vec<MeetingConversationTurn> = turns
        .into_iter()
        .filter(|t| !t.question.trim().is_empty() && !t.answer.trim().is_empty())
        .map(|t| MeetingConversationTurn { question: t.question.trim().to_string(), answer: t.answer.trim().to_string() })
        .collect();
    if history.len() > MAX_HISTORY_TURNS {
        history.drain(..history.len() - MAX_HISTORY_TURNS);
    }
    history
}

#[tauri::command]
pub fn set_mode(state: State<'_, AppState>, mode: Mode) -> Result<(), String> {
    *state.veronica_mode.lock().map_err(|e| e.to_string())? = mode;
    Ok(())
}

#[tauri::command]
pub fn get_mode(state: State<'_, AppState>) -> Result<Mode, String> {
    Ok(*state.veronica_mode.lock().map_err(|e| e.to_string())?)
}

/// One question, answered in whichever mode is active:
///
///     question -> (retrieval, only when it could help) -> ONE LLM call -> stream
///
/// Interview mode reuses `interview_mode::commands`' exact flow (CV/job
/// description full-text fetch, `retrieval_could_help` gate, `AskRequest`,
/// `DirectLlmClient::ask_stream`); Meeting mode reuses `meeting_mode::
/// commands`' exact flow (unconditional retrieval, `MeetingAskRequest`,
/// `DirectLlmClient::meeting_ask_stream`) — see those modules' original
/// commands (now unregistered) for the per-mode reasoning this was copied
/// from. Streams back as `veronica:answer-delta` events, finishing with
/// `veronica:answer-complete`, regardless of mode.
#[tauri::command]
pub async fn ask_veronica(
    app: AppHandle,
    _state: State<'_, AppState>,
    question: String,
    mode: Mode,
    interview_options: Option<AskOptions>,
    meeting_options: Option<MeetingAskOptions>,
    history: Option<Vec<PriorTurn>>,
) -> Result<String, String> {
    use crate::hardware::telemetry::{finish, FirstTokenTracker, PipelineStage, Stopwatch};

    let question_to_answer = Stopwatch::start();

    let trimmed = question.trim();
    if trimmed.is_empty() {
        return Err("no question text to send".into());
    }
    let history = history.unwrap_or_default();

    let answer = match mode {
        Mode::Interview => {
            let options = interview_options.unwrap_or_default();
            let history = trim_interview_history(history);

            let resume_fetch = crate::interview_mode::commands::fetch_document_full_text("RESUME");
            let job_description_fetch = crate::interview_mode::commands::fetch_document_full_text("JOB_DESCRIPTION");

            let retrieved = if crate::interview_mode::commands::retrieval_could_help(trimmed) {
                let cfg = crate::hardware::effective_config_checked(&app);
                let planner = RetrievalPlanner::new()
                    .with_config(cfg.rag_top_k, cfg.rag_similarity_threshold, cfg.rag_max_context_chars)
                    .with_timeout(std::time::Duration::from_millis(cfg.rag_retrieval_timeout_ms));
                let retrieval_timer = Stopwatch::start();
                let results = planner.plan_for_question(trimmed).await;
                finish(retrieval_timer, PipelineStage::RagRetrieval, &crate::hardware::perf_context(&app));
                results
            } else {
                log::debug!("Veronica (Interview mode): skipping retrieval for conceptual question");
                Vec::new()
            };

            let resume_text = resume_fetch.await;
            let uploaded_job_description = job_description_fetch.await;

            let request = AskRequest {
                question: trimmed.to_string(),
                conversation_history: history,
                retrieved_context: retrieved
                    .into_iter()
                    .filter(|r| r.metadata.document_type != "RESUME" && r.metadata.document_type != "JOB_DESCRIPTION")
                    .map(|r| AskRetrievedChunk {
                        text: r.text,
                        source_filename: r.metadata.filename,
                        document_type: r.metadata.document_type,
                        score: r.score,
                    })
                    .collect(),
                candidate_context: resume_text,
                role: options.role,
                job_description: uploaded_job_description.or(options.job_description),
                answer_length: options.answer_length,
                response_style: options.response_style,
                english_level: options.english_level,
                humanization: options.humanization,
                llm_provider: options.llm_provider,
            };

            let app_for_events = app.clone();
            let llm_timer = Stopwatch::start();
            let first_token = FirstTokenTracker::new();
            let first_token_recorder = first_token.recorder();
            let on_delta = move |delta: &str| {
                first_token_recorder.mark();
                let _ = app_for_events.emit("veronica:answer-delta", delta);
            };
            let answer = crate::personal::DirectLlmClient::new(request.llm_provider.as_deref())?
                .ask_stream(&request, on_delta)
                .await?;

            let ctx = crate::hardware::perf_context(&app);
            if let Some(ms) = first_token.elapsed_ms() {
                crate::hardware::telemetry::log_stage_ms(PipelineStage::LlmFirstToken, ms, &ctx);
            }
            finish(llm_timer, PipelineStage::LlmTotal, &ctx);
            answer
        }
        Mode::Meeting => {
            let options = meeting_options.unwrap_or_default();
            let history = trim_meeting_history(history);

            let cfg = crate::hardware::effective_config_checked(&app);
            let planner = RetrievalPlanner::new()
                .with_config(cfg.rag_top_k, cfg.rag_similarity_threshold, cfg.rag_max_context_chars)
                .with_timeout(std::time::Duration::from_millis(cfg.rag_retrieval_timeout_ms));
            let retrieval_timer = Stopwatch::start();
            let retrieved = planner.plan_for_question(trimmed).await;
            finish(retrieval_timer, PipelineStage::RagRetrieval, &crate::hardware::perf_context(&app));

            let request = MeetingAskRequest {
                question: trimmed.to_string(),
                conversation_history: history,
                retrieved_context: retrieved
                    .into_iter()
                    .map(|r| MeetingRetrievedChunk {
                        text: r.text,
                        source_filename: r.metadata.filename,
                        document_type: r.metadata.document_type,
                        score: r.score,
                    })
                    .collect(),
                meeting_title: options.meeting_title,
                participants: options.participants,
                answer_length: options.answer_length,
                response_style: options.response_style,
                humanization: options.humanization,
                llm_provider: options.llm_provider,
            };

            let app_for_events = app.clone();
            let llm_timer = Stopwatch::start();
            let first_token = FirstTokenTracker::new();
            let first_token_recorder = first_token.recorder();
            let on_delta = move |delta: &str| {
                first_token_recorder.mark();
                let _ = app_for_events.emit("veronica:answer-delta", delta);
            };
            let answer = crate::personal::DirectLlmClient::new(request.llm_provider.as_deref())?
                .meeting_ask_stream(&request, on_delta)
                .await?;

            let ctx = crate::hardware::perf_context(&app);
            if let Some(ms) = first_token.elapsed_ms() {
                crate::hardware::telemetry::log_stage_ms(PipelineStage::LlmFirstToken, ms, &ctx);
            }
            finish(llm_timer, PipelineStage::LlmTotal, &ctx);
            answer
        }
    };

    finish(question_to_answer, PipelineStage::QuestionToAnswer, &crate::hardware::perf_context(&app));
    let _ = app.emit("veronica:answer-complete", &answer);
    Ok(answer)
}
