//! The agent loop itself: UNDERSTAND (the streamed response so far) ->
//! DECIDE (does it want a tool, or is it done?) -> EXECUTE TOOL -> OBSERVE
//! RESULT -> UPDATE STATE -> DECIDE NEXT ACTION -> repeat. Provider-agnostic
//! — this function only ever sees `AgentEvent`s (see `types.rs`), never a
//! provider's own wire format; `personal::providers::{anthropic,openai,gemini}`
//! each translate their own streaming response into that shared shape via
//! `stream_agentic`.
//!
//! Bounded by `MAX_ITERATIONS`, not a fixed pre-generated plan: each
//! iteration's tool calls are decided from the model's own response to the
//! PREVIOUS iteration's tool results, not planned upfront — the loop simply
//! stops asking for another iteration once the model returns a normal
//! (non-tool-use) turn, or once the bound is hit as a runaway-loop safety
//! rail.

use base64::Engine;

use crate::actions::{self, ToolOutcome};
use crate::state::CancelToken;

use super::tool_schema::{self, ToolSpec};
use super::types::{AgentContent, AgentEvent, AgentMessage, StopReason};

/// Prevents a genuinely runaway tool-calling loop (a model that keeps
/// calling tools forever without ever settling on a final answer) from
/// hanging a turn indefinitely — not a "generate N steps and execute them
/// blindly" plan; each of these iterations still only exists because the
/// PREVIOUS one asked for it.
const MAX_ITERATIONS: usize = 6;

#[derive(Debug)]
pub struct AgentOutcome {
    pub final_text: String,
    /// One short summary per tool call made this turn, in order — folded
    /// into `WorkingState.recent_actions` by the caller.
    pub actions_taken: Vec<String>,
    pub iterations_used: usize,
}

/// One provider call for one iteration of the loop — implemented by each of
/// `personal::providers::{anthropic,openai,gemini}::stream_agentic`, and the
/// only thing `run_agent_loop` needs to be generic over providers.
/// `Send + Sync` supertrait: Tauri commands run on an async executor that
/// requires the whole command future to be `Send` (it may be polled from a
/// different worker thread than it was created on) — see
/// `veronica::ask_veronica`, which holds a `Box<dyn AgenticProvider>` across
/// its `.await` on `run_agent_loop`.
pub trait AgenticProvider: Send + Sync {
    /// `cancel` must be checked *inside* the streaming loop (per network
    /// chunk), not only by the caller between calls — a single provider
    /// call can run for seconds, and without this, a turn cancelled by a
    /// new utterance (barge-in) or a fast follow-up would keep emitting
    /// text/tool-call deltas into a TTS session and event stream that a
    /// NEWER turn has already reset (`TtsSession::begin_turn`), audibly
    /// mixing the two. Implementations must return `Err("cancelled".into())`
    /// (never `Ok(())`) as soon as `cancel.is_cancelled()` is observed — an
    /// `Ok` here previously let a mid-stream-cancelled turn's partial text
    /// be treated as a normal completed answer instead of being cleanly
    /// discarded.
    fn stream_agentic<'a>(
        &'a self,
        messages: &'a [AgentMessage],
        tools: &'a [ToolSpec],
        cancel: &'a CancelToken,
        on_event: &'a mut (dyn FnMut(AgentEvent) + Send),
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>>;
}

/// Runs the loop for one user turn, given the full message history
/// (including this turn's new user message already appended by the
/// caller). `on_text_delta` is called for every text token as it streams,
/// exactly like `DirectLlmClient::ask_stream`'s callback — so the caller can
/// feed it straight into `veronica:answer-delta` events and the TTS
/// chunker, unchanged from how a plain (non-agentic) answer already streams.
pub async fn run_agent_loop(
    provider: &dyn AgenticProvider,
    mut messages: Vec<AgentMessage>,
    cancel: &CancelToken,
    turn_id: &str,
    mut on_text_delta: impl FnMut(&str) + Send,
    mut on_progress: impl FnMut(&str) + Send,
) -> Result<AgentOutcome, String> {
    let tools = tool_schema::all_tools();
    let mut actions_taken = Vec::new();
    let mut final_text = String::new();
    let mut iterations_used = 0;

    log::info!("[LLM_START] turn_id={turn_id}");

    for iteration in 0..MAX_ITERATIONS {
        if cancel.is_cancelled() {
            log::info!("[INTERRUPT] turn_id={turn_id} agent loop cancelled before iteration {iteration}");
            return Err("cancelled".to_string());
        }
        iterations_used += 1;

        let mut turn_text = String::new();
        let mut token_count = 0u32;
        let mut pending_tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        {
            let mut on_event = |event: AgentEvent| match event {
                AgentEvent::TextDelta(delta) => {
                    token_count += 1;
                    // debug, not info: a single answer can stream dozens to
                    // hundreds of these — [LLM_COMPLETE] below logs the
                    // aggregate (token/char count) at info level instead.
                    log::debug!("[LLM_TOKEN] turn_id={turn_id} iteration={iteration} n={token_count} text={delta:?}");
                    turn_text.push_str(&delta);
                    on_text_delta(&delta);
                }
                AgentEvent::ToolCallReady { id, name, input } => {
                    log::info!("[LLM_COMPLETE] turn_id={turn_id} iteration={iteration} tool_call name={name} id={id}");
                    pending_tool_calls.push((id, name, input));
                }
                AgentEvent::Done { stop_reason: sr } => stop_reason = sr,
            };
            if let Err(err) = provider.stream_agentic(&messages, &tools, cancel, &mut on_event).await {
                if err == "cancelled" {
                    log::info!("[INTERRUPT] turn_id={turn_id} agent loop cancelled mid-stream at iteration {iteration}");
                } else {
                    log::error!("[TURN_ERROR] turn_id={turn_id} iteration={iteration} provider stream failed: {err}");
                }
                return Err(err);
            }
        }

        log::info!(
            "[LLM_COMPLETE] turn_id={turn_id} iteration={iteration} tokens={token_count} chars={} stop_reason={stop_reason:?} tool_calls={}",
            turn_text.len(),
            pending_tool_calls.len()
        );

        final_text = turn_text.clone();

        if pending_tool_calls.is_empty() || stop_reason != StopReason::ToolUse {
            // A normal answer — UNDERSTAND/DECIDE concluded "no tool
            // needed," so the loop is done.
            break;
        }

        // Record the assistant's turn (its text so far, plus every tool
        // call it asked for) before appending results — every provider
        // needs its own prior tool_use turn replayed for the tool_result
        // that follows to make sense.
        let mut assistant_content: Vec<AgentContent> = Vec::new();
        if !turn_text.is_empty() {
            assistant_content.push(AgentContent::Text(turn_text));
        }
        for (id, name, input) in &pending_tool_calls {
            assistant_content.push(AgentContent::ToolUse { id: id.clone(), name: name.clone(), input: input.clone() });
        }
        messages.push(AgentMessage::assistant(assistant_content));

        // EXECUTE TOOL -> OBSERVE RESULT, for every tool call this
        // iteration asked for (a model can ask for more than one at once).
        // `on_progress` fires right before each one runs — a short,
        // human-readable line ("Checking your documents...") the caller
        // streams into the conversation exactly like a text delta, so a
        // multi-step task reads as natural progress rather than going
        // silent while tools execute (requirement 11). Never the raw tool
        // name or JSON arguments — see `tool_schema::progress_message`.
        let mut results = Vec::new();
        for (id, name, input) in pending_tool_calls {
            if cancel.is_cancelled() {
                log::info!("[INTERRUPT] turn_id={turn_id} agent loop cancelled before executing tool call {name}");
                return Err("cancelled".to_string());
            }
            on_progress(&tool_schema::progress_message(&name, &input));
            let (text, image, is_error) = execute_one_tool_call(&name, &input).await;
            actions_taken.push(format!("{name}({input}) -> {text}"));
            results.push(AgentContent::ToolResult { tool_use_id: id, text, image, is_error });
        }
        // UPDATE STATE: the tool results become the next iteration's input,
        // which is what lets the model DECIDE NEXT ACTION from what it just
        // observed instead of a pre-committed plan.
        messages.push(AgentMessage::tool_results(results));
    }

    log::info!("[LLM_COMPLETE] turn_id={turn_id} agent loop done — iterations_used={iterations_used} actions_taken={}", actions_taken.len());
    Ok(AgentOutcome { final_text, actions_taken, iterations_used })
}

async fn execute_one_tool_call(name: &str, input: &serde_json::Value) -> (String, Option<(&'static str, String)>, bool) {
    match tool_schema::parse_tool_call(name, input) {
        Ok(capability) => match actions::execute_tool(&capability).await {
            Ok(ToolOutcome::Text(text)) => (text, None, false),
            Ok(ToolOutcome::Image { media_type, png_bytes }) => {
                let data_base64 = base64::engine::general_purpose::STANDARD.encode(png_bytes);
                ("Screenshot captured.".to_string(), Some((media_type, data_base64)), false)
            }
            Err(err) => (err, None, true),
        },
        Err(err) => (err, None, true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A scripted provider: returns one fixed sequence of `AgentEvent`
    /// batches, one batch per call to `stream_agentic` — lets the loop's
    /// control flow (stop on non-tool-use, bounded iterations, message
    /// accumulation) be tested without any real network call.
    struct ScriptedProvider {
        batches: std::sync::Mutex<Vec<Vec<AgentEvent>>>,
        calls: AtomicUsize,
    }

    impl AgenticProvider for ScriptedProvider {
        fn stream_agentic<'a>(
            &'a self,
            _messages: &'a [AgentMessage],
            _tools: &'a [ToolSpec],
            _cancel: &'a CancelToken,
            on_event: &'a mut (dyn FnMut(AgentEvent) + Send),
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
            Box::pin(async move {
                self.calls.fetch_add(1, Ordering::SeqCst);
                let batch = {
                    let mut guard = self.batches.lock().unwrap();
                    if guard.is_empty() {
                        Vec::new()
                    } else {
                        guard.remove(0)
                    }
                };
                for event in batch {
                    on_event(event);
                }
                Ok(())
            })
        }
    }

    fn text_only_provider(text: &str) -> ScriptedProvider {
        ScriptedProvider {
            batches: std::sync::Mutex::new(vec![vec![
                AgentEvent::TextDelta(text.to_string()),
                AgentEvent::Done { stop_reason: StopReason::EndTurn },
            ]]),
            calls: AtomicUsize::new(0),
        }
    }

    // Plain `#[test]` + `tauri::async_runtime::block_on` throughout (rather
    // than `#[tokio::test]`) since this crate's `tokio` dependency only
    // enables the `time` feature, not `macros`/`rt` — this matches how the
    // rest of the crate already drives async code from sync contexts (e.g.
    // `stt::groq`'s blocking call site) without adding new tokio features
    // just for tests.

    #[test]
    fn a_plain_text_answer_stops_after_one_iteration() {
        tauri::async_runtime::block_on(async {
            let provider = text_only_provider("Hello there.");
            let cancel = CancelToken::new();
            let mut collected = String::new();
            let outcome = run_agent_loop(&provider, vec![AgentMessage::user_text("hi")], &cancel, "test-turn", |d| collected.push_str(d), |_| {})
                .await
                .unwrap();
            assert_eq!(outcome.final_text, "Hello there.");
            assert_eq!(collected, "Hello there.");
            assert_eq!(outcome.iterations_used, 1);
            assert!(outcome.actions_taken.is_empty());
        });
    }

    #[test]
    fn a_tool_call_then_final_answer_takes_two_iterations() {
        tauri::async_runtime::block_on(async {
            let provider = ScriptedProvider {
                batches: std::sync::Mutex::new(vec![
                    vec![
                        AgentEvent::ToolCallReady { id: "1".to_string(), name: "system_info".to_string(), input: serde_json::json!({"kind": "time"}) },
                        AgentEvent::Done { stop_reason: StopReason::ToolUse },
                    ],
                    vec![AgentEvent::TextDelta("It's now.".to_string()), AgentEvent::Done { stop_reason: StopReason::EndTurn }],
                ]),
                calls: AtomicUsize::new(0),
            };
            let cancel = CancelToken::new();
            let outcome = run_agent_loop(&provider, vec![AgentMessage::user_text("what time is it")], &cancel, "test-turn", |_| {}, |_| {}).await.unwrap();
            assert_eq!(outcome.iterations_used, 2);
            assert_eq!(outcome.final_text, "It's now.");
            assert_eq!(outcome.actions_taken.len(), 1);
            assert!(outcome.actions_taken[0].starts_with("system_info"));
        });
    }

    #[test]
    fn cancellation_stops_the_loop_before_the_next_iteration() {
        tauri::async_runtime::block_on(async {
            let provider = ScriptedProvider {
                batches: std::sync::Mutex::new(vec![vec![
                    AgentEvent::ToolCallReady { id: "1".to_string(), name: "system_info".to_string(), input: serde_json::json!({"kind": "time"}) },
                    AgentEvent::Done { stop_reason: StopReason::ToolUse },
                ]]),
                calls: AtomicUsize::new(0),
            };
            let cancel = CancelToken::new();
            cancel.cancel();
            let result = run_agent_loop(&provider, vec![AgentMessage::user_text("what time is it")], &cancel, "test-turn", |_| {}, |_| {}).await;
            assert!(result.is_err());
        });
    }

    #[test]
    fn a_provider_cancelled_mid_stream_never_returns_its_partial_text_as_a_real_answer() {
        // Regression test for the exact live-observed bug: a provider that
        // returns `Ok(())` (instead of `Err("cancelled")`) after observing
        // cancellation mid-stream used to make `run_agent_loop` treat
        // whatever partial text had streamed so far as a normal, complete
        // answer — a stale/interrupted turn's fragment getting spoken and
        // returned as if it were real, racing whatever the newer turn that
        // superseded it was doing.
        struct CancelsMidStreamProvider;
        impl AgenticProvider for CancelsMidStreamProvider {
            fn stream_agentic<'a>(
                &'a self,
                _messages: &'a [AgentMessage],
                _tools: &'a [ToolSpec],
                _cancel: &'a CancelToken,
                on_event: &'a mut (dyn FnMut(AgentEvent) + Send),
            ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send + 'a>> {
                Box::pin(async move {
                    // Streams a plausible-looking partial answer, THEN
                    // discovers cancellation and correctly reports it as an
                    // error instead of quietly returning Ok — this is the
                    // fixed behavior every real provider's per-chunk check
                    // must match.
                    on_event(AgentEvent::TextDelta("This looks like a real an".to_string()));
                    Err("cancelled".to_string())
                })
            }
        }
        tauri::async_runtime::block_on(async {
            let provider = CancelsMidStreamProvider;
            let cancel = CancelToken::new();
            let mut collected = String::new();
            let result = run_agent_loop(&provider, vec![AgentMessage::user_text("hi")], &cancel, "test-turn", |d| collected.push_str(d), |_| {}).await;
            assert!(result.is_err(), "a mid-stream cancellation must surface as an Err, never Ok(partial_text)");
            assert_eq!(result.unwrap_err(), "cancelled");
            // The delta DID reach the caller as it streamed (that's correct
            // — text already spoken/shown can't be un-sent), but the loop's
            // own return value must never claim this partial fragment was
            // the real, final answer.
            assert_eq!(collected, "This looks like a real an");
        });
    }

    #[test]
    fn a_provider_that_always_wants_a_tool_is_bounded_by_max_iterations() {
        tauri::async_runtime::block_on(async {
            // Every batch asks for another tool call and reports ToolUse —
            // an infinite-loop model. The loop must still terminate.
            let infinite_batch = || {
                vec![
                    AgentEvent::ToolCallReady { id: "x".to_string(), name: "system_info".to_string(), input: serde_json::json!({"kind": "time"}) },
                    AgentEvent::Done { stop_reason: StopReason::ToolUse },
                ]
            };
            let provider = ScriptedProvider {
                batches: std::sync::Mutex::new((0..100).map(|_| infinite_batch()).collect()),
                calls: AtomicUsize::new(0),
            };
            let cancel = CancelToken::new();
            let outcome = run_agent_loop(&provider, vec![AgentMessage::user_text("loop forever")], &cancel, "test-turn", |_| {}, |_| {}).await.unwrap();
            assert_eq!(outcome.iterations_used, MAX_ITERATIONS);
            assert_eq!(provider.calls.load(Ordering::SeqCst), MAX_ITERATIONS);
        });
    }

    #[test]
    fn execute_one_tool_call_reports_unknown_tool_as_an_error_result() {
        let result = tauri::async_runtime::block_on(execute_one_tool_call("not_a_real_tool", &serde_json::json!({})));
        assert!(result.2, "unknown tool should be reported as an error result, not silently ignored");
    }
}
