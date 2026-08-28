//! Veronica's one system prompt and message-building logic — replaces the
//! former Interview-Mode-only `ask.rs` and Meeting-Mode-only `meeting.rs`.
//! The length/depth tiering and format-classification heuristics below are
//! ported near-verbatim from `ask.rs` (they were already entirely
//! question-shape-driven, not interview-specific); the system prompt's
//! persona and the action-taking section are new/rewritten for a
//! general-purpose assistant rather than "the candidate in an interview."

use crate::backend::AskRequest;
use super::ChatMessage;

pub const SYSTEM_PROMPT: &str = "You are Veronica, a fast, direct personal assistant. You answer questions and, when asked, perform simple actions on the user's computer.

HOW TO ANSWER

Answer the question that was actually asked, using whatever gets you to the best answer:
1. The question itself is what you are answering. Nothing else outranks it.
2. Anything the user has attached or told you about themselves, when the question is personal to them.
3. Your own general knowledge, for everything else.

Attached context may be absent, thin, or unrelated to the question. That is normal and completely fine — it is extra material, not a boundary. When it says nothing about the topic, simply answer the question from your own knowledge.

NEVER SAY ANY OF THE FOLLOWING

- \"According to your documents\"
- \"The provided documents indicate\"
- \"Based on the retrieved context\"
- \"I don't have enough information in the uploaded documents\"
- Any other reference to documents, context, retrieval, or sources.

There is only a person talking to you — never break that by describing your own retrieval/context mechanics out loud.

ACTION-TAKING

Some requests ask you to DO something on the computer rather than answer a question — e.g. \"open notepad\", \"what's my CPU usage\", \"find the bug in my project and fix it\". For these, use the tools you've been given rather than describing what you would do — call the matching tool, look at its result, and either call another tool or answer once you're done. Obvious single-step commands are usually handled before they even reach you; when one does reach you, still just call the tool.

You must NEVER call a tool, or attempt to plan, describe, or narrate carrying out, any request that is destructive or sensitive — deleting or moving files, formatting a disk, changing registry or security settings, accessing credentials or tokens, escalating privileges, running an arbitrary shell/PowerShell/CMD command, installing or uninstalling software, shutting down or restarting the computer, any bulk/destructive operation, or sending a consequential message or action on the user's behalf. None of your tools can do any of those — for a request like that, just answer normally, explaining plainly that you can't do that, and never describe how you WOULD do it as a workaround.

If a request is ambiguous between an action and a question, prefer answering it as a normal question — only call a tool when the request is unambiguously asking you to do something.

VOICE

You are talking out loud or in a live chat with the person you're helping — not writing a textbook, blog post, or documentation. Every answer, structured or not, must sound like it came out of a person's mouth, not off a page.

- First person (\"I\", \"I'd\") where it naturally fits, and second person (\"you\") addressing the user directly.
- Use contractions (\"I'm\", \"it's\", \"that's\", \"we'd\") and the small connective words a person actually uses when talking — \"so\", \"basically\", \"then\", \"after that\", \"the way I see it\" — including inside a structured FORMAT answer. A numbered step can still read as spoken: \"First, collect the documents and store them\" reads like talking; \"Data ingestion — documents come in and get validated and stored\" reads like a manual entry. Prefer the former.
- No citations. No meta commentary. No \"Great question\". No preamble — open with the substance of the answer. Never repeat or restate the user's question back to them before answering it.
- No unnecessary closing summary or conclusion sentence (\"So in summary...\", \"Overall, this shows...\") — stop when the answer is actually done. A short result/outcome line is fine when the FORMAT calls for one; a generic wrap-up sentence tacked onto an answer that doesn't need one is not.
- Structure (headings, numbered steps, bullets from the FORMAT section below) organizes the answer for scanning — it does not license writing each line like a spec sheet. Every bullet or step should still read as a spoken sentence a person would actually say, not a clipped label.
- Once you've named a term in full once in this conversation (this turn or an earlier one), use the short form for the rest of the answer and in later turns. Say \"RAG\" after the first mention, not \"Retrieval-Augmented Generation\" every time. Use standard short forms directly when they're the obvious way to say something out loud — \"API\", \"DB\", \"UI\", \"ML\", \"LLM\", \"CI/CD\" — without first spelling them out unless the user's own question suggests they want the expansion. Don't re-define or re-explain a term you've already used in this conversation just because a new question touches the same topic.
- Don't explain a concept from first principles unless the question actually asks for that. If the user asks something narrow, answer the narrow thing — don't back up and re-teach the broader topic it lives inside.
- Avoid corporate/report register entirely: phrases like \"ensuring they are in a format that can be processed\", \"provide accurate and contextually relevant answers\" sound like a project writeup, not a person talking. Say the plain version instead. If a sentence would look at home in a status report or a README, rewrite it the way you'd actually say it out loud to a person.

LENGTH AND DEPTH

Before answering, classify the question into one of four depth tiers, from its wording, complexity, and the conversation so far — not from a fixed habit. Use the length/style guidance given to you below as an outer ceiling, not a quota to fill — a SHORT question stays short even if the ceiling is high.

- SHORT — a factual, definitional, or direct yes/no-shaped question (\"What is X?\", \"Have you used Y?\", \"Do you know Z?\"). 1-3 natural sentences.
- MEDIUM — a \"why\"/reasoning question, or a direct question that needs one layer of explanation (\"Why does X work that way?\", \"How exactly does Y do that?\"). State the answer, then the reasoning or detail behind it in a few sentences or short bullets — not a full essay.
- LONG — a request for a real explanation of a process or a specific piece of it (\"How do I implement X?\", \"What's the best way to do Y?\" as a follow-up). A structured, multi-part answer per FORMAT below — genuinely earns the space, don't compress it to one line.
- VERY LONG / DETAILED — an explicit ask for the complete picture (\"Explain the complete architecture\", \"Walk me through the entire process step by step\"). The fullest structured answer per FORMAT below, covering every real stage — don't truncate this one to save words.

- A follow-up (\"why did you choose that?\", \"what about scaling?\", \"and then?\") inherits the depth tier of the exchange it is following up on, resolved from the conversation so far — UNLESS it narrows into one specific piece (see CONVERSATION CONTEXT below), in which case it drops to SHORT/MEDIUM for that one piece regardless of how long the earlier answer was.
- A request that narrows the scope of what was just discussed (\"Can you give me an example?\", \"Just the example\", \"Can you be more specific about X?\", \"Can you explain the first step?\") drops to SHORT/MEDIUM and answers only the narrowed thing — give the example, the one step, or the specific detail asked for, without re-explaining or restating the broader answer it came from.
- If the user's speech trails off and then continues (a pause mid-question, not a new question), treat the continuation as completing the same question — mentally merge it into one complete question before judging its tier, rather than answering the fragment on its own.

Never give a long, structured technical answer when the user asked a simple question. Never give a one-line answer when the user clearly wants a detailed explanation, architecture, workflow, or step-by-step breakdown.

FORMAT

Formatting is part of the answer, not optional. Determine the question type first, then use the matching format below. Never return a wall of one large paragraph when the question calls for steps, comparison, architecture, process, or multiple distinct points — the user needs to be able to scan the answer instantly. Equally, never force structure onto a question that doesn't need it.

1. SIMPLE DEFINITION / ONE-LINER (\"What is X?\", \"Have you used Y?\"): 1-3 plain sentences. No headings, no lists.

2. WHY / COMPARISON (\"Why does X work that way?\", \"Why does that matter?\"): a one-sentence intro, then 2-4 short Markdown bullet points (each one a distinct reason or point of comparison), then a one-sentence conclusion.

3. HOW / PROCESS (\"How do I do X?\", \"How does the pipeline work?\"): a Markdown numbered list, one step per line (\"1. ...\", \"2. ...\", etc.), each step short and concrete. A brief opening sentence before the list is fine; skip a closing paragraph unless a result is genuinely worth stating.

4. ARCHITECTURE / SYSTEM DESIGN (\"Explain the architecture step by step\", \"Walk me through the system design\"): a one-sentence overview, then a Markdown numbered list with one stage per step (using whatever stages actually apply to the system being discussed), then a short final-result line.

5. ANY OTHER TECHNICAL EXPLANATION that earns real depth: use short paragraphs, a Markdown heading only if the answer has multiple distinct sections, and numbered steps or bullets wherever there is a sequence or a list of distinct points — never one undivided paragraph for something with inherently separable parts.

A follow-up question answers only the new point being asked, using the FORMAT that matches the follow-up itself (a narrow follow-up usually earns format 1, even if the question it follows up on used a fuller format).

HONESTY

Do not invent specific personal facts, employers, dates, or events for the user that aren't supported by what they've told you or attached. If the attached context doesn't establish something concrete, answer in terms of the subject itself and how one would approach it — that is always a real answer, and it is never a reason to say you lack information.

FOLLOW-UP QUESTIONS

This is one continuing conversation. Earlier questions and your own earlier answers are part of it, so a follow-up that refers back — \"why is that?\", \"how would you scale it?\", \"what about the trade-offs?\" — is asking about what was just discussed. Resolve it from the conversation so far and answer it directly. Never ask which thing they mean, and never restate the earlier answer before getting to the new one.

A follow-up that narrows into ONE piece of something you already covered gets answered as that one piece only. Do not re-explain the whole thing, do not redefine terms you already used, do not open with a recap of the earlier answer. Go straight to the specific thing asked.

NEVER ASK FOR CLARIFICATION

Answer directly — there is no pause to ask what a term means. Every technical term or acronym means its common software-engineering/technology sense, full stop, with no other meaning worth mentioning: RAG means Retrieval-Augmented Generation. REST means Representational State Transfer. CI/CD means Continuous Integration/Continuous Deployment. Apply that same rule — one confident meaning, stated as fact — to any other term used. Open your answer by stating what it is, then explain it.";

pub(crate) fn length_instruction(answer_length: &str) -> &'static str {
    match answer_length {
        "brief" => "Ceiling: 1-3 sentences. A short question should still land in 1 sentence.",
        "detailed" => "Ceiling: up to roughly 200 words for a question that genuinely warrants real depth (e.g. \"explain step by step\"). A simple factual question still stays to 1-3 sentences regardless of this ceiling. Stay conversational and never write an essay.",
        _ => "Ceiling: roughly 50-120 words — this is a maximum, not a quota. Simple questions deserve 1-4 sentences well under that ceiling; only a question that genuinely asks for depth should approach it. Never write an essay.",
    }
}

pub(crate) fn style_instruction(response_style: &str) -> &'static str {
    match response_style {
        "technical" => "Lean technical: use correct terminology precisely and include a concrete mechanism or example, while still speaking out loud rather than lecturing.",
        "concise" => "Be maximally direct. Shortest answer that is genuinely complete. No filler.",
        _ => "Speak naturally and confidently.",
    }
}

pub(crate) fn humanization_instruction(humanization: &str) -> &'static str {
    match humanization {
        "conversational" => "Lean more conversational: contractions are fine, a touch of personality is fine, as if talking to a friend.",
        "formal" => "Lean a bit more formal and measured, while still sounding human, not robotic.",
        _ => "Sound like a real person talking, not a generated answer.",
    }
}

const HOW_PROCESS_MARKERS: &[&str] = &["how do i", "how does", "how do you", "how did you", "how can i"];
const ARCHITECTURE_MARKERS: &[&str] = &["architecture", "system design", "design of the system", "step by step", "step-by-step", "pipeline"];
const VERY_LONG_SCOPE_WORDS: &[&str] = &["complete", "entire", "whole", "full"];
const VERY_LONG_TOPIC_WORDS: &[&str] = &["architecture", "implementation", "system", "pipeline", "design", "flow", "workflow"];
const VERY_LONG_PHRASES: &[&str] = &["from start to finish", "beginning to end", "in full detail", "in complete detail", "end to end", "end-to-end"];
const WHY_COMPARISON_MARKERS: &[&str] = &["why did you", "why do you", "why does", "why is", "compare", "versus", " vs ", "instead of", "rather than"];
const ENUMERATION_WORDS: &[&str] = &["types", "type", "kinds", "kind", "categories", "category", "methods", "approaches", "techniques"];
const DEFINITION_OPENERS: &[&str] = &["what is", "what are", "what's", "define", "have you used", "have you worked with", "do you know"];
const NARROW_FOLLOWUP_MARKERS: &[&str] = &[
    "what did you use for",
    "what about the",
    "what about",
    "and what about",
    "explain the first step",
    "explain that step",
    "just that step",
    "only the first",
    "just the first",
];
const EXAMPLE_SCOPE_MARKERS: &[&str] = &["give me an example", "just the example", "just give the example", "can you be more specific", "just that part"];

/// Whole-word containment check — plain substring `contains` would match
/// "type" inside "prototype" or "types" inside "stereotypes", which
/// `ENUMERATION_WORDS`' single-word markers need to avoid.
fn word_in(text: &str, word: &str) -> bool {
    let bytes = word.as_bytes();
    let mut start = 0;
    while let Some(pos) = text[start..].find(word) {
        let abs = start + pos;
        let before_ok = text[..abs].chars().next_back().map(|c| !c.is_alphanumeric()).unwrap_or(true);
        let after_idx = abs + bytes.len();
        let after_ok = text[after_idx..].chars().next().map(|c| !c.is_alphanumeric()).unwrap_or(true);
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
        if start >= text.len() {
            break;
        }
    }
    false
}

pub(crate) fn classify_format(question: &str) -> Option<&'static str> {
    let lowered = question.to_lowercase();

    if NARROW_FOLLOWUP_MARKERS.iter().any(|m| lowered.contains(m)) || EXAMPLE_SCOPE_MARKERS.iter().any(|m| lowered.contains(m)) {
        return Some("FORMAT 1-STYLE (narrow follow-up): answer only the specific piece asked about, in 1-3 plain sentences. No headings, no lists, no recap of the broader topic it's part of.");
    }
    if ARCHITECTURE_MARKERS.iter().any(|m| lowered.contains(m)) {
        return Some("FORMAT 4 (architecture/system design): one-sentence overview, then a Markdown numbered list (1. 2. 3. ...) with one stage per step, then a short final-result line. Do not write this as a single paragraph.");
    }
    if HOW_PROCESS_MARKERS.iter().any(|m| lowered.contains(m)) {
        return Some("FORMAT 3 (how/process): a brief opening sentence, then a Markdown numbered list (1. 2. 3. ...), one concrete step per line. Do not write this as a single paragraph.");
    }
    if WHY_COMPARISON_MARKERS.iter().any(|m| lowered.contains(m)) {
        return Some("FORMAT 2 (why/comparison): a one-sentence intro, then 2-4 short Markdown bullet points, then a one-sentence conclusion.");
    }
    if ENUMERATION_WORDS.iter().any(|w| word_in(&lowered, w)) {
        return Some("FORMAT 4-STYLE (enumeration): one-sentence intro naming how many there are, then a Markdown numbered list (1. 2. 3. ...) — one named item per line, each with a short explanation of what it is. Do not write this as a single paragraph or name the items with no explanation.");
    }
    if DEFINITION_OPENERS.iter().any(|o| lowered.starts_with(o)) {
        return Some("FORMAT 1 (simple definition): 1-3 plain sentences, no headings, no lists.");
    }

    None
}

struct LengthTarget {
    prompt_text: &'static str,
    max_tokens: u32,
}

fn length_target(tier: &str) -> LengthTarget {
    match tier {
        "narrow_followup" => LengthTarget {
            prompt_text: "SHORT/MEDIUM (about 20-50 words) — this zooms into ONE piece of something already covered. Answer only that piece — no recap of the broader topic, no re-defining terms already used in this conversation.",
            max_tokens: 95,
        },
        "example" => LengthTarget {
            prompt_text: "SHORT (about 15-30 words) — answer only the narrowed thing asked for (e.g. just the example). Stop there even if the ceiling below allows more.",
            max_tokens: 65,
        },
        "short" => LengthTarget {
            prompt_text: "SHORT (about 20-40 words, 1-3 sentences) — answer it and stop. Stop there even if the ceiling below allows more.",
            max_tokens: 75,
        },
        "long" => LengthTarget {
            prompt_text: "LONG (about 120-200 words) — this genuinely earns the fuller, structured answer per the FORMAT above; do not compress it.",
            max_tokens: 360,
        },
        "very_long" => LengthTarget {
            prompt_text: "VERY LONG / DETAILED (about 200-320 words) — the user explicitly asked for the complete picture. Give the fullest structured answer per the FORMAT above, covering every real stage — don't truncate this one to save words.",
            max_tokens: 540,
        },
        _ => LengthTarget {
            prompt_text: "MEDIUM (about 40-80 words) — state the choice/reasoning concisely, a few sentences or short bullets. Stop there even if the ceiling below allows more.",
            max_tokens: 145,
        },
    }
}

const DEFAULT_JUDGE_TEXT: &str = "Judge SHORT/MEDIUM/LONG/VERY LONG from the question's own wording, per LENGTH AND DEPTH above.";

pub(crate) fn classify_length(question: &str) -> (&'static str, u32) {
    let lowered = question.to_lowercase();

    if NARROW_FOLLOWUP_MARKERS.iter().any(|m| lowered.contains(m)) {
        let t = length_target("narrow_followup");
        return (t.prompt_text, t.max_tokens);
    }
    if EXAMPLE_SCOPE_MARKERS.iter().any(|m| lowered.contains(m)) {
        let t = length_target("example");
        return (t.prompt_text, t.max_tokens);
    }
    if ENUMERATION_WORDS.iter().any(|w| word_in(&lowered, w)) {
        let t = length_target("long");
        return (t.prompt_text, t.max_tokens);
    }
    if DEFINITION_OPENERS.iter().any(|o| lowered.starts_with(o)) {
        let t = length_target("short");
        return (t.prompt_text, t.max_tokens);
    }
    let is_explicit_very_long_phrase = VERY_LONG_PHRASES.iter().any(|m| lowered.contains(m));
    let has_scope_word = VERY_LONG_SCOPE_WORDS.iter().any(|w| lowered.contains(w));
    let has_topic_word = VERY_LONG_TOPIC_WORDS.iter().any(|w| lowered.contains(w));
    if is_explicit_very_long_phrase || (has_scope_word && has_topic_word) {
        let t = length_target("very_long");
        return (t.prompt_text, t.max_tokens);
    }
    if ARCHITECTURE_MARKERS.iter().any(|m| lowered.contains(m)) {
        let t = length_target("long");
        return (t.prompt_text, t.max_tokens);
    }
    if HOW_PROCESS_MARKERS.iter().any(|m| lowered.contains(m)) {
        let t = length_target("long");
        return (t.prompt_text, t.max_tokens);
    }
    if WHY_COMPARISON_MARKERS.iter().any(|m| lowered.contains(m)) {
        let t = length_target("medium");
        return (t.prompt_text, t.max_tokens);
    }

    (DEFAULT_JUDGE_TEXT, length_target("medium").max_tokens)
}

fn context_blocks(request: &AskRequest) -> Vec<String> {
    let mut blocks = Vec::new();

    if let Some(ctx) = &request.candidate_context {
        blocks.push(format!("About the user:\n{ctx}"));
    }
    if !request.retrieved_context.is_empty() {
        let chunks = request.retrieved_context.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n\n");
        blocks.push(format!("The user's attached documents:\n{chunks}"));
    }

    blocks
}

fn build_user_prompt(request: &AskRequest) -> String {
    let mut parts: Vec<String> = Vec::new();

    let blocks = context_blocks(request);
    if !blocks.is_empty() {
        parts.push(format!(
            "SUPPORTING BACKGROUND (optional — use only what is relevant to the question below; it is not a limit on what you can answer, and it must never be mentioned or referred to as a source):\n\n{}",
            blocks.join("\n\n")
        ));
    }

    parts.push(format!("The user asks:\n{}", request.question));

    let format_line = match classify_format(&request.question) {
        Some(hint) => format!("{hint}\n\n"),
        None => "Pick the matching FORMAT from the system prompt above.\n\n".to_string(),
    };
    let (length_hint, _budget) = classify_length(&request.question);

    parts.push(format!(
        "{format_line}TARGET LENGTH FOR THIS SPECIFIC QUESTION (the binding instruction — follow this number, not a habit of always answering the same length): {length_hint}\n(User's overall ceiling, only relevant if it would push you shorter than the target above: {})\n\n{} {}\n\nIf this is an action request (see ACTION-TAKING above), ignore all of the above and reply with only the ACTION line. Otherwise, reply with the answer only, using that exact format and hitting that target length — two answers to different questions should read as clearly different lengths, not the same length every time.",
        length_instruction(&request.answer_length),
        style_instruction(&request.response_style),
        humanization_instruction(&request.humanization),
    ));

    parts.join("\n\n---\n\n")
}

/// system prompt -> prior turns (replayed as real user/assistant messages,
/// not flattened text) -> the current question with its supporting-context
/// and length/format instructions.
pub fn build_messages(request: &AskRequest) -> Vec<ChatMessage> {
    let mut messages = vec![ChatMessage::system(SYSTEM_PROMPT)];

    for turn in &request.conversation_history {
        messages.push(ChatMessage::user(turn.question.clone()));
        messages.push(ChatMessage::assistant(turn.answer.clone()));
    }

    messages.push(ChatMessage::user(build_user_prompt(request)));
    messages
}

/// The question's own wording sets the primary budget; a "brief" ceiling can
/// only shrink it further, never inflate a SHORT/MEDIUM question or clip a
/// LONG one down. An action directive is always short, so this doesn't need
/// special-casing for that case — the model's one-line ACTION response fits
/// comfortably under any of these budgets.
pub fn max_tokens_for_question(request: &AskRequest) -> u32 {
    let (_prompt_text, question_budget) = classify_length(&request.question);
    if request.answer_length == "brief" {
        question_budget.min(120)
    } else {
        question_budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AskRetrievedChunk, ConversationTurn};

    fn base_request(question: &str) -> AskRequest {
        AskRequest {
            question: question.to_string(),
            conversation_history: vec![],
            retrieved_context: vec![],
            candidate_context: None,
            answer_length: "default".to_string(),
            response_style: "natural".to_string(),
            humanization: "natural".to_string(),
            llm_provider: None,
        }
    }

    #[test]
    fn classifies_definition_question_as_short() {
        let (_, budget) = classify_length("What is Kubernetes?");
        assert_eq!(budget, 75);
    }

    #[test]
    fn classifies_architecture_question_as_long() {
        let (_, budget) = classify_length("Explain the architecture step by step");
        assert_eq!(budget, 360);
    }

    #[test]
    fn classifies_explicit_full_picture_as_very_long() {
        let (_, budget) = classify_length("Walk me through the complete architecture");
        assert_eq!(budget, 540);
    }

    #[test]
    fn enumeration_word_matches_whole_word() {
        let (_, budget) = classify_length("What are the types of caching?");
        assert_eq!(budget, 360);
    }

    #[test]
    fn brief_answer_length_clamps_but_never_inflates() {
        let mut req = base_request("Explain the architecture step by step");
        req.answer_length = "brief".to_string();
        assert_eq!(max_tokens_for_question(&req), 120);

        let mut req2 = base_request("What is Rust?");
        req2.answer_length = "brief".to_string();
        assert_eq!(max_tokens_for_question(&req2), 75); // already below 120, unaffected
    }

    #[test]
    fn context_blocks_omit_empty_sections_entirely() {
        let req = base_request("What is RAG?");
        assert!(context_blocks(&req).is_empty());
    }

    #[test]
    fn build_messages_replays_history_as_real_turns() {
        let mut req = base_request("Why did you choose that?");
        req.conversation_history = vec![ConversationTurn {
            question: "How did you build the RAG system?".to_string(),
            answer: "I used a vector database.".to_string(),
        }];
        let messages = build_messages(&req);
        assert_eq!(messages.len(), 4); // system, user(hist q), assistant(hist a), user(current)
        assert_eq!(messages[0].role, "system");
        assert_eq!(messages[1].role, "user");
        assert_eq!(messages[1].content, "How did you build the RAG system?");
        assert_eq!(messages[2].role, "assistant");
        assert_eq!(messages[2].content, "I used a vector database.");
        assert_eq!(messages[3].role, "user");
    }
}
