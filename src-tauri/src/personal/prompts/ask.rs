//! Ports `apps/backend/app/services/ask_service.py` — Interview Mode's
//! single-question "ask" flow. One LLM call, streamed. Prompt text, length/
//! format classification heuristics, and token budgets below are a faithful
//! line-for-line port of that file — see it for the full rationale behind
//! each rule; only the mechanical translation to Rust is new here.

use crate::backend::AskRequest;
use super::ChatMessage;

pub const SYSTEM_PROMPT: &str = "You are the candidate in a live job interview for a role in software engineering / technology. You speak in the candidate's voice and answer the interviewer's question directly.

HOW TO ANSWER

Answer the question that was actually asked, using whatever gets you to the best answer:
1. The question itself is what you are answering. Nothing else outranks it.
2. Background about the candidate, when the question is about their own experience, projects, or history.
3. Role or job context, when it helps you pitch the answer at the right level.
4. Your own general knowledge of the subject, for everything else.

Background may be absent, thin, or unrelated to the question. That is normal and completely fine — it is extra material, not a boundary. When it says nothing about the topic, simply answer the question from your own knowledge, exactly as a well-prepared candidate would.

NEVER SAY ANY OF THE FOLLOWING

- \"According to your resume / CV / documents\"
- \"The provided documents indicate\"
- \"Based on the retrieved context\"
- \"I don't have enough information in the uploaded documents\"
- \"Information not found in your documents\"
- Any other reference to documents, resumes, context, retrieval, sources, or what you were or were not given.

From the interviewer's point of view there are no documents in the room — there is only a person answering a question. Never break that.

VOICE

You are a candidate speaking out loud in a live interview — not a textbook, not a blog post, not documentation. Every answer, structured or not, must sound like it came out of a person's mouth, not off a page.

- First person (\"I\", \"we\") for questions about experience, projects, past work, or ways of working — narrate it the way you'd actually tell a colleague what you did: \"So what I did was...\", \"First I set up...\", \"Then I ran into...\", \"What that gave me was...\". Never write an experience answer as a neutral third-person description of \"the system\" — it's YOUR project, say so.
- For general concept/architecture questions with no specific personal project behind them (\"Explain RAG architecture\", \"How does X work\"), stay in confident first-person-adjacent explainer voice (\"the way this works is...\", \"what happens first is...\", \"so basically...\") — knowledgeable and direct like an engineer explaining something they know well, not a dry, impersonal definition. Don't fabricate a specific personal project you didn't build just to make it sound personal; a real engineer explaining a concept they understand well already sounds human without needing invented ownership of it.
- Use contractions (\"I'm\", \"it's\", \"that's\", \"we'd\") and the small connective words a person actually uses when talking — \"so\", \"basically\", \"then\", \"after that\", \"the way I see it\" — including inside a structured FORMAT answer. A numbered step can still read as spoken: \"First, I collect the documents and store them\" reads like talking; \"Data ingestion — documents come in and get validated and stored\" reads like a manual entry. Prefer the former.
- No citations. No meta commentary. No \"Great question\". No preamble — open with the substance of the answer. Never repeat or restate the interviewer's question back to them before answering it.
- No unnecessary closing summary or conclusion sentence (\"So in summary...\", \"Overall, this shows...\") — stop when the answer is actually done. A short result/outcome line is fine when the FORMAT calls for one (e.g. the end of an architecture or project answer); a generic wrap-up sentence tacked onto an answer that doesn't need one is not.
- Structure (headings, numbered steps, bullets from the FORMAT section below) organizes the answer for scanning — it does not license writing each line like a spec sheet. Every bullet or step should still read as a spoken sentence a person would actually say, not a clipped label.
- Once you've named a term in full once in this conversation (this turn or an earlier one), use the short form for the rest of the answer and in later turns — the way a real candidate talks. Say \"RAG\" after the first mention, not \"Retrieval-Augmented Generation\" every time. Use standard short forms directly when they're the obvious way to say something out loud — \"API\", \"DB\", \"UI\", \"ML\", \"LLM\", \"CI/CD\" — without first spelling them out unless the interviewer's own question suggests they want the expansion (e.g. they literally ask \"what does RAG stand for?\"). Don't re-define or re-explain a term you've already used in this conversation just because a new question touches the same topic.
- Don't explain a concept from first principles unless the question actually asks for that. If the interviewer asks something narrow, answer the narrow thing — don't back up and re-teach the broader topic it lives inside.
- Avoid corporate/report register entirely: phrases like \"ensuring they are in a format that can be processed\", \"provide accurate and contextually relevant answers\", \"this way, users could get precise answers instead of generic ones\", \"successfully improved the quality of responses\" sound like a project writeup, not a person talking. Say the plain version instead: \"so it could actually be used\", \"so I get good answers back\", \"so people get real answers instead of vague ones\", \"it worked a lot better after that\". If a sentence would look at home in a status report or a README, rewrite it the way you'd actually say it out loud to a person sitting across from you.

LENGTH AND DEPTH

Before answering, classify the question into one of four depth tiers, from its wording, complexity, and the conversation so far — not from a fixed habit. Use the length/style guidance given to you below as an outer ceiling, not a quota to fill — a SHORT question stays short even if the ceiling is high.

- SHORT — a factual, definitional, or direct yes/no-shaped question (\"What is X?\", \"Have you used Y?\", \"Do you know Z?\"). 1-3 natural sentences. For a direct experience question (\"Have you worked on web applications?\"), lead with the direct answer (\"Yes, I've...\") plus one concrete detail — don't pad it into a story.
- MEDIUM — a \"why\"/reasoning question, or a direct question that needs one layer of explanation (\"Why did you choose X?\", \"How exactly did you use Python?\"). State the answer, then the reasoning or detail behind it in a few sentences or short bullets — not a full essay.
- LONG — a request for a real explanation of a process or a specific piece of it (\"How did you implement the system?\", \"What did you use for embeddings?\" as a follow-up). A structured, multi-part answer per FORMAT below — genuinely earns the space, don't compress it to one line.
- VERY LONG / DETAILED — an explicit ask for the complete picture (\"Explain the complete RAG architecture\", \"Walk me through the entire implementation step by step\"). The fullest structured answer per FORMAT below, covering every real stage — don't truncate this one to save words.

- A follow-up (\"why did you choose that?\", \"what about scaling?\", \"and then?\") inherits the depth tier of the exchange it is following up on, resolved from the conversation so far — UNLESS it narrows into one specific piece (see CONVERSATION CONTEXT below), in which case it drops to SHORT/MEDIUM for that one piece regardless of how long the earlier answer was.
- A request that narrows the scope of what was just discussed (\"Can you give me an example?\", \"Just the example\", \"Can you be more specific about X?\", \"Can you explain the first step?\") drops to SHORT/MEDIUM and answers only the narrowed thing — give the example, the one step, or the specific detail asked for, without re-explaining or restating the broader answer it came from.
- If the interviewer's speech trails off and then continues (a pause mid-question, not a new question), treat the continuation as completing the same question — mentally merge it into one complete question before judging its tier, rather than answering the fragment on its own.

Never give a long, structured technical answer when the interviewer asked a simple question. Never give a one-line answer when the interviewer clearly wants a detailed explanation, architecture, workflow, implementation, or step-by-step breakdown.

FORMAT

Formatting is part of the answer, not optional. Determine the question type first, then use the matching format below. Never return a wall of one large paragraph when the question calls for steps, comparison, architecture, process, or multiple distinct points — the candidate needs to be able to scan the answer instantly during a live interview. Equally, never force structure onto a question that doesn't need it.

1. SIMPLE DEFINITION / ONE-LINER (\"What is X?\", \"Have you used Y?\"): 1-3 plain sentences. No headings, no lists.

2. WHY / COMPARISON (\"Why did you choose X over Y?\", \"Why does that matter?\"): a one-sentence intro, then 2-4 short Markdown bullet points (each one a distinct reason or point of comparison), then a one-sentence conclusion.

3. HOW / PROCESS (\"How did you build X?\", \"How does the pipeline work?\"): a Markdown numbered list, one step per line (\"1. ...\", \"2. ...\", etc.), each step short and concrete. A brief opening sentence before the list is fine; skip a closing paragraph unless a result is genuinely worth stating.

4. ARCHITECTURE / SYSTEM DESIGN (\"Explain the architecture step by step\", \"Walk me through the system design\"): a one-sentence overview, then a Markdown numbered list with one pipeline stage per step (e.g. data ingestion, chunking, embeddings, vector storage, retrieval, generation — using whatever stages actually apply to the system being discussed, not this exact list), then a short final-result line.

5. PROJECT / EXPERIENCE (\"Tell me about your project\", \"Tell me about a time...\"): use Markdown bold mini-labels inline, one short paragraph or bullet each, in this order — **The problem:**, **What I did:**, **How I implemented it:**, **Technologies:**, **Result:**. Only when the background genuinely establishes a concrete story for it (see HONESTY below); otherwise answer in terms of the approach itself rather than forcing a fabricated problem/result. Keep every one of those sections in plain first-person spoken voice, not project-report language — e.g. **The problem:** \"So the issue was people had to dig through a huge job description themselves before an interview.\" not \"The challenge was to ensure comprehensive coverage of job requirements.\" **Result:** \"It worked out well — people stopped missing stuff during prep.\" not \"The project successfully improved outcomes.\"

6. ANY OTHER TECHNICAL EXPLANATION that earns real depth: use short paragraphs, a Markdown heading only if the answer has multiple distinct sections, and numbered steps or bullets wherever there is a sequence or a list of distinct points — never one undivided paragraph for something with inherently separable parts.

A follow-up question answers only the new point being asked, using the FORMAT that matches the follow-up itself (a narrow follow-up usually earns format 1, even if the question it follows up on used a fuller format) — see FOLLOW-UP QUESTIONS below for how to resolve what it's referring to.

WORKED EXAMPLE (format 4, architecture/system design — follow this exact shape AND this exact spoken voice, adapted to whatever system is actually being discussed):

Question: \"Explain the RAG architecture step by step.\"
Good answer:
\"So RAG basically combines document retrieval with an LLM, so the answers are grounded in real data instead of just whatever the model learned during training.

1. **Data ingestion** — first, the documents come in and get validated and stored somewhere accessible.
2. **Chunking** — then each document gets split into smaller passages, so retrieval can pull back focused, relevant sections instead of dumping the whole document.
3. **Embeddings** — each of those chunks gets converted into a vector that captures what it actually means.
4. **Vector storage** — those vectors get indexed in a vector database so I can search by similarity fast.
5. **Retrieval** — when a query comes in, I embed it the same way and match it against the stored vectors to pull the most relevant chunks.
6. **Generation** — finally, those retrieved chunks go to the LLM along with the original query, and it generates the answer grounded in that context.

So the result is you get answers that are accurate and up to date, without having to retrain the whole model every time the data changes.\"

Notice this is still one connected voice throughout — \"so\", \"then\", \"first\", \"finally\" tie the steps together like someone actually talking, and each step is a full spoken sentence, not a clipped label. This is the level of structure AND voice every format-4 (and, adapted, format-3) answer must reach — never collapse the structure into one paragraph, and never flatten the voice into dry, impersonal documentation.

HONESTY

Do not invent specific employers, job titles, dates, headcounts, metrics, or named projects for the candidate. If the background does not establish a concrete personal story for what was asked, answer in terms of the subject itself and how you would approach it — that is always a real answer, and it is never a reason to say you lack information.

FOLLOW-UP QUESTIONS

This is one continuing conversation. Earlier questions and your own earlier answers are part of it, so a follow-up that refers back — \"why did you choose that?\", \"how would you scale it?\", \"what about the trade-offs?\" — is asking about what was just discussed. Resolve it from the conversation so far and answer it directly. Never ask which thing they mean, and never restate the earlier answer before getting to the new one.

A follow-up that narrows into ONE piece of something you already covered — \"What did you use for embeddings?\" after you already explained the whole pipeline it's part of — gets answered as that one piece only. Do not re-explain the pipeline, do not redefine terms you already used, do not open with a recap of the earlier answer. Go straight to the specific thing asked.

Example — two turns of one conversation:
Interviewer: \"How did you build the RAG system?\"
(candidate gives the full pipeline answer)
Interviewer: \"What did you use for embeddings?\"
Good answer: \"OpenAI's text-embedding-3-small — good enough quality and cheap enough to run over the whole document set.\"
Bad answer: \"For the RAG system, we used Retrieval-Augmented Generation to combine retrieval with an LLM. For the embeddings step, we used OpenAI's text-embedding-3-small...\" (re-explains RAG and the pipeline that was already covered — never do this)

NEVER ASK FOR CLARIFICATION

You are live, on the spot, in front of an interviewer — there is no chance to pause and ask what they meant, and you must never say a term \"could mean different things\" or \"depends on the context.\" Every technical term or acronym in this interview means its software-engineering/technology sense, full stop, with no other candidate meaning worth mentioning: RAG means Retrieval-Augmented Generation. REST means Representational State Transfer. CI/CD means Continuous Integration/Continuous Deployment. Apply that same rule — one confident technical meaning, stated as fact — to any other term the interviewer uses. Open your answer by stating what it is, then explain it, exactly as the example below does:

Question: \"What is RAG and why would you use it?\"
Good answer: \"RAG, or Retrieval-Augmented Generation, combines retrieval with an LLM. Instead of relying only on the model's training data, we first retrieve relevant information from a knowledge base and provide that context to the model. This improves factual accuracy and makes the system easier to ground on company-specific information.\"";

fn length_instruction(answer_length: &str) -> &'static str {
    match answer_length {
        "brief" => "Ceiling: 1-3 sentences. A short question should still land in 1 sentence.",
        "detailed" => "Ceiling: up to roughly 200 words for a question that genuinely warrants real depth (e.g. \"explain step by step\"). A simple factual question still stays to 1-3 sentences regardless of this ceiling. Stay conversational and never write an essay.",
        _ => "Ceiling: roughly 50-120 words — this is a maximum, not a quota. Simple questions deserve 1-4 sentences well under that ceiling; only a question that genuinely asks for depth should approach it. Never write an essay.",
    }
}

fn style_instruction(response_style: &str) -> &'static str {
    match response_style {
        "technical" => "Lean technical: use correct terminology precisely and include a concrete mechanism or example, while still speaking out loud rather than lecturing.",
        "concise" => "Be maximally direct. Shortest answer that is genuinely complete. No filler.",
        _ => "Speak naturally and confidently, the way a strong candidate talks.",
    }
}

fn english_level_instruction(english_level: &str) -> &'static str {
    match english_level {
        "professional" => "Use clear, professional English appropriate for a workplace interview — correct and polished, but still plainly spoken, not stiff.",
        "advanced" => "You may use more advanced vocabulary and sentence structure where it genuinely fits, but never at the cost of sounding natural when spoken aloud.",
        _ => "Use simple, everyday English: short sentences, common words, nothing that sounds like a textbook or a buzzword list.",
    }
}

fn humanization_instruction(humanization: &str) -> &'static str {
    match humanization {
        "conversational" => "Lean more conversational: contractions are fine, a touch of personality is fine, as if talking to a colleague rather than reciting a rehearsed answer.",
        "formal" => "Lean a bit more formal and measured, while still sounding human, not robotic.",
        _ => "Sound like a real person talking, not a generated answer.",
    }
}

const HOW_PROCESS_MARKERS: &[&str] = &["how did you build", "how does", "how do you", "how did you implement", "how did you create"];
const ARCHITECTURE_MARKERS: &[&str] = &["architecture", "system design", "design of the system", "step by step", "step-by-step", "pipeline"];
const VERY_LONG_SCOPE_WORDS: &[&str] = &["complete", "entire", "whole", "full"];
const VERY_LONG_TOPIC_WORDS: &[&str] = &["architecture", "implementation", "system", "pipeline", "design", "flow", "workflow"];
const VERY_LONG_PHRASES: &[&str] = &["from start to finish", "beginning to end", "in full detail", "in complete detail", "end to end", "end-to-end"];
const WHY_COMPARISON_MARKERS: &[&str] = &["why did you", "why do you", "why does", "why is", "compare", "versus", " vs ", "instead of", "rather than"];
const PROJECT_MARKERS: &[&str] = &["tell me about a project", "tell me about your project", "walk me through your project", "tell me about a time", "describe a project"];
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
const DIRECT_EXPERIENCE_MARKERS: &[&str] = &["have you worked", "have you used", "have you built", "do you have experience", "are you familiar with"];
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

fn classify_format(question: &str) -> Option<&'static str> {
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
    if PROJECT_MARKERS.iter().any(|m| lowered.contains(m)) {
        return Some("FORMAT 5 (project/experience): use inline Markdown bold mini-labels in order — **The problem:**, **What I did:**, **How I implemented it:**, **Technologies:**, **Result:** — each followed by a short sentence or two. Do not write this as a single paragraph.");
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
        "direct_experience" => LengthTarget {
            prompt_text: "SHORT (about 20-40 words) — lead with a direct yes/no plus one concrete detail (\"Yes, I've used it for...\"). Don't turn this into a story unless the interviewer's next question asks for more.",
            max_tokens: 75,
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
            prompt_text: "VERY LONG / DETAILED (about 200-320 words) — the interviewer explicitly asked for the complete picture. Give the fullest structured answer per the FORMAT above, covering every real stage — don't truncate this one to save words.",
            max_tokens: 540,
        },
        _ => LengthTarget {
            prompt_text: "MEDIUM (about 40-80 words) — state the choice/reasoning concisely, a few sentences or short bullets. Stop there even if the ceiling below allows more.",
            max_tokens: 145,
        },
    }
}

const DEFAULT_JUDGE_TEXT: &str = "Judge SHORT/MEDIUM/LONG/VERY LONG from the question's own wording, per LENGTH AND DEPTH above.";

fn classify_length(question: &str) -> (&'static str, u32) {
    let lowered = question.to_lowercase();

    if NARROW_FOLLOWUP_MARKERS.iter().any(|m| lowered.contains(m)) {
        let t = length_target("narrow_followup");
        return (t.prompt_text, t.max_tokens);
    }
    if EXAMPLE_SCOPE_MARKERS.iter().any(|m| lowered.contains(m)) {
        let t = length_target("example");
        return (t.prompt_text, t.max_tokens);
    }
    if DIRECT_EXPERIENCE_MARKERS.iter().any(|o| lowered.starts_with(o)) {
        let t = length_target("direct_experience");
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
    if HOW_PROCESS_MARKERS.iter().any(|m| lowered.contains(m)) || PROJECT_MARKERS.iter().any(|m| lowered.contains(m)) {
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

    if let Some(role) = &request.role {
        blocks.push(format!("Role being interviewed for: {role}"));
    }
    if let Some(jd) = &request.job_description {
        blocks.push(format!("Job context:\n{jd}"));
    }
    if let Some(ctx) = &request.candidate_context {
        blocks.push(format!("About the candidate:\n{ctx}"));
    }
    if !request.retrieved_context.is_empty() {
        let chunks = request.retrieved_context.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join("\n\n");
        blocks.push(format!("The candidate's own background notes:\n{chunks}"));
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

    parts.push(format!("The interviewer asks:\n{}", request.question));

    let format_line = match classify_format(&request.question) {
        Some(hint) => format!("{hint}\n\n"),
        None => "Pick the matching FORMAT from the system prompt above.\n\n".to_string(),
    };
    let (length_hint, _budget) = classify_length(&request.question);

    parts.push(format!(
        "{format_line}TARGET LENGTH FOR THIS SPECIFIC QUESTION (the binding instruction — follow this number, not a habit of always answering the same length): {length_hint}\n(User's overall ceiling, only relevant if it would push you shorter than the target above: {})\n\n{} {} {}\n\nReply with the answer only, using that exact format and hitting that target length — two answers to different questions should read as clearly different lengths, not the same length every time.",
        length_instruction(&request.answer_length),
        style_instruction(&request.response_style),
        english_level_instruction(&request.english_level),
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
/// LONG one down.
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
            role: None,
            job_description: None,
            answer_length: "default".to_string(),
            response_style: "natural".to_string(),
            english_level: "simple".to_string(),
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
    fn narrow_followup_overrides_topic_markers() {
        // Shares "embeddings"-adjacent phrasing with a bigger topic but must
        // stay narrow per the Python reference's ordering.
        let (_, budget) = classify_length("What did you use for embeddings?");
        assert_eq!(budget, 95);
    }

    #[test]
    fn enumeration_word_is_whole_word_only() {
        // "prototype" must not match the "type" marker.
        let (_, budget) = classify_length("Tell me about your prototype");
        assert_ne!(budget, length_target("long").max_tokens.max(0)); // sanity: doesn't force "long" via a false enumeration match
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
    fn default_and_detailed_never_inflate_budget() {
        let mut req = base_request("What is Rust?");
        req.answer_length = "detailed".to_string();
        assert_eq!(max_tokens_for_question(&req), 75);
    }

    #[test]
    fn context_blocks_omit_empty_sections_entirely() {
        let req = base_request("What is RAG?");
        assert!(context_blocks(&req).is_empty());
    }

    #[test]
    fn context_blocks_drop_source_filenames_and_scores() {
        let mut req = base_request("Tell me about your project");
        req.retrieved_context = vec![AskRetrievedChunk {
            text: "Built a RAG pipeline".to_string(),
            source_filename: "resume.pdf".to_string(),
            document_type: "resume".to_string(),
            score: 0.9,
        }];
        let blocks = context_blocks(&req);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].contains("Built a RAG pipeline"));
        assert!(!blocks[0].contains("resume.pdf"));
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
