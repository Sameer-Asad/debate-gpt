"""LLM clients and node factories for the debate graph.

Per PRD §3.4:
- Pro  → Groq `allam-2-7b`, temperature 0.9
- Con  → Groq `llama-3.1-8b-instant`, temperature 0.9
- Judge → OpenRouter `openai/gpt-oss-120b`, temperature 0.2

Each node is built as a factory `(llm) -> node_fn` so Day 5 tests can inject
fake LLMs without monkey-patching globals.

Length-imbalance fix (Day 6): a prompt-only instruction ("respond in
180-220 words") was unreliable — llama-3.1-8b-instant (Con) would
routinely stop well short of that, while allam-2-7b (Pro) complied more
often, giving Pro a systematic evidence/persuasion edge unrelated to
argument quality. `_ensure_min_words` enforces the floor in CODE: if the
model's output is short, it is fed back its own partial argument and
asked to continue, up to `_MAX_CONTINUATION_CALLS` times, until the
minimum is met (or the retry budget is exhausted, in which case we log
a warning and use what we have rather than looping forever).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from .config import load_settings
from .prompts import build_con_messages, build_judge_messages, build_pro_messages
from .schemas import RoundScore
from .state import DebateState
from .verdict import con_total_for_round, pro_total_for_round

logger = logging.getLogger(__name__)


# Signature of the streaming sink the runtime passes in. Receives the
# caller-supplied event name (e.g. "pro_token"), the round number, and the
# newly-flushed content chunk. Called from inside a sync thread; must not
# raise. Used by Day 3 SSE.
ChunkSink = Callable[[str, int, str], None]


# ---------- Length enforcement (Day 6 fix) ----------

_MIN_WORDS = 180
_MAX_WORDS = 220
_MAX_CONTINUATION_CALLS = 5  # raised from 2 — some models need more nudges

# Sentence-boundary splitter for truncation. Deliberately simple (no NLP
# dependency) — splits on ". ", "! ", "? " followed by a capital letter,
# which is good enough for the debater agents' own generated prose.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _continue_prompt(current_words: int) -> str:
    """Build a continuation prompt with the EXACT remaining word count
    needed, rather than a vague "80-100 more words" ask. Telling the
    model precisely how far short it is (rather than a generic range)
    closes the gap far more reliably — a vague target lets a terse
    model round down to "close enough"."""
    remaining = max(_MIN_WORDS - current_words, 40)
    return (
        f"Your argument so far is only {current_words} words. You MUST "
        f"write at least {remaining} MORE words before you are done — "
        f"this is a strict requirement. Continue elaborating on the SAME "
        f"argument: add a second piece of supporting evidence, address an "
        f"obvious counterargument, cite another concrete example, or "
        f"explain the mechanism behind your claim in more depth. Do not "
        f"repeat what you already said. Do not summarize or conclude — "
        f"just keep building the argument. Output ONLY the additional "
        f"text, with no preamble."
    )


def _word_count(text: str) -> int:
    return len(text.split())


def _enforce_max_words(
    content: str, event_name: str, round_no: int
) -> str:
    """If `content` exceeds `_MAX_WORDS`, truncate at the last complete
    sentence boundary at or before the limit — never mid-sentence.

    This is the ceiling counterpart to `_ensure_min_words`'s floor: Pro
    (allam-2-7b) tends to run long on its own, and without an explicit
    cap it could keep growing well past 220 while Con (after continuation
    calls) lands closer to 200-220 — reintroducing the same length
    imbalance from the other direction. Both agents should land in the
    same 180-220 band.
    """
    if _word_count(content) <= _MAX_WORDS:
        return content

    sentences = _SENTENCE_BOUNDARY.split(content.strip())
    kept: list[str] = []
    word_total = 0
    for sentence in sentences:
        sentence_words = _word_count(sentence)
        if word_total + sentence_words > _MAX_WORDS and kept:
            # Adding this sentence would exceed the cap, and we already
            # have at least one sentence kept — stop here.
            break
        kept.append(sentence)
        word_total += sentence_words
        if word_total >= _MAX_WORDS:
            break

    truncated = " ".join(kept)
    logger.info(
        "%s round=%d truncated %d -> %d words (cap %d)",
        event_name, round_no, _word_count(content), _word_count(truncated), _MAX_WORDS,
    )
    return truncated


def _ensure_min_words(
    llm,
    base_messages: list,
    content: str,
    event_name: str,
    round_no: int,
    chunk_sink: ChunkSink | None,
) -> str:
    """If `content` is under `_MIN_WORDS`, ask the model to continue (up
    to `_MAX_CONTINUATION_CALLS` times) and append the result.

    This is a code-level enforcement, not a prompt instruction — the
    Day 6 length-imbalance bug showed that asking nicely in the system
    prompt is not reliable across models (llama-3.1-8b-instant in
    particular stayed under the 180-word floor even with a vague
    "add 80-100 words" continuation ask). Each retry now tells the model
    the EXACT word count so far and the exact number of words still
    needed, and the retry budget is higher (5 instead of 2) so a model
    that only adds ~20-30 words per call still converges.

    Continuation calls are plain `invoke()` (not streamed) since they're
    short follow-ups; if a `chunk_sink` is attached, the appended text
    is flushed as one additional chunk under the same
    `event_name`/`round_no` so the SSE client sees it as a natural
    continuation of the same argument.

    NOTE: this only enforces the FLOOR. The caller must also call
    `_enforce_max_words` afterward, since a continuation call could
    overshoot the ceiling — the two are separate, symmetric checks.
    """
    attempts = 0
    while _word_count(content) < _MIN_WORDS and attempts < _MAX_CONTINUATION_CALLS:
        attempts += 1
        current_words = _word_count(content)
        convo = list(base_messages) + [
            AIMessage(content=content),
            HumanMessage(content=_continue_prompt(current_words)),
        ]
        response = llm.invoke(convo)
        addition = _extract_content(response).strip()
        if not addition:
            logger.warning(
                "%s round=%d continuation_call=%d/%d returned empty — stopping",
                event_name, round_no, attempts, _MAX_CONTINUATION_CALLS,
            )
            break
        content = f"{content} {addition}"
        if chunk_sink is not None:
            chunk_sink(event_name, round_no, f" {addition}")
        logger.info(
            "%s round=%d continuation_call=%d/%d prev_words=%d new_word_count=%d",
            event_name, round_no, attempts, _MAX_CONTINUATION_CALLS,
            current_words, _word_count(content),
        )

    final_count = _word_count(content)
    if final_count < _MIN_WORDS:
        logger.warning(
            "%s round=%d still under min words after %d continuation "
            "call(s): %d words (target %d) — using as-is",
            event_name, round_no, attempts, final_count, _MIN_WORDS,
        )
    return content


# ---------- LLM clients ----------

def build_llms() -> dict[str, Any]:
    settings = load_settings()

    pro = ChatOpenAI(
    model="nvidia/nemotron-3.5-lightning:free",
    temperature=0.9,
    api_key=settings["openrouter_api_key"],
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/SameerAsad/debate-gpt",
        "X-Title": "debate-gpt",
    }
    )
    con = ChatGroq(
    model="openai/gpt-oss-20b",   
    temperature=0.9,
    api_key=settings["groq_api_key"],
    )
    judge = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.2,
        max_tokens=2500,
        api_key=settings["groq_api_key"],
    )
    return {"pro": pro, "con": con, "judge": judge}


# ---------- Node factories ----------

NodeFn = Callable[[DebateState], dict]

# How many characters of streaming buffer to flush per chunk. Smaller =
# more events; larger = fewer round-trips. ~50 chars is a good balance
# for human-readable token flow.
_CHUNK_FLUSH_CHARS = 50


def _stream_and_collect(
    llm,
    messages: list,
    event_name: str,
    round_no: int,
    chunk_sink: ChunkSink | None,
) -> str:
    """Run `llm.stream(messages)`; flush incremental chunks to `chunk_sink`
    and return the joined final content.

    Each flush sends the *new* text accumulated since the last flush
    (e.g. ~50 chars at a time). The sink is called with
    (event_name, round_no, content). Final-flush is always called so the
    last partial buffer is never lost.

    If `chunk_sink` is None (e.g. dry-run, pytest), no flushes happen —
    the function still returns the full content for state accumulation.
    """
    full: list[str] = []
    pending: list[str] = []  # unflushed accumulator
    pending_len = 0
    for chunk in llm.stream(messages):
        text = _extract_content(chunk) if chunk is not None else ""
        if not text:
            continue
        full.append(text)
        if chunk_sink is not None:
            pending.append(text)
            pending_len += len(text)
            if pending_len >= _CHUNK_FLUSH_CHARS:
                chunk_sink(event_name, round_no, "".join(pending))
                pending = []
                pending_len = 0

    final = "".join(full)
    if chunk_sink is not None and pending:
        # Final flush so nothing is dropped.
        chunk_sink(event_name, round_no, "".join(pending))
    return final


def make_pro_node(llm, chunk_sink: ChunkSink | None = None) -> NodeFn:
    def pro_node(state: DebateState) -> dict:
        messages = build_pro_messages(state)
        if chunk_sink is None:
            response = llm.invoke(messages)
            content = _extract_content(response)
        else:
            content = _stream_and_collect(
                llm, messages, "pro_token", state["round"], chunk_sink
            )
        content = _ensure_min_words(
            llm, messages, content, "pro_token", state["round"], chunk_sink
        )
        content = _enforce_max_words(content, "pro_token", state["round"])
        logger.info(
            "pro round=%d chars=%d words=%d",
            state["round"], len(content), _word_count(content),
        )
        return {"messages": [AIMessage(content=content, name="Pro")]}

    return pro_node


def make_con_node(llm, chunk_sink: ChunkSink | None = None) -> NodeFn:
    def con_node(state: DebateState) -> dict:
        messages = build_con_messages(state)
        if chunk_sink is None:
            response = llm.invoke(messages)
            content = _extract_content(response)
        else:
            content = _stream_and_collect(
                llm, messages, "con_token", state["round"], chunk_sink
            )
        content = _ensure_min_words(
            llm, messages, content, "con_token", state["round"], chunk_sink
        )
        content = _enforce_max_words(content, "con_token", state["round"])
        logger.info(
            "con round=%d chars=%d words=%d",
            state["round"], len(content), _word_count(content),
        )
        return {"messages": [AIMessage(content=content, name="Con")]}

    return con_node


def make_judge_node(llm) -> NodeFn:
    def judge_node(state: DebateState) -> dict:
        pro_text = _latest_text(state["messages"], name="Pro")
        con_text = _latest_text(state["messages"], name="Con")
        if pro_text is None or con_text is None:
            raise RuntimeError(
                f"judge_node received no Pro/Con arguments "
                f"(round={state['round']}, messages={len(state['messages'])})"
            )

        swap = state["round"] % 2 == 0
        bundle = build_judge_messages(state, pro_text, con_text, swap=swap)
        score_dict = _invoke_judge_with_retry(llm, bundle.messages, max_retries=2)

        # On a successful parse, translate the Speaker A/B verdict back to
        # pro/con and compute the derived totals for the verdict node (Day 3).
        if "round_winner" in score_dict and "parse_error" not in score_dict:
            score_dict["round_winner"] = _translate_winner(
                score_dict["round_winner"], swap
            )
            score_dict["pro_score"] = pro_total_for_round(score_dict, swap)
            score_dict["con_score"] = con_total_for_round(score_dict, swap)

        next_round = state["round"] + 1
        logger.info(
            "judge round=%d swap=%s winner=%s pro=%s con=%s next_round=%d",
            state["round"],
            swap,
            score_dict.get("round_winner"),
            score_dict.get("pro_score"),
            score_dict.get("con_score"),
            next_round,
        )
        return {
            "round_scores": [score_dict],
            "round": next_round,
        }

    return judge_node


# ---------- Bidirectional judging (order-bias fix) ----------

def judge_once(
    llm,
    state: DebateState,
    pro_text: str,
    con_text: str,
    swap: bool,
    stats: dict | None = None,
) -> dict:
    """Judge ONE presentation order, chosen explicitly (not by round parity).

    swap=False: Pro is Speaker A (shown first).
    swap=True:  Con is Speaker A (shown first).
    Returns the RoundScore dict plus swap-corrected pro_score / con_score,
    or the {"raw", "parse_error"} dict if the judge never produced valid JSON.
    """
    bundle = build_judge_messages(state, pro_text, con_text, swap=swap)
    score = _invoke_judge_with_retry(llm, bundle.messages, max_retries=2, stats=stats)
    if "parse_error" not in score:
        score["round_winner"] = _translate_winner(score["round_winner"], swap)
        score["pro_score"] = pro_total_for_round(score, swap)
        score["con_score"] = con_total_for_round(score, swap)
    return score


def judge_bidirectional(
    llm,
    state: DebateState,
    pro_text: str,
    con_text: str,
) -> dict:
    """Judge BOTH orders (Pro first, then Con first) and average per side.

    Any fixed "second speaker gets a bonus" effect lands on Pro in one call
    and on Con in the other, so it cancels in the averaged margin.
    Scores are floats (average of two ints). The verdict uses the averaged
    totals, never the judge's own round_winner label.
    """
    pro_first = judge_once(llm, state, pro_text, con_text, swap=False)
    con_first = judge_once(llm, state, pro_text, con_text, swap=True)

    for label, s in (("pro_first", pro_first), ("con_first", con_first)):
        if "parse_error" in s:
            return {"parse_error": f"{label}: {s['parse_error']}",
                    "pro_first": pro_first, "con_first": con_first}

    pro_avg = (pro_first["pro_score"] + con_first["pro_score"]) / 2
    con_avg = (pro_first["con_score"] + con_first["con_score"]) / 2
    if pro_avg > con_avg:
        winner = "pro"
    elif con_avg > pro_avg:
        winner = "con"
    else:
        winner = "tie"
    return {
        "pro_score": pro_avg,
        "con_score": con_avg,
        "round_winner": winner,
        "orders_agree": pro_first["round_winner"] == con_first["round_winner"],
        "pro_first": pro_first,
        "con_first": con_first,
    }


# ---------- Helpers ----------

def _extract_content(response: Any) -> str:
    """Pull the string content out of a chat-model response, regardless of
    whether it returns a string, an AIMessage, or has a `.content` attr."""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is None:
        raise RuntimeError(f"LLM response had no .content: {response!r}")
    if isinstance(content, list):
        # Some providers return content as a list of parts; join text parts.
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _latest_text(messages: list, name: str) -> str | None:
    """Return the most recent AIMessage.content with the given `name`, or None."""
    for msg in reversed(messages):
        if getattr(msg, "name", None) == name:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_STRUCTURED_UNSUPPORTED = (NotImplementedError, AttributeError, ValueError)

# Concise feedback injected on retry. The model sees its own mistake and
# a terse instruction to return ONLY the JSON object.
_RETRY_FEEDBACK_TMPL = (
    "Your previous output failed validation: {err}\n"
    "Return ONLY the JSON object exactly matching the schema above. "
    "No markdown fences, no commentary, no trailing prose."
)


def _invoke_judge_with_retry(
    llm, messages: list, max_retries: int = 2, stats: dict | None = None
) -> dict:
    """Invoke the judge and return a RoundScore dict.

    Strategy:
      1) Try `llm.with_structured_output(RoundScore)` once (tool-calling path).
         If the provider doesn't support it, or the model emits malformed
         tool args, fall through to path 2.
      2) Manual parse: invoke plain LLM, extract the first {...} blob,
         validate with RoundScore. On any failure, append a feedback
         HumanMessage and re-invoke, up to `max_retries` additional times.

    Always returns a dict. On full failure, returns
    `{"raw": ..., "parse_error": "..."}` so the graph keeps running.

    Optional `stats` (a dict you pass in) is filled with schema-validity
    info for the benchmark:
      tier:      "structured"      valid on the very first call
                 "fallback_first"  valid on the first plain-text parse
                 "fallback_retry"  valid only after feedback retry
                 "failed"          never valid
      llm_calls: number of model calls used
      errors:    short error strings (first 200 chars each)
    """
    if stats is None:
        stats = {}
    stats.update(tier="failed", llm_calls=0, errors=[])

    # --- Path 1: structured output (tool calling) ---
    try:
        structured = llm.with_structured_output(RoundScore)
        stats["llm_calls"] += 1
        result = structured.invoke(messages)
        if isinstance(result, RoundScore):
            out = result.model_dump()
        else:
            # Some adapters return a dict already.
            out = RoundScore.model_validate(result).model_dump()
        stats["tier"] = "structured"
        return out
    except _STRUCTURED_UNSUPPORTED as exc:
        # NOTE: pydantic's ValidationError is a ValueError subclass, so
        # validation failures also land here. Behavior is the same
        # (fall back to path 2); only the log wording differs.
        stats["errors"].append(f"structured: {exc}"[:200])
        logger.info(
            "structured_output unsupported (%s); falling back to manual parse", exc
        )
    except (ValidationError, OutputParserException, json.JSONDecodeError) as exc:
        stats["errors"].append(f"structured: {exc}"[:200])
        logger.warning(
            "structured_output validation failed: %s; falling back to manual parse",
            exc,
        )

    # --- Path 2: manual parse with retry ---
    convo = list(messages)  # copy; we may append feedback
    last_err: str | None = None
    last_raw: str | None = None
    for attempt in range(max_retries + 1):  # initial + max_retries retries
        stats["llm_calls"] += 1
        response = llm.invoke(convo)
        raw = _extract_content(response)
        last_raw = raw
        try:
            out = _parse_judge_output_strict(raw)
            stats["tier"] = "fallback_first" if attempt == 0 else "fallback_retry"
            return out
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            last_err = str(exc)
            stats["errors"].append(f"fallback{attempt + 1}: {exc}"[:200])
            logger.warning(
                "judge parse attempt %d/%d failed: %s",
                attempt + 1,
                max_retries + 1,
                exc,
            )
            if attempt < max_retries:
                convo = convo + [
                    HumanMessage(content=_RETRY_FEEDBACK_TMPL.format(err=last_err))
                ]

    return {"raw": last_raw or "", "parse_error": last_err or "unknown parse failure"}


def _parse_judge_output_strict(raw: str) -> dict:
    """Strict parse: extract the first JSON object, validate with RoundScore.

    Raises on failure (the retry helper owns the fallback decision).
    """
    text = _JSON_FENCE.sub("", raw.strip()).strip()
    match = _JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError("no JSON object found in judge output")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge JSON is not an object")
    return RoundScore.model_validate(parsed).model_dump()


def _translate_winner(winner: str, swap: bool) -> str:
    """Translate Speaker A/B verdict back to pro/con/tie.

    swap=True (even rounds): Con was Speaker A, Pro was Speaker B.
    swap=False (odd rounds): Pro was Speaker A, Con was Speaker B.
    """
    if winner == "tie":
        return "tie"
    if swap:
        # Con = A, Pro = B
        return "con" if winner == "A" else "pro"
    # Pro = A, Con = B
    return "pro" if winner == "A" else "con"