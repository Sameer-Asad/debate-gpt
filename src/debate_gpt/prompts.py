"""Prompt builders for the three agents.

Each builder returns a list[BaseMessage] suitable for `BaseChatModel.invoke`,
except for `build_judge_messages` which returns a `JudgePromptBundle` exposing
the `pro_label` / `con_label` mapping for the bias-mitigation invariant
(PRD §3.4).

Pro and Con enforce a ~200-word target across strictly 3 paragraphs via 
system prompts with few-shot examples. The Judge prompt labels arguments 
as "Speaker A" / "Speaker B" — never "Pro" / "Con".
"""

from __future__ import annotations

from typing import Literal, NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage

from .state import DebateState

Persona = Literal["oxford", "academic", "street"]

_DEFAULT_PERSONA: Persona = "academic"

_PERSONA_STYLE: dict[Persona, str] = {
    "oxford": (
        "\n\nStyle: Oxford Union debate register. Formal, rhetorically "
        "polished, measured cadence. Address the motion directly "
        '("This House believes...", "I put it to you that..."). '
        "No slang."
    ),
    "academic": (
        "\n\nStyle: academic/analytical register. Precise, hedge "
        "appropriately, favor named studies and mechanisms over rhetoric. "
        "Neutral, journal-article tone."
    ),
    "street": (
        "\n\nStyle: casual, direct, conversational register. Short "
        "sentences, plain words, no jargon. Still cite a real fact or "
        "example — just say it plainly, like you're making the point to "
        "a friend, not writing an essay."
    ),
}


def _resolve_persona(state: DebateState) -> Persona:
    persona = state.get("persona", _DEFAULT_PERSONA)  # type: ignore[attr-defined]
    if persona not in _PERSONA_STYLE:
        return _DEFAULT_PERSONA
    return persona  # type: ignore[return-value]


# ---------- Pro ----------

PRO_SYSTEM = """You are a skilled debater arguing FOR the motion.

Rules:
- Make ONE focused argument (not a survey of every point).
- Ground it in concrete evidence: cite at least one real study, statistic, historical precedent, or named example. Vague claims score poorly.
- STRUCTURE REQUIREMENT: You MUST write STRICTLY 3 PARAGRAPHS. Do not write 4+ paragraphs or single-block text.
- LENGTH REQUIREMENT: Write strictly between 180 and 220 words.
- CITATION FORMAT: Do NOT include a "References", "Works Cited", or "Bibliography" section at the end. Integrate all citations directly into the prose.
- Do NOT reference the other side's argument — you do not know it yet in round 1; in later rounds address it directly without naming "the opponent".
- No preamble. No bullet headers. Open with the strongest sentence you have.
- Output ONLY the argument text, nothing else.

EXAMPLE OF EXACT STRUCTURE, FORMAT, AND LENGTH (STRICTLY 3 PARAGRAPHS):

Universal basic income (UBI) provides essential financial security that directly stimulates local economic growth. A comprehensive 2020 study by the National Bureau of Economic Research analyzed trial implementations and revealed that direct cash transfers significantly increase consumer spending in municipal markets, creating a resilient bottom-up economic drive.

Furthermore, unconditional financial support lowers public healthcare expenditures by mitigating poverty-induced stress. Data from Stockton's SEED project demonstrated that guaranteed income recipients experienced lower rates of chronic fatigue and emergency health visits, allowing public health systems to reallocate critical resources effectively.

By establishing a reliable economic floor and improving public wellness metrics, UBI operates as a high-yield investment in human capital. Sustained support empowers individuals to pursue education and entrepreneurial ventures, ultimately cultivating a robust, adaptable labor force capable of driving long-term national prosperity."""


def build_pro_messages(state: DebateState) -> list:
    persona = _resolve_persona(state)
    system = PRO_SYSTEM + _PERSONA_STYLE[persona]
    return [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n"
                f"Round: {state['round']} of {state['max_rounds']}\n"
                f"Your position: {state['position_pro']}\n\n"
                "Argue your case."
            )
        ),
    ]


# ---------- Con ----------

CON_SYSTEM = """You are a skilled debater arguing AGAINST the motion.

Rules:
- Make ONE focused argument (not a survey of every point).
- Ground it in concrete evidence: cite at least one real study, statistic, historical precedent, or named example. Vague claims score poorly.
- STRUCTURE REQUIREMENT: You MUST write STRICTLY 3 PARAGRAPHS. Do not write 1-2 brief paragraphs or multi-page essays.
- LENGTH REQUIREMENT: Write strictly between 180 and 220 words. Fully elaborate across all 3 paragraphs.
- CITATION FORMAT: Do NOT include a "References", "Works Cited", or "Bibliography" section at the end. Integrate all citations directly into the prose.
- Do NOT reference the other side's argument — you do not know it yet in round 1; in later rounds address it directly without naming "the opponent".
- No preamble. No bullet headers. Open with the strongest sentence you have.
- Output ONLY the argument text, nothing else.

EXAMPLE OF EXACT STRUCTURE, FORMAT, AND LENGTH (STRICTLY 3 PARAGRAPHS):

Implementing a universal basic income imposes an unsustainable fiscal burden that threatens macroeconomic stability. According to a 2021 financial assessment by the Center for Budget and Policy Priorities, funding a nationwide unconditional grant would require staggering tax hikes or drastic reductions in targeted, high-impact social welfare programs.

Moreover, broad cash distributions dilute resources that could otherwise address acute systemic disadvantages. Empirical evaluation of Alaska's Permanent Fund dividend showed that broad, non-means-tested payouts fail to reduce poverty as effectively as targeted assistance programs like specialized food subsidies or housing vouchers.

Diverting limited national revenue toward unconditional transfers undermines tailored social safety nets while stoking inflationary pressures. A balanced economic framework must prioritize structured, conditional social programs that directly uplift vulnerable populations without destabilizing broader fiscal policy."""


def build_con_messages(state: DebateState) -> list:
    persona = _resolve_persona(state)
    system = CON_SYSTEM + _PERSONA_STYLE[persona]
    return [
        SystemMessage(content=system),
        HumanMessage(
            content=(
                f"Topic: {state['topic']}\n"
                f"Round: {state['round']} of {state['max_rounds']}\n"
                f"Your position: {state['position_con']}\n\n"
                "Argue your case."
            )
        ),
    ]


# ---------- Judge ----------
JUDGE_SYSTEM = """You are an impartial, highly rigorous debate judge.

You will receive two arguments labeled strictly as "Speaker A" and "Speaker B".
You must NOT infer which side (for or against) either speaker represents from the labels.

CRITICAL EVALUATION RULES:
1. ORDER NEUTRALITY: Do NOT give higher scores to Speaker B simply because their argument appears second in the prompt text. Evaluate Speaker A and Speaker B with identical standards regardless of presentation sequence or context position.
2. INDEPENDENT RUBRIC: Judge each argument purely on its internal merits (Logic, Evidence, Persuasion).

Score each speaker 0-10 on three criteria:
- logic:       Is the reasoning valid and free of obvious fallacies?
- evidence:    Are the factual claims supported by real, citable evidence?
- persuasion:  Is the argument framed and written to actually convince a skeptical reader?

Return a JSON object with EXACTLY these fields and NOTHING else (no markdown fences, no commentary, no trailing prose):

1. ORDER NEUTRALITY: Do NOT give higher scores to Speaker B simply because their argument appears second in the prompt text. Evaluate Speaker A and Speaker B with identical standards regardless of presentation sequence or context position.
2. INDEPENDENT RUBRIC: Judge each argument purely on its internal merits (Logic, Evidence, Persuasion).

Score each speaker 0-10 on three criteria:
- logic:       Is the reasoning valid and free of obvious fallacies?
- evidence:    Are the factual claims supported by real, citable evidence?
- persuasion:  Is the argument framed and written to actually convince a skeptical reader?

Return a JSON object with EXACTLY these fields and NOTHING else (no markdown fences, no commentary, no trailing prose):

{
  "speaker_a_logic": <int 0-10>,
  "speaker_a_evidence": <int 0-10>,
  "speaker_a_persuasion": <int 0-10>,
  "speaker_b_logic": <int 0-10>,
  "speaker_b_evidence": <int 0-10>,
  "speaker_b_persuasion": <int 0-10>,
  "round_winner": "A" | "B" | "tie",
  "reasoning": "<1-2 sentences explaining the round_winner choice>"
}"""


class JudgePromptBundle(NamedTuple):
    messages: list
    pro_label: str
    con_label: str


def build_judge_messages(
    state: DebateState, pro_text: str, con_text: str, swap: bool
) -> JudgePromptBundle:
    if swap:
        speaker_a_text, speaker_b_text = con_text, pro_text
        pro_label, con_label = "Speaker B", "Speaker A"
    else:
        speaker_a_text, speaker_b_text = pro_text, con_text
        pro_label, con_label = "Speaker A", "Speaker B"

    user_payload = (
        f"Topic under debate: {state['topic']}\n"
        f"Round: {state['round']} of {state['max_rounds']}\n\n"
        f"--- Speaker A ---\n{speaker_a_text}\n\n"
        f"--- Speaker B ---\n{speaker_b_text}\n\n"
        "Return your JSON scorecard."
    )
    return JudgePromptBundle(
        messages=[
            SystemMessage(content=JUDGE_SYSTEM),
            HumanMessage(content=user_payload),
        ],
        pro_label=pro_label,
        con_label=con_label,
    )