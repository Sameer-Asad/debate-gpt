"""Day 6 eval: position bias (PRD §9.4 row 2 / §3.4 bias mitigation note).

Scores one fixed, real argument pair twice through the REAL judge node
(agents.make_judge_node): once at round=1 (swap=False, Pro=Speaker A) and
once at round=2 (swap=True, Pro=Speaker B). Compares the swap-corrected
pro_score/con_score totals — per PRD §3.4, "scores should not change by
more than ±1 point" once corrected for the swap.

Pass threshold: delta < ±1.0 (PRD §9.4).

This directly validates the Speaker A/B bias-mitigation invariant
documented in CLAUDE.md. A failure here is a real finding about the
judge model's behavior, not a flaky test — do not adjust the threshold
to force a pass.

Real Groq + OpenRouter calls — NOT mocked. Run separately from pytest:
    python tests/evals/test_position_bias.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.messages import AIMessage

from debate_gpt.agents import build_llms, make_judge_node, make_pro_node, make_con_node

DELTA_THRESHOLD = 1.0

FIXED_TOPIC = "Social media should be regulated as a public utility"


def _base_state(round_no: int, messages: list) -> dict:
    return {
        "session_id": "eval-bias",
        "topic": FIXED_TOPIC,
        "position_pro": f"For: {FIXED_TOPIC}",
        "position_con": f"Against: {FIXED_TOPIC}",
        "round": round_no,
        "max_rounds": 3,
        "messages": messages,
        "round_scores": [],
        "winner": None,
        "trace_id": None,
    }


def _generate_fixed_argument_pair(llms: dict) -> tuple[str, str]:
    pro_node = make_pro_node(llms["pro"])
    state = _base_state(round_no=1, messages=[])
    pro_out = pro_node(state)
    pro_msg = pro_out["messages"][0]

    con_node = make_con_node(llms["con"])
    state_with_pro = _base_state(round_no=1, messages=[pro_msg])
    con_out = con_node(state_with_pro)
    con_msg = con_out["messages"][0]

    return pro_msg.content, con_msg.content


def run_bias_eval() -> dict:
    llms = build_llms()
    pro_text, con_text = _generate_fixed_argument_pair(llms)

    judge_node = make_judge_node(llms["judge"])
    messages = [
        AIMessage(content=pro_text, name="Pro"),
        AIMessage(content=con_text, name="Con"),
    ]

    # round=1 -> swap=False (Pro=Speaker A). round=2 -> swap=True (Pro=Speaker B).
    state_no_swap = _base_state(round_no=1, messages=messages)
    out_no_swap = judge_node(state_no_swap)
    score_no_swap = out_no_swap["round_scores"][0]

    state_swap = _base_state(round_no=2, messages=messages)
    out_swap = judge_node(state_swap)
    score_swap = out_swap["round_scores"][0]

    for label, score in (("no_swap", score_no_swap), ("swap", score_swap)):
        if "parse_error" in score:
            raise RuntimeError(
                f"judge call ({label}) failed to parse: {score.get('parse_error')}"
            )

    # pro_score / con_score are already swap-corrected by judge_node
    # (agents.py calls pro_total_for_round/con_total_for_round with the
    # right swap flag before returning). If the judge is truly unbiased,
    # these totals should be nearly identical regardless of which
    # speaker label the model saw as "Pro" underneath.
    pro_delta = abs(score_no_swap["pro_score"] - score_swap["pro_score"])
    con_delta = abs(score_no_swap["con_score"] - score_swap["con_score"])
    max_delta = max(pro_delta, con_delta)
    passed = max_delta < DELTA_THRESHOLD

    return {
        "eval_name": "position_bias",
        "metric": "max_pro_con_delta_after_swap_correction",
        "value": round(max_delta, 3),
        "threshold": f"< {DELTA_THRESHOLD}",
        "passed": passed,
        "detail": {
            "pro_score_no_swap": score_no_swap["pro_score"],
            "pro_score_swap": score_swap["pro_score"],
            "con_score_no_swap": score_no_swap["con_score"],
            "con_score_swap": score_swap["con_score"],
            "pro_delta": pro_delta,
            "con_delta": con_delta,
        },
        "raw": {
            "pro_text": pro_text,
            "con_text": con_text,
            "score_no_swap": score_no_swap,
            "score_swap": score_swap,
        },
    }


if __name__ == "__main__":
    result = run_bias_eval()
    print(f"position_bias: max_delta={result['value']} "
          f"(threshold {result['threshold']}) -> "
          f"{'PASS' if result['passed'] else 'FAIL'}")
    for k, v in result["detail"].items():
        print(f"  {k}: {v}")