"""Day 6 eval: judge consistency (PRD §9.4 row 1).

Calls the REAL judge node (agents.make_judge_node) 5 times against one
fixed, real argument pair, and measures the standard deviation of each
scoring criterion across the 5 runs.

Pass threshold: std dev < 1.5 per criterion (PRD §9.4).

This uses real Groq + OpenRouter calls — NOT mocked. Run separately from
the pytest CI suite:
    python tests/evals/test_judge_consistency.py
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from langchain_core.messages import AIMessage

from debate_gpt.agents import build_llms, make_judge_node, make_pro_node, make_con_node

CRITERIA = (
    "speaker_a_logic", "speaker_a_evidence", "speaker_a_persuasion",
    "speaker_b_logic", "speaker_b_evidence", "speaker_b_persuasion",
)

STD_DEV_THRESHOLD = 1.5
N_RUNS = 5

FIXED_TOPIC = "Universal basic income should be adopted worldwide"


def _base_state(round_no: int, messages: list) -> dict:
    return {
        "session_id": "eval-consistency",
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
    """Generate one real Pro/Con exchange (via Groq) that all 5 judge
    calls will reuse, so we're testing judge consistency in isolation
    from debater variance."""
    pro_node = make_pro_node(llms["pro"])
    state = _base_state(round_no=1, messages=[])
    pro_out = pro_node(state)
    pro_msg = pro_out["messages"][0]

    con_node = make_con_node(llms["con"])
    state_with_pro = _base_state(round_no=1, messages=[pro_msg])
    con_out = con_node(state_with_pro)
    con_msg = con_out["messages"][0]

    return pro_msg.content, con_msg.content


def run_consistency_eval() -> dict:
    """Returns a dict suitable for eval_report.py's summary + raw sidecar."""
    llms = build_llms()
    pro_text, con_text = _generate_fixed_argument_pair(llms)

    judge_node = make_judge_node(llms["judge"])
    messages = [
        AIMessage(content=pro_text, name="Pro"),
        AIMessage(content=con_text, name="Con"),
    ]

    raw_scores: list[dict] = []
    for i in range(N_RUNS):
        # round=1 (odd) => swap=False every time, so all 5 calls see the
        # identical Speaker A/B framing — we're isolating judge variance,
        # not swap variance (that's test_position_bias.py's job).
        state = _base_state(round_no=1, messages=messages)
        out = judge_node(state)
        score = out["round_scores"][0]
        if "parse_error" in score:
            raise RuntimeError(
                f"judge call {i + 1}/{N_RUNS} failed to parse: {score.get('parse_error')}"
            )
        raw_scores.append(score)

    per_criterion_stdev = {}
    for crit in CRITERIA:
        values = [s[crit] for s in raw_scores]
        per_criterion_stdev[crit] = statistics.stdev(values) if len(values) > 1 else 0.0

    max_stdev = max(per_criterion_stdev.values())
    passed = max_stdev < STD_DEV_THRESHOLD

    return {
        "eval_name": "judge_consistency",
        "metric": "max_stdev_across_criteria",
        "value": round(max_stdev, 3),
        "threshold": f"< {STD_DEV_THRESHOLD}",
        "passed": passed,
        "detail": per_criterion_stdev,
        "raw": {
            "pro_text": pro_text,
            "con_text": con_text,
            "runs": raw_scores,
        },
    }


if __name__ == "__main__":
    result = run_consistency_eval()
    print(f"judge_consistency: max_stdev={result['value']} "
          f"(threshold {result['threshold']}) -> "
          f"{'PASS' if result['passed'] else 'FAIL'}")
    for crit, sd in result["detail"].items():
        print(f"  {crit}: stdev={sd:.3f}")