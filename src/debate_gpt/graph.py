"""LangGraph wiring for the debate graph.

Topology (PRD §3.2, §3.4):

    START -> pro_node -> con_node -> judge_node -> (conditional)
                                                     |
                                            round <= max_rounds ? pro_node : END

`judge_node` increments `round` to N+1 in its return value. The conditional
edge reads the post-increment value, so round 1 -> 2 -> 3 with judge
returning 2, 3, 4; the fourth call exits.

LangSmith tracing (PRD §10): every node is wrapped with `@traceable` so
each round's pro/con/judge calls show up as distinct child spans under
one parent trace per debate session, rather than a single opaque graph
run. This requires `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true`, and
`LANGCHAIN_PROJECT` to be set (see .env.example) — if they're unset,
`traceable` is a no-op passthrough and the graph runs exactly as before,
just untraced.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from .agents import ChunkSink, build_llms, make_con_node, make_judge_node, make_pro_node
from .state import DebateState


def should_continue(state: DebateState) -> str:
    """Conditional edge after judge_node. Loop while there are rounds left."""
    if state["round"] <= state["max_rounds"]:
        return "pro_node"
    return END


def build_graph(
    llms: dict[str, Any] | None = None,
    chunk_sink: ChunkSink | None = None,
):
    """Compile the debate graph.

    Parameters
    ----------
    llms : dict, optional
        Override the real LLM clients. Day 5's pytest suite (and
        `dry_run.py`) inject a `FakeLLM` here.
    chunk_sink : callable, optional
        `(event, round, content) -> None` invoked as pro/con tokens
        stream. Day 3's runtime passes a sink that XADDs each chunk to
        the session's Upstash Stream.
    """
    if llms is None:
        llms = build_llms()

    g = StateGraph(DebateState)

    # Each node function is wrapped with @traceable (PRD §10.1: "Input
    # and output of every node... LangGraph state transitions"). This
    # gives one named child span per node execution — "pro_node",
    # "con_node", "judge_node" — nested under the parent trace for the
    # whole debate, rather than tracing only the raw LLM calls inside
    # them. `traceable` also captures exceptions/retries per PRD §10.1.
    pro_fn = traceable(name="pro_node", run_type="chain")(
        make_pro_node(llms["pro"], chunk_sink=chunk_sink)
    )
    con_fn = traceable(name="con_node", run_type="chain")(
        make_con_node(llms["con"], chunk_sink=chunk_sink)
    )
    judge_fn = traceable(name="judge_node", run_type="chain")(
        make_judge_node(llms["judge"])
    )

    g.add_node("pro_node", pro_fn)
    g.add_node("con_node", con_fn)
    g.add_node("judge_node", judge_fn)

    g.add_edge(START, "pro_node")
    g.add_edge("pro_node", "con_node")
    g.add_edge("con_node", "judge_node")
    g.add_conditional_edges(
        "judge_node",
        should_continue,
        {"pro_node": "pro_node", END: END},
    )
    return g.compile()