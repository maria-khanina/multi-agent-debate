from typing import Any
import uuid

from debate_graph import debate_app
from entities import DebateState, FactCheckStats
from langchain_core.runnables import RunnableConfig
from tracing import build_run_config, flush_traces

SEPARATOR = "=================================================="


def _get_empty_factcheck_stats() -> FactCheckStats:
    return {
        "SUPPORTED": 0,
        "FABRICATED": 0,
        "EXAGGERATED": 0,
        "UNVERIFIABLE": 0,
    }


def _get_initial_test_state() -> DebateState:
    return {
        "debate_id": uuid.uuid4().hex,
        "motion": "AI will replace human",
        "transcript": [],
        "turn_count": 0,
        "max_turns": 2,
        "pro_sources": [],
        "against_sources": [],
        "pro_contradictions": [],
        "against_contradictions": [],
        "pro_mistakes": [],
        "against_mistakes": [],
        "pro_factcheck_stats": _get_empty_factcheck_stats(),
        "against_factcheck_stats": _get_empty_factcheck_stats(),
        "final_argument_stats": {"PRO": 0, "AGAINST": 0},
    }


def _print_banner(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}\n")


def _print_events(initial_state: DebateState, run_config: RunnableConfig) -> None:
    for event in debate_app.stream(initial_state, config=run_config):
        state_updates: dict[str, dict[str, Any]] = event
        for node, state_update in state_updates.items():
            print(f"\n--- [ NODE EXECUTED: {node} ] ---")
            transcript: list[str] = state_update.get("transcript") or []
            if transcript:
                print(f"\n{transcript[-1]}\n")


def run_debate_test() -> None:
    initial_state = _get_initial_test_state()
    _print_banner(f"STARTING DEBATE: {initial_state['motion']}")
    run_config = build_run_config(
        initial_state["debate_id"],
        initial_state["motion"],
    )
    _print_events(initial_state, run_config)
    flush_traces()
    _print_banner("DEBATE FINISHED")


if __name__ == "__main__":
    run_debate_test()
