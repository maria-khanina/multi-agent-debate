from typing import Any
import uuid

from debate_settings import SEPARATOR
from entities import ResearcherState
from research import research_app


def _get_initial_state() -> ResearcherState:
    return {
        "debate_id": uuid.uuid4().hex,
        "motion": "AI will replace human",
        "query": "transformers",
        "agent_role": "PRO",
        "raw_papers": [],
        "approved_papers": [],
    }


def _print_update(node_name: str, state_update: dict[str, Any]) -> None:
    print(f"\n[ Node: {node_name} ]")
    if "raw_papers" in state_update:
        print(f" -> Found {len(state_update['raw_papers'])} papers.")
    if "approved_papers" in state_update:
        print(f" -> Total approved: {len(state_update['approved_papers'])}")


def run_test() -> None:
    initial_state = _get_initial_state()
    print(f"Starting test for: '{initial_state['query']}'")
    print(SEPARATOR)
    for step in research_app.stream(initial_state, {}):
        for node_name, state_update in step.items():
            _print_update(node_name, state_update)
    print(f"\n{SEPARATOR}\nGraph Execution Complete.")


if __name__ == "__main__":
    run_test()
