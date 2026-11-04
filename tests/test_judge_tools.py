import json
from typing import TypedDict

from arguments_tracking import check_internal_contradiction
from entities import Contradiction, TopicState

CASES_PATH = "tests/consistency_cases.json"


class ConsistencyCase(TypedDict):
    name: str
    topic: str
    old_stance: str
    new_stance: str


def _load_cases() -> list[ConsistencyCase]:
    with open(CASES_PATH, "r", encoding="utf-8") as file_stream:
        return json.load(file_stream)


def _run_case(case: ConsistencyCase) -> list[Contradiction]:
    topic_state: TopicState = {
        "subject": case["topic"],
        "pro_stance": case["old_stance"],
        "against_stance": None,
        "brought_up_by": "PRO",
        "current_winner": "PRO",
        "winning_claim": case["old_stance"],
    }
    contradictions: list[Contradiction] = []
    check_internal_contradiction(
        {"topic_key": case["topic"], "stance": case["new_stance"]},
        topic_state,
        "PRO",
        contradictions,
    )
    return contradictions


def run_consistency_tests() -> None:
    print("=== STARTING CONSISTENCY TESTS ===")
    for case in _load_cases():
        print(f"\n>>> RUNNING TEST: {case['name']}")
        contradictions = _run_case(case)
        print(f"Contradictions logged: {len(contradictions)}")


if __name__ == "__main__":
    run_consistency_tests()
