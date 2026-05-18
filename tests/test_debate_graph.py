import sys
import os
import uuid
from typing import Any


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

from debate_graph import debate_app


def _get_initial_test_state() -> dict[str, Any]:
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
        "pro_factcheck_stats": {
            "SUPPORTED": 0, 
            "FABRICATED": 0, 
            "EXAGGERATED": 0, 
            "UNVERIFIABLE": 0,
        },
        "against_factcheck_stats": {
            "SUPPORTED": 0, 
            "FABRICATED": 0, 
            "EXAGGERATED": 0, 
            "UNVERIFIABLE": 0,
        },
        "final_argument_stats": {"PRO": 0, "AGAINST": 0},
    }


def run_debate_test() -> None:
    initial_state = _get_initial_test_state()

    print(f"\n{'='*50}")
    print(f"STARTING DEBATE: {initial_state['motion']}")
    print(f"{'='*50}\n")
    
    for event in debate_app.stream(initial_state):
        for node, state_update in event.items():
            print(f"\n--- [ NODE EXECUTED: {node} ] ---")
            if "transcript" in state_update and state_update["transcript"]:
                print(f"\n{state_update['transcript'][-1]}\n")
                    
    print(f"\n{'='*50}")
    print("DEBATE FINISHED")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_debate_test()
