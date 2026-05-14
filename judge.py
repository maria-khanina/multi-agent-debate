import json
from typing import TypedDict

from langchain_core.messages import HumanMessage

from configurations.config_manager import get_chat_model


class Contradiction(TypedDict):
    concept: str
    old_stance: str
    new_stance: str
    explanation: str


class BeliefTrackerResult(TypedDict):
    turn_analysis: str
    updated_beliefs: dict[str, str]
    contradictions: list[Contradiction]


def build_tracker_prompt(
    latest_turn: str,
    motion: str,
    agent_role: str,
    current_beliefs: dict[str, str] = None,
    past_contradictions: list[dict] = None,
) -> str:
    current_beliefs = current_beliefs or {}
    past_contradictions = past_contradictions or []
    
    beliefs_str = json.dumps(current_beliefs, indent=2)
    past_contra_str = json.dumps(past_contradictions, indent=2)
    
    prompt_text = (
        f"Debate Motion: {motion}\n"
        f"Agent {agent_role} just spoke.\n"
        f"Latest Turn:\n{latest_turn}\n\n"
        "Current Beliefs Ledger (State before this turn):\n"
        f"{beliefs_str}\n\n"
        "Previously Logged Contradictions:\n"
        f"{past_contra_str}\n\n"
        "Step 1 (Analyze): Think step-by-step in the 'turn_analysis' "
        "field. What is the core philosophical claim here? How does it "
        "logically interact with the Current Beliefs Ledger? Does the "
        "Latest Turn contain an internal logical paradox?\n\n"
        "Step 2 (Contradictions): Look for explicit AND implicit "
        "philosophical contradictions. This includes contradictions "
        "between the Latest Turn and the Current Beliefs Ledger, AND "
        "internal contradictions entirely within the Latest Turn itself. "
        "If a statement logically breaks a recorded stance or its own "
        "internal logic, AND this shift is not already in the Previously "
        "Logged Contradictions, log it ONCE. Do not create multiple "
        "entries for the same underlying logical shift.\n\n"
        "Step 3 (Update): Update the Beliefs Ledger. You MUST overwrite "
        "existing keys if the stance shifted. If a new field of thought "
        "is brought on, make a new pair of key and value. Keep the keys "
        "broad."
    )
    return prompt_text


def track_debator_beliefs(
    latest_turn: str,
    motion: str,
    agent_role: str,
    current_beliefs: dict[str, str] = None,
    past_contradictions: list[dict] = None,
) -> BeliefTrackerResult:
    current_beliefs = current_beliefs or {}
    past_contradictions = past_contradictions or []
    
    language_model = get_chat_model("judge").with_structured_output(
        BeliefTrackerResult
    )
    
    prompt_text = build_tracker_prompt(
        latest_turn, 
        motion,
        agent_role,
        current_beliefs,
        past_contradictions
    )
    
    response = language_model.invoke([HumanMessage(content=prompt_text)])
    return response


def _update_tracker(
    turn_text: str,
    motion: str,
    agent_role: str,
    tracker: dict = None,
) -> None:
    tracker = tracker or {}
    beliefs = tracker.get("updated_beliefs", {})
    contradictions = tracker.get("contradictions", [])
    
    result = track_debator_beliefs(
        turn_text,
        motion,
        agent_role,
        beliefs,
        contradictions
    )
    
    tracker["updated_beliefs"] = result.get("updated_beliefs", {})
    tracker["contradictions"].extend(result.get("contradictions", []))


def _process_debate(debate: dict = None) -> dict:
    debate = debate or {}
    motion = debate.get("motion", "")
    turns = debate.get("turns", [])
    
    pro_tracker = {"updated_beliefs": {}, "contradictions": []}
    against_tracker = {"updated_beliefs": {}, "contradictions": []}
    
    for turn in turns:
        if turn.startswith("PRO:"):
            _update_tracker(turn, motion, "PRO", pro_tracker)
        else:
            _update_tracker(turn, motion, "AGAINST", against_tracker)
            
    return {
        "debate": debate,
        "pro_tracker": pro_tracker,
        "against_tracker": against_tracker
    }


def run_belief_tracker_test() -> None:
    with open("mock_debates.json", "r", encoding="utf-8") as f:
        debates_data = json.load(f)

    all_results = []
    for debate in debates_data:
        debate_result = _process_debate(debate)
        print("PRO: \n\n", debate_result["pro_tracker"])
        print("AGAINST: \n\n", debate_result["against_tracker"])
        all_results.append(debate_result)
        

    with open(
        "consistency_checker_results.json", 
        "w", 
        encoding="utf-8"
    ) as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    run_belief_tracker_test()
