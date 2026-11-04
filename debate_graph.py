import json
from typing import Any, Literal, TypedDict, cast

from agents import execute_researcher, generate_debater_response
from arguments_tracking import (
    deduplicate_contradictions,
    get_topic_store,
    process_debate_turn,
)
from configurations.config_manager import config, get_chat_model
from entities import (
    Contradiction,
    DebateState,
    DebaterMistake,
    FactCheckStats,
    Side,
    TopicState,
)
from graph_types import RunnableGraph
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from language_calls import ask_structured_with_backoff
from tools import extract_llm_response


class FilteredTopics(TypedDict):
    keys_to_drop: list[str]


class TurnOutcome(TypedDict):
    entry: str
    contradictions: list[Contradiction]
    mistakes: list[DebaterMistake]
    factcheck_stats: FactCheckStats


def prepare_debators(state: DebateState) -> dict[str, Any]:
    debate_id = state["debate_id"]
    motion = state["motion"]
    pro = config.read_text("agents.pro_stance")
    against = config.read_text("agents.against_stance")
    return {
        "pro_sources": execute_researcher(debate_id, pro, motion),
        "against_sources": execute_researcher(debate_id, against, motion),
    }


def _run_turn(
    state: DebateState,
    side: Side,
    factcheck_stats: FactCheckStats,
) -> TurnOutcome:
    stance_key = "agents.pro_stance" if side == "PRO" else "agents.against_stance"
    entry = generate_debater_response(state, stance_key)
    outcome: TurnOutcome = {
        "entry": entry,
        "contradictions": [],
        "mistakes": [],
        "factcheck_stats": factcheck_stats.copy(),
    }
    process_debate_turn(
        state["debate_id"],
        entry,
        state["motion"],
        side,
        get_topic_store(state["debate_id"]),
        outcome["contradictions"],
        outcome["mistakes"],
        outcome["factcheck_stats"],
    )
    return outcome


def pro_node(state: DebateState) -> dict[str, Any]:
    outcome = _run_turn(state, "PRO", state["pro_factcheck_stats"])
    return {
        "transcript": [outcome["entry"]],
        "pro_contradictions": outcome["contradictions"],
        "pro_mistakes": outcome["mistakes"],
        "pro_factcheck_stats": outcome["factcheck_stats"],
    }


def against_node(state: DebateState) -> dict[str, Any]:
    outcome = _run_turn(state, "AGAINST", state["against_factcheck_stats"])
    return {
        "transcript": [outcome["entry"]],
        "turn_count": state["turn_count"] + 1,
        "against_contradictions": outcome["contradictions"],
        "against_mistakes": outcome["mistakes"],
        "against_factcheck_stats": outcome["factcheck_stats"],
    }


def _get_topics_for_llm(all_topics: list[TopicState]) -> list[dict[str, str]]:
    return [
        {"topic_key": topic["subject"], "winning_claim": topic["winning_claim"]}
        for topic in all_topics
    ]


def _invoke_rhetoric_filter(topics_json: str) -> list[str]:
    prompt_text = config.read_text("prompts.filter_rhetoric").format(
        topics_json=topics_json,
    )
    try:
        result = ask_structured_with_backoff("judge", FilteredTopics, prompt_text)
    except Exception as error:
        print(f"[Filter] Error during rhetoric filtering: {error}")
        return []
    return result.get("keys_to_drop", [])


def _calculate_final_stats(
    all_topics: list[TopicState],
    keys_to_drop: list[str],
) -> dict[str, int]:
    stats: dict[str, int] = {"PRO": 0, "AGAINST": 0}
    for topic in all_topics:
        if topic["subject"] not in keys_to_drop:
            stats[topic["current_winner"]] += 1
    return stats


def filter_claims_node(state: DebateState) -> dict[str, Any]:
    print("\n--- Filtering Rhetoric & Calculating Stats ---")
    all_topics = get_topic_store(state["debate_id"]).get_all_topics()
    topics_for_llm = _get_topics_for_llm(all_topics)
    if not topics_for_llm:
        return {"final_argument_stats": {"PRO": 0, "AGAINST": 0}}
    keys_to_drop = _invoke_rhetoric_filter(json.dumps(topics_for_llm, indent=2))
    print(f"[Filter] Dropped {len(keys_to_drop)} purely rhetorical topics.")
    stats = _calculate_final_stats(all_topics, keys_to_drop)
    print(f"[Stats] Final Scoreboard: {stats}")
    return {"final_argument_stats": stats}


def route_debate(state: DebateState) -> Literal["pro_node", "filter_claims_node"]:
    if state["turn_count"] >= state["max_turns"]:
        return "filter_claims_node"
    return "pro_node"


def _format_contradictions_for_judge(contradictions: list[Contradiction]) -> str:
    if not contradictions:
        return "None detected."
    return "\n".join(
        f"- Topic: {contradiction['topic_key']}\n"
        f"  Old Stance: {contradiction['old_stance']}\n"
        f"  New Stance: {contradiction['new_stance']}\n"
        f"  Explanation: {contradiction['explanation']}"
        for contradiction in contradictions
    )


def _format_mistakes_for_judge(mistakes: list[DebaterMistake]) -> str:
    if not mistakes:
        return "None recorded."
    return "\n".join(
        f"- Topic: {mistake['topic']}\n"
        f"  Attempted Claim: {mistake['attempted_claim']}\n"
        f"  Failure Type: {mistake['failure_type']}\n"
        f"  Explanation: {mistake['explanation']}"
        for mistake in mistakes
    )


def _format_sources(titles: list[str]) -> str:
    return "\n".join(f"- {title}" for title in titles) or "None"


def _get_judge_prompt_arguments(
    history_text: str,
    state: DebateState,
    cleaned_pro: list[Contradiction],
    cleaned_against: list[Contradiction],
) -> dict[str, str | int]:
    stats = state["final_argument_stats"]
    return {
        "motion": state["motion"],
        "pro_sources": _format_sources(state["pro_sources"]),
        "against_sources": _format_sources(state["against_sources"]),
        "pro_contradictions": _format_contradictions_for_judge(cleaned_pro),
        "against_contradictions": _format_contradictions_for_judge(cleaned_against),
        "pro_mistakes": _format_mistakes_for_judge(state["pro_mistakes"]),
        "against_mistakes": _format_mistakes_for_judge(state["against_mistakes"]),
        "pro_factcheck_stats": json.dumps(state["pro_factcheck_stats"]),
        "against_factcheck_stats": json.dumps(state["against_factcheck_stats"]),
        "pro_score": stats.get("PRO", 0),
        "against_score": stats.get("AGAINST", 0),
        "transcript": history_text,
    }


def _build_judge_messages(
    history_text: str,
    state: DebateState,
    cleaned_pro_contradictions: list[Contradiction],
    cleaned_against_contradictions: list[Contradiction],
) -> list[BaseMessage]:
    prompt_arguments = _get_judge_prompt_arguments(
        history_text,
        state,
        cleaned_pro_contradictions,
        cleaned_against_contradictions,
    )
    prompt = config.read_text("prompts.judge").format(**prompt_arguments)
    return [HumanMessage(content=prompt)]


def judge_node(state: DebateState) -> dict[str, Any]:
    print("\n--- Auditing Contradictions ---")
    cleaned_pro = deduplicate_contradictions(state["pro_contradictions"])
    cleaned_against = deduplicate_contradictions(state["against_contradictions"])
    messages = _build_judge_messages(
        "\n".join(state["transcript"]),
        state,
        cleaned_pro,
        cleaned_against,
    )
    print("\n--- Judge Thinking ---")
    judge_response = extract_llm_response(get_chat_model("judge").invoke(messages))
    return {"transcript": [f"\nVERDICT:\n{judge_response}"]}


builder = StateGraph(DebateState)

builder.add_node("prepare_debators", prepare_debators)
builder.add_node("pro_node", pro_node)
builder.add_node("against_node", against_node)
builder.add_node("filter_claims_node", filter_claims_node)
builder.add_node("judge_node", judge_node)

builder.add_edge(START, "prepare_debators")
builder.add_edge("prepare_debators", "pro_node")
builder.add_edge("pro_node", "against_node")
builder.add_conditional_edges(
    "against_node",
    route_debate,
    {
        "pro_node": "pro_node",
        "filter_claims_node": "filter_claims_node",
    },
)
builder.add_edge("filter_claims_node", "judge_node")
builder.add_edge("judge_node", END)

debate_app = cast(RunnableGraph[DebateState], builder.compile())
