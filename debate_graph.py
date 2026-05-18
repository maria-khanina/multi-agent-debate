import json
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agents import execute_researcher, generate_debater_response
from configurations.config_manager import config, get_chat_model
from arguments_tracking import (
    deduplicate_contradictions,
    execute_with_backoff,
    get_topic_store,
    process_debate_turn,
)
from entities import Contradiction, DebateState, DebaterMistake
from tools import extract_llm_response


class FilteredTopics(TypedDict):
    keys_to_drop: list[str]


def prepare_debators(state: DebateState) -> dict[str, Any]:
    debate_id = state["debate_id"]
    motion = state.get("motion", "")
    pro = config.get("agents.pro_stance")
    against = config.get("agents.against_stance")
    
    return {
        "pro_sources": execute_researcher(debate_id, pro, motion),
        "against_sources": execute_researcher(debate_id, against, motion),
    }


def pro_node(state: DebateState) -> dict[str, Any]:
    entry = generate_debater_response(state, "agents.pro_stance")
    topic_store = get_topic_store(state["debate_id"])
    
    new_contradictions = []
    new_mistakes = []
    factcheck_stats = state.get("pro_factcheck_stats", {
        "SUPPORTED": 0, 
        "FABRICATED": 0, 
        "EXAGGERATED": 0, 
        "UNVERIFIABLE": 0
    })
    
    process_debate_turn(
        state["debate_id"],
        entry,
        state["motion"],
        "PRO",
        topic_store,
        new_contradictions,
        new_mistakes,
        factcheck_stats,
    )
    
    return {
        "transcript": [entry],
        "pro_contradictions": new_contradictions,
        "pro_mistakes": new_mistakes,
        "pro_factcheck_stats": factcheck_stats,
    }


def against_node(state: DebateState) -> dict[str, Any]:
    entry = generate_debater_response(state, "agents.against_stance")
    topic_store = get_topic_store(state["debate_id"])
    
    new_contradictions = []
    new_mistakes = []
    factcheck_stats = state.get("against_factcheck_stats", {
        "SUPPORTED": 0, 
        "FABRICATED": 0, 
        "EXAGGERATED": 0, 
        "UNVERIFIABLE": 0
    })
    
    process_debate_turn(
        state["debate_id"],
        entry,
        state["motion"],
        "AGAINST",
        topic_store,
        new_contradictions,
        new_mistakes,
        factcheck_stats,
    )

    return {
        "transcript": [entry],
        "turn_count": state.get("turn_count", 0) + 1,
        "against_contradictions": new_contradictions,
        "against_mistakes": new_mistakes,
        "against_factcheck_stats": factcheck_stats,
    }


def _get_topics_for_llm(all_topics: list[dict]) -> list[dict]:
    return [
        {
            "topic_key": t["subject"], 
            "winning_claim": t["winning_claim"]
        } 
        for t in all_topics
    ]


def _invoke_rhetoric_filter(topics_json: str) -> list[str]:
    language_model = get_chat_model("judge").with_structured_output(
        FilteredTopics
    )
    
    prompt_template = config.get("prompts.filter_rhetoric")
    prompt_text = prompt_template.format(topics_json=topics_json)
    
    def _invoke() -> FilteredTopics:
        return language_model.invoke([HumanMessage(content=prompt_text)])
        
    try:
        filter_result = execute_with_backoff(_invoke)
        return filter_result.get("keys_to_drop", [])
    except Exception as error:
        print(f"[Filter] Error during rhetoric filtering: {error}")
        return []


def _calculate_final_stats(
    all_topics: list[dict], 
    keys_to_drop: list[str],
) -> dict[str, int]:
    stats = {"PRO": 0, "AGAINST": 0}
    for topic in all_topics:
        if topic["subject"] not in keys_to_drop:
            winner = topic.get("current_winner")
            if winner in stats:
                stats[winner] += 1
    return stats


def filter_claims_node(state: DebateState) -> dict[str, Any]:
    print("\n--- Filtering Rhetoric & Calculating Stats ---")
    topic_store = get_topic_store(state["debate_id"])
    all_topics = topic_store.get_all_topics()
    
    topics_for_llm = _get_topics_for_llm(all_topics)
    if not topics_for_llm:
        return {"final_argument_stats": {"PRO": 0, "AGAINST": 0}}
        
    topics_json = json.dumps(topics_for_llm, indent=2)
    keys_to_drop = _invoke_rhetoric_filter(topics_json)
        
    print(f"[Filter] Dropped {len(keys_to_drop)} purely rhetorical topics.")
    
    stats = _calculate_final_stats(all_topics, keys_to_drop)
                
    print(f"[Stats] Final Scoreboard: {stats}")
    return {"final_argument_stats": stats}


def route_debate(
    state: DebateState,
) -> Literal["pro_node", "filter_claims_node"]:
    if state.get("turn_count", 0) >= state.get("max_turns", 3):
        return "filter_claims_node"
    return "pro_node"


def _format_contradictions_for_judge(
    contradictions: list[Contradiction],
) -> str:
    if not contradictions:
        return "None detected."
    formatted = []
    for contra in contradictions:
        formatted.append(
            f"- Topic: {contra['topic_key']}\n"
            f"  Old Stance: {contra['old_stance']}\n"
            f"  New Stance: {contra['new_stance']}\n"
            f"  Explanation: {contra['explanation']}"
        )
    return "\n".join(formatted)


def _format_mistakes_for_judge(
    mistakes: list[DebaterMistake],
) -> str:
    if not mistakes:
        return "None recorded."
    formatted = []
    for mistake in mistakes:
        formatted.append(
            f"- Topic: {mistake['topic']}\n"
            f"  Attempted Claim: {mistake['attempted_claim']}\n"
            f"  Failure Type: {mistake['failure_type']}\n"
            f"  Explanation: {mistake['explanation']}"
        )
    return "\n".join(formatted)


def _get_judge_prompt_kwargs(
    history_str: str,
    state: DebateState,
    cleaned_pro: list[Contradiction],
    cleaned_against: list[Contradiction],
) -> dict[str, Any]:
    pro_sources = "\n".join([f"- {t}" for t in state.get("pro_sources", [])])
    aga_sources = "\n".join(
        [f"- {t}" for t in state.get("against_sources", [])]
    )
    
    stats = state.get("final_argument_stats", {"PRO": 0, "AGAINST": 0})
    pro_fc_str = json.dumps(state.get("pro_factcheck_stats", {}))
    against_fc_str = json.dumps(state.get("against_factcheck_stats", {}))
    
    return {
        "motion": state.get("motion", ""),
        "pro_sources": pro_sources or "None",
        "against_sources": aga_sources or "None",
        "pro_contradictions": _format_contradictions_for_judge(cleaned_pro),
        "against_contradictions": _format_contradictions_for_judge(
            cleaned_against
        ),
        "pro_mistakes": _format_mistakes_for_judge(
            state.get("pro_mistakes", [])
        ),
        "against_mistakes": _format_mistakes_for_judge(
            state.get("against_mistakes", [])
        ),
        "pro_factcheck_stats": pro_fc_str,
        "against_factcheck_stats": against_fc_str,
        "pro_score": stats.get("PRO", 0),
        "against_score": stats.get("AGAINST", 0),
        "transcript": history_str,
    }


def _build_judge_messages(
    history_str: str,
    state: DebateState,
    cleaned_pro_contradictions: list[Contradiction],
    cleaned_against_contradictions: list[Contradiction],
) -> list[BaseMessage]:
    prompt_template = config.get("prompts.judge")
    
    kwargs = _get_judge_prompt_kwargs(
        history_str,
        state,
        cleaned_pro_contradictions,
        cleaned_against_contradictions,
    )
    prompt = prompt_template.format(**kwargs)
    
    return [HumanMessage(content=prompt)]


def judge_node(state: DebateState) -> dict[str, Any]:
    print("\n--- Auditing Contradictions ---")
    cleaned_pro = deduplicate_contradictions(
        state.get("pro_contradictions", [])
    )
    cleaned_against = deduplicate_contradictions(
        state.get("against_contradictions", [])
    )
    
    history_str = "\n".join(state["transcript"])
    
    llm = get_chat_model("judge")
    messages = _build_judge_messages(
        history_str, 
        state,
        cleaned_pro,
        cleaned_against,
    )
    
    print("\n--- Judge Thinking ---")
    response = llm.invoke(messages)
    judge_response = extract_llm_response(response)

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
    }
)
builder.add_edge("filter_claims_node", "judge_node")
builder.add_edge("judge_node", END)

debate_app = builder.compile()
