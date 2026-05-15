import uuid
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from agents import execute_researcher, generate_debater_response
from configurations.config_manager import config, get_chat_model
from debators.consistency_tracking import deduplicate_contradictions, get_memory_store, \
    process_debate_turn
from entities import Contradiction, DebateState
from tools import create_audit_tool, extract_llm_response


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
    memory_store = get_memory_store(state["debate_id"], "PRO")
    
    new_contradictions = []
    process_debate_turn(
        entry,
        state["motion"],
        "PRO",
        memory_store,
        new_contradictions,
    )
    
    return {
        "transcript": [entry],
        "pro_contradictions": new_contradictions,
    }


def against_node(state: DebateState) -> dict[str, Any]:
    entry = generate_debater_response(state, "agents.against_stance")
    memory_store = get_memory_store(state["debate_id"], "AGAINST")
    
    new_contradictions = []
    process_debate_turn(
        entry,
        state["motion"],
        "AGAINST",
        memory_store,
        new_contradictions,
    )

    return {
        "transcript": [entry],
        "turn_count": state.get("turn_count", 0) + 1,
        "against_contradictions": new_contradictions,
    }


def audit_node(state: DebateState) -> dict[str, Any]:
    print("\n--- Auditing Phase ---")
    cleaned_pro = deduplicate_contradictions(
        state.get("pro_contradictions", [])
    )
    cleaned_against = deduplicate_contradictions(
        state.get("against_contradictions", [])
    )
    
    return {
        "cleaned_pro_contradictions": cleaned_pro,
        "cleaned_against_contradictions": cleaned_against,
    }


def route_debate(state: DebateState) -> Literal["pro_node", "audit_node"]:
    if state.get("turn_count", 0) >= state.get("max_turns", 3):
        return "audit_node"
    return "pro_node"


def _format_contradictions_for_judge(
    contradictions: list[Contradiction],
) -> str:
    if not contradictions:
        return "None detected."
    formatted = []
    for c in contradictions:
        formatted.append(
            f"- Topic: {c['topic_key']}\n"
            f"  Old Stance: {c['old_stance']}\n"
            f"  New Stance: {c['new_stance']}\n"
            f"  Explanation: {c['explanation']}"
        )
    return "\n".join(formatted)


def _build_judge_messages(
    history_str: str,
    state: DebateState,
) -> list[BaseMessage]:
    pro_sources_str = "\n".join(
        [f"- {t}" for t in state.get("pro_sources", [])]
    )
    against_sources_str = "\n".join(
        [f"- {t}" for t in state.get("against_sources", [])]
    )
    
    pro_contra_str = _format_contradictions_for_judge(
        state.get("cleaned_pro_contradictions", [])
    )
    against_contra_str = _format_contradictions_for_judge(
        state.get("cleaned_against_contradictions", [])
    )

    prompt = config.get("prompts.judge").format(
        motion=state.get("motion", ""),
        pro_sources=pro_sources_str or "None",
        against_sources=against_sources_str or "None",
        pro_contradictions=pro_contra_str,
        against_contradictions=against_contra_str,
        transcript=history_str,
    )
    return [HumanMessage(content=prompt)]


def _run_judge_reasoning_loop(
    llm_with_tools: Runnable,
    messages: list[BaseMessage],
    factchecking_tool: BaseTool,
) -> str:
    print("\n--- Judge Thinking ---")
    while True:
        raw_response = llm_with_tools.invoke(messages)
        messages.append(raw_response)
        
        if not raw_response.tool_calls:
            return extract_llm_response(raw_response)
            
        for tool_call in raw_response.tool_calls:
            result = factchecking_tool.invoke(tool_call)
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )


def judge_node(state: DebateState) -> dict[str, Any]:
    history_str = "\n".join(state["transcript"])
    factchecking_tool = create_audit_tool(state["debate_id"])
    
    llm = get_chat_model("judge")
    llm_with_tools = llm.bind_tools([factchecking_tool])
    messages = _build_judge_messages(history_str, state)
    
    judge_response = _run_judge_reasoning_loop(
        llm_with_tools,
        messages,
        factchecking_tool,
    )

    return {"transcript": [f"\nVERDICT:\n{judge_response}"]}


builder = StateGraph(DebateState)

builder.add_node("prepare_debators", prepare_debators)
builder.add_node("pro_node", pro_node)
builder.add_node("against_node", against_node)
builder.add_node("audit_node", audit_node)
builder.add_node("judge_node", judge_node)

builder.add_edge(START, "prepare_debators")
builder.add_edge("prepare_debators", "pro_node")
builder.add_edge("pro_node", "against_node")
builder.add_conditional_edges(
    "against_node",
    route_debate,
    {
        "pro_node": "pro_node",
        "audit_node": "audit_node",
    }
)
builder.add_edge("audit_node", "judge_node")
builder.add_edge("judge_node", END)

debate_app = builder.compile()


def run_debate_test() -> None:
    current_debate_id = uuid.uuid4().hex

    initial_state = {
        "debate_id": current_debate_id,
        "motion": "AI will replace human",
        "transcript": [],
        "turn_count": 0,
        "max_turns": 2,
        "pro_sources": [],
        "against_sources": [],
        "pro_contradictions": [],
        "against_contradictions": [],
        "cleaned_pro_contradictions": [],
        "cleaned_against_contradictions": [],
    }

    print(f"\n{'='*50}")
    print(f"STARTING DEBATE: {initial_state['motion']}")
    print(f"{'='*50}\n")
    
    for event in debate_app.stream(initial_state):
        for node, state_update in event.items():
            print(f"\n--- [ NODE EXECUTED: {node} ] ---")
            if "transcript" in state_update:
                if state_update["transcript"]:
                    print(f"\n{state_update['transcript'][-1]}\n")
                    
    print(f"\n{'='*50}")
    print("DEBATE FINISHED")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    run_debate_test()
