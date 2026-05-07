import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph

import configurations.small_local_llm as config
from tools import create_evidence_tool
from research import research_app


class DebateState(TypedDict):
    motion: str
    transcript: Annotated[list[str], operator.add]
    turn_count: int
    max_turns: int


def _execute_researcher(
    stance: str,
    agent_id: str, 
    motion: str, 
    max_tries=2,
    max_knowledge_base=2,
) -> None:
    llm = ChatOllama(
        model=config.MODEL,
        temperature=config.TEMPERATURE,
        num_ctx=8192,
    )

    prompt = config.RESEARCH_INIT_PROMPT.format(
        motion=motion,
        stance=stance,
        max_tries=max_tries,
    )
    messages = [HumanMessage(content=prompt)]
    total_approved = 0

    for attempt in range(max_tries):
        response = llm.invoke(messages)
        query = response.content.strip().strip("'\"")
        print(f"Agent {agent_id}, searching attempt {attempt}: {query}")
        
        search_result = research_app.invoke({
            "query": query,
            "agent_id": agent_id,
            "search_count": 0,
            "raw_papers": [],
            "approved_papers": [],
        })

        approved = len(search_result.get("approved_papers", []))
        total_approved += approved
        if total_approved >= max_knowledge_base:
            break
        
        messages.append(AIMessage(content=query))
        
        next_prompt = config.RESEARCH_MIDDLE_PROMPT.format(
            approved=approved,
            attempts_left=max_tries - attempt - 1,
        )
        messages.append(HumanMessage(content=next_prompt))


def prepare_debators(state: DebateState) -> dict[str, Any]:
    motion = state.get("motion", "")
    prep_tries = 3
    
    _execute_researcher(
        "PRO", 
        "PRO", 
        motion, 
        prep_tries
    )
    
    _execute_researcher(
        "AGAINST", 
        "AGAINST", 
        motion, 
        prep_tries
    )

    return {}


def _execute_debator(
    stance: str,
    agent_id: str,
    state: DebateState,
) -> str:
    llm = ChatOllama(
        model=config.MODEL,
        temperature=config.TEMPERATURE,
        num_ctx=8192,
    )
    evidence_tool = create_evidence_tool(agent_id)
    llm_with_tools = llm.bind_tools([evidence_tool])

    history = "\n".join(state.get("transcript", []))

    if not history:
        content = config.OPENING_PROMPT.format(
            motion=state["motion"],
            stance=stance,
        )
    else:
        content = config.DEBATOR_PROMPT.format(
            motion=state["motion"], 
            stance=stance, 
            transcript=history,
        )

    messages = [HumanMessage(content=content)]
    response = llm_with_tools.invoke(messages)

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return f"{stance}: {response.content}"

        for tool_call in response.tool_calls:
            result = evidence_tool.invoke(tool_call)
            messages.append(ToolMessage(
                content=str(result), 
                tool_call_id=tool_call["id"],
            ))


def pro_node(state: DebateState) -> dict[str, Any]:
    entry = _execute_debator("PRO", "PRO", state)
    return {"transcript": [entry]}


def against_node(state: DebateState) -> dict[str, Any]:
    entry = _execute_debator("AGAINST", "AGAINST", state)
    return {
        "transcript": [entry],
        "turn_count": state.get("turn_count", 0) + 1
    }


def route_debate(state: DebateState) -> str:
    if state["turn_count"] >= state["max_turns"]:
        return "judge_node"
    return "pro_node"


def judge_node(state: DebateState) -> dict[str, Any]:
    llm = ChatOllama(model=config.MODEL, temperature=config.TEMPERATURE)
    history = "\n".join(state.get("transcript", []))

    prompt = config.JUDGE_PROMPT.format(
        motion=state["motion"], 
        transcript=history
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"transcript": [f"\nJUDGE'S VERDICT:\n{response.content}"]}


builder = StateGraph(DebateState)

builder.add_node("prepare_debators", prepare_debators)
builder.add_node("pro_node", pro_node)
builder.add_node("against_node", against_node)
builder.add_node("judge_node", judge_node)

builder.add_edge(START, "prepare_debators")
builder.add_edge("prepare_debators", "pro_node")
builder.add_edge("pro_node", "against_node")

builder.add_conditional_edges(
    "against_node",
    route_debate,
    {
        "pro_node": "pro_node",
        "judge_node": "judge_node",
    }
)

builder.add_edge("judge_node", END)

debate_app = builder.compile()


def run_debate_test() -> None:
    initial_state = {
        "motion": "Men are smarter than women",
        "transcript": [],
        "turn_count": 0,
        "max_turns": 2,
    }

    print(f"--- DEBATE STARTED: {initial_state['motion']} ---")
    
    for event in debate_app.stream(initial_state):
        for node, data in event.items():
            print(f"\n[ NODE: {node} ]")
            
            if data and "transcript" in data:
                last_msg = data["transcript"][-1]
                print(f"{last_msg}")

    print("\n--- DEBATE FINISHED ---")


if __name__ == "__main__":
    run_debate_test()
