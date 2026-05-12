import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage, \
    AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import START, END, StateGraph

from configurations.config_manager import config, get_chat_model
from tools import audit_cited_claim, create_evidence_tool, \
    extract_llm_response, create_consistency_tool
from research import research_app, SciencePaper


class DebateState(TypedDict):
    motion: str
    transcript: Annotated[list[str], operator.add]
    turn_count: int
    max_turns: int
    pro_sources: list[str]
    against_sources: list[str]


class ProJudgeResult(TypedDict):
    transcript: list[str]


class AgainstResult(TypedDict):
    transcript: list[str]
    turn_count: int


def _build_research_prompt(stance: str, motion: str) -> str:
    template = config.get("prompts.research_init")
    max_tries = config.get("search.max_tries")
    return template.format(
        motion=motion,
        stance=stance,
        max_tries=max_tries,
    )


def _run_search_attempt(
    agent_id: str,
    messages: list[BaseMessage],
    motion: str,
    current_approved: list[dict],
) -> tuple[str, list[SciencePaper]]:
    llm = get_chat_model("researcher")
    raw_response = llm.invoke(messages)
    query = extract_llm_response(raw_response)
    print(f"-----Researcher {agent_id} Agent searchs for {query}-----")

    search_result = research_app.invoke({
        "motion": motion,
        "query": query,
        "agent_id": agent_id,
        "raw_papers": [],
        "approved_papers": current_approved,
    })

    return query, search_result.get("approved_papers", [])


def _extract_approved_titles(
        current_approved_papers: list[SciencePaper]
) -> list[str]:
    approved_titles: list[str] = []
    for paper in current_approved_papers:
        if paper.get("title"):
            approved_titles.append(paper["title"])
    return approved_titles


def _update_messages(
        messages: list[BaseMessage],
        query: str,
        current_approved_papers: list[SciencePaper],
        max_tries: int,
        attempt: int,
) -> None:
    messages.append(AIMessage(content=query))   
    approved_titles = _extract_approved_titles(current_approved_papers)
    titles_str = "\n".join(approved_titles) or "None"
    next_prompt = config.get("prompts.research_middle").format(
        attempts_left=max_tries - attempt - 1,
        approved_titles=titles_str,
    )
    messages.append(HumanMessage(content=next_prompt))


def _run_research_loop(
    agent_id: str,
    messages: list[BaseMessage],
    motion: str,
) -> list[str]:
    max_tries = config.get("search.max_tries")
    max_knowedge_base = config.get("search.max_knowledge_base")
    current_approved_papers: list[SciencePaper] = []

    for attempt in range(max_tries):
        query, current_approved_papers = _run_search_attempt(
            agent_id,
            messages,
            motion,
            current_approved_papers,
        )
        
        if len(current_approved_papers) >= max_knowedge_base:
            break

        _update_messages(
            messages,
            query,
            current_approved_papers,
            max_tries,
            attempt,
        )

    return _extract_approved_titles(current_approved_papers)


def _execute_researcher(stance: str, agent_id: str, motion: str) -> list[str]:
    prompt = _build_research_prompt(stance, motion)
    messages = [HumanMessage(content=prompt)]
    return _run_research_loop(agent_id, messages, motion)


def prepare_debators(state: DebateState) -> dict[str, list[str]]:
    motion = state.get("motion", "")
    pro = config.get("agents.pro_stance")
    against = config.get("agents.against_stance")
    
    pro_sources = _execute_researcher(pro, pro, motion)
    against_sources = _execute_researcher(against, against, motion)

    return {
        "pro_sources": pro_sources,
        "against_sources": against_sources
    }


def _build_debator_messages(
    stance: str,
    state: DebateState,
) -> list[list[BaseMessage]]:
    history = "\n".join(state.get("transcript", []))

    if not history:
        content = config.get("prompts.opening").format(
            motion=state["motion"],
            stance=stance,
        )
    else:
        content = config.get("prompts.debator").format(
            motion=state["motion"], 
            stance=stance, 
            transcript=history,
        )

    return [HumanMessage(content=content)]


def _run_tool_loop(
    stance: str,
    agent_id: str,
    messages: list[BaseMessage],
) -> str:
    llm = get_chat_model("debater")
    evidence_tool = create_evidence_tool(agent_id)
    llm_with_tools = llm.bind_tools([evidence_tool])

    while True:
        raw_response = llm_with_tools.invoke(messages)
        messages.append(raw_response)

        if not raw_response.tool_calls:
            output_response = extract_llm_response(raw_response)
            return output_response

        for tool_call in raw_response.tool_calls:
            result = evidence_tool.invoke(tool_call)
            messages.append(ToolMessage(
                content=str(result), 
                tool_call_id=tool_call["id"],
            ))


def _execute_debator(
    stance: str,
    agent_id: str,
    state: DebateState,
) -> str:
    messages = _build_debator_messages(stance, state)
    return _run_tool_loop(stance, agent_id, messages)


def _generate_debater_response(
    state: DebateState,
    stance_key: str,
) -> str:
    stance = config.get(stance_key)
    entry = _execute_debator(stance, stance, state)
    turn = state.get("turn_count", 0)

    return f"[Turn {turn}] {stance}: {entry}"


def pro_node(state: DebateState) -> ProJudgeResult:
    entry = _generate_debater_response(state, "agents.pro_stance")
    return {"transcript": [entry]}


def against_node(state: DebateState) -> AgainstResult:
    entry = _generate_debater_response(state, "agents.against_stance")
    return {
        "transcript": [entry],
        "turn_count": state.get("turn_count", 0) + 1
    }


def route_debate(state: DebateState) -> Literal["pro_node", "judge_node"]:
    if state.get("turn_count", 0) >= state.get("max_turns", 3):
        return "judge_node"
    return "pro_node"


def _build_judge_messages(
    history_str: str,
    state: DebateState,
) -> list[BaseMessage]:
    pro_sources_str = "\n".join(
        [f"- {title}" for title in state.get("pro_sources", [])]
    )
    against_sources_str = "\n".join(
        [f"- {title}" for title in state.get("against_sources", [])]
    )

    prompt = config.get("prompts.judge").format(
        motion=state.get("motion", ""),
        pro_sources=pro_sources_str or "None",
        against_sources=against_sources_str or "None",
        transcript=history_str,
    )
    return [HumanMessage(content=prompt)]


def _run_judge_reasoning_loop(
    llm_with_tools: Runnable,
    messages: list[BaseMessage],
    consistency_tool: BaseTool,
) -> str:
    while True:
        raw_response = llm_with_tools.invoke(messages)
        messages.append(raw_response)
        
        if not raw_response.tool_calls:
            return extract_llm_response(raw_response)
            
        for tool_call in raw_response.tool_calls:
            if tool_call["name"] == "check_logical_consistency":
                result = consistency_tool.invoke(tool_call)
            else:
                result = audit_cited_claim.invoke(tool_call)
                
            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )


def judge_node(state: DebateState) -> ProJudgeResult:
    history = state.get("transcript", [])
    history_str = "\n".join(history)

    consistency_tool = create_consistency_tool(history)
    messages = _build_judge_messages(history_str, state)

    llm = get_chat_model("judge")
    llm_with_tools = llm.bind_tools([consistency_tool, audit_cited_claim])

    judge_response = _run_judge_reasoning_loop(
        llm_with_tools,
        messages,
        consistency_tool,
    )

    return {"transcript": [f"\nVERDICT:\n{judge_response}"]}


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
        "motion": "Sugar replacements are bad for your health.",
        "transcript": [],
        "turn_count": 0,
        "max_turns": 2,
        "pro_sources": [],
        "against_sources": [],
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
