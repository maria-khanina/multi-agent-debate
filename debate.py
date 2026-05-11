import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage, \
    AIMessage, BaseMessage
from langgraph.graph import START, END, StateGraph

from configurations.config_manager import config, get_chat_model
from tools import create_evidence_tool, extract_llm_response
from research import research_app


class DebateState(TypedDict):
    motion: str
    transcript: Annotated[list[str], operator.add]
    turn_count: int
    max_turns: int


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
    attempt: int,
) -> tuple[str, int]:
    llm = get_chat_model("researcher")
    raw_response = llm.invoke(messages)
    query = extract_llm_response(raw_response)
    print(f"Agent {agent_id}, searching attempt {attempt}: {query}")

    search_result = research_app.invoke({
        "query": query,
        "agent_id": agent_id,
        "raw_papers": [],
        "approved_papers": [],
    })

    approved = len(search_result.get("approved_papers", []))
    return query, approved


def _run_research_loop(
    agent_id: str,
    messages: list[BaseMessage],
) -> None:
    max_tries = config.get("search.max_tries")
    max_knowledge_base = config.get("search.max_knowledge_base")
    total_approved = 0

    for attempt in range(max_tries):
        query, approved = _run_search_attempt(agent_id, messages, attempt)
        
        total_approved += approved
        if total_approved >= max_knowledge_base:
            break
        
        messages.append(AIMessage(content=query))
        
        next_prompt = config.get("prompts.research_middle").format(
            approved=approved,
            attempts_left=max_tries - attempt - 1,
        )
        messages.append(HumanMessage(content=next_prompt))


def _execute_researcher(stance: str, agent_id: str, motion: str) -> None:
    prompt = _build_research_prompt(stance, motion)
    messages = [HumanMessage(content=prompt)]
    _run_research_loop(agent_id, messages)


def prepare_debators(state: DebateState) -> dict[str, Any]:
    motion = state.get("motion", "")
    pro = config.get("agents.pro_stance")
    against = config.get("agents.against_stance")
    
    _execute_researcher(pro, pro, motion)
    _execute_researcher(against, against, motion)

    return {}


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
        output_response = extract_llm_response(raw_response)

        if not raw_response.tool_calls:
            return f"{stance}: {output_response}"

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





def pro_node(state: DebateState) -> ProJudgeResult:
    pro = config.get("agents.pro_stance")
    entry = _execute_debator(pro, pro, state)
    return {"transcript": [entry]}


def against_node(state: DebateState) -> AgainstResult:
    against = config.get("agents.against_stance")
    entry = _execute_debator(against, against, state)
    return {
        "transcript": [entry],
        "turn_count": state.get("turn_count", 0) + 1
    }


def route_debate(state: DebateState) -> Literal["pro_node", "judge_node"]:
    if state.get("turn_count", 0) >= state.get("max_turns", 3):
        return "judge_node"
    return "pro_node"


def judge_node(state: DebateState) -> ProJudgeResult:
    llm = get_chat_model("judge")
    history = "\n".join(state.get("transcript", []))

    prompt = config.get("prompts.judge").format(
        motion=state["motion"], 
        transcript=history
    )
    
    raw_response = llm.invoke([HumanMessage(content=prompt)])
    judge_response = extract_llm_response(raw_response)
    return {"transcript": [f"\nJUDGE'S VERDICT:\n{judge_response}"]}


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
