from langchain_core.messages import AIMessage, BaseMessage, \
    HumanMessage, ToolMessage

from configurations.config_manager import config, get_chat_model
from research import research_app
from entities import DebateState, SciencePaper
from tools import create_evidence_tool, extract_llm_response


def _build_research_prompt(agent_role: str, motion: str) -> str:
    template = config.get("prompts.research_init")
    max_tries = config.get("search.max_tries")
    return template.format(
        motion=motion,
        stance=agent_role,
        max_tries=max_tries,
    )


def _run_search_attempt(
    debate_id: str,
    agent_role: str,
    messages: list[BaseMessage],
    motion: str,
    current_approved: list[dict],
) -> tuple[str, list[SciencePaper]]:
    llm = get_chat_model("researcher")
    raw_response = llm.invoke(messages)
    query = extract_llm_response(raw_response)

    print(f"\n[Researcher {agent_role}] Formulated query: '{query}'")

    search_result = research_app.invoke({
        "debate_id": debate_id,
        "motion": motion,
        "query": query,
        "agent_role": agent_role,
        "raw_papers": [],
        "approved_papers": current_approved,
    })

    return query, search_result.get("approved_papers", [])


def _extract_approved_titles(
    current_approved_papers: list[SciencePaper],
) -> list[str]:
    return [
        paper["title"] 
        for paper in current_approved_papers 
        if paper.get("title")
    ]


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
    debate_id: str,
    agent_role: str,
    messages: list[BaseMessage],
    motion: str,
) -> list[str]:
    max_tries = config.get("search.max_tries")
    max_knowedge_base = config.get("search.max_knowledge_base")
    current_approved_papers = []

    print(f"\n--- Starting Research Phase for {agent_role} ---")

    for attempt in range(max_tries):
        query, current_approved_papers = _run_search_attempt(
            debate_id,
            agent_role,
            messages,
            motion,
            current_approved_papers,
        )
        
        print(
            f"[Researcher {agent_role}] Approved papers so far: "
            f"{len(current_approved_papers)} / {max_knowedge_base}"
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


def execute_researcher(
    debate_id: str,
    agent_role: str,
    motion: str,
) -> list[str]:
    prompt = _build_research_prompt(agent_role, motion)
    messages = [HumanMessage(content=prompt)]
    return _run_research_loop(debate_id, agent_role, messages, motion)


def _build_debater_messages(
    agent_role: str,
    state: DebateState,
) -> list[BaseMessage]:
    history = "\n".join(state.get("transcript", []))
    if not history:
        content = config.get("prompts.opening").format(
            motion=state["motion"],
            stance=agent_role,
        )
    else:
        content = config.get("prompts.debator").format(
            motion=state["motion"],
            stance=agent_role,
            transcript=history,
        )
    return [HumanMessage(content=content)]


def _run_tool_loop(
    debate_id: str,
    agent_role: str,
    messages: list[BaseMessage],
) -> str:
    llm = get_chat_model("debater")
    evidence_tool = create_evidence_tool(agent_role, debate_id)
    llm_with_tools = llm.bind_tools([evidence_tool])

    while True:
        raw_response = llm_with_tools.invoke(messages)
        messages.append(raw_response)

        if not raw_response.tool_calls:
            return extract_llm_response(raw_response)

        for tool_call in raw_response.tool_calls:
            result = evidence_tool.invoke(tool_call)
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            ))


def execute_debater(agent_role: str, state: DebateState) -> str:
    messages = _build_debater_messages(agent_role, state)
    return _run_tool_loop(state["debate_id"], agent_role, messages)


def generate_debater_response(state: DebateState, stance_key: str) -> str:
    stance = config.get(stance_key)
    entry = execute_debater(stance, state)
    turn = state.get("turn_count", 0)
    return f"[Turn {turn}] {stance}: {entry}"
