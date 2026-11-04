from typing import cast

from configurations.config_manager import config, get_chat_model
from entities import DebateState, ResearcherState, SciencePaper
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from research import research_app
from tools import create_evidence_tool, extract_llm_response


def _build_research_prompt(agent_role: str, motion: str) -> str:
    return config.read_text("prompts.research_init").format(
        motion=motion,
        stance=agent_role,
        max_tries=config.read_integer("search.max_tries"),
    )


def _run_search_attempt(
    debate_id: str,
    agent_role: str,
    messages: list[BaseMessage],
    motion: str,
    current_approved: list[SciencePaper],
) -> tuple[str, list[SciencePaper]]:
    raw_response = get_chat_model("researcher").invoke(messages)
    query = extract_llm_response(raw_response)
    print(f"\n[Researcher {agent_role}] Formulated query: '{query}'")
    initial_state: ResearcherState = {
        "debate_id": debate_id,
        "motion": motion,
        "query": query,
        "agent_role": agent_role,
        "raw_papers": [],
        "approved_papers": current_approved,
    }
    search_result = cast(ResearcherState, research_app.invoke(initial_state))
    return query, search_result.get("approved_papers", [])


def _extract_approved_titles(current_approved_papers: list[SciencePaper]) -> list[str]:
    return [paper["title"] for paper in current_approved_papers if paper["title"]]


def _update_messages(
    messages: list[BaseMessage],
    query: str,
    current_approved_papers: list[SciencePaper],
    attempts_left: int,
) -> None:
    messages.append(AIMessage(content=query))
    approved_titles = _extract_approved_titles(current_approved_papers)
    next_prompt = config.read_text("prompts.research_middle").format(
        attempts_left=attempts_left,
        approved_titles="\n".join(approved_titles) or "None",
    )
    messages.append(HumanMessage(content=next_prompt))


def _has_enough_papers(
    agent_role: str,
    approved_papers: list[SciencePaper],
    max_knowledge_base: int,
) -> bool:
    print(
        f"[Researcher {agent_role}] Approved papers so far: "
        f"{len(approved_papers)} / {max_knowledge_base}"
    )
    return len(approved_papers) >= max_knowledge_base


def _run_research_loop(
    debate_id: str,
    agent_role: str,
    messages: list[BaseMessage],
    motion: str,
) -> list[str]:
    max_tries = config.read_integer("search.max_tries")
    max_knowledge_base = config.read_integer("search.max_knowledge_base")
    approved_papers: list[SciencePaper] = []
    print(f"\n--- Starting Research Phase for {agent_role} ---")
    for attempt in range(max_tries):
        query, approved_papers = _run_search_attempt(
            debate_id,
            agent_role,
            messages,
            motion,
            approved_papers,
        )
        if _has_enough_papers(agent_role, approved_papers, max_knowledge_base):
            break
        _update_messages(messages, query, approved_papers, max_tries - attempt - 1)
    return _extract_approved_titles(approved_papers)


def execute_researcher(debate_id: str, agent_role: str, motion: str) -> list[str]:
    prompt = _build_research_prompt(agent_role, motion)
    messages: list[BaseMessage] = [HumanMessage(content=prompt)]
    return _run_research_loop(debate_id, agent_role, messages, motion)


def _build_debater_messages(agent_role: str, state: DebateState) -> list[BaseMessage]:
    history = "\n".join(state["transcript"])
    if not history:
        content = config.read_text("prompts.opening").format(
            motion=state["motion"],
            stance=agent_role,
        )
    else:
        content = config.read_text("prompts.debator").format(
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
    evidence_tool = create_evidence_tool(agent_role, debate_id)
    model_with_tools = get_chat_model("debater").bind_tools([evidence_tool])
    while True:
        raw_response = model_with_tools.invoke(messages)
        messages.append(raw_response)
        if not raw_response.tool_calls:
            return extract_llm_response(raw_response)
        for tool_call in raw_response.tool_calls:
            result = evidence_tool.invoke(tool_call)
            messages.append(ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"] or "",
            ))


def execute_debater(agent_role: str, state: DebateState) -> str:
    messages = _build_debater_messages(agent_role, state)
    return _run_tool_loop(state["debate_id"], agent_role, messages)


def generate_debater_response(state: DebateState, stance_key: str) -> str:
    stance = config.read_text(stance_key)
    entry = execute_debater(stance, state)
    return f"[Turn {state['turn_count']}] {stance}: {entry}"
