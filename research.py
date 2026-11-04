import io
from typing import cast

from configurations.config_manager import config, get_chat_model, get_retriever
from debate_settings import REQUEST_TIMEOUT_SECONDS, TITLE_PREVIEW_CHARACTERS
from entities import ApprovalResult, ResearcherState, SciencePaper, SearchResult
from graph_types import RunnableGraph
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from openalex_search import search_papers
from pypdf import PdfReader
import requests
from tools import extract_llm_response


def search_node(state: ResearcherState) -> SearchResult:
    papers_per_search = config.read_integer("search.papers_per_search")
    return {"raw_papers": search_papers(state["query"], papers_per_search)}


def _cut_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half_limit = limit // 2
    return f"{text[:half_limit]}\n\n...\n\n{text[-half_limit:]}"


def _build_approval_prompt(paper: SciencePaper, motion: str) -> str:
    prompt_template = config.read_text("prompts.paper_approver_prompt")
    limit = config.read_integer("embedding.truncation_length")
    return prompt_template.format(
        motion=motion,
        title=paper["title"],
        year=paper["year"],
        citations=paper["citations"],
        affiliations=", ".join(paper["affiliations"]),
        abstract=_cut_middle(paper["abstract"], limit),
    )


def approve_paper(paper: SciencePaper, motion: str) -> bool:
    prompt = _build_approval_prompt(paper, motion)
    raw_response = get_chat_model("approver").invoke(prompt)
    verdict = extract_llm_response(raw_response)
    print(f"[Approver] '{paper['title'][:TITLE_PREVIEW_CHARACTERS]}...' -> {verdict}")
    pass_word = config.read_text("prompts.paper_approver_pass_word")
    return pass_word in verdict.upper()


def _download_pdf(url: str) -> bytes:
    print(f"[PDF] Downloading {url}...")
    response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    if "application/pdf" not in response.headers.get("Content-Type", ""):
        return b""
    return response.content


def _read_pdf_text(content: bytes) -> str:
    if not content:
        return ""
    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = [page.extract_text() for page in reader.pages]
    return "".join(f"{page}\n" for page in pages if page)


def extract_text_from_pdf(url: str) -> str:
    try:
        return _read_pdf_text(_download_pdf(url))
    except Exception as error:
        print(f"[PDF Error] Failed to extract text: {error}")
        return ""


def _choose_content(paper: SciencePaper) -> tuple[str, str]:
    pdf_url = paper["pdf_url"]
    full_text = extract_text_from_pdf(pdf_url) if pdf_url else ""
    title = paper["title"][:TITLE_PREVIEW_CHARACTERS]
    if full_text.strip():
        print(f"[Database] Saved full PDF for '{title}...'")
        return full_text, "pdf"
    print(f"[Database] Saved abstract for '{title}...'")
    return paper["abstract"], "abstract"


def process_and_store_paper(
    paper: SciencePaper,
    agent_role: str,
    debate_id: str,
) -> None:
    content, source = _choose_content(paper)
    if not content.strip():
        return
    document = Document(
        page_content=content,
        metadata={"title": paper["title"], "source": source},
    )
    get_retriever(agent_role, debate_id).add_documents([document])


def _approve_new_paper(
    paper: SciencePaper,
    state: ResearcherState,
    approved_papers: list[SciencePaper],
    existing_titles: set[str],
) -> None:
    if paper["title"] in existing_titles:
        return
    if approve_paper(paper, state["motion"]):
        approved_papers.append(paper.copy())
        existing_titles.add(paper["title"])
        process_and_store_paper(paper, state["agent_role"], state["debate_id"])


def approve_node(state: ResearcherState) -> ApprovalResult:
    approved_papers = state.get("approved_papers", [])
    max_knowledge_base = config.read_integer("search.max_knowledge_base")
    existing_titles: set[str] = {
        paper["title"] for paper in approved_papers if paper["title"]
    }
    for paper in state.get("raw_papers", []):
        _approve_new_paper(paper, state, approved_papers, existing_titles)
        if len(approved_papers) >= max_knowledge_base:
            break
    return {"approved_papers": approved_papers}


research_workflow = StateGraph(ResearcherState)

research_workflow.add_node("search_node", search_node)
research_workflow.add_node("approve_node", approve_node)
research_workflow.set_entry_point("search_node")
research_workflow.add_edge("search_node", "approve_node")
research_workflow.add_edge("approve_node", END)

research_app = cast(RunnableGraph[ResearcherState], research_workflow.compile())
