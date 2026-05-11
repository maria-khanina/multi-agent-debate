import io
import requests
from typing import NotRequired, Literal, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.documents import Document
from pypdf import PdfReader

from openalex_search import search_papers, RawPaper
from configurations.config_manager import config, get_chat_model, get_retriever


class ApprovedPaper(TypedDict):
    title: str
    year: int | None
    citations: int
    affiliations: list[str]
    abstract: str
    abstract_embedding: list[float]
    pdf_url: str | None


class ResearcherState(TypedDict):
    query: str
    agent_id: str
    raw_papers: NotRequired[list[RawPaper]]
    approved_papers: NotRequired[list[ApprovedPaper]]
    search_count: NotRequired[int]


class SearchResult(TypedDict):
    raw_papers: list[RawPaper]
    search_count: int


def search_node(state: ResearcherState) -> SearchResult:
    query = state.get("query", "")
    count = state.get("search_count", 0)
    papers_per_search = config.get("search.papers_per_search")

    new_papers = search_papers(query, papers_per_search)

    return {"raw_papers": new_papers, "search_count": count + 1}


def _cut_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
        
    half_limit = limit // 2
    head = text[:half_limit]
    tail = text[-half_limit:]
    
    return f"{head}\n\n...\n\n{tail}"


def _build_approval_prompt(paper: RawPaper) -> str:
    affiliations_str = ", ".join(paper.get("affiliations", []))
    prompt_template = config.get("prompts.paper_approver_prompt")
    limit = config.get("llm.truncation_length")
    abstract = paper.get("abstract", "")
    
    return prompt_template.format(
        title=paper.get("title"),
        year=paper.get("year"),
        citations=paper.get("citations"),
        affiliations=affiliations_str,
        abstract=_cut_middle(abstract, limit),
    )


def approve_paper(paper: RawPaper) -> bool:
    if paper is None:
        return False

    prompt = _build_approval_prompt(paper)
    llm = get_chat_model()
    response = llm.invoke(prompt)
    content = str(response.content)
    print(paper["title"], paper["citations"], content)
    
    pass_word = config.get("prompts.paper_approver_pass_word")
    return pass_word in content.upper()


def extract_text_from_pdf(url: str) -> str:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" not in content_type:
            return ""
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception:
        return ""


def _process_and_store_paper(
    paper_data: RawPaper,
    agent_id: str,
) -> None:
    title = str(paper_data.get("title", ""))
    pdf_url = paper_data.get("pdf_url")
    abstract = str(paper_data.get("abstract", ""))
    
    full_text = ""
    if pdf_url:
        full_text = extract_text_from_pdf(pdf_url)
        
    content = full_text if full_text.strip() else abstract
    if not content.strip():
        return
        
    doc = Document(
        page_content=content,
        metadata={
            "title": title,
            "source": "pdf" if full_text.strip() else "abstract"
        }
    )
    retriever = get_retriever(agent_id)
    retriever.add_documents([doc])


def approve_node(
    state: ResearcherState,
) -> dict[Literal["approved_papers"], list[ApprovedPaper]]:
    raw_papers = state.get("raw_papers", [])
    approved_papers = state.get("approved_papers", [])
    agent_id = state.get("agent_id", "default")
    
    for paper in raw_papers:
        if approve_paper(paper):
            approved_papers.append(dict(paper))
            _process_and_store_paper(paper, agent_id)
            
    return {"approved_papers": approved_papers}


research_workflow = StateGraph(ResearcherState)
research_workflow.add_node("search_node", search_node)
research_workflow.add_node("approve_node", approve_node)
research_workflow.set_entry_point("search_node")
research_workflow.add_edge("search_node", "approve_node")
research_workflow.add_edge("approve_node", END)

research_app = research_workflow.compile()
