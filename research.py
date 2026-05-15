import io
from typing import Literal

import requests
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from pypdf import PdfReader

from configurations.config_manager import config, get_chat_model, \
    get_retriever
from entities import ResearcherState, SciencePaper, SearchResult
from openalex_search import search_papers
from tools import extract_llm_response


def search_node(state: ResearcherState) -> SearchResult:
    query = state.get("query", "")
    papers_per_search = config.get("search.papers_per_search")
    new_papers = search_papers(query, papers_per_search)
    return {"raw_papers": new_papers}


def _cut_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
        
    half_limit = limit // 2
    head = text[:half_limit]
    tail = text[-half_limit:]
    
    return f"{head}\n\n...\n\n{tail}"


def _build_approval_prompt(paper: SciencePaper, motion: str) -> str:
    affiliations_str = ", ".join(paper.get("affiliations", []))
    prompt_template = config.get("prompts.paper_approver_prompt")
    limit = config.get("embedding.truncation_length")
    abstract = paper.get("abstract", "")
    
    return prompt_template.format(
        motion=motion,
        title=paper.get("title"),
        year=paper.get("year"),
        citations=paper.get("citations"),
        affiliations=affiliations_str,
        abstract=_cut_middle(abstract, limit),
    )


def approve_paper(paper: SciencePaper, motion: str) -> bool:
    if paper is None:
        return False

    prompt = _build_approval_prompt(paper, motion)
    llm = get_chat_model("approver")
    raw_response = llm.invoke(prompt)
    verdict = extract_llm_response(raw_response)
    
    print(f"[Approver] '{paper['title'][:30]}...' -> {verdict}")
    
    pass_word = config.get("prompts.paper_approver_pass_word")
    return pass_word in verdict.upper()


def extract_text_from_pdf(url: str) -> str:
    try:
        print(f"[PDF] Downloading {url}...")
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
    except Exception as e:
        print(f"[PDF Error] Failed to extract text: {e}")
        return ""


def _process_and_store_paper(
    paper_data: SciencePaper,
    agent_role: str,
    debate_id: str,
) -> None:
    title = str(paper_data.get("title", ""))
    pdf_url = paper_data.get("pdf_url")
    abstract = str(paper_data.get("abstract", ""))
    
    full_text = ""
    if pdf_url:
        full_text = extract_text_from_pdf(pdf_url)
        
    if full_text.strip():
        content = full_text
        print(f"[Database] Saved full PDF for '{title[:30]}...'")
    else:
        content = abstract
        print(f"[Database] Saved abstract for '{title[:30]}...'")
        
    if not content.strip():
        return None

    doc = Document(
        page_content=content,
        metadata={
            "title": title,
            "source": "pdf" if full_text.strip() else "abstract",
        },
    )
    retriever = get_retriever(agent_role, debate_id)
    retriever.add_documents([doc])


def approve_node(
    state: ResearcherState,
) -> dict[Literal["approved_papers"], list[SciencePaper]]:
    raw_papers = state.get("raw_papers", [])
    approved_papers = state.get("approved_papers", [])
    agent_role = state.get("agent_role", "default")
    max_knowledge_base = config.get("search.max_knowledge_base", 2)
    
    existing_titles = set()
    for paper in approved_papers:
        if paper.get("title"):
            existing_titles.add(paper["title"])
    
    for paper in raw_papers:
        title = paper.get("title")
        if title and title in existing_titles:
            continue
            
        if approve_paper(paper, state["motion"]):
            approved_papers.append(dict(paper))
            existing_titles.add(title)
            _process_and_store_paper(
                paper, 
                agent_role, 
                state["debate_id"],
            )
        
        if len(approved_papers) >= max_knowledge_base:
            break
            
    return {"approved_papers": approved_papers}


research_workflow = StateGraph(ResearcherState)

research_workflow.add_node("search_node", search_node)
research_workflow.add_node("approve_node", approve_node)
research_workflow.set_entry_point("search_node")
research_workflow.add_edge("search_node", "approve_node")
research_workflow.add_edge("approve_node", END)

research_app = research_workflow.compile()
