from typing import Any, NotRequired, Literal, TypedDict

import chromadb
from chromadb.api.models.Collection import Collection
import ollama
from langgraph.graph import END, StateGraph

from openalex_search import search_papers, RawPaper
import configurations.small_local_llm as config


SEARCH_BATCH_SIZE = 2


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


def get_embedding(text: str) -> list[float]:
    if not text:
        return []
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text[:8000],
    )
    return response.get("embedding", [])


def search_node(state: ResearcherState) -> dict[str, Any]:
    query = state.get("query", "")
    count = state.get("search_count", 0)

    new_papers = search_papers(query, SEARCH_BATCH_SIZE)

    return {"raw_papers": new_papers, "search_count": count + 1}


def call_llm(prompt: str) -> str:
    response = ollama.chat(
        model=config.MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={
            "num_predict": config.MAX_TOKENS_PER_TURN,
            "temperature": config.TEMPERATURE,
        }
    )
    return response['message']['content'].strip()


def approve_paper(paper: RawPaper) -> bool:
    if paper is None:
        return False

    affiliations_str = ", ".join(paper.get("affiliations", []))

    prompt = f"""Evaluate this academic paper.
        Title: {paper.get("title")}
        Year: {paper.get("year")}
        Citations: {paper.get("citations")}
        Affiliations: {affiliations_str}
        Abstract: {paper.get("abstract")[:8000]}

        Decision Criteria from most necessary to optional:
        1. Impact: Is the citation count sufficient?
        (Take into account field and publication year,
        but never approve a paper with less than 3 citations).
        2. Methodological Rigor: Does the abstract cite specific,
         verifiable findings (e.g., statistical significance, 
         clear qualitative methodologies)?
        3. Reject papers that rely solely on vague adjectives 
        without detailing the data or method.
        4. Trust: Are the institutions recognized?

        Output EXACTLY ONE WORD: PASS if approved, FAIL if rejected.
    """

    response = call_llm(prompt)
    print(paper["title"], paper["citations"], response)
    
    return "PASS" in response.upper()


def _traverse_raw_papers(
    raw_papers: list[RawPaper],
    approved_papers: list[ApprovedPaper],
    collection: Collection,
) -> None:
    for paper in raw_papers:
        if approve_paper(paper):
            content = paper.get("abstract", "")
            vector = get_embedding(content)

            paper_data = dict(paper)
            paper_data["abstract_embedding"] = vector

            approved_papers.append(paper_data)
            
            title = paper_data.get("title", "")
            collection.upsert(
                documents=[content],
                metadatas=[{"title": title}],
                embeddings=[vector],
                ids=[title],
            )


def evaluate_node(
        state: ResearcherState,
) -> dict[Literal["approved_papers"], list[ApprovedPaper]]:
    raw_papers = state.get("raw_papers", [])
    approved_papers = state.get("approved_papers", [])
    agent_id = state.get("agent_id", "default")

    client = chromadb.PersistentClient(path="./debate_vectors")
    collection = client.get_or_create_collection(name=f"agent_{agent_id}")

    _traverse_raw_papers(
        raw_papers, 
        approved_papers, 
        collection,
    )

    return {"approved_papers": approved_papers}


research_workflow = StateGraph(ResearcherState)

research_workflow.add_node("search_node", search_node)
research_workflow.add_node("evaluate_node", evaluate_node)

research_workflow.set_entry_point("search_node")
research_workflow.add_edge("search_node", "evaluate_node")

research_workflow.add_edge("evaluate_node", END)

research_app = research_workflow.compile()
