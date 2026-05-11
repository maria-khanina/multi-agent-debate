from typing import TypedDict

import arxiv
from langchain_core.tools import tool


class _Paper(TypedDict):
    title: str
    year: int
    authors: list[str]
    abstract: str
    pdf_url: str


@tool
def find_publications(query: str, top_num: int = 3) -> list[_Paper]:
    """
    Searches the arXiv database for scientific preprints and papers.
    Returns list of top_num papers most relevant to given query.
    You MUST use the following arXiv-specific query syntax to find accurate results.
    
    1. Field Prefixes (Use these to narrow your search):
       - ti: Title (e.g., "ti:Attention")
       - au: Author (e.g., 'au:"Yann LeCun"')
       - abs: Abstract (e.g., "abs:transformer AND abs:vision")
       - cat: Category (e.g., cat:cs.AI for Artificial Intelligence, 
        cat:cs.CL for Comp Linguistics)

    2. Exact Matching:
       - Use double quotes for exact phrases: 'ti:"machine learning"'

    3. Boolean Operators (MUST BE UPPERCASE):
       - AND, OR, ANDNOT
       - Example: 'abs:"reinforcement learning" AND cat:cs.AI ANDNOT au:"Smith"'

    4. Grouping:
       - Use parentheses for complex queries: au:"Vaswani" AND (ti:"Attention" OR ti:"Transformer")

    If exploring a general topic without a specific paper in mind, you can use standard keywords 
    combined with the AND operator and a category filter for better relevance.
    """
    print(f"Searching arXiv for '{query}'...")

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=top_num,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    
    results = []
    try:
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "year": paper.published.year,
                "abstract": paper.summary.replace('\n', ' '),
                "pdf_url": paper.pdf_url
            })
        return results
    except Exception as e:
        print(f"arXiv search failed: {e}")
        return []
