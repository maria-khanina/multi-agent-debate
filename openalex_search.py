import requests
from typing import Any, TypedDict


class RawPaper(TypedDict):
    title: str
    year: int | None
    citations: int
    affiliations: list[str]
    abstract: str
    pdf_url: str | None


def _find_max_index(
    inverted_index: dict[str, list[int]],
) -> int:
    max_index = 0
    for positions in inverted_index.values():
        for pos in positions:
            if pos > max_index:
                max_index = pos
    return max_index


def revert_index(
    inverted_index: dict[str, list[int]],
    max_index: int,
):
    words = [""] * (max_index + 1)
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return words


def _reconstruct_abstract(
    inverted_index: dict[str, list[int]] | None,
) -> str:
    if inverted_index is None:
        return "No abstract available."

    max_index = _find_max_index(inverted_index)
    words = revert_index(inverted_index, max_index)

    return " ".join(words)


def _extract_affiliations(
    authorships: list[dict[str, Any]] | None,
) -> list[str]:
    if authorships is None:
        authorships = []

    affiliations = []
    for authorship in authorships:
        institutions = authorship.get('institutions', [])
        for inst in institutions:
            name = inst.get('display_name')
            if name and name not in affiliations:
                affiliations.append(name)

    return affiliations


def _wrap_paper(
        paper: dict[str, Any],
        results: list[RawPaper],
) -> None:
    if paper.get('open_access') is None:
        return None
    if paper['open_access'].get('oa_url') is None:
        return None
    
    pdf_url = paper['open_access']['oa_url']

    affiliations = _extract_affiliations(
        paper.get('authorships', [])
    )
    abstract = _reconstruct_abstract(
        paper.get('abstract_inverted_index')
    )

    results.append({
        "title": paper.get('title', ""),
        "year": paper.get('publication_year'),
        "citations": paper.get('cited_by_count', 0),
        "affiliations": affiliations,
        "abstract": abstract,
        "pdf_url": pdf_url,
    })


def search_papers(query: str, top_num=3) -> list[RawPaper]:
    """
    Search the OpenAlex database for scientific papers and metadata.
    Provide standard keywords as the query. Use double quotes for 
    exact phrases (e.g., "large language models"). 
    The tool returns metadata including citation counts and 
    institutional affiliations to aid in evaluating paper validity.
    """
    try:
        url = "https://api.openalex.org/works"
        params = {
            "search": query,
            "per-page": top_num,
            "sort": "relevance_score:desc",
            "mailto": "maria.khanina.ola@gmail.com"
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        for paper in data.get('results', []):
            _wrap_paper(paper, results)
        return results
    except Exception as e:
        print(f"OpenAlex search failed: {e}")
        return []
