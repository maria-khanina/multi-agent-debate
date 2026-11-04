import os
from typing import Any

from debate_settings import OPENALEX_URL, REQUEST_TIMEOUT_SECONDS
from entities import SciencePaper
import requests


def _find_max_index(inverted_index: dict[str, list[int]]) -> int:
    max_index = 0
    for positions in inverted_index.values():
        max_index = max([max_index, *positions])
    return max_index


def revert_index(
    inverted_index: dict[str, list[int]],
    max_index: int,
) -> list[str]:
    words = [""] * (max_index + 1)
    for word, positions in inverted_index.items():
        for position in positions:
            words[position] = word
    return words


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if inverted_index is None:
        return "No abstract available."
    max_index = _find_max_index(inverted_index)
    return " ".join(revert_index(inverted_index, max_index))


def _extract_affiliations(authorships: list[dict[str, Any]] | None) -> list[str]:
    affiliations: list[str] = []
    for authorship in authorships or []:
        for institution in authorship.get("institutions", []):
            name = institution.get("display_name")
            if name and name not in affiliations:
                affiliations.append(name)
    return affiliations


def _wrap_paper(paper: dict[str, Any], results: list[SciencePaper]) -> None:
    pdf_url = paper.get("open_access", {}).get("oa_url")
    if pdf_url is None:
        return
    results.append({
        "title": paper.get("title", ""),
        "year": paper.get("publication_year"),
        "citations": paper.get("cited_by_count", 0),
        "affiliations": _extract_affiliations(paper.get("authorships", [])),
        "abstract": _reconstruct_abstract(paper.get("abstract_inverted_index")),
        "pdf_url": pdf_url,
    })


def _request_papers(query: str, paper_count: int) -> list[dict[str, Any]]:
    parameters: dict[str, str | int] = {
        "search": query,
        "per-page": paper_count,
        "sort": "relevance_score:desc",
        "mailto": os.getenv("OPENALEX_EMAIL", ""),
    }
    response = requests.get(
        OPENALEX_URL,
        params=parameters,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _collect_papers(query: str, paper_count: int) -> list[SciencePaper]:
    results: list[SciencePaper] = []
    for paper in _request_papers(query, paper_count):
        _wrap_paper(paper, results)
    return results


def search_papers(query: str, paper_count: int) -> list[SciencePaper]:
    print(f"\n[OpenAlex] Searching for: '{query}'")
    try:
        results = _collect_papers(query, paper_count)
    except Exception as error:
        print(f"[OpenAlex Error] Search failed: {error}")
        return []
    print(f"[OpenAlex] Found {len(results)} valid papers.")
    return results
