import asyncio
from typing import Any, TypedDict

import aiohttp


class _Paper(TypedDict):
    title: str
    year: int
    citations: int
    affiliations: list[str]
    abstract: str
    pdf_url: str


def _extract_affiliations(
    authors: list[dict[str, Any]] | None
) -> list[str]:
    if authors is None:
        return []
    affiliations = []
    for author in authors:
        for affil in author.get('affiliations', []):
            if affil not in affiliations:
                affiliations.append(affil)
                
    return affiliations


def format_paper(
    raw_paper: dict[str, Any],
) -> _Paper | None:
    if not raw_paper.get('abstract'):
        return None
    if not raw_paper.get('openAccessPdf'):
        return None

    return {
        "title": raw_paper.get('title', ""),
        "year": raw_paper.get('year'),
        "citations": raw_paper.get('citationCount', 0),
        "affiliations": _extract_affiliations(
            raw_paper.get('authors', []),
        ),
        "abstract": raw_paper['abstract'],
        "pdf_url": raw_paper['openAccessPdf'].get('url')
    }


def _gather_formatted_results(
    raw_papers: list[dict[str, Any]],
) -> list[_Paper]:
    results = []
    for paper in raw_papers:
        formatted = format_paper(paper)
        if formatted is not None:
            results.append(formatted)
            
    return results


async def _get_matching_papers(
    query: str,
    top_num: int,
    session: aiohttp.ClientSession,
) -> list[dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": top_num,
        "fields": (
            "title,abstract,authors.affiliations,year,"
            "citationCount,openAccessPdf"
        )
    }

    async with session.get(url, params=params, timeout=10) as response:
        response.raise_for_status()
        data = await response.json()
        return data.get('data', [])


async def find_publications(
    query: str,
    top_num=3,
) -> list[_Paper]:
    async with aiohttp.ClientSession() as session:
        try:
            raw_papers = await _get_matching_papers(
                query,
                top_num, 
                session,
            )
            return _gather_formatted_results(raw_papers)
        except Exception as e:
            print(f"Search for {query} has failed: {e}")
            return []


if __name__ == "__main__":
    asyncio.run(find_publications("Transformers"))
