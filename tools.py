from typing import Any, NotRequired, TypedDict

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_core.tools import tool, BaseTool

from research import get_embedding


MetadataValue = str | int | float | bool


class ChromaQueryResult(TypedDict):
    ids: list[list[str]]
    distances: NotRequired[list[list[float]] | None]
    metadatas: NotRequired[list[list[dict[str, MetadataValue]]] | None]
    embeddings: NotRequired[list[list[list[float]]] | None]
    documents: NotRequired[list[list[str]] | None]
    uris: NotRequired[list[list[str]] | None]
    data: NotRequired[list[list[str]] | None]


class Evidence(TypedDict):
    title: str
    content: str


def _get_agent_collection(agent_id: str) -> Collection:
    client = chromadb.PersistentClient(
        path="./debate_vectors"
    )
    return client.get_or_create_collection(
        name=f"agent_{agent_id}"
    )


def _format_evidence(matches: ChromaQueryResult) -> list[Evidence]:
    if matches is None:
        return []

    evidence = []
    contents = matches.get("documents", [[]])
    details = matches.get("metadatas", [[]])

    if contents and contents[0]:
        for i in range(len(contents[0])):
            evidence.append({
                "title": details[0][i].get("title", ""),
                "content": contents[0][i],
            })
    print(evidence)
            
    return evidence


def create_evidence_tool(agent_id: str) -> BaseTool:

    @tool
    def find_evidence(
        key_words: str,
        chunks_num=3,
    ) -> list[Evidence]:
        """
        Search the private database for peer-reviewed paper abstracts.
        
        Use this tool when you need empirical evidence, specific facts, or
        methodological details to support your stance or refute an opponent.
        
        Args:
            key_words: A string of descriptive terms or a short phrase 
                representing the search intent.
            chunks_num: The number of relevant abstracts to retrieve. 
                Default is 3.
                
        Returns:
            A list of dictionaries, each containing the 'title' and 
            'abstract' of a relevant scientific paper.
        """
        print(f"-------Agent {agent_id} looks for evidence: {key_words}")
        collection = _get_agent_collection(agent_id)
        query_vector = get_embedding(key_words)

        raw_matches = collection.query(
            query_embeddings=[query_vector],
            n_results=chunks_num,
        )

        return _format_evidence(raw_matches)

    return find_evidence
