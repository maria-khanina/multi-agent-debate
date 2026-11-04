import json
import uuid

from configurations.config_manager import get_retriever
from debate_settings import CHILD_PREVIEW_CHARACTERS, PARENT_PREVIEW_CHARACTERS
from entities import SciencePaper
from langchain_core.documents import Document
from research import process_and_store_paper

SAMPLE_PAPER_PATH = "tests/sample_paper.json"
AGENT_ROLE = "chunk_tester"
QUERY = "self-attention mechanism"
SHOWN_CHUNKS = 2


def _load_sample_paper() -> SciencePaper:
    with open(SAMPLE_PAPER_PATH, "r", encoding="utf-8") as file_stream:
        return json.load(file_stream)


def _print_chunks(label: str, documents: list[Document], preview: int) -> None:
    print(f"\n--- {label} ---")
    for position, document in enumerate(documents[:SHOWN_CHUNKS], start=1):
        print(f"Chunk {position} | Length: {len(document.page_content)} chars")
        print(f"{document.page_content[:preview]}...\n")


def run_chunking_test() -> None:
    debate_id = uuid.uuid4().hex
    paper = _load_sample_paper()
    process_and_store_paper(paper, AGENT_ROLE, debate_id)
    print(f"Paper {paper['title']} is downloaded")
    retriever = get_retriever(AGENT_ROLE, debate_id)
    print(f"\n--- Searching for: '{QUERY}' ---")
    children = retriever.vectorstore.similarity_search(QUERY, k=SHOWN_CHUNKS)
    _print_chunks("CHILD CHUNKS (From ChromaDB)", children, CHILD_PREVIEW_CHARACTERS)
    parents = retriever.invoke(QUERY)
    _print_chunks("PARENT CHUNKS (From Redis)", parents, PARENT_PREVIEW_CHARACTERS)


if __name__ == "__main__":
    run_chunking_test()
