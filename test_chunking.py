import pprint

from configurations.config_manager import get_retriever
from openalex_search import RawPaper
from research import extract_text_from_pdf, _process_and_store_paper


def run_chunking_test() -> None:
    test_agent = "chunk_tester"
    
    sample_paper: RawPaper = {
        "title": "Attention Is All You Need",
        "year": 2017,
        "citations": 246_307,
        "affiliations": ["Google"],
        "abstract": "Transformers transformers transformers",
        "pdf_url": "https://arxiv.org/pdf/1706.03762.pdf"
    }

    _process_and_store_paper(sample_paper, test_agent)
    print(f"Paper {sample_paper['title']} is downloaded")
    
    retriever = get_retriever(test_agent)
    query = "self-attention mechanism"
    
    print(f"\n--- Searching for: '{query}' ---")
    
    child_docs = retriever.vectorstore.similarity_search(query, k=2)
    print("\n--- CHILD CHUNKS (From ChromaDB) ---")
    for idx, doc in enumerate(child_docs):
        length = len(doc.page_content)
        print(f"Child {idx + 1} | Length: {length} chars")
        print(f"{doc.page_content[:200]}...\n")
        
    parent_docs = retriever.invoke(query)
    print("\n--- PARENT CHUNKS (From Redis) ---")
    for idx, doc in enumerate(parent_docs[:2]):
        length = len(doc.page_content)
        print(f"Parent {idx + 1} | Length: {length} chars")
        print(f"{doc.page_content[:300]}...\n")


if __name__ == "__main__":
    run_chunking_test()
