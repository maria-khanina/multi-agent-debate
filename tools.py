from typing import TypedDict

from langchain_core.tools import tool, BaseTool

from configurations.config_manager import get_retriever


class Evidence(TypedDict):
    title: str
    content: str
    source: str


def create_evidence_tool(agent_id: str) -> BaseTool:

    @tool
    def find_evidence(key_words: str) -> list[Evidence]:
        """
        Search the private database for peer-reviewed paper abstracts.
        
        Use this tool when you need empirical evidence, specific facts, or
        methodological details to support your stance or refute an opponent.
        
        Args:
            key_words: A string of descriptive terms or a short phrase 
                representing the search intent.

        Returns:
            A list of dictionaries, each containing the 'title' and 
            'abstract' of a relevant scientific paper.
        """
        retriever = get_retriever(agent_id)
        retrieved_docs = retriever.invoke(key_words)

        evidence_list = []
        for doc in retrieved_docs:
            evidence_list.append({
                "title": doc.metadata.get("title", "Unknown"),
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
            })
            
        return evidence_list

    return find_evidence
