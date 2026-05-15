from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool, tool

from configurations.config_manager import config, get_chat_model, \
    get_retriever
from entities import AuditResult, Evidence


def extract_llm_response(response: AIMessage) -> str:
    if isinstance(response.content, list):
        raw_query = ""
        for block in response.content:
            if block.get("type") == "text":
                raw_query = block.get("text", "")
                break
    else:
        raw_query = response.content
    return str(raw_query).strip()


def create_evidence_tool(agent_role: str, debate_id: str) -> BaseTool:
    
    @tool
    def find_evidence(query: str, goal: str) -> list[Evidence]:
        """
        Search the private vector database for empirical evidence.
        
        This database uses semantic embeddings. Do NOT use isolated 
        keywords or boolean operators. Pass full sentences, specific 
        factual claims, or natural language questions.
        
        Args:
            query: A complete sentence or specific claim to 
                semantically match against the database text.
            goal: A brief explanation of what you are trying 
                to prove or refute with this search.
        """
        print(f"\n[Tool] {agent_role} searching evidence: '{query}'")
        retriever = get_retriever(agent_role, debate_id)
        retrieved_docs = retriever.invoke(query)

        evidence_list = []
        for doc in retrieved_docs:
            doc_title = doc.metadata.get("title", "Unknown")
            print(f"[Tool] Found source: {doc_title}")
            evidence_list.append({
                "title": doc_title,
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
            })
            
        return evidence_list

    return find_evidence


def _extract_matching_chunks(
    retriever: BaseRetriever,
    source_title: str,
    claim: str,
) -> list[Document]:
    try:
        return retriever.vectorstore.similarity_search(
            query=claim, 
            k=config.get("models.judge.audit_chunk_limit"), 
            filter={"title": source_title},
        )
    except Exception as e:
        print(f"[Database Error] Audit failed: {e}")
        return []


def create_audit_tool(debate_id: str) -> BaseTool:
    
    @tool
    def audit_cited_claim(
        agent_role: str,
        claim: str,
        exact_source_title: str,
    ) -> AuditResult:
        """
        Verifies a debater's claim against the raw text of a cited 
        source document.

        Args:
            agent_role: The debater who made the claim.
            claim: The EXACT, COMPLETE factual assertion or statistic used.
            exact_source_title: The exact title of the referenced 
                academic paper. Must perfectly match a title from
                the available sources list.
        """
        print(
            f"\n[Audit] Checking {agent_role} claim: '{claim}' "
            f"against '{exact_source_title}'"
        )
            
        retriever = get_retriever(agent_role, debate_id)
        matching_chunks = _extract_matching_chunks(
            retriever, 
            exact_source_title,
            claim,
        )

        if not matching_chunks:
            print(f"[Audit] Source '{exact_source_title}' not found.")
            return {
                "verdict": "UNVERIFIABLE",
                "reasoning": "Source not found in agent database.",
            }
            
        source_text = "...\n...".join(
            [chunk.page_content for chunk in matching_chunks]
        )

        llm = get_chat_model("judge").with_structured_output(AuditResult)
        
        prompt = config.get("prompts.judge_factcheck").format(
            source_text=source_text,
            claim=claim,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"[Audit] Verdict: {response['verdict']}")
        
        return response
        
    return audit_cited_claim
