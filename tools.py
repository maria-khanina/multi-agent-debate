from configurations.config_manager import config, get_retriever
from entities import AuditResult, Evidence
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool
from language_calls import ask_structured


def extract_llm_response(response: BaseMessage) -> str:
    if isinstance(response.content, str):
        return response.content.strip()
    for block in response.content:
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", "")).strip()
    return ""


def _to_evidence(document: Document) -> Evidence:
    title = str(document.metadata.get("title", "Unknown"))
    print(f"[Tool] Found source: {title}")
    return {
        "title": title,
        "source": str(document.metadata.get("source", "Unknown")),
        "content": document.page_content,
    }


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
        return [_to_evidence(document) for document in retriever.invoke(query)]

    return find_evidence


def _extract_matching_chunks(
    retriever: ParentDocumentRetriever,
    source_title: str,
    claim: str,
) -> list[Document]:
    try:
        return retriever.vectorstore.similarity_search(
            query=claim,
            k=config.read_integer("models.judge.audit_chunk_limit"),
            filter={"title": source_title},
        )
    except Exception as error:
        print(f"[Database Error] Audit failed: {error}")
        return []


def _judge_against_source(claim: str, chunks: list[Document]) -> AuditResult:
    source_text = "...\n...".join([chunk.page_content for chunk in chunks])
    prompt = config.read_text("prompts.judge_factcheck").format(
        source_text=source_text,
        claim=claim,
    )
    response = ask_structured("judge", AuditResult, prompt)
    print(f"[Audit] Verdict: {response['verdict']}")
    return response


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
        chunks = _extract_matching_chunks(retriever, exact_source_title, claim)
        if not chunks:
            print(f"[Audit] Source '{exact_source_title}' not found.")
            return {
                "verdict": "UNVERIFIABLE",
                "reasoning": "Source not found in agent database.",
            }
        return _judge_against_source(claim, chunks)

    return audit_cited_claim
