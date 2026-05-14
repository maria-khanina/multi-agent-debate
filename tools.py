from typing import Literal, TypedDict

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool, BaseTool

from configurations.config_manager import config, get_retriever, \
                        get_chat_model


class Evidence(TypedDict):
    title: str
    content: str
    source: str


def create_evidence_tool(agent_role: str, debate_id: str) -> BaseTool:

    @tool
    def find_evidence(
        query: str,
        goal: str,
    ) -> list[Evidence]:
        """
        Search the private vector database for empirical evidence.
        
        This database uses semantic embeddings. Do NOT use isolated 
        keywords or boolean operators. Pass full sentences, specific 
        factual claims, or natural language questions.
        
        Args:
            query: A complete sentence or specific claim to 
                semantically match against the database text.
                (e.g., "Automation reduces the share of human labor.")
            goal: A brief explanation of what you are trying 
                to prove or refute with this search.

        Returns:
            A list of dictionaries containing scientific paper excerpts.
        """
        print(f"----Agent {agent_role} searchs for evidence: {query}----")
        retriever = get_retriever(agent_role, debate_id)
        retrieved_docs = retriever.invoke(query)

        evidence_list = []
        for doc in retrieved_docs:
            doc_title = doc.metadata.get("title", "Unknown")
            print(f"---Found evidence from paper: {doc_title}")
            evidence_list.append({
                "title": doc_title,
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
            })
            
        return evidence_list

    return find_evidence


def extract_llm_response(
        response: AIMessage,
) -> str:
    if isinstance(response.content, list):
        raw_query = ""
        for block in response.content:
            if block.get("type", None) == "text":
                raw_query = block.get("text", "")
                break
    else:
        raw_query = response.content
    return str(raw_query).strip()


class ConsistencyResult(TypedDict):
    is_consistent: bool
    contradiction_details: str


class AuditResult(TypedDict):
    verdict: Literal[
        "SUPPORTED", 
        "FABRICATED", 
        "EXAGGERATED", 
        "UNVERIFIABLE"
    ]
    reasoning: str


def _extract_agent_past_history(
    full_transcript: list[str],
    turn_number: int,
    agent_role: str,
) -> str:
    past_history = full_transcript[:turn_number]
    agent_history = [
        msg for msg in past_history 
        if f"{agent_role}:" in msg
    ]
    return "\n".join(agent_history)


def create_consistency_tool(
    full_transcript: list[str],
) -> BaseTool:
    @tool
    def check_logical_consistency(
        agent_role: Literal["PRO", "AGAINST"],
        turn_number: int,
        validated_claim: str,
    ) -> ConsistencyResult:
        """
        Evaluates a claim or a complete turn for logical contradictions
        against the agent's previous transcript.

        Args:
            agent_role: The debater to evaluate ("PRO" or "AGAINST").
            turn_number: The integer index of the turn being evaluated.
            validated_claim: A specific EXTRACTED FROM THAT TURN claim.
                If left empty, the function defaults to evaluating
                the entire turn.
        """
        history_str = _extract_agent_past_history(
            full_transcript, 
            turn_number, 
            agent_role,
        )

        print(
            f"----Judge validates consistency of '{validated_claim}'"
            f" of {agent_role} agent at {turn_number} turn.----"
        )
        
        if validated_claim == "":
            validated_claim = full_transcript[turn_number]

        llm = get_chat_model(
            "judge"
        ).with_structured_output(ConsistencyResult)
        
        prompt = config.get("prompts.judge_consistency").format(
            history_str=history_str,
            validated_claim=validated_claim,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"----Verdict: {response['is_consistent']}")
        print(f"----Explanation: {response['contradiction_details']}")

        return response

    return check_logical_consistency


def _extract_matching_chunks(
        retriever: BaseRetriever,
        source_title: str,
        claim: str,
) -> list[Document]:
    try:
        return retriever.vectorstore.similarity_search(
            query=claim, 
            k=config.get("models.judge.audit_chunk_limit"), 
            filter={"title": source_title}
        )
    except Exception as e:
        print(f"Database error during audit: {e}")
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
            claim: The EXACT factual assertion or statistic debater used.
            exact_source_title: The exact title of the referenced 
                academic paper. Must perfectly match a title from
                the available sources list.
        """
        print(
            f"---Judge factchecks '{claim}' of "
            f"{agent_role} based on {exact_source_title}"
        )
            
        retriever = get_retriever(agent_role, debate_id)
        matching_chunks = _extract_matching_chunks(
            retriever, 
            exact_source_title,
            claim,
        )

        if not matching_chunks:
            print("----Declared source wasn't found.----")
            return {
                "verdict": "UNVERIFIABLE",
                "reasoning": f"'{exact_source_title}' not found in agent's database."
            }
            
        source_text = "...\n...".join(
            [
                chunk.page_content for chunk in matching_chunks
            ]
        )

        llm = get_chat_model(
            "judge"
        ).with_structured_output(AuditResult)
        
        prompt = config.get("prompts.judge_factcheck").format(
            source_text=source_text,
            claim=claim,
        )

        response = llm.invoke([HumanMessage(content=prompt)])
        print(f"-----Verdict: {response['verdict']} -----")
        print(f"-----Explanation: {response['reasoning']} ----")

        return response
    return audit_cited_claim
