import json
from typing import Any, TypedDict
import uuid

from configurations.config_manager import get_chat_model, get_retriever
from debate_settings import DEBATE_ID_CHARACTERS, LOG_INDENT, RAG_PAPER_COUNT
from langchain_core.messages import HumanMessage
from language_calls import ask_structured
from openalex_search import search_papers
from research import process_and_store_paper
from tools import extract_llm_response


class RagLog(TypedDict):
    goal: str
    query: str
    chunks: list[str]
    output: str


class ChunkScore(TypedDict):
    score: float


class AtomicClaims(TypedDict):
    claims: list[str]


class ClaimSupport(TypedDict):
    is_supported: bool


class ChunkUtilization(TypedDict):
    is_utilized: bool


class ChunkTotals(TypedDict):
    chunks: int
    relevance: float
    utilized: int


class ClaimTotals(TypedDict):
    claims: int
    supported: int


def evaluate_chunk_relevance(motion: str, stance: str, goal: str, chunk: str) -> float:
    prompt_text = (
        f"Debator participates in debate with motion {motion} and stance: {stance}\n"
        f"Right now debator has a goal: {goal}\n"
        "This is some text debeator may use or not use for their argument:\n"
        f"Text:\n{chunk}\n\n"
        "Score relevance based on these strict criteria:\n"
        "- 1.0: The chunk comprehensively provides the empirical data, facts, "
        "or conclusions necessary to fully satisfy the agent's search goal.\n"
        "- 0.5: The chunk is on-topic and tangentially related, but only "
        "partially answers the goal or lacks definitive empirical data.\n"
        "- 0.0: The chunk is completely irrelevant to the goal, or it "
        "provides information that supports the opposing stance."
    )
    response = ask_structured("judge", ChunkScore, prompt_text)
    return float(response.get("score", 0.0))


def evaluate_chunk_utilization(output_text: str, chunk_text: str) -> bool:
    prompt_text = (
        f"Output:\n{output_text}\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        "Does the output explicitly use empirical information from "
        "this specific chunk?"
    )
    response = ask_structured("judge", ChunkUtilization, prompt_text)
    return response.get("is_utilized", False)


def extract_claims(text: str) -> list[str]:
    prompt_text = (
        "Extract atomic, empirically verifiable claims from:\n"
        f"{text}\n\n"
        "Ignore rhetoric and transition words."
    )
    return ask_structured("judge", AtomicClaims, prompt_text).get("claims", [])


def verify_claim(claim: str, chunks: list[str]) -> bool:
    context_text = "\n".join(chunks)
    prompt_text = (
        f"Context:\n{context_text}\n\n"
        f"Claim: {claim}\n\n"
        "Is this claim strictly backed by the context without "
        "any exaggeration?"
    )
    return ask_structured("judge", ClaimSupport, prompt_text).get("is_supported", False)


def generate_rag_output(goal: str, chunks: list[str]) -> str:
    context_text = "\n".join(chunks)
    prompt_text = (
        f"Goal: {goal}\n\n"
        f"Context:\n{context_text}\n\n"
        "Write a short, dense argument achieving the goal using "
        "the provided Context."
    )
    response = get_chat_model("debater").invoke([HumanMessage(content=prompt_text)])
    return extract_llm_response(response)


def populate_debate_database(subject: str, agent_role: str, debate_id: str) -> None:
    for paper in search_papers(subject, RAG_PAPER_COUNT):
        process_and_store_paper(paper, agent_role, debate_id)


def generate_evaluation_logs(
    queries_and_goals: list[dict[str, str]],
    agent_role: str,
    debate_id: str,
) -> list[RagLog]:
    retriever = get_retriever(agent_role, debate_id)
    evaluation_logs: list[RagLog] = []
    for item in queries_and_goals:
        documents = retriever.invoke(item["query"])
        retrieved_chunks: list[str] = [document.page_content for document in documents]
        evaluation_logs.append({
            "goal": item["goal"],
            "query": item["query"],
            "chunks": retrieved_chunks,
            "output": generate_rag_output(item["goal"], retrieved_chunks),
        })
    return evaluation_logs


def _score_chunks(log: RagLog, motion: str, stance: str, totals: ChunkTotals) -> None:
    for chunk in log["chunks"]:
        totals["relevance"] += evaluate_chunk_relevance(
            motion,
            stance,
            log["goal"],
            chunk,
        )
        if evaluate_chunk_utilization(log["output"], chunk):
            totals["utilized"] += 1
        totals["chunks"] += 1


def _score_claims(log: RagLog, totals: ClaimTotals) -> None:
    for claim in extract_claims(log["output"]):
        if verify_claim(claim, log["chunks"]):
            totals["supported"] += 1
        totals["claims"] += 1


def _divide(part: float, total: int) -> float:
    return part / total if total > 0 else 0.0


def calculate_rag_metrics(
    evaluation_logs: list[RagLog],
    motion: str,
    stance: str,
) -> dict[str, float]:
    chunk_totals: ChunkTotals = {"chunks": 0, "relevance": 0.0, "utilized": 0}
    claim_totals: ClaimTotals = {"claims": 0, "supported": 0}
    for log in evaluation_logs:
        _score_chunks(log, motion, stance, chunk_totals)
        _score_claims(log, claim_totals)
    return {
        "relevance_precision": _divide(
            chunk_totals["relevance"],
            chunk_totals["chunks"],
        ),
        "context_utilization": _divide(
            chunk_totals["utilized"],
            chunk_totals["chunks"],
        ),
        "faithfulness_ratio": _divide(
            claim_totals["supported"],
            claim_totals["claims"],
        ),
    }


def evaluate_debate_rag(config_file_path: str) -> dict[str, float]:
    with open(config_file_path, "r", encoding="utf-8") as file_stream:
        config_data: dict[str, Any] = json.load(file_stream)
    debate_id = str(uuid.uuid4())[:DEBATE_ID_CHARACTERS]
    agent_role = "EVAL"
    populate_debate_database(config_data["subject"], agent_role, debate_id)
    evaluation_logs = generate_evaluation_logs(
        config_data["queries_and_goals"],
        agent_role,
        debate_id,
    )
    with open("rag_eval_logs.json", "w", encoding="utf-8") as log_file:
        json.dump(evaluation_logs, log_file, indent=LOG_INDENT)
    return calculate_rag_metrics(
        evaluation_logs,
        config_data["motion"],
        config_data["stance"],
    )


if __name__ == "__main__":
    print(evaluate_debate_rag("rag_evaluation_sample.json"))
