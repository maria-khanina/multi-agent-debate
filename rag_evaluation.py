import json
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage

from configurations.config_manager import get_chat_model, get_retriever
from openalex_search import search_papers
from research import _process_and_store_paper
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


def evaluate_chunk_relevance(
    motion: str,
    stance: str,
    goal: str,
    chunk: str,
) -> float:
    language_model = get_chat_model("judge").with_structured_output(ChunkScore)
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
    response = language_model.invoke([HumanMessage(content=prompt_text)])
    return float(response.get("score", 0.0))


def evaluate_chunk_utilization(
    output_text: str,
    chunk_text: str,
) -> bool:
    expected_output = ChunkUtilization
    language_model = get_chat_model("judge").with_structured_output(
        expected_output
    )
    prompt_text = (
        f"Output:\n{output_text}\n\n"
        f"Chunk:\n{chunk_text}\n\n"
        "Does the output explicitly use empirical information from "
        "this specific chunk?"
    )
    response = language_model.invoke([HumanMessage(content=prompt_text)])
    return response.get("is_utilized", False)


def extract_claims(text: str) -> list[str]:
    language_model = get_chat_model("judge").with_structured_output(
        AtomicClaims
    )
    prompt_text = (
        "Extract atomic, empirically verifiable claims from:\n"
        f"{text}\n\n"
        "Ignore rhetoric and transition words."
    )
    response = language_model.invoke([HumanMessage(content=prompt_text)])
    return response.get("claims", [])


def verify_claim(claim: str, chunks: list[str]) -> bool:
    language_model = get_chat_model("judge").with_structured_output(
        ClaimSupport
    )
    context_text = "\n".join(chunks)
    prompt_text = (
        f"Context:\n{context_text}\n\n"
        f"Claim: {claim}\n\n"
        "Is this claim strictly backed by the context without "
        "any exaggeration?"
    )
    response = language_model.invoke([HumanMessage(content=prompt_text)])
    return response.get("is_supported", False)


def generate_rag_output(goal: str, chunks: list[str]) -> str:
    language_model = get_chat_model("debater")
    context_text = "\n".join(chunks)
    prompt_text = (
        f"Goal: {goal}\n\n"
        f"Context:\n{context_text}\n\n"
        "Write a short, dense argument achieving the goal using "
        "the provided Context."
    )
    response = language_model.invoke([HumanMessage(content=prompt_text)])
    return extract_llm_response(response)


def populate_debate_database(
    subject: str,
    agent_role: str,
    debate_id: str,
) -> None:
    papers = search_papers(subject, 10)
    for paper in papers:
        _process_and_store_paper(paper, agent_role, debate_id)


def generate_evaluation_logs(
    queries_and_goals: list[dict[str, str]],
    agent_role: str,
    debate_id: str,
) -> list[RagLog]:
    retriever = get_retriever(agent_role, debate_id)
    evaluation_logs = []

    for item in queries_and_goals:
        documents = retriever.invoke(item["query"])
        
        retrieved_chunks = []
        for document in documents:
            retrieved_chunks.append(document.page_content)
            
        generated_output = generate_rag_output(
            item["goal"], 
            retrieved_chunks
        )
        
        log_entry: RagLog = {
            "goal": item["goal"],
            "query": item["query"],
            "chunks": retrieved_chunks,
            "output": generated_output
        }
        evaluation_logs.append(log_entry)

    return evaluation_logs


def calculate_rag_metrics(
    evaluation_logs: list[RagLog],
    motion: str,
    stance: str,
) -> dict[str, float]:
    total_chunks = 0
    total_relevance_score = 0.0
    utilized_chunks = 0
    
    total_claims = 0
    supported_claims = 0

    for log in evaluation_logs:
        output_text = log["output"]
        
        for chunk in log["chunks"]:
            relevance_score = evaluate_chunk_relevance(
                motion, 
                stance, 
                log["goal"], 
                chunk
            )
            total_relevance_score += relevance_score
            
            is_utilized = evaluate_chunk_utilization(output_text, chunk)
            if is_utilized:
                utilized_chunks += 1
                
            total_chunks += 1

        extracted_claims = extract_claims(output_text)
        for claim in extracted_claims:
            is_supported = verify_claim(claim, log["chunks"])
            if is_supported:
                supported_claims += 1
            total_claims += 1

    relevance_precision = 0.0
    context_utilization = 0.0
    faithfulness_ratio = 0.0

    if total_chunks > 0:
        relevance_precision = total_relevance_score / total_chunks
        context_utilization = utilized_chunks / total_chunks
        
    if total_claims > 0:
        faithfulness_ratio = supported_claims / total_claims

    return {
        "relevance_precision": relevance_precision,
        "context_utilization": context_utilization,
        "faithfulness_ratio": faithfulness_ratio,
    }


def evaluate_debate_rag(config_file_path: str) -> dict[str, float]:
    with open(config_file_path, "r", encoding="utf-8") as file_stream:
        config_data = json.load(file_stream)

    debate_subject = config_data["subject"]
    debate_motion = config_data["motion"]
    stance = config_data["stance"]
    queries_and_goals = config_data["queries_and_goals"]

    debate_id = str(uuid.uuid4())[:8]
    agent_role = "EVAL"
    
    populate_debate_database(
        debate_subject, 
        agent_role, 
        debate_id
    )

    evaluation_logs = generate_evaluation_logs(
        queries_and_goals,
        agent_role,
        debate_id
    )

    with open("rag_eval_logs.json", "w", encoding="utf-8") as log_file:
        json.dump(evaluation_logs, log_file, indent=4)

    metrics_report = calculate_rag_metrics(
        evaluation_logs, 
        debate_motion, 
        stance
    )

    return metrics_report


if __name__ == "__main__":
    print(evaluate_debate_rag("rag_evaluation_sample.json"))
