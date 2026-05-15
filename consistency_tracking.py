import json
import time
from typing import Callable, TypeVar

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from configurations.config_manager import config, get_chat_model, \
    get_embeddings_model
from entities import AtomizedTurn, BeliefComparisonResult, \
    CleanedContradictions, Contradiction, ExtractedBelief


T = TypeVar('T')


def execute_with_backoff(
    api_call: Callable[[], T],
    max_retries=5,
    base_delay=2,
) -> T:
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as error_instance:
            print(f"[Backoff] API call failed, attempt {attempt + 1}")
            if attempt == max_retries - 1:
                raise error_instance
            time.sleep(base_delay * (2 ** attempt))


class BeliefsStore:
    def __init__(self, debate_id: str, agent_role: str) -> None:
        embedder = get_embeddings_model()
        db_path = config.get("database.path", "./debate_vectors")
        col_name = f"tracker_{debate_id}_{agent_role}".lower()
        
        self.vectorstore = Chroma(
            collection_name=col_name,
            embedding_function=embedder,
            persist_directory=db_path,
        )

    def upsert_belief(self, topic_key: str, stance: str) -> None:
        new_document = Document(
            page_content=topic_key,
            metadata={"stance": stance},
        )
        self.vectorstore.add_documents(
            documents=[new_document],
            ids=[topic_key],
        )

    def find_similar_topics(
        self,
        query_topic: str,
        top_k=4,
    ) -> list[ExtractedBelief]:
        search_results = self.vectorstore.similarity_search(
            query=query_topic,
            k=top_k,
        )
        extracted_results = []
        for document in search_results:
            extracted_results.append({
                "topic_key": document.page_content,
                "stance": document.metadata.get("stance", ""),
            })
        return extracted_results


_BELIEFS_STORE = {}


def get_memory_store(
    debate_id: str,
    agent_role: str,
) -> BeliefsStore:
    key = (debate_id, agent_role)
    if key not in _BELIEFS_STORE:
        _BELIEFS_STORE[key] = BeliefsStore(debate_id, agent_role)
    return _BELIEFS_STORE[key]


def extract_beliefs_from_turn(
    latest_turn: str,
    motion: str,
    agent_role: str,
) -> AtomizedTurn:
    language_model = get_chat_model("judge").with_structured_output(
        AtomizedTurn
    )
    prompt_template = config.get("prompts.consistency_extraction")
    prompt_text = prompt_template.format(
        motion=motion,
        agent_role=agent_role,
        latest_turn=latest_turn,
    )
    
    def _invoke_llm() -> AtomizedTurn:
        return language_model.invoke([HumanMessage(content=prompt_text)])
        
    return execute_with_backoff(_invoke_llm)


def evaluate_belief_relationship(
    new_belief: ExtractedBelief,
    historical_belief: ExtractedBelief,
) -> BeliefComparisonResult:
    language_model = get_chat_model("judge").with_structured_output(
        BeliefComparisonResult
    )
    prompt_template = config.get("prompts.consistency_evaluation")
    prompt_text = prompt_template.format(
        old_topic=historical_belief["topic_key"],
        old_stance=historical_belief["stance"],
        new_topic=new_belief["topic_key"],
        new_stance=new_belief["stance"],
    )
    
    def _invoke_llm() -> BeliefComparisonResult:
        return language_model.invoke([HumanMessage(content=prompt_text)])
        
    return execute_with_backoff(_invoke_llm)


def process_belief(
    new_belief: ExtractedBelief,
    memory_store: BeliefsStore,
    contradictions_list: list[Contradiction],
) -> None:
    topic = new_belief["topic_key"]
    print(f"[Tracker] Evaluating belief topic: '{topic}'")
    
    similar_beliefs = memory_store.find_similar_topics(topic)
    
    if not similar_beliefs:
        print(f"[Tracker] No similar topics. Inserting: '{topic}'")
        memory_store.upsert_belief(topic, new_belief["stance"])
        return

    found_redundancy_or_contradiction = False

    for historical_belief in similar_beliefs:
        evaluation = evaluate_belief_relationship(
            new_belief,
            historical_belief,
        )
        category = evaluation["category"]

        if category == "DIFFERENT_TOPIC":
            continue

        found_redundancy_or_contradiction = True
        
        print(
            f"[Tracker] Compared to '{historical_belief['topic_key']}' "
            f"-> {category}"
        )

        if category == "CONTRADICT":
            print(f"[Tracker] !!! Contradiction logged for: '{topic}'")
            contradictions_list.append({
                "topic_key": historical_belief["topic_key"],
                "old_stance": historical_belief["stance"],
                "new_stance": new_belief["stance"],
                "explanation": evaluation.get("explanation", ""),
            })
            memory_store.upsert_belief(
                historical_belief["topic_key"],
                new_belief["stance"],
            )
            break

        if category == "REDUNDANT":
            break

    if not found_redundancy_or_contradiction:
        print(f"[Tracker] Topic differs from matches. Inserting '{topic}'")
        memory_store.upsert_belief(topic, new_belief["stance"])


def process_debate_turn(
    turn_text: str,
    motion: str,
    agent_role: str,
    memory_store: BeliefsStore,
    contradictions_list: list[Contradiction],
) -> None:
    print(f"\n--- Checking consistency for {agent_role} ---")
    atomized_result = extract_beliefs_from_turn(
        turn_text,
        motion,
        agent_role,
    )
    
    internal_contras = atomized_result.get("internal_contradictions", [])
    extracted_beliefs = atomized_result.get("extracted_beliefs", [])
    
    print(f"[Tracker] Extracted {len(extracted_beliefs)} beliefs.")
    if internal_contras:
        print(f"[Tracker] Found {len(internal_contras)} internal contras!")
        
    for internal_contra in internal_contras:
        contradictions_list.append(internal_contra)
    
    for belief in extracted_beliefs:
        process_belief(
            belief,
            memory_store,
            contradictions_list,
        )


def deduplicate_contradictions(
    contradictions_list: list[Contradiction],
) -> list[Contradiction]:
    if len(contradictions_list) <= 1:
        return contradictions_list

    language_model = get_chat_model("judge").with_structured_output(
        CleanedContradictions
    )
    json_string = json.dumps(contradictions_list, indent=2)
    prompt_template = config.get("prompts.consistency_deduplication")
    prompt_text = prompt_template.format(contradictions_json=json_string)
    
    def _invoke_llm() -> CleanedContradictions:
        return language_model.invoke([HumanMessage(content=prompt_text)])
        
    response = execute_with_backoff(_invoke_llm)
    cleaned = response.get("unique_contradictions", [])
    
    print(
        f"[Deduplication] Reduced {len(contradictions_list)} "
        f"contradictions to {len(cleaned)}"
    )
    
    return cleaned
