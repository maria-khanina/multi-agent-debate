import json
import os
import time
import uuid
from typing import Callable, Literal, TypedDict, TypeVar

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from configurations.config_manager import config
from configurations.config_manager import get_chat_model
from configurations.config_manager import get_embeddings_model


ComparisonResult = Literal[
    "CONTRADICT",
    "REDUNDANT",
    "DIFFERENT_TOPIC"
]


class Contradiction(TypedDict):
    topic_key: str
    old_stance: str
    new_stance: str
    explanation: str


class ExtractedBelief(TypedDict):
    topic_key: str
    stance: str


class AtomizedTurn(TypedDict):
    extracted_beliefs: list[ExtractedBelief]
    internal_contradictions: list[Contradiction]


class TurnContradictionResult(TypedDict):
    category: ComparisonResult
    explanation: str | None


class TrackerOutput(TypedDict):
    final_beliefs: dict[str, str]
    contradictions: list[Contradiction]


class DebateResult(TypedDict):
    debate_motion: str
    pro_tracker: TrackerOutput
    against_tracker: TrackerOutput


class DebateEntry(TypedDict):
    motion: str
    turns: list[str]


T = TypeVar('T')


def execute_with_backoff(
    api_call: Callable[[], T],
    max_retries: int = 5,
    base_delay: int = 2
) -> T:
    for attempt in range(max_retries):
        try:
            return api_call()
        except Exception as error_instance:
            print(f"API call failed (attempt {attempt + 1}). Retrying...")
            if attempt == max_retries - 1:
                raise error_instance
            time.sleep(base_delay * (2 ** attempt))


class VectorSemanticMemoryStore:
    def __init__(self, debate_id: str, agent_role: str) -> None:
        embedder = get_embeddings_model()
        db_path = config.get("database.path", "./debate_vectors")
        
        safe_id = "".join(c for c in debate_id if c.isalnum())
        safe_role = "".join(c for c in agent_role if c.isalnum())
        col_name = f"contradictions_{safe_id}_{safe_role}"
        
        self.vectorstore = Chroma(
            collection_name=col_name,
            embedding_function=embedder,
            persist_directory=db_path
        )

    def upsert_belief(self, topic_key: str, stance: str) -> None:
        new_document = Document(
            page_content=topic_key,
            metadata={"stance": stance}
        )
        self.vectorstore.add_documents(
            documents=[new_document],
            ids=[topic_key]
        )

    def get_all_records(self) -> dict[str, str]:
        all_documents = self.vectorstore.get()
        records_dictionary: dict[str, str] = {}
        
        contents = all_documents.get("documents", [])
        metadatas = all_documents.get("metadatas", [])
        
        for index in range(len(contents)):
            topic_key = contents[index]
            metadata = metadatas[index]
            records_dictionary[topic_key] = metadata.get("stance", "")
            
        return records_dictionary

    def find_similar_topics(
        self,
        query_topic: str,
        top_k=4,
    ) -> list[ExtractedBelief]:
        search_results = self.vectorstore.similarity_search(
            query=query_topic,
            k=top_k
        )
        
        extracted_results: list[ExtractedBelief] = []
        for document in search_results:
            extracted_results.append({
                "topic_key": document.page_content,
                "stance": document.metadata.get("stance", "")
            })
            
        return extracted_results


def build_extraction_prompt(
    latest_turn: str,
    motion: str,
    agent_role: str,
) -> str:
    return (
        f"Debate Motion: {motion}\n"
        f"Agent {agent_role} just spoke.\n"
        f"Latest Turn:\n{latest_turn}\n\n"
        "Task 1: Identify if this turn contains any internal logical "
        "contradictions where the agent opposes their own statements "
        "within this single turn. Log them in internal_contradictions.\n"
        "Task 2: Extract the core claims from this turn "
        "into distinct topic and stance pairs. Keep topics broad."
        "Formatting Rule: All topic keys MUST be lowercase, use spaces "
        "instead of underscores, and be 2 to 4 words maximum. "
    )


def extract_beliefs_from_turn(
    latest_turn: str,
    motion: str,
    agent_role: str,
) -> AtomizedTurn:
    language_model = get_chat_model("judge").with_structured_output(
        AtomizedTurn
    )
    prompt_text = build_extraction_prompt(latest_turn, motion, agent_role)
    
    def _invoke_llm() -> AtomizedTurn:
        messages = [HumanMessage(content=prompt_text)]
        return language_model.invoke(messages)
        
    return execute_with_backoff(_invoke_llm)


def build_evaluation_prompt(
    new_topic: str,
    new_stance: str,
    old_topic: str,
    old_stance: str,
) -> str:
    return (
        f"Historical Topic: {old_topic}\n"
        f"Historical Stance: {old_stance}\n\n"
        f"New Topic: {new_topic}\n"
        f"New Stance: {new_stance}\n\n"
        "Compare the New Stance against the Historical Stance.\n"
        "Classify the relationship into one of these categories:\n"
        "- CONTRADICT: The new stance logically opposes the old one.\n"
        "- REDUNDANT: The new stance repeats the old one entirely.\n"
        "- DIFFERENT_TOPIC: The new stance covers a different nuance, "
        "expands on the topic, or is completely unrelated.\n\n"
        "If CONTRADICT, provide an explanation. Otherwise, leave empty."
    )


def evaluate_belief_relationship(
    new_belief: ExtractedBelief,
    historical_belief: ExtractedBelief,
) -> TurnContradictionResult:
    language_model = get_chat_model("judge").with_structured_output(
        TurnContradictionResult
    )
    prompt_text = build_evaluation_prompt(
        new_belief["topic_key"],
        new_belief["stance"],
        historical_belief["topic_key"],
        historical_belief["stance"],
    )
    
    def _invoke_llm() -> TurnContradictionResult:
        messages = [HumanMessage(content=prompt_text)]
        return language_model.invoke(messages)
        
    return execute_with_backoff(_invoke_llm)


def process_single_extracted_belief(
    new_belief: ExtractedBelief,
    memory_store: VectorSemanticMemoryStore,
    contradictions_list: list[Contradiction],
) -> None:
    topic = new_belief["topic_key"]
    print(f"Evaluating belief topic: {topic}")
    
    similar_beliefs = memory_store.find_similar_topics(topic)
    
    if not similar_beliefs:
        print(f"No similar past topics found. Inserting new: {topic}")
        memory_store.upsert_belief(topic, new_belief["stance"])
        return

    found_redundancy_or_contradiction = False

    for historical_belief in similar_beliefs:
        evaluation = evaluate_belief_relationship(
            new_belief,
            historical_belief
        )

        category = evaluation["category"]
        print(f"Compared to historical '{historical_belief['topic_key']}' "
              f"-> Result: {category}")

        if category == "DIFFERENT_TOPIC":
            continue

        found_redundancy_or_contradiction = True

        if category == "CONTRADICT":
            print(f"!!! Contradiction logged for topic: {topic}")
            contradictions_list.append({
                "topic_key": historical_belief["topic_key"],
                "old_stance": historical_belief["stance"],
                "new_stance": new_belief["stance"],
                "explanation": evaluation.get("explanation", "")
            })
            memory_store.upsert_belief(
                historical_belief["topic_key"],
                new_belief["stance"]
            )
            break

        if category == "REDUNDANT":
            print(f"Stance is redundant. Skipping insertion.")
            break

    if not found_redundancy_or_contradiction:
        print(f"Topic is different from retrieved matches. Inserting new.")
        memory_store.upsert_belief(topic, new_belief["stance"])


def process_turn_through_pipeline(
    turn_text: str,
    motion: str,
    agent_role: str,
    memory_store: VectorSemanticMemoryStore,
    contradictions_list: list[Contradiction],
) -> None:
    print(f"\nProcessing {agent_role} turn...")
    atomized_result = extract_beliefs_from_turn(
        turn_text,
        motion,
        agent_role
    )
    
    internal_contras = atomized_result.get("internal_contradictions", [])
    extracted_beliefs = atomized_result.get("extracted_beliefs", [])
    
    print(f"Extracted {len(extracted_beliefs)} beliefs.")
    print(f"Found {len(internal_contras)} internal contradictions.")
    
    for internal_contra in internal_contras:
        contradictions_list.append(internal_contra)
    
    for belief in extracted_beliefs:
        process_single_extracted_belief(
            belief,
            memory_store,
            contradictions_list
        )


def simulate_single_debate(debate_data: DebateEntry) -> DebateResult:
    motion = debate_data.get("motion", "")
    turns = debate_data.get("turns", [])
    
    debate_id = str(uuid.uuid4()).replace("-", "")
    
    print(f"\n{'='*50}\nStarting debate simulation: {motion}\n{'='*50}")
    
    pro_memory = VectorSemanticMemoryStore(debate_id, "PRO")
    pro_contradictions: list[Contradiction] = []
    
    against_memory = VectorSemanticMemoryStore(debate_id, "AGAINST")
    against_contradictions: list[Contradiction] = []

    for turn_index, turn in enumerate(turns):
        print(f"\n--- Turn {turn_index + 1} ---")
        if turn.startswith("PRO:"):
            process_turn_through_pipeline(
                turn,
                motion,
                "PRO",
                pro_memory,
                pro_contradictions
            )
        else:
            process_turn_through_pipeline(
                turn,
                motion,
                "AGAINST",
                against_memory,
                against_contradictions
            )

    print(f"\nFinished debate simulation: {motion}")

    return {
        "debate_motion": motion,
        "pro_tracker": {
            "final_beliefs": pro_memory.get_all_records(),
            "contradictions": pro_contradictions
        },
        "against_tracker": {
            "final_beliefs": against_memory.get_all_records(),
            "contradictions": against_contradictions
        }
    }


def load_debate_dataset(file_path: str) -> list[DebateEntry]:
    print(f"Loading datasets from {file_path}...")
    if not os.path.exists(file_path):
        print("Dataset file not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as file_stream:
        return json.load(file_stream)


def save_debate_result(
    result: DebateResult,
    file_path: str
) -> None:
    print(f"Appending current result to {file_path}...")
    with open(file_path, "a", encoding="utf-8") as output_stream:
        json_string = json.dumps(result)
        output_stream.write(json_string + "\n")


def run_consistency_checker_test() -> None:
    input_path = "mock_debates.json"
    output_path = "pipeline_results.jsonl"
    sleep_duration = 60
    
    debates_dataset = load_debate_dataset(input_path)
    total_debates = len(debates_dataset)

    print(f"Found {total_debates} debates to process.")

    for index in range(total_debates):
        print(f"\nProcessing Debate {index + 1} of {total_debates}")
        debate_entry = debates_dataset[index]
        
        debate_result = simulate_single_debate(debate_entry)
        
        save_debate_result(debate_result, output_path)
        
        if index < total_debates - 1:
            print(f"Sleeping for {sleep_duration} seconds...")
            time.sleep(sleep_duration)


if __name__ == "__main__":
    run_consistency_checker_test()
