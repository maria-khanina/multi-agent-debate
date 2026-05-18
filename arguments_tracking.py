import json
import time
from typing import Callable, TypeVar

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from configurations.config_manager import config, get_chat_model, \
    get_embeddings_model
from entities import (
    ArgumentResult, AtomizedTurn, CleanedContradictions, 
    Contradiction, DebaterMistake, ExtractedBelief, TopicState,
    FactCheckStats, SourceCheckResult, AuditResult
)
from tools import create_audit_tool


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


class TopicStore:
    def __init__(self, debate_id: str) -> None:
        embedder = get_embeddings_model()
        db_path = config.get("database.path", "./debate_vectors")
        col_name = f"tracker_{debate_id}_global".lower()
        
        self.vectorstore = Chroma(
            collection_name=col_name,
            embedding_function=embedder,
            persist_directory=db_path,
        )

    def upsert_topic(self, topic_state: TopicState) -> None:
        new_document = Document(
            page_content=topic_state["subject"],
            metadata={
                "pro_stance": topic_state.get("pro_stance") or "",
                "against_stance": topic_state.get("against_stance") or "",
                "brought_up_by": topic_state["brought_up_by"],
                "current_winner": topic_state["current_winner"],
                "winning_claim": topic_state["winning_claim"],
            },
        )
        self.vectorstore.add_documents(
            documents=[new_document],
            ids=[topic_state["subject"]],
        )

    def _document_to_topic_state(
        self, 
        content: str, 
        meta: dict,
    ) -> TopicState:
        return {
            "subject": content,
            "pro_stance": meta.get("pro_stance") or None,
            "against_stance": meta.get("against_stance") or None,
            "brought_up_by": meta.get("brought_up_by", "PRO"),
            "current_winner": meta.get("current_winner", "PRO"),
            "winning_claim": meta.get("winning_claim", ""),
        }

    def find_similar_topics(
        self,
        query_topic: str,
        top_k: int = 4,
    ) -> list[TopicState]:
        search_results = self.vectorstore.similarity_search(
            query=query_topic,
            k=top_k,
        )
        extracted_results = []
        for document in search_results:
            extracted_results.append(
                self._document_to_topic_state(
                    document.page_content, 
                    document.metadata,
                )
            )
        return extracted_results

    def get_all_topics(self) -> list[TopicState]:
        all_docs = self.vectorstore.get()
        extracted_results = []
        docs = all_docs["documents"]
        metas = all_docs["metadatas"]
        
        for content, metadata in zip(docs, metas):
            extracted_results.append(
                self._document_to_topic_state(content, metadata)
            )
        return extracted_results


_TOPIC_STORE = {}


def get_topic_store(debate_id: str) -> TopicStore:
    if debate_id not in _TOPIC_STORE:
        _TOPIC_STORE[debate_id] = TopicStore(debate_id)
    return _TOPIC_STORE[debate_id]


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


def _get_source_check_result(
    belief: ExtractedBelief,
    turn_text: str,
) -> SourceCheckResult:
    language_model = get_chat_model("judge").with_structured_output(
        SourceCheckResult
    )
    prompt_template = config.get("prompts.check_source_requirement")
    prompt_text = prompt_template.format(
        belief=json.dumps(belief),
        transcript_turn=turn_text,
    )
    
    def _invoke_llm() -> SourceCheckResult:
        return language_model.invoke([HumanMessage(content=prompt_text)])
        
    return execute_with_backoff(_invoke_llm)


def _run_audit_tool(
    debate_id: str,
    agent_role: str,
    claim: str,
    exact_source_title: str,
) -> AuditResult:
    factchecking_tool = create_audit_tool(debate_id)
    
    def _run_audit() -> AuditResult:
        res = factchecking_tool.invoke({
            "agent_role": agent_role,
            "claim": claim,
            "exact_source_title": exact_source_title,
        })
        if isinstance(res, str):
            try:
                import ast
                return ast.literal_eval(res)
            except Exception:
                return {"verdict": "UNVERIFIABLE", "reasoning": str(res)}
        return res
        
    return execute_with_backoff(_run_audit)


def check_belief_facts(
    debate_id: str,
    belief: ExtractedBelief,
    turn_text: str,
    agent_role: str,
    stats: FactCheckStats,
) -> AuditResult | None:
    source_check = _get_source_check_result(belief, turn_text)
    
    if not source_check.get("requires_source"):
        return None
        
    has_source = source_check.get("has_source")
    exact_title = source_check.get("exact_source_title")
    
    if not has_source or not exact_title:
        stats["UNVERIFIABLE"] += 1
        return {
            "verdict": "UNVERIFIABLE",
            "reasoning": "Requires source but none was provided.",
        }
        
    audit_result = _run_audit_tool(
        debate_id, 
        agent_role, 
        source_check.get("claim", belief["stance"]), 
        exact_title,
    )
    
    verdict = audit_result.get("verdict", "UNVERIFIABLE")
    if verdict in stats:
        stats[verdict] += 1
        
    return audit_result


def evaluate_argument_loop(
    winning_claim: str,
    new_argument: str,
    audit_result: AuditResult | None,
) -> ArgumentResult:
    language_model = get_chat_model("judge").with_structured_output(
        ArgumentResult
    )
    prompt_template = config.get("prompts.argument_evaluation")
    
    if audit_result:
        verdict = audit_result['verdict']
        reasoning = audit_result['reasoning']
        audit_str = f"Fact-Check Result: {verdict} - {reasoning}"
    else:
        audit_str = "No external references used."
        
    prompt_text = prompt_template.format(
        winning_claim=winning_claim,
        new_argument=new_argument,
        audit_result=audit_str,
    )
    
    def _invoke_eval() -> ArgumentResult:
        return language_model.invoke([HumanMessage(content=prompt_text)])
        
    return execute_with_backoff(_invoke_eval)


def _check_internal_contradiction(
    new_belief: ExtractedBelief,
    topic_state: TopicState,
    agent_role: str,
    contradictions_list: list[Contradiction],
) -> None:
    past_stance = topic_state.get(f"{agent_role.lower()}_stance")
    if not past_stance:
        return
        
    from entities import BeliefComparisonResult
    
    language_model = get_chat_model("judge").with_structured_output(
        BeliefComparisonResult
    )
    prompt_template = config.get("prompts.consistency_evaluation")
    prompt_text = prompt_template.format(
        old_topic=topic_state["subject"],
        old_stance=past_stance,
        new_topic=new_belief["topic_key"],
        new_stance=new_belief["stance"],
    )
    
    def _invoke_llm() -> BeliefComparisonResult:
        return language_model.invoke([HumanMessage(content=prompt_text)])
        
    evaluation = execute_with_backoff(_invoke_llm)
    
    if evaluation["category"] == "CONTRADICT":
        subj = topic_state['subject']
        print(f"[Tracker] !!! Contradiction logged for: '{subj}'")
        contradictions_list.append({
            "topic_key": subj,
            "old_stance": past_stance,
            "new_stance": new_belief["stance"],
            "explanation": evaluation.get("explanation", ""),
        })


def _create_new_topic(
    topic_key: str,
    stance: str,
    agent_role: str,
) -> TopicState:
    return {
        "subject": topic_key,
        "pro_stance": stance if agent_role == "PRO" else None,
        "against_stance": stance if agent_role == "AGAINST" else None,
        "brought_up_by": agent_role,
        "current_winner": agent_role,
        "winning_claim": stance,
    }


def _handle_clash(
    best_match: TopicState,
    stance: str,
    agent_role: str,
    audit_result: AuditResult | None,
    topic_store: TopicStore,
    mistakes_list: list[DebaterMistake],
) -> None:
    topic_key = best_match["subject"]
    evaluation = evaluate_argument_loop(
        best_match["winning_claim"],
        stance,
        audit_result,
    )
    
    category = evaluation["category"]
    print(f"[Tracker] Clash evaluation: {category}")
    
    if category == "DEFEATS_CURRENT":
        best_match["current_winner"] = agent_role
        best_match["winning_claim"] = stance
        topic_store.upsert_topic(best_match)
    elif category == "FAILS_TO_DEFEAT":
        mistakes_list.append({
            "topic": topic_key,
            "attempted_claim": stance,
            "failure_type": "UNCONVINCING", 
            "explanation": evaluation.get("explanation", ""),
        })
        topic_store.upsert_topic(best_match) 
    elif category == "NEW_VALID_POINT":
        topic_store.upsert_topic(best_match)
        new_topic = _create_new_topic(topic_key, stance, agent_role)
        topic_store.upsert_topic(new_topic)


def process_belief(
    debate_id: str,
    new_belief: ExtractedBelief,
    turn_text: str,
    agent_role: str,
    topic_store: TopicStore,
    contradictions_list: list[Contradiction],
    mistakes_list: list[DebaterMistake],
    factcheck_stats: FactCheckStats,
) -> None:
    topic_key = new_belief["topic_key"]
    stance = new_belief["stance"]
    print(f"[Tracker] Evaluating belief topic: '{topic_key}'")
    
    audit_result = check_belief_facts(
        debate_id, new_belief, turn_text, agent_role, factcheck_stats
    )
    
    if audit_result and audit_result["verdict"] in (
        "FABRICATED", "UNVERIFIABLE"
    ):
        print(f"[Tracker] Automatic failure for claim: {topic_key}")
        mistakes_list.append({
            "topic": topic_key,
            "attempted_claim": stance,
            "failure_type": audit_result["verdict"], 
            "explanation": audit_result.get(
                "reasoning", "Fact-check failed."
            ),
        })
        return

    similar_topics = topic_store.find_similar_topics(topic_key)
    if not similar_topics:
        print(f"[Tracker] No similar topics. Inserting: '{topic_key}'")
        new_topic = _create_new_topic(topic_key, stance, agent_role)
        topic_store.upsert_topic(new_topic)
        return

    best_match = similar_topics[0]
    _check_internal_contradiction(
        new_belief, best_match, agent_role, contradictions_list
    )
    
    if agent_role == "PRO":
        best_match["pro_stance"] = stance
    else:
        best_match["against_stance"] = stance

    opponent = "AGAINST" if agent_role == "PRO" else "PRO"

    if best_match["current_winner"] == opponent:
        _handle_clash(
            best_match, 
            stance, 
            agent_role, 
            audit_result, 
            topic_store, 
            mistakes_list,
        )
    else:
        best_match["winning_claim"] = stance
        topic_store.upsert_topic(best_match)


def process_debate_turn(
    debate_id: str,
    turn_text: str,
    motion: str,
    agent_role: str,
    topic_store: TopicStore,
    contradictions_list: list[Contradiction],
    mistakes_list: list[DebaterMistake],
    factcheck_stats: FactCheckStats,
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
            debate_id,
            belief,
            turn_text,
            agent_role,
            topic_store,
            contradictions_list,
            mistakes_list,
            factcheck_stats,
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
