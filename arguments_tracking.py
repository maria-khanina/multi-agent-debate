import ast
import json
from typing import Any, cast

from configurations.config_manager import config, get_embeddings_model
from debate_settings import SIMILAR_TOPIC_COUNT
from entities import (
    ArgumentResult,
    AtomizedTurn,
    AuditResult,
    BeliefComparisonResult,
    CleanedContradictions,
    Contradiction,
    DebaterMistake,
    ExtractedBelief,
    FactCheckStats,
    Side,
    SourceCheckResult,
    TopicState,
)
from langchain_chroma import Chroma
from langchain_core.documents import Document
from language_calls import ask_structured_with_backoff, execute_with_backoff
from tools import create_audit_tool


class TopicStore:
    def __init__(self, debate_id: str) -> None:
        self.vectorstore = Chroma(
            collection_name=f"tracker_{debate_id}_global".lower(),
            embedding_function=get_embeddings_model(),
            persist_directory=config.read_text("database.path"),
        )

    def upsert_topic(self, topic_state: TopicState) -> None:
        new_document = Document(
            page_content=topic_state["subject"],
            metadata={
                "pro_stance": topic_state["pro_stance"] or "",
                "against_stance": topic_state["against_stance"] or "",
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
        metadata: dict[str, Any],
    ) -> TopicState:
        return {
            "subject": content,
            "pro_stance": metadata.get("pro_stance") or None,
            "against_stance": metadata.get("against_stance") or None,
            "brought_up_by": cast(Side, metadata.get("brought_up_by", "PRO")),
            "current_winner": cast(Side, metadata.get("current_winner", "PRO")),
            "winning_claim": metadata.get("winning_claim", ""),
        }

    def find_similar_topics(self, query_topic: str) -> list[TopicState]:
        search_results = self.vectorstore.similarity_search(
            query=query_topic,
            k=SIMILAR_TOPIC_COUNT,
        )
        return [
            self._document_to_topic_state(document.page_content, document.metadata)
            for document in search_results
        ]

    def get_all_topics(self) -> list[TopicState]:
        stored = self.vectorstore.get()
        contents: list[str] = stored["documents"]
        metadatas: list[dict[str, Any]] = stored["metadatas"]
        return [
            self._document_to_topic_state(content, metadata)
            for content, metadata in zip(contents, metadatas)
        ]


TOPIC_STORES: dict[str, TopicStore] = {}


def get_topic_store(debate_id: str) -> TopicStore:
    if debate_id not in TOPIC_STORES:
        TOPIC_STORES[debate_id] = TopicStore(debate_id)
    return TOPIC_STORES[debate_id]


def extract_beliefs_from_turn(
    latest_turn: str,
    motion: str,
    agent_role: str,
) -> AtomizedTurn:
    prompt_text = config.read_text("prompts.consistency_extraction").format(
        motion=motion,
        agent_role=agent_role,
        latest_turn=latest_turn,
    )
    return ask_structured_with_backoff("judge", AtomizedTurn, prompt_text)


def _get_source_check_result(
    belief: ExtractedBelief,
    turn_text: str,
) -> SourceCheckResult:
    prompt_text = config.read_text("prompts.check_source_requirement").format(
        belief=json.dumps(belief),
        transcript_turn=turn_text,
    )
    return ask_structured_with_backoff("judge", SourceCheckResult, prompt_text)


def _parse_audit_answer(answer: object) -> AuditResult:
    if not isinstance(answer, str):
        return cast(AuditResult, answer)
    try:
        return cast(AuditResult, ast.literal_eval(answer))
    except Exception:
        return {"verdict": "UNVERIFIABLE", "reasoning": answer}


def _run_audit_tool(
    debate_id: str,
    agent_role: str,
    claim: str,
    exact_source_title: str,
) -> AuditResult:
    factchecking_tool = create_audit_tool(debate_id)
    arguments: dict[str, str] = {
        "agent_role": agent_role,
        "claim": claim,
        "exact_source_title": exact_source_title,
    }

    def _run_audit() -> AuditResult:
        return _parse_audit_answer(factchecking_tool.invoke(arguments))

    return execute_with_backoff(_run_audit)


def _count_verdict(stats: FactCheckStats, audit_result: AuditResult) -> None:
    verdict = audit_result.get("verdict", "UNVERIFIABLE")
    if verdict in stats:
        stats[verdict] += 1


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
    exact_title = source_check.get("exact_source_title")
    if not source_check.get("has_source") or not exact_title:
        stats["UNVERIFIABLE"] += 1
        return {
            "verdict": "UNVERIFIABLE",
            "reasoning": "Requires source but none was provided.",
        }
    audit_result = _run_audit_tool(
        debate_id,
        agent_role,
        source_check.get("claim") or belief["stance"],
        exact_title,
    )
    _count_verdict(stats, audit_result)
    return audit_result


def evaluate_argument_loop(
    winning_claim: str,
    new_argument: str,
    audit_result: AuditResult | None,
) -> ArgumentResult:
    if audit_result:
        verdict = audit_result["verdict"]
        audit_text = f"Fact-Check Result: {verdict} - {audit_result['reasoning']}"
    else:
        audit_text = "No external references used."
    prompt_text = config.read_text("prompts.argument_evaluation").format(
        winning_claim=winning_claim,
        new_argument=new_argument,
        audit_result=audit_text,
    )
    return ask_structured_with_backoff("judge", ArgumentResult, prompt_text)


def _read_stance(topic_state: TopicState, agent_role: Side) -> str | None:
    if agent_role == "PRO":
        return topic_state["pro_stance"]
    return topic_state["against_stance"]


def _compare_stances(
    new_belief: ExtractedBelief,
    topic_state: TopicState,
    past_stance: str,
) -> BeliefComparisonResult:
    prompt_text = config.read_text("prompts.consistency_evaluation").format(
        old_topic=topic_state["subject"],
        old_stance=past_stance,
        new_topic=new_belief["topic_key"],
        new_stance=new_belief["stance"],
    )
    return ask_structured_with_backoff("judge", BeliefComparisonResult, prompt_text)


def check_internal_contradiction(
    new_belief: ExtractedBelief,
    topic_state: TopicState,
    agent_role: Side,
    contradictions_list: list[Contradiction],
) -> None:
    past_stance = _read_stance(topic_state, agent_role)
    if not past_stance:
        return
    evaluation = _compare_stances(new_belief, topic_state, past_stance)
    if evaluation["category"] == "CONTRADICT":
        subject = topic_state["subject"]
        print(f"[Tracker] !!! Contradiction logged for: '{subject}'")
        contradictions_list.append({
            "topic_key": subject,
            "old_stance": past_stance,
            "new_stance": new_belief["stance"],
            "explanation": evaluation.get("explanation", ""),
        })


def _create_new_topic(topic_key: str, stance: str, agent_role: Side) -> TopicState:
    return {
        "subject": topic_key,
        "pro_stance": stance if agent_role == "PRO" else None,
        "against_stance": stance if agent_role == "AGAINST" else None,
        "brought_up_by": agent_role,
        "current_winner": agent_role,
        "winning_claim": stance,
    }


def _apply_clash_result(
    evaluation: ArgumentResult,
    best_match: TopicState,
    stance: str,
    agent_role: Side,
    mistakes_list: list[DebaterMistake],
) -> None:
    if evaluation["category"] == "DEFEATS_CURRENT":
        best_match["current_winner"] = agent_role
        best_match["winning_claim"] = stance
    elif evaluation["category"] == "FAILS_TO_DEFEAT":
        mistakes_list.append({
            "topic": best_match["subject"],
            "attempted_claim": stance,
            "failure_type": "UNCONVINCING",
            "explanation": evaluation.get("explanation", ""),
        })


def _handle_clash(
    best_match: TopicState,
    stance: str,
    agent_role: Side,
    audit_result: AuditResult | None,
    topic_store: TopicStore,
    mistakes_list: list[DebaterMistake],
) -> None:
    evaluation = evaluate_argument_loop(
        best_match["winning_claim"],
        stance,
        audit_result,
    )
    print(f"[Tracker] Clash evaluation: {evaluation['category']}")
    _apply_clash_result(evaluation, best_match, stance, agent_role, mistakes_list)
    topic_store.upsert_topic(best_match)
    if evaluation["category"] == "NEW_VALID_POINT":
        new_topic = _create_new_topic(best_match["subject"], stance, agent_role)
        topic_store.upsert_topic(new_topic)


def _record_failed_fact_check(
    new_belief: ExtractedBelief,
    audit_result: AuditResult | None,
    mistakes_list: list[DebaterMistake],
) -> bool:
    if not audit_result:
        return False
    verdict = audit_result["verdict"]
    if verdict != "FABRICATED" and verdict != "UNVERIFIABLE":
        return False
    print(f"[Tracker] Automatic failure for claim: {new_belief['topic_key']}")
    mistakes_list.append({
        "topic": new_belief["topic_key"],
        "attempted_claim": new_belief["stance"],
        "failure_type": verdict,
        "explanation": audit_result.get("reasoning", "Fact-check failed."),
    })
    return True


def _insert_if_new(
    new_belief: ExtractedBelief,
    agent_role: Side,
    topic_store: TopicStore,
) -> TopicState | None:
    topic_key = new_belief["topic_key"]
    similar_topics = topic_store.find_similar_topics(topic_key)
    if similar_topics:
        return similar_topics[0]
    print(f"[Tracker] No similar topics. Inserting: '{topic_key}'")
    new_topic = _create_new_topic(topic_key, new_belief["stance"], agent_role)
    topic_store.upsert_topic(new_topic)
    return None


def _update_matched_topic(
    best_match: TopicState,
    new_belief: ExtractedBelief,
    agent_role: Side,
    audit_result: AuditResult | None,
    topic_store: TopicStore,
    mistakes_list: list[DebaterMistake],
) -> None:
    stance = new_belief["stance"]
    if agent_role == "PRO":
        best_match["pro_stance"] = stance
    else:
        best_match["against_stance"] = stance
    if best_match["current_winner"] != agent_role:
        _handle_clash(
            best_match,
            stance,
            agent_role,
            audit_result,
            topic_store,
            mistakes_list,
        )
        return
    best_match["winning_claim"] = stance
    topic_store.upsert_topic(best_match)


def process_belief(
    debate_id: str,
    new_belief: ExtractedBelief,
    turn_text: str,
    agent_role: Side,
    topic_store: TopicStore,
    contradictions_list: list[Contradiction],
    mistakes_list: list[DebaterMistake],
    factcheck_stats: FactCheckStats,
) -> None:
    print(f"[Tracker] Evaluating belief topic: '{new_belief['topic_key']}'")
    audit_result = check_belief_facts(
        debate_id,
        new_belief,
        turn_text,
        agent_role,
        factcheck_stats,
    )
    if _record_failed_fact_check(new_belief, audit_result, mistakes_list):
        return
    best_match = _insert_if_new(new_belief, agent_role, topic_store)
    if best_match is None:
        return
    check_internal_contradiction(
        new_belief,
        best_match,
        agent_role,
        contradictions_list,
    )
    _update_matched_topic(
        best_match,
        new_belief,
        agent_role,
        audit_result,
        topic_store,
        mistakes_list,
    )


def process_debate_turn(
    debate_id: str,
    turn_text: str,
    motion: str,
    agent_role: Side,
    topic_store: TopicStore,
    contradictions_list: list[Contradiction],
    mistakes_list: list[DebaterMistake],
    factcheck_stats: FactCheckStats,
) -> None:
    print(f"\n--- Checking consistency for {agent_role} ---")
    atomized_result = extract_beliefs_from_turn(turn_text, motion, agent_role)
    internal_contradictions = atomized_result.get("internal_contradictions", [])
    extracted_beliefs = atomized_result.get("extracted_beliefs", [])
    print(f"[Tracker] Extracted {len(extracted_beliefs)} beliefs.")
    if internal_contradictions:
        print(f"[Tracker] Found {len(internal_contradictions)} internal contras!")
    contradictions_list.extend(internal_contradictions)
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
    prompt_text = config.read_text("prompts.consistency_deduplication").format(
        contradictions_json=json.dumps(contradictions_list, indent=2),
    )
    response = ask_structured_with_backoff("judge", CleanedContradictions, prompt_text)
    cleaned = response.get("unique_contradictions", [])
    print(
        f"[Deduplication] Reduced {len(contradictions_list)} "
        f"contradictions to {len(cleaned)}"
    )
    return cleaned
