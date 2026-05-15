import operator
from typing import Annotated, Literal, NotRequired, TypedDict


BeliefRelationship = Literal["CONTRADICT", "REDUNDANT", "DIFFERENT_TOPIC"]


class Contradiction(TypedDict):
    topic_key: str
    old_stance: str
    new_stance: str
    explanation: str


class CleanedContradictions(TypedDict):
    unique_contradictions: list[Contradiction]


class ExtractedBelief(TypedDict):
    topic_key: str
    stance: str


class AtomizedTurn(TypedDict):
    extracted_beliefs: list[ExtractedBelief]
    internal_contradictions: list[Contradiction]


class BeliefComparisonResult(TypedDict):
    category: BeliefRelationship
    explanation: str


class SciencePaper(TypedDict):
    title: str
    year: int | None
    citations: int
    affiliations: list[str]
    abstract: str
    pdf_url: str | None


class ResearcherState(TypedDict):
    debate_id: str
    motion: str
    query: str
    agent_role: str
    raw_papers: NotRequired[list[SciencePaper]]
    approved_papers: NotRequired[list[SciencePaper]]


class SearchResult(TypedDict):
    raw_papers: list[SciencePaper]


class Evidence(TypedDict):
    title: str
    content: str
    source: str


class AuditResult(TypedDict):
    verdict: Literal[
        "SUPPORTED", 
        "FABRICATED", 
        "EXAGGERATED", 
        "UNVERIFIABLE"
    ]
    reasoning: str


class DebateState(TypedDict):
    debate_id: str
    motion: str
    transcript: Annotated[list[str], operator.add]
    turn_count: int
    max_turns: int
    pro_sources: list[str]
    against_sources: list[str]
    pro_contradictions: Annotated[list[Contradiction], operator.add]
    against_contradictions: Annotated[list[Contradiction], operator.add]
    cleaned_pro_contradictions: list[Contradiction]
    cleaned_against_contradictions: list[Contradiction]
