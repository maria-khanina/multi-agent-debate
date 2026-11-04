import operator
from typing import Annotated, Literal, NotRequired, TypedDict

BeliefRelationship = Literal["CONTRADICT", "REDUNDANT", "DIFFERENT_TOPIC"]
Side = Literal["PRO", "AGAINST"]
FactCheckVerdict = Literal["SUPPORTED", "FABRICATED", "EXAGGERATED", "UNVERIFIABLE"]


class BeliefComparisonResult(TypedDict):
    category: BeliefRelationship
    explanation: str


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


class TopicState(TypedDict):
    subject: str
    pro_stance: str | None
    against_stance: str | None
    brought_up_by: Side
    current_winner: Side
    winning_claim: str


class ArgumentResult(TypedDict):
    category: Literal["DEFEATS_CURRENT", "FAILS_TO_DEFEAT", "NEW_VALID_POINT"]
    explanation: str


class DebaterMistake(TypedDict):
    topic: str
    attempted_claim: str
    failure_type: Literal["IRRELEVANT", "UNCONVINCING", "FABRICATED", "UNVERIFIABLE"]
    explanation: str


class FactCheckStats(TypedDict):
    SUPPORTED: int
    FABRICATED: int
    EXAGGERATED: int
    UNVERIFIABLE: int


class SourceCheckResult(TypedDict):
    requires_source: bool
    has_source: bool
    exact_source_title: str | None
    claim: str | None


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


class ApprovalResult(TypedDict):
    approved_papers: list[SciencePaper]


class Evidence(TypedDict):
    title: str
    content: str
    source: str


class AuditResult(TypedDict):
    verdict: FactCheckVerdict
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
    pro_mistakes: Annotated[list[DebaterMistake], operator.add]
    against_mistakes: Annotated[list[DebaterMistake], operator.add]
    pro_factcheck_stats: FactCheckStats
    against_factcheck_stats: FactCheckStats
    final_argument_stats: dict[str, int]
