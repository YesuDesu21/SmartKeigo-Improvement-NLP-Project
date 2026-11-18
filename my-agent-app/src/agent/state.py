import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # Input definition (normalized to NFKC on entry)
    casual_input: str
    raw_input: str  # backward-compatible alias
    target_domain: Optional[str]  # "Email", "Verbal", "Chat", "Customer Support"
    scenario: Optional[str]  # backward-compatible alias
    requested_register: Optional[str]  # "teineigo" | "sonkeigo" | "kenjogo" | "no_change" | None

    # Parallel extraction contexts
    hierarchy_metadata: Dict[str, Any]  # addressee, tone, intent
    context: Dict[str, Any]  # AnalysisResult.model_dump()
    retrieved_exemplars: List[Dict[str, str]]  # dynamically retrieved train.jsonl pairs
    retrieved_examples: List[Dict[str, str]]  # backward-compatible alias
    retrieved_ids: List[str]
    suppletive_rules: List[str]  # matched irregular verb constraints (補充法)
    domain_guidelines: Dict[str, Any]  # loaded domain template guidelines

    # Generation, Critique & Routing
    current_candidate: str
    drafts: Annotated[List[str], operator.add]  # full candidate history
    critique_history: Annotated[List[str], operator.add]
    feedback: List[Dict[str, str]]  # structured issues from latest review
    iteration_count: int
    revision_count: int  # backward-compatible alias
    is_approved: bool
    status: str  # "approved" | "needs_review" | "unreviewed" | "fallback"
    rejection_reason: Optional[str]
    final_output: Optional[str]
    explanation: str

    # Empirical research logging & RQ4 telemetry
    execution_trace: Annotated[List[Dict[str, Any]], operator.add]
    usage: Annotated[List[Dict[str, Any]], operator.add]
    model_id: str
    fallback_triggered: bool


KeigoState = AgentState
