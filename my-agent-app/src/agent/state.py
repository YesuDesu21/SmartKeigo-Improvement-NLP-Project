import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class KeigoState(TypedDict, total=False):
    # inputs (never overwritten after START)
    raw_input: str
    scenario: str                       # optional free-text situation from the user
    requested_register: Optional[str]   # teineigo | sonkeigo | kenjogo | None (controlled track)
    # analyzer output
    context: Dict[str, Any]             # AnalysisResult.model_dump(); {} when analyzer is off
    # retrieval
    retrieved_ids: List[str]
    retrieved_examples: List[Dict[str, str]]
    # generation / review
    drafts: Annotated[List[str], operator.add]       # full draft history
    explanation: str
    feedback: List[Dict[str, str]]                   # issues from the latest review
    revision_count: int
    is_approved: bool
    status: str                                       # approved | needs_review | unreviewed
    # bookkeeping for RQ4 (time, calls, tokens)
    usage: Annotated[List[Dict[str, Any]], operator.add]
    model_id: str
