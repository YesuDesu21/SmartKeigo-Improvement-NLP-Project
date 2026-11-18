"""Run-time switches. Passed per call via config={"configurable": {...}} so one compiled
graph can run every paper condition (B1, A1-A4) without code changes."""
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class RunConfig:
    use_analyzer: bool = True
    use_retrieval: bool = True
    use_reviewer: bool = True
    max_revisions: int = 2          # revisions AFTER the initial draft
    top_k: int = 3
    model: str = "openai/gpt-oss-120b"
    temperature: float = 0.0

    @classmethod
    def from_runnable(cls, config: Optional[Mapping[str, Any]]) -> "RunConfig":
        c = (config or {}).get("configurable", {}) or {}
        return cls(**{k: v for k, v in c.items() if k in cls.__dataclass_fields__})


# Paper conditions (Section 4a). B0 = copy source and B2 = T5 are run outside the graph.
CONDITIONS = {
    "B1": dict(use_analyzer=False, use_retrieval=False, use_reviewer=False),
    "A1": dict(use_analyzer=False, use_retrieval=True, use_reviewer=False),
    "A2": dict(use_analyzer=True, use_retrieval=True, use_reviewer=False),
    "A3": dict(use_analyzer=True, use_retrieval=True, use_reviewer=True),
    "A4": dict(use_analyzer=True, use_retrieval=False, use_reviewer=True),
}
