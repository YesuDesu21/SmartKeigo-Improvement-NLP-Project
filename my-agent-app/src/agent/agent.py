"""Node implementations. Dependencies (LLM, retriever) are injected so the graph is unit-testable
without network access, and so import never requires API keys."""
import time
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_core.runnables import RunnableConfig

from . import prompts
from .config import RunConfig
from .schemas import AnalysisResult, DraftResult, ReviewResult
from .state import KeigoState


@lru_cache(maxsize=8)
def default_llm_factory(model: str, temperature: float):
    import os
    from dotenv import find_dotenv, load_dotenv
    from langchain_groq import ChatGroq
    load_dotenv(find_dotenv())
    return ChatGroq(model=model, temperature=temperature, api_key=os.getenv("GROQ_API_KEY"))


_retriever = None


def default_retriever_factory():
    global _retriever
    if _retriever is None:
        from dotenv import find_dotenv, load_dotenv
        from .retrieval import Retriever
        load_dotenv(find_dotenv())
        _retriever = Retriever()
    return _retriever


def call_structured(llm, schema, system: str, user: str, node: str, retries: int = 1) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Structured call with usage logging. Failures are recorded, never silently dropped."""
    runner = llm.with_structured_output(schema, include_raw=True)
    rec: Dict[str, Any] = {"node": node, "calls": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "failed": False}
    parsed = None
    for _ in range(retries + 1):
        t0 = time.perf_counter()
        try:
            out = runner.invoke([("system", system), ("human", user)])
        except Exception as e:  # network / rate limit
            rec["calls"] += 1
            rec["seconds"] += time.perf_counter() - t0
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            continue
        rec["calls"] += 1
        rec["seconds"] += time.perf_counter() - t0
        um = getattr(out.get("raw"), "usage_metadata", None) or {}
        rec["input_tokens"] += um.get("input_tokens", 0) or 0
        rec["output_tokens"] += um.get("output_tokens", 0) or 0
        parsed = out.get("parsed")
        if parsed is not None:
            rec.pop("error", None)
            break
        rec["error"] = f"parse_error: {out.get('parsing_error')}"[:200]
    rec["failed"] = parsed is None
    return parsed, rec


def make_nodes(llm_factory: Callable = default_llm_factory, retriever_factory: Callable = default_retriever_factory):
    def _llm(cfg: RunConfig):
        return llm_factory(cfg.model, cfg.temperature)

    def analyzer(state: KeigoState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        if not cfg.use_analyzer:
            return {"context": {}, "model_id": cfg.model}
        parsed, rec = call_structured(_llm(cfg), AnalysisResult, prompts.ANALYZER_SYSTEM,
                                      prompts.analyzer_user(state["raw_input"], state.get("scenario", ""), state.get("requested_register")), "analyzer")
        ctx = parsed.model_dump() if parsed else {"unknowns": ["analysis_failed"]}
        if state.get("requested_register"):  # controlled track: the given register is authoritative
            ctx["overall_register"] = state["requested_register"]
        return {"context": ctx, "usage": [rec], "model_id": cfg.model}

    def retrieve(state: KeigoState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        if not cfg.use_retrieval:
            return {"retrieved_ids": [], "retrieved_examples": []}
        reg = state.get("requested_register") or (state.get("context") or {}).get("overall_register")
        t0 = time.perf_counter()
        rec = {"node": "retrieval", "calls": 1, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "failed": False}
        try:
            ex = retriever_factory().retrieve(state["raw_input"], cfg.top_k, reg)
        except Exception as e:
            ex, rec["failed"], rec["error"] = [], True, f"{type(e).__name__}: {e}"[:200]
        rec["seconds"] = time.perf_counter() - t0
        return {"retrieved_ids": [e["id"] for e in ex], "retrieved_examples": ex, "usage": [rec]}

    def specialist(state: KeigoState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        drafts = state.get("drafts", [])
        feedback = state.get("feedback") or []
        is_revision = bool(drafts) and bool(feedback)
        user = prompts.specialist_user(
            state["raw_input"], state.get("context"), state.get("retrieved_examples"), state.get("requested_register"),
            prior_draft=drafts[-1] if is_revision else None, feedback=feedback if is_revision else None)
        parsed, rec = call_structured(_llm(cfg), DraftResult, prompts.SPECIALIST_SYSTEM, user,
                                      "specialist_revision" if is_revision else "specialist")
        text = parsed.keigo_text.strip() if parsed and parsed.keigo_text.strip() else state["raw_input"]
        return {"drafts": [text], "explanation": parsed.explanation if parsed else "",
                "revision_count": state.get("revision_count", 0) + (1 if is_revision else 0),
                "usage": [rec], "feedback": []}

    def reviewer(state: KeigoState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        if not cfg.use_reviewer:
            return {"is_approved": True, "status": "unreviewed", "feedback": []}
        parsed, rec = call_structured(_llm(cfg), ReviewResult, prompts.REVIEWER_SYSTEM,
                                      prompts.reviewer_user(state["raw_input"], state.get("context"), state["drafts"][-1], state.get("requested_register")), "reviewer")
        if parsed is None:  # do not loop on a broken reviewer
            return {"is_approved": False, "status": "needs_review", "usage": [rec],
                    "feedback": [{"category": "reviewer_error", "description": rec.get("error", "")}]}
        if parsed.approved:
            return {"is_approved": True, "status": "approved", "feedback": [], "usage": [rec]}
        issues = [i.model_dump() for i in parsed.issues] or [{"category": "other", "description": "Reviewer rejected without details", "suggested_fix": ""}]
        return {"is_approved": False, "status": "needs_review", "feedback": issues, "usage": [rec]}

    def route_after_review(state: KeigoState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        if state.get("is_approved"):
            return "end"
        if any(i.get("category") == "reviewer_error" for i in state.get("feedback", [])):
            return "end"
        return "revise" if state.get("revision_count", 0) < cfg.max_revisions else "end"

    return analyzer, retrieve, specialist, reviewer, route_after_review
