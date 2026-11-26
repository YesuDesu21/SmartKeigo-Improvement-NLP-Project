"""Node implementations. Dependencies (LLM, retriever) are injected so the graph is unit-testable
without network access, and so import never requires API keys."""
import time
import unicodedata
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_core.runnables import RunnableConfig

from . import prompts
from .config import RunConfig
from .guardrails.honorific_rules import (audit_nijuu_keigo, audit_casual_markers, convert_to_conservative_teineigo,
                                         is_already_polite)
from .knowledge import get_domain_template, get_suppletive_rules
from .schemas import AnalysisResult, DraftResult, ReviewResult
from .state import AgentState, KeigoState


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

    def analyzer(state: AgentState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        raw = unicodedata.normalize("NFKC", state.get("casual_input") or state.get("raw_input") or "")
        scenario = state.get("target_domain") or state.get("scenario") or ""
        suppletive_rules = get_suppletive_rules(raw)
        domain_guidelines = get_domain_template(scenario) or {}

        if not cfg.use_analyzer:
            return {
                "casual_input": raw,
                "raw_input": raw,
                "context": {},
                "hierarchy_metadata": {},
                "suppletive_rules": suppletive_rules,
                "domain_guidelines": domain_guidelines,
                "model_id": cfg.model,
            }

        parsed, rec = call_structured(
            _llm(cfg),
            AnalysisResult,
            prompts.ANALYZER_SYSTEM,
            prompts.analyzer_user(raw, scenario, state.get("requested_register")),
            "analyzer",
        )
        ctx = parsed.model_dump() if parsed else {"unknowns": ["analysis_failed"]}
        if state.get("requested_register"):  # controlled track: the given register is authoritative
            ctx["overall_register"] = state["requested_register"]
        elif ctx.get("overall_register") in (None, "no_change", "unknown") and not is_already_polite(raw):
            # Free input (no register given) means the user wants it formalized. If the analyzer
            # could not tell who is addressed and called it no_change, default to teineigo
            # instead of returning casual text unchanged.
            ctx["overall_register"] = "teineigo"

        hierarchy = {
            "addressee": getattr(parsed, "addressee", "unknown") if parsed else "unknown",
            "tone": getattr(parsed, "tone", "casual") if parsed else "casual",
            "intent": getattr(parsed, "intent", "") if parsed else "",
        }
        return {
            "casual_input": raw,
            "raw_input": raw,
            "context": ctx,
            "hierarchy_metadata": hierarchy,
            "suppletive_rules": suppletive_rules,
            "domain_guidelines": domain_guidelines,
            "usage": [rec],
            "execution_trace": [rec],
            "model_id": cfg.model,
        }

    def retrieve(state: AgentState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        raw = unicodedata.normalize("NFKC", state.get("casual_input") or state.get("raw_input") or "")
        if not cfg.use_retrieval:
            return {"retrieved_ids": [], "retrieved_examples": [], "retrieved_exemplars": []}

        reg = state.get("requested_register") or (state.get("context") or {}).get("overall_register")
        t0 = time.perf_counter()
        rec = {"node": "retrieval", "calls": 1, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "failed": False}
        try:
            ex = retriever_factory().retrieve(raw, cfg.top_k, reg)
        except Exception as e:
            ex, rec["failed"], rec["error"] = [], True, f"{type(e).__name__}: {e}"[:200]
        rec["seconds"] = time.perf_counter() - t0
        return {
            "retrieved_ids": [e["id"] for e in ex],
            "retrieved_examples": ex,
            "retrieved_exemplars": ex,
            "usage": [rec],
            "execution_trace": [rec],
        }

    def specialist(state: AgentState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        raw = unicodedata.normalize("NFKC", state.get("casual_input") or state.get("raw_input") or "")
        drafts = state.get("drafts", [])
        feedback = state.get("feedback") or []
        is_revision = bool(drafts) and bool(feedback)

        exemplars = state.get("retrieved_exemplars") or state.get("retrieved_examples") or []
        suppletive_rules = state.get("suppletive_rules") or get_suppletive_rules(raw)
        domain_template = state.get("domain_guidelines") or get_domain_template(
            state.get("target_domain") or state.get("scenario")
        )

        user = prompts.specialist_user(
            raw,
            state.get("context"),
            exemplars,
            state.get("requested_register"),
            prior_draft=drafts[-1] if is_revision else None,
            feedback=feedback if is_revision else None,
            suppletive_rules=suppletive_rules,
            domain_template=domain_template,
        )
        parsed, rec = call_structured(
            _llm(cfg), DraftResult, prompts.SPECIALIST_SYSTEM, user, "specialist_revision" if is_revision else "specialist"
        )

        # --- Safeguard: if LLM parse failed or returned the unchanged casual input,
        #     apply the deterministic conservative fallback immediately instead of
        #     passing casual text to the reviewer (which may incorrectly approve it).
        if parsed is None or not parsed.keigo_text.strip():
            # LLM failed to produce structured output – fall back deterministically.
            fallback_text = convert_to_conservative_teineigo(raw)
            rec["error"] = rec.get("error", "specialist parse failed – used conservative fallback")
            cur_rev = state.get("iteration_count", state.get("revision_count", 0))
            return {
                "drafts": [fallback_text],
                "current_candidate": fallback_text,
                "final_output": fallback_text,
                "status": "needs_review",
                "explanation": "LLM specialist did not return a valid result. Applied deterministic conservative Teineigo fallback.",
                "iteration_count": cur_rev,
                "revision_count": cur_rev,
                "fallback_triggered": True,
                "usage": [rec],
                "execution_trace": [rec],
                "feedback": [],
            }

        text = parsed.keigo_text.strip()
        text = unicodedata.normalize("NFKC", text)

        # --- Safeguard: reject output that is identical to the casual input.
        #     This catches cases where the LLM parsed successfully but returned
        #     the original text unchanged (silent no-op conversion).
        raw_norm = unicodedata.normalize("NFKC", raw)
        if text == raw_norm:
            # Force the conservative fallback instead of passing unchanged casual text onward.
            fallback_text = convert_to_conservative_teineigo(raw)
            cur_rev = state.get("iteration_count", state.get("revision_count", 0))
            return {
                "drafts": [fallback_text],
                "current_candidate": fallback_text,
                "final_output": fallback_text,
                "status": "needs_review",
                "explanation": "Specialist returned the input unchanged. Applied deterministic conservative Teineigo fallback.",
                "iteration_count": cur_rev,
                "revision_count": cur_rev,
                "fallback_triggered": True,
                "usage": [rec],
                "execution_trace": [rec],
                "feedback": [],
            }

        cur_rev = state.get("iteration_count", state.get("revision_count", 0)) + (1 if is_revision else 0)
        return {
            "drafts": [text],
            "current_candidate": text,
            "final_output": text,
            "explanation": parsed.explanation if parsed else "",
            "iteration_count": cur_rev,
            "revision_count": cur_rev,
            "usage": [rec],
            "execution_trace": [rec],
            "feedback": [],
        }

    def reviewer(state: AgentState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        raw = unicodedata.normalize("NFKC", state.get("casual_input") or state.get("raw_input") or "")
        candidate = state.get("current_candidate") or (state.get("drafts") or [raw])[-1]

        if not cfg.use_reviewer:
            return {
                "is_approved": True,
                "status": "unreviewed",
                "feedback": [],
                "critique_history": ["Unreviewed (reviewer disabled)"],
                "rejection_reason": None,
                "final_output": candidate,
            }

        # Stage 1a: Deterministic casual-marker guardrail – reject if text is still casual.
        # This prevents the LLM reviewer from incorrectly approving an unconverted candidate.
        has_casual, casual_violations = audit_casual_markers(candidate)
        if has_casual:
            rejection_desc = f"Candidate still contains casual Japanese: {'; '.join(casual_violations)}"
            issues = [
                {
                    "category": "not_converted",
                    "description": rejection_desc,
                    "suggested_fix": "Convert all casual endings and address forms to appropriate keigo.",
                }
            ]
            rec = {
                "node": "reviewer_casual_guardrail",
                "calls": 0,
                "seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "failed": False,
            }
            return {
                "is_approved": False,
                "status": "needs_review",
                "rejection_reason": rejection_desc,
                "critique_history": [f"Casual-marker guardrail rejected: {rejection_desc}"],
                "feedback": issues,
                "usage": [rec],
                "execution_trace": [rec],
                "final_output": candidate,
            }

        # Stage 1b: Deterministic regex guardrail for Nijuu Keigo (二重敬語) - Zero token fast fail
        has_violations, violations = audit_nijuu_keigo(candidate)
        if has_violations:
            rejection_desc = f"Detected Nijuu Keigo (二重敬語): {'; '.join(violations)}"
            issues = [
                {
                    "category": "double_honorific",
                    "description": rejection_desc,
                    "suggested_fix": "Use standard canonical non-redundant honorific (e.g. お書きになった, おっしゃいました)",
                }
            ]
            rec = {
                "node": "reviewer_regex_guardrail",
                "calls": 0,
                "seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "failed": False,
            }
            return {
                "is_approved": False,
                "status": "needs_review",
                "rejection_reason": rejection_desc,
                "critique_history": [f"Regex guardrail rejected: {rejection_desc}"],
                "feedback": issues,
                "usage": [rec],
                "execution_trace": [rec],
                "final_output": candidate,
            }

        # Stage 2: LLM Evaluation Criteria (Groq Llama 3.3 70B)
        suppletive_rules = state.get("suppletive_rules") or get_suppletive_rules(raw)
        parsed, rec = call_structured(
            _llm(cfg),
            ReviewResult,
            prompts.REVIEWER_SYSTEM,
            prompts.reviewer_user(
                raw,
                state.get("context"),
                candidate,
                state.get("requested_register"),
                suppletive_rules=suppletive_rules,
            ),
            "reviewer",
        )
        if parsed is None:  # do not loop on a broken reviewer
            issue = [{"category": "reviewer_error", "description": rec.get("error", "")}]
            return {
                "is_approved": False,
                "status": "needs_review",
                "rejection_reason": rec.get("error", "reviewer error"),
                "critique_history": [f"Reviewer Error: {rec.get('error', '')}"],
                "feedback": issue,
                "usage": [rec],
                "execution_trace": [rec],
                "final_output": candidate,
            }

        if parsed.approved:
            return {
                "is_approved": True,
                "status": "approved",
                "rejection_reason": None,
                "critique_history": ["Approved"],
                "feedback": [],
                "usage": [rec],
                "execution_trace": [rec],
                "final_output": candidate,
            }

        issues = [i.model_dump() for i in parsed.issues] or [
            {"category": "other", "description": "Reviewer rejected without details", "suggested_fix": ""}
        ]
        rejection_desc = "; ".join(i.get("description", "") for i in issues)
        return {
            "is_approved": False,
            "status": "needs_review",
            "rejection_reason": rejection_desc,
            "critique_history": [f"Iteration critique: {rejection_desc}"],
            "feedback": issues,
            "usage": [rec],
            "execution_trace": [rec],
            "final_output": candidate,
        }

    def conservative_fallback(state: AgentState, config: RunnableConfig):
        """Deterministic conservative fallback to polite Teineigo (〜です / 〜ます)
        Guarantees 100% convergence and zero pipeline failure when max revisions are reached.
        """
        raw = unicodedata.normalize("NFKC", state.get("casual_input") or state.get("raw_input") or "")
        candidate = state.get("current_candidate") or (state.get("drafts") or [raw])[-1]
        fallback_text = convert_to_conservative_teineigo(candidate or raw)
        rec = {
            "node": "conservative_fallback",
            "calls": 0,
            "seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "failed": False,
        }
        return {
            "status": "needs_review",
            "fallback_triggered": True,
            "is_approved": False,
            "final_output": fallback_text,
            "current_candidate": fallback_text,
            "explanation": "Reached maximum critique iterations. Applied deterministic conservative Teineigo fallback.",
            "critique_history": ["Conservative fallback applied"],
            "usage": [rec],
            "execution_trace": [rec],
        }

    def route_after_review(state: AgentState, config: RunnableConfig):
        cfg = RunConfig.from_runnable(config)
        if state.get("is_approved"):
            return "end"
        cur_rev = state.get("iteration_count", state.get("revision_count", 0))
        max_rev = getattr(cfg, "max_revisions", 2)
        if cur_rev >= max_rev or cur_rev >= 3:
            return "fallback"
        feedback = state.get("feedback", [])
        # Immediately fall back for unrecoverable reviewer failures or
        # repeated not_converted (casual-marker guardrail) rejections.
        if any(i.get("category") in ("reviewer_error", "not_converted") for i in feedback):
            return "fallback"
        return "revise"

    return analyzer, retrieve, specialist, reviewer, conservative_fallback, route_after_review

    return analyzer, retrieve, specialist, reviewer, route_after_review
