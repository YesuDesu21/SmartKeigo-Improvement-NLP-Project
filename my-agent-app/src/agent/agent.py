"""Node implementations. Dependencies (LLM, retriever) are injected so the graph is unit-testable
without network access, and so import never requires API keys."""
import time
import unicodedata
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

from langchain_core.runnables import RunnableConfig

from . import prompts
from .config import RunConfig
from .guardrails.honorific_rules import (audit_honorific_target, audit_nijuu_keigo, audit_casual_markers,
                                         convert_to_conservative_teineigo, fix_title_honorifics, is_already_polite)
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


def _is_format_failure(err: str) -> bool:
    """The model answered but not in the required structure (Groq 'tool_use_failed' / parse errors)."""
    return "tool_use_failed" in err or "parse_error" in err or "did not call a tool" in err


def call_structured(llm, schema, system: str, user: str, node: str, retries: int = 2) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Structured call with usage logging. Failures are recorded, never silently dropped.
    If the model ignores the tool call (a format failure), the next attempt switches to Groq's
    JSON-schema mode, which constrains the output to the schema directly."""
    runner = llm.with_structured_output(schema, include_raw=True)
    rec: Dict[str, Any] = {"node": node, "calls": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "failed": False}
    parsed = None
    for _ in range(retries + 1):
        if _is_format_failure(rec.get("error", "")):
            try:
                runner = llm.with_structured_output(schema, method="json_schema", include_raw=True)
            except TypeError:  # LLM without a json_schema mode (e.g. test fakes): keep the current runner
                pass
        t0 = time.perf_counter()
        try:
            out = runner.invoke([("system", system), ("human", user)])
        except Exception as e:  # network / rate limit / format failure
            rec["calls"] += 1
            rec["seconds"] += time.perf_counter() - t0
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            if "rate_limit" in rec["error"] or "RateLimit" in rec["error"]:
                break  # retrying immediately only burns more of the limit
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
        text = fix_title_honorifics(text)  # deterministic: 部長様 -> 部長 (titles are already honorific)

        # --- Safeguard: reject output that is identical to the casual input.
        #     This catches cases where the LLM parsed successfully but returned
        #     the original text unchanged (silent no-op conversion).
        raw_norm = unicodedata.normalize("NFKC", raw)
        target_reg = state.get("requested_register") or (state.get("context") or {}).get("overall_register")
        # Unchanged output is the correct answer for no_change targets and already-polite input,
        # so only treat it as a failed conversion when a real register change was needed.
        unchanged_ok = target_reg in (None, "no_change", "unknown") or (
            target_reg == "teineigo" and is_already_polite(raw_norm))
        if text == raw_norm and not unchanged_ok:
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

        # Stage 1b: Deterministic regex guardrail for Nijuu Keigo (二重敬語) and sonkeigo on things - zero-token fast fail
        has_violations, violations = audit_nijuu_keigo(candidate)
        wrong_target, target_desc = audit_honorific_target(candidate)
        if wrong_target and not has_violations:
            has_violations, violations = True, target_desc
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
        drafts = [d for d in (state.get("drafts") or []) if d and d.strip()]
        # Prefer the newest LLM draft that passes the deterministic checks: an LLM draft is almost always
        # better keigo than the rule-based converter, which only adds です/ます endings.
        target_reg = state.get("requested_register") or (state.get("context") or {}).get("overall_register")
        keep_as_is = target_reg in ("no_change", None, "unknown")
        clean = [d for d in drafts if not audit_casual_markers(d)[0] and not audit_nijuu_keigo(d)[0]
                 and (is_already_polite(d) or (keep_as_is and d == raw))]
        if clean:
            fallback_text = clean[-1]
            how = "Used the latest draft that passed the rule checks"
        else:
            candidate = state.get("current_candidate") or (drafts or [raw])[-1]
            fallback_text = convert_to_conservative_teineigo(candidate or raw)
            how = "Applied deterministic conservative Teineigo fallback"
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
            "explanation": f"Reached maximum critique iterations. {how}.",
            "critique_history": [f"Conservative fallback applied ({how.lower()})"],
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
        # Immediately fall back only when the reviewer itself failed. A casual-marker rejection
        # (e.g. a bare name 田中、) is easy for the specialist to fix, so it gets a revision like any other issue.
        if any(i.get("category") == "reviewer_error" for i in feedback):
            return "fallback"
        return "revise"

    return analyzer, retrieve, specialist, reviewer, conservative_fallback, route_after_review

    return analyzer, retrieve, specialist, reviewer, route_after_review
