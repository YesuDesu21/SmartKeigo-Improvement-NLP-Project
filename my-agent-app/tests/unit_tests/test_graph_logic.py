"""Offline tests with a scripted fake LLM: loop bound, feedback propagation, ablation flags."""
from types import SimpleNamespace

from src.agent.config import CONDITIONS
from src.agent.graph import build_graph
from src.agent.schemas import AnalysisResult, DraftResult, ReviewResult, Issue


class FakeLLM:
    def __init__(self, approve_on_review=None, log=None):
        self.approve_on_review = approve_on_review  # review number (1-based) that approves; None = never
        self.reviews, self.log = 0, log if log is not None else []

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class R:
            def invoke(self, msgs):
                user = msgs[1][1]
                outer.log.append((schema.__name__, user))
                raw = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5})
                if schema is AnalysisResult:
                    p = AnalysisResult(addressee="client", intent="x", tone="casual", actions=[], overall_register="sonkeigo")
                elif schema is DraftResult:
                    p = DraftResult(keigo_text=f"draft{sum(1 for s, _ in outer.log if s == 'DraftResult')}", explanation="e")
                else:
                    outer.reviews += 1
                    ok = outer.approve_on_review is not None and outer.reviews >= outer.approve_on_review
                    p = ReviewResult(approved=ok, issues=[] if ok else [Issue(category="wrong_direction", description="bad", suggested_fix="fix it")])
                return {"raw": raw, "parsed": p, "parsing_error": None}
        return R()


class FakeRetriever:
    def retrieve(self, q, k, reg):
        return [{"id": "t1", "source_text": "a", "target_text": "b", "register": "teineigo"}]


def run(llm, cond=None, **cfg):
    g = build_graph(lambda m, t: llm, lambda: FakeRetriever())
    conf = {**(CONDITIONS[cond] if cond else {}), **cfg}
    return g.invoke({"raw_input": "明日行く", "drafts": [], "usage": []}, config={"configurable": conf})


def test_approved_first_try():
    out = run(FakeLLM(approve_on_review=1))
    assert out["status"] == "approved" and out["revision_count"] == 0 and len(out["drafts"]) == 1


def test_one_revision_then_approved_and_feedback_reaches_specialist():
    log = []
    out = run(FakeLLM(approve_on_review=2, log=log))
    assert out["status"] == "approved" and out["revision_count"] == 1 and len(out["drafts"]) == 2
    revision_prompts = [u for s, u in log if s == "DraftResult"][1]
    assert "PREVIOUS DRAFT" in revision_prompts and "fix it" in revision_prompts


def test_loop_is_bounded_at_two_revisions():
    out = run(FakeLLM(approve_on_review=None))
    assert out["status"] == "needs_review" and out["revision_count"] == 2 and len(out["drafts"]) == 3


def test_conditions_toggle_nodes():
    a2 = run(FakeLLM(), "A2")
    assert a2["status"] == "unreviewed" and a2["context"] and a2["retrieved_ids"] == ["t1"]
    b1 = run(FakeLLM(), "B1")
    assert b1["context"] == {} and b1["retrieved_ids"] == [] and len(b1["usage"]) == 1  # single specialist call
    a4 = run(FakeLLM(approve_on_review=1), "A4")
    assert a4["retrieved_ids"] == [] and a4["status"] == "approved"
