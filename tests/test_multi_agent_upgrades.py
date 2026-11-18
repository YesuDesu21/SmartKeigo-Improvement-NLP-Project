"""Comprehensive Test Suite for Multi-Agent System Upgrades (Phase 4.1).
Covers the 5 canonical linguistic edge cases and acceptance criteria.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

# Ensure my-agent-app is on path
app_path = str(Path(__file__).resolve().parent.parent / "my-agent-app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from src.agent.guardrails.honorific_rules import audit_nijuu_keigo
from src.agent.graph import build_graph
from src.agent.knowledge import get_suppletive_rules, load_suppletive_verbs
from src.agent.schemas import AnalysisResult, DraftResult, Issue, ReviewResult


class EdgeCaseFakeLLM:
    """Scripted fake LLM simulating specialist and reviewer across edge cases."""

    def __init__(self, mode="suppletion"):
        self.mode = mode
        self.iteration = 0
        self.recorded_prompts = []

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class R:
            def invoke(self, msgs):
                user = msgs[1][1]
                outer.recorded_prompts.append((schema.__name__, user))
                raw = SimpleNamespace(usage_metadata={"input_tokens": 20, "output_tokens": 15})

                if schema is AnalysisResult:
                    if outer.mode == "directionality":
                        p = AnalysisResult(
                            addressee="client",
                            intent="report that the president reviewed document",
                            tone="casual",
                            actions=[],
                            overall_register="sonkeigo",
                        )
                    else:
                        p = AnalysisResult(
                            addressee="superior",
                            intent="communicate with teacher",
                            tone="casual",
                            actions=[],
                            overall_register="kenjogo",
                        )
                elif schema is DraftResult:
                    outer.iteration += 1
                    if outer.mode == "suppletion":
                        p = DraftResult(
                            keigo_text="明日、先生のオフィスに伺います。",
                            explanation="Suppletive kenjougo 伺う used for 行く.",
                        )
                    elif outer.mode == "directionality":
                        p = DraftResult(
                            keigo_text="社長が書類をご覧になりました。",
                            explanation="Sonkeigo ご覧になる used because action was performed by superior (社長).",
                        )
                    elif outer.mode == "nijuu_keigo":
                        if outer.iteration == 1:
                            # First draft produces a double honorific intentionally
                            p = DraftResult(
                                keigo_text="先生がお書きになられました。",
                                explanation="Attempt with honorific.",
                            )
                        else:
                            # Second draft fixes it after critique
                            p = DraftResult(
                                keigo_text="先生がお書きになりました。",
                                explanation="Corrected double honorific to standard sonkeigo.",
                            )
                    elif outer.mode == "email_domain":
                        p = DraftResult(
                            keigo_text="いつも大変お世話になっております。プロジェクトの進捗をご報告いたします。何卒よろしくお願い申し上げます。",
                            explanation="Email business frame with formal opening and closing salutations.",
                        )
                    else:  # loop_bound
                        p = DraftResult(
                            keigo_text="難しい文章です",
                            explanation="Stubborn candidate.",
                        )
                else:  # ReviewResult
                    if outer.mode == "nijuu_keigo":
                        ok = outer.iteration >= 2
                        p = ReviewResult(
                            approved=ok,
                            issues=[]
                            if ok
                            else [Issue(category="double_honorific", description="Avoid double honorific", suggested_fix="use お書きになりました")],
                        )
                    elif outer.mode == "loop_bound":
                        p = ReviewResult(
                            approved=False,
                            issues=[Issue(category="unnatural", description="Never satisfactory")],
                        )
                    else:
                        p = ReviewResult(approved=True, issues=[])

                return {"raw": raw, "parsed": p, "parsing_error": None}

        return R()


class DummyRetriever:
    def retrieve(self, q, k, reg):
        return []


# =====================================================================
# Canonical Linguistic Edge-Case Test Suite (Checklist Section 4.1)
# =====================================================================


def test_edge_case_1_suppletion():
    """Test 1 (Suppletion): 明日、先生のオフィスに行く -> Must produce 伺う / 参る, never *お行きになる."""
    rules = get_suppletive_rules("明日、先生のオフィスに行く")
    assert any("行く" in r and "伺う" in r for r in rules)

    fake_llm = EdgeCaseFakeLLM(mode="suppletion")
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke({"casual_input": "明日、先生のオフィスに行く", "drafts": [], "usage": []})

    final_text = out["final_output"]
    assert "伺" in final_text or "参" in final_text
    assert "お行きにな" not in final_text
    assert "お行きする" not in final_text
    assert out["status"] == "approved"


def test_edge_case_2_honorific_directionality():
    """Test 2 (Honorific Directionality): 社長が書類を見た -> Must produce ご覧になった, never 拝見した."""
    rules = get_suppletive_rules("社長が書類を見た")
    assert any("見る" in r and "ご覧になる" in r for r in rules)

    fake_llm = EdgeCaseFakeLLM(mode="directionality")
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke({"casual_input": "社長が書類を見た", "drafts": [], "usage": []})

    final_text = out["final_output"]
    assert "ご覧に" in final_text
    assert "拝見" not in final_text
    assert out["status"] == "approved"


def test_edge_case_3_nijuu_keigo_detection_and_correction():
    """Test 3 (Nijuu Keigo Detection): Candidate 先生がお書きになられた -> Must be rejected and corrected to お書きになった."""
    fake_llm = EdgeCaseFakeLLM(mode="nijuu_keigo")
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke({"casual_input": "先生が書く", "drafts": [], "usage": []})

    assert len(out["drafts"]) == 2
    assert "お書きになられました" in out["drafts"][0]
    assert "お書きになりました" in out["final_output"]
    has_err, _ = audit_nijuu_keigo(out["final_output"])
    assert not has_err
    assert out["status"] == "approved"
    assert out["iteration_count"] == 1


def test_edge_case_4_domain_salutation():
    """Test 4 (Domain Salutation): Input with domain Email -> Must include business opening/closing."""
    fake_llm = EdgeCaseFakeLLM(mode="email_domain")
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke(
        {
            "casual_input": "プロジェクトの進捗を報告する",
            "target_domain": "Email",
            "drafts": [],
            "usage": [],
        }
    )

    final_text = out["final_output"]
    assert "いつも大変お世話になっております" in final_text
    assert "よろしくお願い申し上げます" in final_text
    assert out["status"] == "approved"


def test_edge_case_5_loop_bound_and_convergence():
    """Test 5 (Loop Bound): Adversarial / ambiguous input -> Must terminate within <= 3 iterations."""
    fake_llm = EdgeCaseFakeLLM(mode="loop_bound")
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke(
        {"casual_input": "これは難しい問題だ", "drafts": [], "usage": []},
        config={"configurable": {"max_revisions": 2}},
    )

    assert out["iteration_count"] <= 3
    assert out["fallback_triggered"] is True
    assert out["final_output"].endswith("です。") or out["final_output"].endswith("ます。")
    assert len(out["final_output"]) > 0


def test_acceptance_criteria_13_irregular_verb_lexicon():
    """Acceptance: 0% unhandled irregular verb roots across 13 core suppletive items."""
    verbs = load_suppletive_verbs().get("verbs", {})
    expected_roots = [
        "行く", "来る", "いる", "言う", "する", "食べる", "飲む", "見る", "知る", "会う", "聞く", "尋ねる", "訪問する"
    ]
    for root in expected_roots:
        assert root in verbs
        assert len(verbs[root]["kenjougo"]) > 0 or len(verbs[root]["sonkeigo"]) > 0
