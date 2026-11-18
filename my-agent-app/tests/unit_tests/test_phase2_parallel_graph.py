"""Unit tests for Phase 2: LangGraph State Schema & Graph Parallelism."""
from types import SimpleNamespace

from src.agent.config import CONDITIONS
from src.agent.graph import build_graph
from src.agent.schemas import AnalysisResult, DraftResult, Issue, ReviewResult


class ParallelFakeLLM:
    def __init__(self, approve_on_review=1):
        self.approve_on_review = approve_on_review
        self.reviews = 0
        self.recorded_prompts = []

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class R:
            def invoke(self, msgs):
                user = msgs[1][1]
                outer.recorded_prompts.append((schema.__name__, user))
                raw = SimpleNamespace(usage_metadata={"input_tokens": 15, "output_tokens": 10})
                if schema is AnalysisResult:
                    p = AnalysisResult(
                        addressee="superior",
                        intent="report to teacher",
                        tone="casual",
                        actions=[],
                        overall_register="kenjogo",
                    )
                elif schema is DraftResult:
                    p = DraftResult(keigo_text="先生のオフィスに伺います。", explanation="伺う is suppletive kenjogo for 行く.")
                else:
                    outer.reviews += 1
                    ok = outer.reviews >= outer.approve_on_review
                    p = ReviewResult(
                        approved=ok,
                        issues=[] if ok else [Issue(category="wrong_direction", description="Need humble form", suggested_fix="use 伺う")],
                    )
                return {"raw": raw, "parsed": p, "parsing_error": None}

        return R()


class ParallelFakeRetriever:
    def __init__(self):
        self.called = False

    def retrieve(self, q, k, reg):
        self.called = True
        return [{"id": "t100", "source_text": "オフィスに行く", "target_text": "オフィスに伺います", "register": "kenjogo"}]


def test_parallel_fanout_execution():
    fake_retriever = ParallelFakeRetriever()
    fake_llm = ParallelFakeLLM(approve_on_review=1)
    g = build_graph(lambda m, t: fake_llm, lambda: fake_retriever)

    # Invoke with casual_input and target_domain
    out = g.invoke(
        {
            "casual_input": "明日、先生のオフィスに行く",
            "target_domain": "Email",
            "drafts": [],
            "usage": [],
            "execution_trace": [],
        }
    )

    # 1. Check parallel retrieval ran
    assert fake_retriever.called
    assert out["retrieved_ids"] == ["t100"]

    # 2. Check analyzer ran and extracted hierarchy & suppletive rules
    assert out["hierarchy_metadata"]["addressee"] == "superior"
    assert len(out["suppletive_rules"]) >= 1
    assert any("行く" in r and "伺う" in r for r in out["suppletive_rules"])

    # 3. Check specialist synthesized both branches
    specialist_calls = [u for s, u in fake_llm.recorded_prompts if s == "DraftResult"]
    assert len(specialist_calls) >= 1
    # Check that suppletive rules were fed into specialist prompt
    assert "CANONICAL SUPPLETIVE VERB CONSTRAINTS" in specialist_calls[0]
    assert "行く" in specialist_calls[0]
    # Check that domain guidelines were fed into specialist prompt
    assert "WORKPLACE DOMAIN GUIDELINES" in specialist_calls[0]

    # 4. Check approval and output convergence
    assert out["is_approved"] is True
    assert out["status"] == "approved"
    assert out["current_candidate"] == "先生のオフィスに伺います。"
    assert out["final_output"] == "先生のオフィスに伺います。"
    assert out["iteration_count"] == 0


def test_unicode_nfkc_normalization():
    fake_llm = ParallelFakeLLM(approve_on_review=1)
    g = build_graph(lambda m, t: fake_llm, lambda: ParallelFakeRetriever())

    # Input with half-width katakana: ﾃｽﾄ (test) -> should normalize to テスト
    out = g.invoke(
        {
            "casual_input": "ﾃｽﾄに行く",
            "drafts": [],
            "usage": [],
        }
    )
    # casual_input and raw_input should be normalized to NFKC
    assert out["casual_input"] == "テストに行く"
    assert out["raw_input"] == "テストに行く"


def test_backward_compatibility_with_raw_input():
    fake_llm = ParallelFakeLLM(approve_on_review=1)
    g = build_graph(lambda m, t: fake_llm, lambda: ParallelFakeRetriever())

    # Legacy invocation using raw_input and scenario
    out = g.invoke(
        {
            "raw_input": "明日行く",
            "scenario": "取引先の部長へのメール",
            "requested_register": "kenjogo",
            "drafts": [],
            "usage": [],
        }
    )
    assert out["status"] == "approved"
    assert len(out["drafts"]) == 1
    assert out["revision_count"] == 0
