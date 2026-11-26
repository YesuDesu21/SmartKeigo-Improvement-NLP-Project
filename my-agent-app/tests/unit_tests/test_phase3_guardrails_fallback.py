"""Unit tests for Phase 3: Linguistic Guardrails, Critique Loop & Conservative Fallback."""
from types import SimpleNamespace

from src.agent.guardrails.honorific_rules import convert_to_conservative_teineigo
from src.agent.graph import build_graph
from src.agent.schemas import AnalysisResult, DraftResult, Issue, ReviewResult


class CountingFakeLLM:
    def __init__(self, draft_text="先生がお書きになられた", approve_on_review=None):
        self.draft_text = draft_text
        self.approve_on_review = approve_on_review
        self.reviewer_calls = 0

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class R:
            def invoke(self, msgs):
                raw = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5})
                if schema is AnalysisResult:
                    p = AnalysisResult(
                        addressee="superior",
                        intent="test",
                        tone="casual",
                        actions=[],
                        overall_register="sonkeigo",
                    )
                elif schema is DraftResult:
                    p = DraftResult(keigo_text=outer.draft_text, explanation="test")
                else:
                    outer.reviewer_calls += 1
                    ok = outer.approve_on_review is not None and outer.reviewer_calls >= outer.approve_on_review
                    p = ReviewResult(approved=ok, issues=[] if ok else [Issue(category="wrong_direction", description="bad")])
                return {"raw": raw, "parsed": p, "parsing_error": None}

        return R()


class DummyRetriever:
    def retrieve(self, q, k, reg):
        return []


def test_dual_stage_reviewer_regex_fast_fail_zero_llm_tokens():
    # Candidate contains double honorific: お書きになられた
    fake_llm = CountingFakeLLM(draft_text="先生がお書きになられた", approve_on_review=1)
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke(
        {"casual_input": "先生が書く", "drafts": [], "usage": []},
        config={"configurable": {"max_revisions": 0}},  # test single review stage
    )

    # 1. Check that regex guardrail caught the violation
    assert out["is_approved"] is False
    assert "Nijuu Keigo" in out["rejection_reason"]
    assert any("お/ご〜になられる" in f["description"] for f in out["feedback"])

    # 2. Check zero LLM tokens spent on review: fake_llm.reviewer_calls MUST be 0!
    assert fake_llm.reviewer_calls == 0


def test_dual_stage_reviewer_clean_candidate_calls_llm():
    # Clean candidate without double honorific: お書きになりました
    fake_llm = CountingFakeLLM(draft_text="先生がお書きになりました", approve_on_review=1)
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke(
        {"casual_input": "先生が書く", "drafts": [], "usage": []},
        config={"configurable": {"max_revisions": 0}},
    )

    # Regex guardrail passed, LLM reviewer was called and approved
    assert fake_llm.reviewer_calls == 1
    assert out["is_approved"] is True
    assert out["status"] == "approved"


def test_conservative_fallback_node_triggered_on_max_revisions():
    # Never approve -> must trigger conservative fallback
    fake_llm = CountingFakeLLM(draft_text="明日、先生のオフィスに行く", approve_on_review=None)
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())

    out = g.invoke(
        {"casual_input": "明日、先生のオフィスに行く", "drafts": [], "usage": []},
        config={"configurable": {"max_revisions": 2}},
    )

    # Check fallback was triggered
    assert out["fallback_triggered"] is True
    assert out["status"] == "needs_review"
    # Final output was converted to polite Teineigo
    assert out["final_output"].endswith("行きます。") or out["final_output"].endswith("行きます")
    assert "Applied deterministic conservative Teineigo fallback" in out["explanation"]


def test_convert_to_conservative_teineigo_cases():
    assert convert_to_conservative_teineigo("明日行く").endswith("行きます。")
    assert convert_to_conservative_teineigo("了解した").endswith("しました。")
    assert convert_to_conservative_teineigo("寝ている").endswith("寝ています。")
    assert convert_to_conservative_teineigo("本を持ってくる").endswith("持ってきます。")
    assert convert_to_conservative_teineigo("先生だ").endswith("先生です。")
    # Cleans double honorifics to standard polite
    assert "お書きになりました" in convert_to_conservative_teineigo("先生がお書きになられた")
    assert "おっしゃいました" in convert_to_conservative_teineigo("部長がおっしゃられました")
    # Volitional to polite volitional
    assert "頑張りましょう！" in convert_to_conservative_teineigo("こんにちは、先生、今日も頑張ろうな！")
    # Bare surname and casual negotiation
    sato_result = convert_to_conservative_teineigo("佐藤、そのレポートは明日じゃダメ？今日はもう帰りたいんだよね。")
    assert "佐藤さん、" in sato_result
    assert "ではいけませんか？" in sato_result
    assert "帰りたいのです。" in sato_result
    assert "。 です。" not in sato_result


def test_conservative_teineigo_keeps_polite_questions_and_titles():
    # A polite LLM draft ending in でしょうか must not get an extra です, and 先生 must not become 先生さん.
    draft = "先生、こんにちは。今週の宿題の期限をお伺いしたいのですが、いつでしょうか？"
    assert convert_to_conservative_teineigo(draft) == draft
    for ending in ("よろしいですか？", "お願いできますか？", "いかがでしょう。", "ご確認ください。", "参りましょうか？"):
        assert convert_to_conservative_teineigo(ending) == ending
    assert convert_to_conservative_teineigo("社長、資料をお持ちしました").startswith("社長、")
    assert convert_to_conservative_teineigo("明日、会議に行く").startswith("明日、")
    assert convert_to_conservative_teineigo("田中、資料を準備した").startswith("田中さん、")


def test_conservative_teineigo_drops_oi_and_handles_kke_questions():
    out = convert_to_conservative_teineigo("おい、こんにちは先生。今週の宿題の期限はいつだっけ？")
    assert out == "こんにちは先生。今週の宿題の期限はいつでしたでしょうか？"


def test_is_already_polite():
    from src.agent.guardrails.honorific_rules import is_already_polite
    assert not is_already_polite("昨日送ってくれた資料をもう見たから、あとで電話で連絡するね")
    assert not is_already_polite("資料を見ました。あとで電話するね。")
    assert is_already_polite("資料を拝見しました。後ほどお電話いたします。")
    assert is_already_polite("よろしいでしょうか？")


def test_casual_marker_guardrail_allows_titles_but_flags_bare_names():
    from src.agent.guardrails.honorific_rules import audit_casual_markers
    assert audit_casual_markers("先生、こんにちは。お伺いしたいのですが、今週の宿題の期限はいつでしょうか。") == (False, [])
    assert audit_casual_markers("社長、資料をお持ちしました。") == (False, [])
    assert audit_casual_markers("田中、資料はもう準備しましたか。")[0]
    assert not audit_casual_markers("田中さん、資料はもう準備しましたか。")[0]


def test_unchanged_polite_input_is_not_a_fallback():
    # Already-polite input, target no_change: returning it unchanged is correct, not a failure.
    fake_llm = CountingFakeLLM(draft_text="資料を拝見しました。", approve_on_review=1)
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())
    out = g.invoke({"casual_input": "資料を拝見しました。", "requested_register": "no_change", "drafts": [], "usage": []},
                   config={"configurable": {"max_revisions": 2}})
    assert not out.get("fallback_triggered")
    assert out["final_output"] == "資料を拝見しました。"


def test_fallback_keeps_clean_llm_draft():
    # Reviewer never approves, but the draft passes the rule checks: keep the LLM draft, not the rule-based one.
    draft = "明日、先生のオフィスに伺います。"
    fake_llm = CountingFakeLLM(draft_text=draft, approve_on_review=None)
    g = build_graph(lambda m, t: fake_llm, lambda: DummyRetriever())
    out = g.invoke({"casual_input": "明日、先生のオフィスに行く", "drafts": [], "usage": []},
                   config={"configurable": {"max_revisions": 2}})
    assert out["fallback_triggered"] is True
    assert out["final_output"] == draft
    assert "latest draft" in out["explanation"]


def test_format_failure_retries_with_json_schema():
    from src.agent.agent import call_structured
    calls = []

    class FlakyLLM:
        def with_structured_output(self, schema, include_raw=True, method="function_calling"):
            class R:
                def invoke(self, msgs):
                    calls.append(method)
                    if method != "json_schema":
                        raise RuntimeError("Error code: 400 - tool_use_failed: model did not call a tool")
                    return {"raw": SimpleNamespace(usage_metadata={}),
                            "parsed": DraftResult(keigo_text="伺います。", explanation=""), "parsing_error": None}
            return R()

    parsed, rec = call_structured(FlakyLLM(), DraftResult, "s", "u", "specialist")
    assert parsed.keigo_text == "伺います。"
    assert calls == ["function_calling", "json_schema"]
    assert not rec["failed"] and "error" not in rec


def test_conservative_teineigo_leaves_comma_fragment():
    s = "原因は友達が不定時に猫に餌をやるからで、プリンはお腹がすきすぎて、"
    assert convert_to_conservative_teineigo(s) == s
