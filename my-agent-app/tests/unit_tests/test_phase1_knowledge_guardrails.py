"""Unit tests for Phase 1: Suppletive Verbs Lexicon, Domain Templates, and Guardrails."""
from src.agent.guardrails.honorific_rules import audit_nijuu_keigo
from src.agent.knowledge import (
    format_few_shot_exemplars,
    get_domain_template,
    get_suppletive_rules,
    load_domain_templates,
    load_suppletive_verbs,
)


def test_suppletive_verbs_loaded():
    verbs = load_suppletive_verbs().get("verbs", {})
    assert "行く" in verbs
    assert "見る" in verbs
    assert "言う" in verbs
    assert "食べる" in verbs
    assert "参る" in verbs["行く"]["kenjougo"]
    assert "ご覧になる" in verbs["見る"]["sonkeigo"]


def test_get_suppletive_rules_extraction():
    # Test 1: 行く inflection
    rules_iku = get_suppletive_rules("明日、先生のオフィスに行く")
    assert len(rules_iku) >= 1
    assert any("行く" in r and "参る" in r for r in rules_iku)

    # Test 2: 見る inflection
    rules_miru = get_suppletive_rules("社長が書類を見た")
    assert len(rules_miru) >= 1
    assert any("見る" in r and "ご覧になる" in r for r in rules_miru)

    # Test 3: regular verb with no suppletive constraint
    rules_none = get_suppletive_rules("プレゼントを受け取る")
    assert len(rules_none) == 0


def test_domain_templates():
    kb = load_domain_templates().get("domains", {})
    assert "Email" in kb
    assert "Verbal" in kb
    assert "Chat" in kb
    assert "Customer Support" in kb

    email_tmpl = get_domain_template("Email")
    assert email_tmpl is not None
    assert any("お世話になっております" in o for o in email_tmpl["openings"])
    assert any("よろしくお願い申し上げます" in c for c in email_tmpl["closings"])

    verbal_tmpl = get_domain_template("verbal")
    assert verbal_tmpl is not None
    assert any("恐れ入りますが" in s for s in verbal_tmpl["softeners"])


def test_nijuu_keigo_validator():
    # Should detect violations
    has_err, errs = audit_nijuu_keigo("先生がお書きになられた")
    assert has_err
    assert len(errs) > 0

    has_err2, errs2 = audit_nijuu_keigo("部長がおっしゃられました")
    assert has_err2

    has_err3, errs3 = audit_nijuu_keigo("資料をご覧になられましたか？")
    assert has_err3

    # Should pass valid keigo without false positives
    has_err_clean, clean_errs = audit_nijuu_keigo("先生がお書きになりました")
    assert not has_err_clean
    assert len(clean_errs) == 0

    has_err_clean2, _ = audit_nijuu_keigo("部長がおっしゃいました")
    assert not has_err_clean2


def test_few_shot_formatting():
    exemplars = [
        {"source_text": "明日行く", "target_text": "明日伺います", "register": "kenjogo"},
        {"source_text": "本を読む", "target_text": "本をお読みになります", "register": "sonkeigo"},
    ]
    formatted = format_few_shot_exemplars(exemplars)
    assert "Input (Casual): 明日行く -> Output (Formal Keigo) [kenjogo]: 明日伺います" in formatted
    assert "本をお読みになります" in formatted
