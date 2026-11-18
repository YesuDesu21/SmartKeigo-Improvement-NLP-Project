"""Deterministic guardrails for Japanese honorific speech (敬語).
Includes detection for double honorifics (二重敬語) and redundant forms.
"""
import re
from typing import List, Tuple

NIJUU_KEIGO_PATTERNS = [
    # Double honorific: お/ご + stem + になられる (e.g. お書きになられた, お書きになられました)
    (
        r"(お|ご)[ぁ-んァ-ヶー一-龠]{1,6}になられ(る|ます|まし|た|ない|ば|て)",
        "お/ご〜になられる (Double Honorific: use お/ご〜になる/〜になられた)",
    ),
    # Irregular verb double honorifics
    (
        r"おっしゃられ(る|ます|まし|た|ない|ば|て)",
        "おっしゃられる (Double Honorific on 言う: use おっしゃる/おっしゃいました)",
    ),
    (
        r"ご覧になられ(る|ます|まし|た|ない|ば|て)",
        "ご覧になられる (Double Honorific on 見る: use ご覧になる/ご覧になりました)",
    ),
    (
        r"お召し上がりになられ(る|ます|まし|た|ない|ば|て)",
        "お召し上がりになられる (Double Honorific on 食べる/飲む: use 召し上がる/召し上がりました)",
    ),
    (
        r"いらっしゃられ(る|ます|まし|た|ない|ば|て)",
        "いらっしゃられる (Double Honorific on 行く/来る/いる: use いらっしゃる/いらっしゃいました)",
    ),
    # Redundant Kenjougo combinations
    (
        r"お伺いいた(す|します|した|して)",
        "お伺いいたす (Redundant Kenjougo: use 伺います)",
    ),
    (
        r"ご案内いた(す|します|した|して)",
        "ご案内いたす (Potential redundancy: use ご案内します/ご案内申し上げます)",
    ),
]

# Patterns that definitively mark text as still casual/informal.
# Any match means the candidate has NOT been converted to keigo.
CASUAL_MARKERS = [
    (r"じゃダメ(か|？|\?)?",     "じゃダメ – casual negotiation; use ではいけませんか/ではご都合が悪いでしょうか"),
    (r"(んだよね|んだよ|んだな)(\s|。|$)", "〜んだよね/よ/な – casual explanatory ending; use 〜のです/〜のですが"),
    (r"(たいんだよ|たいんだ)(\s|。|$)", "〜たいんだよ – casual desiderative; use 〜たく存じます/〜たいと存じます"),
    (r"(だよ|だぜ|だな|だよな)(\s|。|$)", "〜だよ/ぜ/な – casual copula ending; use です"),
    (r"(するの|したの|行くの|来るの)\?",  "〜するの？ – casual question; use 〜されますか/〜なさいますか"),
    # Bare surname (kanji 2-4 chars) at start, addressed without さん/様
    (r"^[一-龠]{2,4}(、|,|\s+)(?!さん|様|先生)",
     "Bare surname without honorific suffix (呼び捨て); must add さん or 様"),
]


def audit_nijuu_keigo(text: str) -> Tuple[bool, List[str]]:
    """Audits text for deterministic Nijuu Keigo (二重敬語) violations.
    
    Returns:
        (has_violations, list_of_violation_descriptions)
    """
    violations = []
    for pattern, desc in NIJUU_KEIGO_PATTERNS:
        if re.search(pattern, text):
            violations.append(desc)
    return len(violations) > 0, violations


def audit_casual_markers(text: str) -> Tuple[bool, List[str]]:
    """Detect clearly casual language that proves the text was NOT converted to keigo.

    Returns:
        (has_casual, list_of_descriptions)
    """
    found = []
    for pattern, desc in CASUAL_MARKERS:
        if re.search(pattern, text):
            found.append(desc)
    return len(found) > 0, found




def audit_nijuu_keigo(text: str) -> Tuple[bool, List[str]]:
    """Audits text for deterministic Nijuu Keigo (二重敬語) violations.
    
    Returns:
        (has_violations, list_of_violation_descriptions)
    """
    violations = []
    for pattern, desc in NIJUU_KEIGO_PATTERNS:
        if re.search(pattern, text):
            violations.append(desc)
    return len(violations) > 0, violations


def convert_to_conservative_teineigo(text: str) -> str:
    """Deterministic, robust fallback converter to standard polite Teineigo (〜です / 〜ます).
    Used as an emergency conservative fallback to guarantee 100% convergence without pipeline failure.
    """
    text = text.strip()
    if not text:
        return text

    # 1. Clean common double honorifics if present
    text = re.sub(r"おっしゃられ(る|ます|ました|た)", r"おっしゃいました", text)
    text = re.sub(r"ご覧になられ(る|ます|ました|た)", r"ご覧になりました", text)
    text = re.sub(r"お召し上がりになられ(る|ます|ました|た)", r"召し上がりました", text)
    text = re.sub(r"いらっしゃられ(る|ます|ました|た)", r"いらっしゃいました", text)
    text = re.sub(r"(お|ご)([ぁ-んァ-ヶー一-龠]{1,6})になられ(る|ます|ました|た)", r"\1\2になりました", text)

    # 2. Add 'さん' to bare Japanese surname/name at beginning if followed by punctuation/space
    # e.g. 佐藤、 -> 佐藤さん、 | 田中、 -> 田中さん、
    text = re.sub(r"^([一-龠]{2,4})(、|,|\s+)", r"\1さん\2", text)

    # 3. Handle casual negotiation 'じゃダメ？' -> 'ではいけませんか？'
    text = re.sub(r"じゃダメ(\?|？)", r"ではいけませんか\1", text)
    text = re.sub(r"じゃダメ", r"ではいけません", text)

    # 4. If already ending in polite form, ensure single trailing punctuation
    if re.search(r"(です|ます|でした|ました|ません|ございます|ください)[。！!？?]*$", text):
        if not re.search(r"[。！!？?]$", text):
            text += "。"
        return text

    # 5. Extract and strip trailing punctuation
    orig_punct = "。"
    m_p = re.search(r"([。！!？?]+)$", text)
    if m_p:
        orig_punct = m_p.group(1)[-1]
        text = text[: m_p.start()].strip()

    # 6. Handle casual explanatory/emotional sentence endings (〜んだよね, 〜んだよ, 〜んだ)
    text = re.sub(r"(んだよね|んだよ|んだな|んだ)$", r"のです", text)
    # Clean trailing casual particles (な, ぜ, ぞ, さ, ね, よ, よね)
    text = re.sub(r"(よね|よ|ね|な|ぜ|ぞ|さ)$", "", text).strip()

    if re.search(r"(です|ます|でした|ました|ません|ございます|ください)$", text):
        return text + orig_punct

    # 7. Volitional ending (ろう/よう -> ましょう)
    if re.search(r"([ぁ-んァ-ヶー一-龠]+)ろう$", text):
        return re.sub(r"ろう$", "りましょう", text) + orig_punct
    if re.search(r"([ぁ-んァ-ヶー一-龠]+)よう$", text):
        return re.sub(r"よう$", "いましょう", text) + orig_punct

    # 8. Plain copula
    if text.endswith("だ"):
        return text[:-1] + "です" + orig_punct
    if text.endswith("だった"):
        return text[:-3] + "でした" + orig_punct

    # 9. Verbs and Desideratives (-たい)
    if text.endswith("たい"):
        return text + "です" + orig_punct
    if text.endswith("ている"):
        return text[:-3] + "ています" + orig_punct
    if text.endswith("てある"):
        return text[:-3] + "てあります" + orig_punct
    if text.endswith("ておく"):
        return text[:-3] + "ておきます" + orig_punct
    if text.endswith("ていく"):
        return text[:-3] + "ていきます" + orig_punct
    if text.endswith("てくる"):
        return text[:-3] + "てきます" + orig_punct
    if text.endswith("くる"):
        return text[:-2] + "きます" + orig_punct
    if text.endswith("する"):
        return text[:-2] + "します" + orig_punct
    if text.endswith("した"):
        return text[:-2] + "しました" + orig_punct

    # Godan / Ichidan standard endings
    verb_endings = [
        ("う", "います"),
        ("く", "きます"),
        ("ぐ", "ぎます"),
        ("す", "します"),
        ("つ", "ちます"),
        ("ぬ", "にます"),
        ("ぶ", "びます"),
        ("む", "みます"),
        ("る", "ます"),
        ("た", "ました"),
    ]
    for v_end, t_end in verb_endings:
        if text.endswith(v_end):
            return text[:-len(v_end)] + t_end + orig_punct

    return text + "です" + orig_punct

