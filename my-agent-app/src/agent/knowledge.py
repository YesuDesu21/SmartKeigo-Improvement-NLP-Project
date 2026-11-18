"""Knowledge base loader and helpers for suppletive verbs, domain templates, and exemplars."""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _find_kb_file(filename: str) -> Path:
    """Finds knowledge base file across local and repo-root paths."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base" / filename,
        Path(__file__).resolve().parent / "knowledge_base" / filename,
        Path("data/knowledge_base") / filename,
        Path("my-agent-app/data/knowledge_base") / filename,
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


_SUPPLETIVE_CACHE: Optional[Dict[str, Any]] = None
_DOMAIN_CACHE: Optional[Dict[str, Any]] = None

VERB_PATTERNS = {
    "行く": r"(行[くきかないったってこけ]|いく|いった|いって)",
    "来る": r"(来[るますたてないよう]|くる|きた|きて|こない|こよう)",
    "いる": r"(?<![ぁ-んァ-ヶー一-龠])いる|(?<![ぁ-んァ-ヶー一-龠])いた|(?<![ぁ-んァ-ヶー一-龠])いて|にいる",
    "言う": r"(言[ういわないったっておえ]|いう|いった|いって)",
    "する": r"(する|しない|します|した|して|しよう|すれ)",
    "食べる": r"(食[べるべますべたべてべないべよう]|たべる|たべた)",
    "飲む": r"(飲[むみますんだんでまないもう]|のむ|のんだ)",
    "見る": r"(見[るますたてないようれ]|みる|みた|みて)",
    "知る": r"(知[るりますったってらないろう]|しる|しった|しって)",
    "会う": r"(会[ういわないったっておえ]|あう|あった|あって)",
    "聞く": r"(聞[くきかないいたいてこけ]|きく|きいた|きいて)",
    "尋ねる": r"(尋[ねるねますねたねてねない]|たずねる)",
    "訪問する": r"(訪問[するしたしてしない])",
}


def load_suppletive_verbs() -> Dict[str, Any]:
    global _SUPPLETIVE_CACHE
    if _SUPPLETIVE_CACHE is None:
        path = _find_kb_file("suppletive_verbs.json")
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                _SUPPLETIVE_CACHE = json.load(f)
        else:
            _SUPPLETIVE_CACHE = {"verbs": {}}
    return _SUPPLETIVE_CACHE


def load_domain_templates() -> Dict[str, Any]:
    global _DOMAIN_CACHE
    if _DOMAIN_CACHE is None:
        path = _find_kb_file("domain_templates.json")
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                _DOMAIN_CACHE = json.load(f)
        else:
            _DOMAIN_CACHE = {"domains": {}}
    return _DOMAIN_CACHE


def get_suppletive_rules(text: str) -> List[str]:
    """Extracts canonical suppletion constraints (補充法) for irregular verbs in the input.
    Prevents hallucinated regularized honorifics (e.g. *お行きになる).
    """
    kb = load_suppletive_verbs().get("verbs", {})
    rules = []
    for verb, pattern in VERB_PATTERNS.items():
        if re.search(pattern, text):
            vinfo = kb.get(verb, {})
            sonkeigo = vinfo.get("sonkeigo", [])
            kenjougo = vinfo.get("kenjougo", [])
            s_str = "、".join(sonkeigo) if sonkeigo else "なし（一般敬語表現を使用）"
            k_str = "、".join(kenjougo) if kenjougo else "なし（一般敬語表現を使用）"
            rules.append(
                f"【不規則動詞「{verb}」の補充法制約】尊敬語（相手側の動作）: {s_str} | 謙譲語（自分・身内の動作）: {k_str} "
                f"※「*お{verb}になる」「*お{verb}する」等の不自然な規則化・二重敬語は厳禁"
            )
    return rules


def get_domain_template(domain: Optional[str]) -> Optional[Dict[str, Any]]:
    """Retrieves structured template and tone guidelines for a workplace domain."""
    if not domain:
        return None
    domains = load_domain_templates().get("domains", {})
    # Case-insensitive or partial match
    norm = domain.strip().lower()
    for key, data in domains.items():
        if key.lower() == norm or norm in key.lower() or norm in data.get("display_name", "").lower():
            return data
    return None


def format_few_shot_exemplars(exemplars: List[Dict[str, str]]) -> str:
    """Formats retrieved exemplars into clean few-shot prompt text."""
    if not exemplars:
        return ""
    lines = []
    for e in exemplars:
        src = e.get("source_text", "").strip()
        tgt = e.get("target_text", "").strip()
        reg = e.get("register", "")
        reg_tag = f" [{reg}]" if reg else ""
        lines.append(f"- Input (Casual): {src} -> Output (Formal Keigo){reg_tag}: {tgt}")
    return "\n".join(lines)
