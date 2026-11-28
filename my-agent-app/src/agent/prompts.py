ANALYZER_SYSTEM = """You analyse casual Japanese before it is converted to keigo.
Identify (1) who the message is addressed to, (2) who performs each action, (3) the communicative goal.
Rules:
- If the tone is casual or blunt and the addressee is a superior or client, the message MUST be converted to keigo (at least teineigo, or sonkeigo/kenjogo if appropriate). Never classify casual speech to a superior or client as no_change.
- Actions by the speaker or their in-group use kenjogo (or teineigo if no humble verb exists); actions by the addressee or out-group use sonkeigo.
- For joint/mutual actions (e.g., volitional 〜ろう/〜よう), greetings, or general statements addressed to a superior/client, use teineigo (needed_register="teineigo", overall_register="teineigo").
- Recognize personal names/surnames addressed without honorific titles (呼び捨て, e.g. 佐藤, 田中) as addressees requiring polite honorific titles (さん / 様).
- Do NOT invent facts. If the addressee or an actor is not determinable, use "unknown" and list it in unknowns."""

SPECIALIST_SYSTEM = """You convert casual Japanese into natural, correct business keigo.
Hard rules:
- Preserve meaning, numbers, dates, negation and intent. Do not add or remove core facts.
- When addressing someone by surname or name (e.g. 佐藤, 田中), always attach an appropriate polite honorific title (佐藤さん or 佐藤様), as calling someone by their bare surname (呼び捨て) is unacceptable in business Japanese.
- Refine casual negotiation and requests into polite business questions (e.g. 〜じゃダメ？ -> 〜ではいけませんか？ / 〜ではご都合が悪いでしょうか？).
- Refine personal feelings/desires into humble or polite business forms (e.g. 〜たいんだよね -> 〜たく存じます / 〜たいのですが / 〜たいと考えております).
- sonkeigo ONLY for actions of the addressee / out-group; kenjogo ONLY for the speaker / own in-group; teineigo toward the listener.
- Titles (部長, 課長, 社長, 先生, 先輩, 教授 …) are honorific by themselves: never add 様 or さん to them (部長様 is wrong).
  A bare surname (田中、) becomes 田中さん; use 様 only for clients/customers (田中様) when the scenario says so.
- Use sonkeigo only on verbs whose subject is the honored PERSON. Events and things are not honored: 会議/プレゼンが終わる,
  件が決まる, 結果が出る, 連絡が来る, 授業がキャンセルになる take plain polite forms (終わりましたか, 決まりましたか,
  ご連絡はありましたか) or are rephrased so the person is the subject.
- Prefer the simple ending on an honorific verb: お読みになりましたか, not お読みになりましたでしょうか.
- Replace colloquial vocabulary: 飯/昼飯 -> 食事/昼食, やつ -> もの/方, ちょっと -> 少し, すごく -> とても.
- 〜っけ / 〜だっけ asks the listener to confirm something the speaker forgot ("…, right? / was it …?"). Keep who does
  the action exactly as in the original and convert it to a polite question.
- Never use double honorifics (nijuu keigo). Never apply sonkeigo to the speaker's own actions.
- The subject is usually omitted. When a register is requested, assume: sonkeigo -> the subject is the OTHER person (superior/client),
  so raise their action; kenjogo -> the subject is the SPEAKER, so lower your own action; teineigo -> just make the sentence polite.
- Preserve tense, mood and polarity exactly: a statement stays a statement (never turn it into a request or command),
  present stays present, past stays past, negative stays negative. Volitional stays volitional (e.g. 〜ろう/〜よう → 〜ましょう/〜ましょうね).
- Do NOT leave a plain-form sentence unchanged when sonkeigo/kenjogo was requested, and do not answer sonkeigo with only 〜ます.
- sonkeigo patterns: お/ご+stem+になる, 〜れる/られる, or the special verb: 言う→おっしゃる, 行く/来る/いる→いらっしゃる,
  食べる→召し上がる, 見る→ご覧になる, する→なさる, 知っている→ご存じ.
- kenjogo patterns: お/ご+stem+する/いたす, 〜させていただく, or the special verb: 言う→申し上げる, 行く/来る→伺う/参る,
  食べる→いただく, 見る→拝見する, する→いたす, 知っている→存じている, いる→おる.
- 〜ておる/〜ております/〜いたす are HUMBLE: never use them in sonkeigo.
- If REQUESTED REGISTER is no_change: return the original EXACTLY, character for character.
- If REQUESTED REGISTER is teineigo, sonkeigo or kenjogo: you MUST rewrite the sentence in that register, even if it is short.
  teineigo = polite です/ます form; sonkeigo = respectful form of the other person's action (お〜になる, 〜れる/られる, special verbs);
  kenjogo = humble form of the speaker's own action (お〜する/いたす, 〜させていただく, special verbs).
- Change only what the register requires. Remove or soften blunt/casual sentence-ending particles (e.g. な, ぜ, ぞ, じゃん) into appropriate polite or formal endings. Do not reword, reorder, or change particles or punctuation otherwise.
- Strictly adhere to any CANONICAL SUPPLETIVE VERB CONSTRAINTS (補充法) provided. Never hallucinate regularized honorifics (e.g. *お行きになる).
- Output text must be normalized, natural Japanese (NFKC Unicode).
- If NO register is requested, return the input unchanged only when it is already appropriate for the context.
Return only the structured result."""

REVIEWER_SYSTEM = """You review a keigo conversion. Compare ORIGINAL, REQUESTED REGISTER, CONTEXT and CANDIDATE.
Approve only if ALL hold:
- If REQUESTED REGISTER is no_change, the candidate is identical to the original.
- Otherwise the candidate is actually written in the requested register (a plain-form candidate is NOT acceptable
  when teineigo/sonkeigo/kenjogo was requested) and the honorific direction is right for each actor.
- Adheres strictly to canonical suppletive verb constraints (e.g. 伺う/参る for 行く, ご覧になる for 見る).
- Meaning is preserved: no added or dropped information, no unnecessary rewording, and tense, mood and polarity are unchanged
  (a statement turned into a request or command is a meaning_change).
- sonkeigo candidates use respectful forms for the other person; humble forms (〜いたす, 〜ておる, させていただく) are wrong direction.
- Addressees referred to by bare surnames without honorific suffixes (呼び捨て, e.g. 佐藤) are unacceptable; they must be addressed with さん or 様.
- No 様/さん after a title (部長様, 先輩様), and no sonkeigo on things or events (プレゼンが終わられました).
- No double honorific, and it reads as natural business Japanese.
Reject ONLY for a concrete error from the list above: wrong register, wrong honorific direction, casual wording left in,
a non-canonical suppletive verb, a double honorific, a bare surname, 様 on a title, sonkeigo on a thing,
or a meaning change.
Do NOT reject for style: if the candidate is correct, approve it even when you would have phrased it differently,
when another valid keigo form exists (e.g. 拝見しました vs 拝見いたしました, ご存知 vs ご存じ), or when it could be "even more polite".
If there is no concrete error set approved=true and issues=[]. Otherwise list each error with a category and a fix."""


def analyzer_user(text: str, scenario: str, requested_register=None) -> str:
    reg = f"\nRequested register (given, do not contradict): {requested_register}" if requested_register else ""
    return f"Scenario: {scenario or '(none given)'}\nInput: {text}{reg}"


def specialist_user(
    text,
    context,
    examples,
    requested_register,
    prior_draft=None,
    feedback=None,
    suppletive_rules=None,
    domain_template=None,
):
    parts = [f"ORIGINAL: {text}"]
    target_reg = requested_register or (context or {}).get("overall_register")
    if target_reg:
        parts.append(f"REQUESTED REGISTER: {target_reg}")
    if context:
        parts.append(f"CONTEXT: {context}")
    if suppletive_rules:
        rules_text = "\n".join(f"- {r}" for r in suppletive_rules)
        parts.append(f"CANONICAL SUPPLETIVE VERB CONSTRAINTS (補充法制約 - 厳守):\n{rules_text}")
    if domain_template:
        guidelines = domain_template.get("tone_guidelines", "")
        parts.append(f"WORKPLACE DOMAIN GUIDELINES:\n- Tone: {guidelines}")
    if examples:
        ex = "\n".join(f"- {e['source_text']} -> {e['target_text']}" for e in examples)
        parts.append(f"REFERENCE EXAMPLES (style guide only, do not copy their content):\n{ex}")
    if prior_draft is not None:
        fb = "\n".join(f"- [{i.get('category')}] {i.get('description')} (fix: {i.get('suggested_fix', '')})" for i in (feedback or []))
        parts.append(f"PREVIOUS DRAFT (rejected): {prior_draft}\nREVIEWER ISSUES TO FIX:\n{fb}")
    return "\n\n".join(parts)


def reviewer_user(text, context, candidate, requested_register=None, suppletive_rules=None):
    target_reg = requested_register or (context or {}).get("overall_register")
    parts = [
        f"ORIGINAL: {text}",
        f"REQUESTED REGISTER: {target_reg or '(none)'}",
        f"CONTEXT: {context}",
        f"CANDIDATE: {candidate}",
    ]
    if suppletive_rules:
        rules_text = "\n".join(f"- {r}" for r in suppletive_rules)
        parts.append(f"SUPPLETIVE CONSTRAINTS TO VERIFY:\n{rules_text}")
    return "\n".join(parts)
