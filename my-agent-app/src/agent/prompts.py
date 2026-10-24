ANALYZER_SYSTEM = """You analyse casual Japanese before it is converted to keigo.
Identify (1) who the message is addressed to, (2) who performs each action, (3) the communicative goal.
Rules: do NOT invent facts. If the addressee or an actor is not determinable, use "unknown" and list it in unknowns.
Actions by the speaker or their in-group (kenjogo), by the addressee or out-group people (sonkeigo)."""

SPECIALIST_SYSTEM = """You convert casual Japanese into natural, correct business keigo.
Hard rules:
- Preserve meaning, names, numbers, dates, negation and intent. Do not add or remove information.
- sonkeigo ONLY for actions of the addressee / out-group; kenjogo ONLY for the speaker / own in-group; teineigo toward the listener.
- Never use double honorifics (nijuu keigo). Never apply sonkeigo to the speaker's own actions.
- The subject is usually omitted. When a register is requested, assume: sonkeigo -> the subject is the OTHER person (superior/client),
  so raise their action; kenjogo -> the subject is the SPEAKER, so lower your own action; teineigo -> just make the sentence polite.
- Preserve tense, mood and polarity exactly: a statement stays a statement (never turn it into a request or command),
  present stays present, past stays past, negative stays negative.
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
- Change only what the register requires. Do not reword, reorder, or change particles or punctuation otherwise.
- If NO register is requested, return the input unchanged only when it is already appropriate for the context.
Return only the structured result."""

REVIEWER_SYSTEM = """You review a keigo conversion. Compare ORIGINAL, REQUESTED REGISTER, CONTEXT and CANDIDATE.
Approve only if ALL hold:
- If REQUESTED REGISTER is no_change, the candidate is identical to the original.
- Otherwise the candidate is actually written in the requested register (a plain-form candidate is NOT acceptable
  when teineigo/sonkeigo/kenjogo was requested) and the honorific direction is right for each actor.
- Meaning is preserved: no added or dropped information, no unnecessary rewording, and tense, mood and polarity are unchanged
  (a statement turned into a request or command is a meaning_change).
- sonkeigo candidates use respectful forms for the other person; humble forms (〜いたす, 〜ておる, させていただく) are wrong direction.
- No double honorific, and it reads as natural business Japanese.
List every concrete problem with a category and a fix. If there is no problem set approved=true and issues=[]."""


def analyzer_user(text: str, scenario: str, requested_register=None) -> str:
    reg = f"\nRequested register (given, do not contradict): {requested_register}" if requested_register else ""
    return f"Scenario: {scenario or '(none given)'}\nInput: {text}{reg}"


def specialist_user(text, context, examples, requested_register, prior_draft=None, feedback=None):
    parts = [f"ORIGINAL: {text}"]
    if requested_register:
        parts.append(f"REQUESTED REGISTER: {requested_register}")
    if context:
        parts.append(f"CONTEXT: {context}")
    if examples:
        ex = "\n".join(f"- {e['source_text']} -> {e['target_text']}" for e in examples)
        parts.append(f"REFERENCE EXAMPLES (style guide only, do not copy their content):\n{ex}")
    if prior_draft is not None:
        fb = "\n".join(f"- [{i.get('category')}] {i.get('description')} (fix: {i.get('suggested_fix', '')})" for i in (feedback or []))
        parts.append(f"PREVIOUS DRAFT (rejected): {prior_draft}\nREVIEWER ISSUES TO FIX:\n{fb}")
    return "\n\n".join(parts)


def reviewer_user(text, context, candidate, requested_register=None):
    return (f"ORIGINAL: {text}\nREQUESTED REGISTER: {requested_register or '(none)'}\n"
            f"CONTEXT: {context}\nCANDIDATE: {candidate}")
