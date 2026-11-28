"""Generate new casual -> keigo training pairs for T5 (Track B), for HUMAN REVIEW before training.

Run from my-agent-app/. Three steps, each safe to re-run:

  1. Write casual sentences (cheap: about 1.5K Groq tokens per 20 sentences):
       python -m scripts.generate_pairs sentences --n 100
     Or add your own: put one sentence per line in a .txt file (optionally "register<TAB>sentence") and run
       python -m scripts.generate_pairs sentences --from-file my_sentences.txt

  2. Convert them with the agent (resumable; stops cleanly at the Groq daily limit, re-run to continue):
       python -m scripts.generate_pairs convert
     --condition A2 (default) skips the LLM reviewer, about half the tokens of A3; your team is the reviewer.

  3. Pre-screen them with a second model (qwen, about 600 tokens per pair) so the reviewer can skip the safe ones:
       python -m scripts.generate_pairs judge

  4. Export the converted pairs that have not been exported yet to a new CSV for the team:
       python -m scripts.generate_pairs export
     -> ../training/generated/review_001.csv, review_002.csv, ...  (open in Excel / Google Sheets)
     Rows are sorted into buckets (2-check, 1-likely-fine, 3-drop); see training/generated/REVIEW_GUIDE.md.
     Reviewers fill in: keep (y/n/j), corrected_keigo (only if the agent was wrong), corrected_register, notes.
     Then run scripts/merge_reviewed.py to turn the reviewed rows into training data.

Nothing here reads val.jsonl / test.jsonl content into a prompt, and sentences that already appear in any corpus
split are dropped, so the test set stays unseen.
"""
import argparse, csv, hashlib, json, random, re, sys, time, unicodedata
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

APP = Path(__file__).resolve().parents[1]
ROOT = APP.parent
load_dotenv(APP / ".env")
OUT = ROOT / "training" / "generated"
SENTENCES = OUT / "sentences.jsonl"
CONVERTED = OUT / "converted.jsonl"
JUDGED = OUT / "judged.jsonl"
JUDGE_MODEL = "qwen/qwen3.8-27b"  # a different model family from the agent (gpt-oss), so it is less likely to share its mistakes
SPLITS = ROOT / "corpus-dataset" / "splits"
REGISTERS = ("teineigo", "sonkeigo", "kenjogo")
CSV_COLUMNS = ["sid", "bucket", "spot_check", "register", "casual", "agent_keigo", "auto_flags", "judge_verdict",
               "judge_issues", "judge_keigo", "english_casual", "english_keigo", "keep", "corrected_keigo",
               "corrected_register", "notes"]
BUCKET_ORDER = {"2-check": 0, "1-likely-fine": 1, "3-drop": 2}

# What each register needs from the casual sentence, so the agent has a real conversion to make.
REGISTER_BRIEF = {
    "teineigo": "Casual sentences said to a teacher, customer, senior or stranger where plain polite です/ます is the right "
                "conversion: statements, questions, greetings, opinions, plans, small talk.",
    "sonkeigo": "Casual sentences about the LISTENER's or a superior's/customer's OWN actions, which need respectful "
                "language: asking the boss if they already saw/ate/went/said something, whether the client is coming, "
                "what the teacher thinks, telling someone a superior did something (e.g. 部長もう帰った？, 先生これ見た？).",
    "kenjogo": "Casual sentences about the SPEAKER's OWN actions toward a superior or customer, which need humble language: "
               "I'll call/send/visit/check/explain, I saw/received/heard something (e.g. あとで電話する, 資料送っとくね, "
               "明日そっち行く, それ見たよ).",
}
SCENES = ["office with the boss", "talking to a client or customer", "university / school with a teacher",
          "part-time job at a shop or restaurant", "phone call or email to another company",
          "senior colleague (先輩)", "meeting or presentation", "hospital, bank or city office"]
STYLE = ("Make them sound like real everyday casual speech: mix plain forms and contractions such as 〜てた, 〜ちゃった, "
         "〜とく, 〜なきゃ, じゃん, 〜っけ/だっけ, 〜よね, 〜な, 〜の？, 〜てる？, rough openers like おい/ねえ, and sometimes a "
         "bare surname such as 田中、 at the start. 8 to 40 characters each. Every sentence must be different; vary verbs, "
         "topics, tense, questions vs statements. Do NOT write any polite です/ます sentences.")


class SentenceBatch(BaseModel):
    sentences: List[str] = Field(description="Casual Japanese sentences, one per item, no numbering or translations")


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s).strip()


def sid_for(text: str, register: str) -> str:
    return "gen_" + hashlib.sha1(f"{register}|{nfkc(text)}".encode("utf-8")).hexdigest()[:10]


def read_jsonl(p: Path):
    return [json.loads(l) for l in p.open(encoding="utf-8")] if p.exists() else []


def corpus_sources() -> set:
    """Every source sentence already in the corpus, so generated ones never duplicate (or leak into) a split."""
    seen = set()
    for split in ("train", "val", "test"):
        for r in read_jsonl(SPLITS / f"{split}.jsonl"):
            seen.add(nfkc(r["source_text"]))
    return seen


def add_sentences(new_rows):
    """Append unique, non-polite, non-corpus sentences to sentences.jsonl. Returns how many were added."""
    from src.agent.guardrails.honorific_rules import is_already_polite
    OUT.mkdir(parents=True, exist_ok=True)
    have = {r["sid"] for r in read_jsonl(SENTENCES)}
    have_text = {nfkc(r["text"]) for r in read_jsonl(SENTENCES)}
    corpus = corpus_sources()
    added = 0
    with SENTENCES.open("a", encoding="utf-8") as f:
        for r in new_rows:
            t = nfkc(r["text"])
            if not t or t in corpus or t in have_text or is_already_polite(t):
                continue
            r = {**r, "text": t, "sid": sid_for(t, r["register"] or "auto")}
            if r["sid"] in have:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            have.add(r["sid"]); have_text.add(t); added += 1
    return added


def cmd_sentences(a):
    if a.from_file:
        rows = []
        for line in Path(a.from_file).read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            reg, _, text = line.partition("\t") if "\t" in line else ("", "", line)
            reg = reg.strip() if reg.strip() in REGISTERS else None
            rows.append({"text": text, "register": reg, "scene": "", "origin": "team_written"})
        print(f"added {add_sentences(rows)} of {len(rows)} sentences from {a.from_file}")
        return

    from src.agent.agent import call_structured, default_llm_factory
    from src.agent.config import RunConfig
    llm = default_llm_factory(RunConfig().model, a.temperature)
    rng = random.Random()
    total, per_reg = 0, max(1, a.n // len(REGISTERS))
    for reg in REGISTERS:
        made = 0
        while made < per_reg:
            k = min(20, per_reg - made)
            scene = rng.choice(SCENES)
            parsed, rec = call_structured(
                llm, SentenceBatch,
                "You write realistic casual (タメ口) Japanese sentences for a keigo-conversion training set.",
                f"Write {k} casual Japanese sentences. Situation: {scene}.\n{REGISTER_BRIEF[reg]}\n{STYLE}",
                "sentence_generator")
            if parsed is None:
                print("generation failed:", rec.get("error", "")[:200])
                if "RateLimit" in rec.get("error", "") or "invalid_api_key" in rec.get("error", ""):
                    sys.exit(1)
                continue
            added = add_sentences([{"text": s, "register": reg, "scene": scene, "origin": "llm_generated"}
                                   for s in parsed.sentences[:k]])
            made += added; total += added
            print(f"  {reg}: +{added} ({made}/{per_reg})  [{scene}]", flush=True)
            if added == 0:  # the model keeps repeating itself; move on rather than loop forever
                break
            time.sleep(2)
    print(f"added {total} sentences -> {SENTENCES}")


def limit_kind(errors):
    e = " ".join(errors)
    if "invalid_api_key" in e or "AuthenticationError" in e:
        return "auth"
    if "RateLimit" in e or "429" in e:
        return "day" if "per day" in e else "minute"
    return None


def cmd_convert(a):
    from src.agent.config import CONDITIONS
    from src.agent.graph import graph
    todo_all = read_jsonl(SENTENCES)
    done = {r["sid"] for r in read_jsonl(CONVERTED)}
    todo = [r for r in todo_all if r["sid"] not in done][: a.limit]
    print(f"{len(done)} converted, {len(todo)} to convert now (condition {a.condition})", flush=True)
    with CONVERTED.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            for attempt in range(4):
                try:
                    s = graph.invoke({"raw_input": r["text"], "casual_input": r["text"], "scenario": r.get("scene", ""),
                                      "requested_register": r["register"], "drafts": [], "usage": []},
                                     config={"configurable": CONDITIONS[a.condition]})
                    errors = [f'{u["node"]}: {u["error"]}' for u in s.get("usage", []) if u.get("error")]
                except Exception as e:
                    s, errors = {}, [f"{type(e).__name__}: {e}"[:300]]
                kind = limit_kind(errors)
                if kind != "minute" or attempt == 3:
                    break
                print("  per-minute rate limit, waiting 60s...", flush=True)
                time.sleep(60)
            if kind in ("day", "auth"):
                msg = "Groq daily token limit reached" if kind == "day" else "Groq rejected the API key (check .env)"
                print(f"{msg} after {i - 1} sentences. Run the same command again later; finished ones are kept.")
                sys.exit(1)
            if kind == "minute":
                print(f"  still rate-limited, skipped {r['sid']} (re-run to retry)")
                continue
            reg = r["register"] or ((s.get("context") or {}).get("overall_register"))
            out = s.get("final_output") or (s.get("drafts") or [""])[-1]
            status = "llm_failed" if errors else ("fallback" if s.get("fallback_triggered") else s.get("status", ""))
            f.write(json.dumps({"sid": r["sid"], "register": reg, "casual": r["text"], "agent_keigo": out,
                                "agent_status": status, "scene": r.get("scene", ""), "origin": r.get("origin", ""),
                                "condition": a.condition,
                                "tokens": sum(u.get("input_tokens", 0) + u.get("output_tokens", 0)
                                              for u in s.get("usage", []))}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] {reg:9} {status}", flush=True)
            time.sleep(a.sleep)
    print("done. Next: python -m scripts.generate_pairs export")


_SONKEIGO_HINTS = ("いらっしゃ", "おっしゃ", "召し上が", "ご覧にな", "なさ", "になり", "になる", "られ", "ご存じ", "ご存知")
_KENJOGO_HINTS = ("いたし", "いたす", "申し", "伺", "参り", "拝見", "拝読", "せていただ", "せて頂", "存じ", "おり", "致し", "頂戴", "お送り", "ご連絡")


def auto_flags(register: str, casual: str, keigo: str) -> str:
    """Rule-based warnings for reviewers. Empty does not mean correct; it means no rule fired."""
    from src.agent.guardrails.honorific_rules import audit_casual_markers, audit_nijuu_keigo, is_already_polite
    flags = []
    if audit_nijuu_keigo(keigo)[0] or "られましたでしょうか" in keigo or "られますでしょうか" in keigo:
        flags.append("double/over-polite honorific?")
    if audit_casual_markers(keigo)[0]:
        flags.append("casual wording left?")
    if not is_already_polite(keigo):
        flags.append("ending not polite?")
    if re.search(r"(上司|部下|同僚|社員|お客|部長|課長|社長|係長|店長|先輩|先生|教授)様", keigo):
        flags.append("様 on a title/role word?")
    if re.search(r"(が|は)(決まって|出て|届いて|終わって|始まって)(いらっしゃ|おられ)", keigo):
        flags.append("honorific on a thing, not a person?")
    if register == "teineigo" and any(h in keigo for h in _SONKEIGO_HINTS + _KENJOGO_HINTS):
        flags.append("teineigo row uses sonkeigo/kenjogo?")
    if register == "sonkeigo" and not any(h in keigo for h in _SONKEIGO_HINTS) and not re.search(r"お[一-龠]+に", keigo):
        flags.append("no respectful verb found?")
    if register == "kenjogo" and not any(h in keigo for h in _KENJOGO_HINTS) and not re.search(r"(お|ご)[一-龠]+(し|いた)", keigo):
        flags.append("no humble verb found?")
    if len(keigo) > 2.2 * len(casual) + 10:
        flags.append("much longer than the original: meaning added?")
    return "; ".join(flags)


class Judgment(BaseModel):
    verdict: str = Field(description="ok = correct and natural; fix = has a concrete error you can correct; "
                                     "bad = the casual sentence is unnatural/ungrammatical or the pair cannot be saved")
    issues: str = Field(description="Short description of each concrete error, in English. Empty if ok.")
    suggested_keigo: str = Field(description="Corrected keigo in the SAME register with the SAME meaning, only if "
                                             "verdict is fix. Empty otherwise.")
    english_casual: str = Field(description="Short literal English translation of the casual sentence")
    english_keigo: str = Field(description="Short literal English translation of the agent's keigo")


JUDGE_SYSTEM = """You are a strict native-level checker of Japanese keigo training data. For each pair you get the
target REGISTER, a CASUAL sentence and the agent's KEIGO. Registers:
- teineigo: polite です/ます only. Respectful verbs (いらっしゃる, おっしゃる, 召し上がる, ご覧になる, お〜になる) or humble verbs
  (伺う, 参る, 拝見する, いたす, 申し上げる, お〜する) do NOT belong in a teineigo answer.
- sonkeigo: raises the OTHER person's (listener, superior, client) action: いらっしゃる, おっしゃる, ご覧になる, 召し上がる,
  なさる, お〜になる, 〜れる/られる.
- kenjogo: lowers the SPEAKER's own action: 伺う, 参る, 拝見する, いたす, 申し上げる, おる, お/ご〜する, 〜させていただく.
Call it an error (verdict fix) if any of these is wrong:
1. Double honorific (二重敬語), e.g. お出になられる, おっしゃられる, ご覧になられる, お伺いいたす, or stacked forms like
   終わられましたでしょうか.
2. Honorific direction: respectful verbs for the speaker's own action, or humble verbs for the other person's action.
3. Register mismatch with REGISTER (see above).
4. Wrong verb conjugation (e.g. 伺ります instead of 伺います) or broken forms (e.g. お電話をかけいたします).
5. 様 attached to a title or role word (部長様, 課長様, 先輩様, 先生様, 上司様): titles are already honorific, so drop 様.
   Also a bare surname with no さん/様.
8. Sonkeigo used on a thing instead of a person (出張が決まっていらっしゃる, 結果が出ていらっしゃる).
9. Plain/colloquial words left in a polite sentence (昼飯 -> 昼食/お昼, めし, やつ).
6. Meaning change: added or dropped information, a statement turned into a request, a question turned into a statement,
   lost hearsay (〜って / 〜んだって = "I heard"), changed tense or negation. Note: 〜っけ / 〜だっけ is NOT hearsay; it
   means "what was it again? / did ... ?", and a plain polite question is a correct conversion of it.
7. Casual wording left in.
These are correct, NOT double honorifics: お〜になる, お〜になっている, お〜になりました, お聞きになる, お読みになる,
〜ていらっしゃる, ご覧になりましたか. Only count it as 二重敬語 when two honorific mechanisms apply to the same verb
(お〜になられる, おっしゃられる). Any suggested_keigo must itself follow every rule above.
Use verdict ok when the keigo is correct and natural, even if another valid phrasing exists (拝見しました and
拝見いたしました are both fine). Use bad when the CASUAL sentence itself is unnatural or ungrammatical Japanese
(e.g. 大丈夫っけ), or when the pair makes no sense. Be concrete; do not invent problems."""


def cmd_judge(a):
    from src.agent.agent import call_structured, default_llm_factory
    llm = default_llm_factory(a.model, 0.0)
    done = {r["sid"] for r in read_jsonl(JUDGED)}
    todo = [r for r in read_jsonl(CONVERTED) if r["sid"] not in done and r["agent_status"] != "llm_failed"][: a.limit]
    print(f"{len(done)} judged, {len(todo)} to judge now with {a.model}", flush=True)
    counts = {}
    with JUDGED.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            user = f"REGISTER: {r['register']}\nCASUAL: {r['casual']}\nKEIGO: {r['agent_keigo']}"
            for attempt in range(4):
                parsed, rec = call_structured(llm, Judgment, JUDGE_SYSTEM, user, "judge")
                kind = limit_kind([rec.get("error", "")]) if parsed is None else None
                if kind != "minute" or attempt == 3:
                    break
                print("  per-minute rate limit, waiting 60s...", flush=True)
                time.sleep(60)
            if kind in ("day", "auth"):
                msg = "Groq daily token limit reached" if kind == "day" else "Groq rejected the API key (check .env)"
                print(f"{msg} after {i - 1} rows. Run the same command again later; finished ones are kept.")
                sys.exit(1)
            if parsed is None:
                print(f"  [{i}/{len(todo)}] judge failed ({rec.get('error', '')[:80]}); will retry next run", flush=True)
                continue
            verdict = parsed.verdict.strip().lower()
            verdict = verdict if verdict in ("ok", "fix", "bad") else "fix"
            if verdict == "fix" and nfkc(parsed.suggested_keigo) in ("", nfkc(r["agent_keigo"])):
                verdict = "ok" if not parsed.suggested_keigo.strip() and not parsed.issues.strip() else verdict
            counts[verdict] = counts.get(verdict, 0) + 1
            f.write(json.dumps({"sid": r["sid"], "verdict": verdict, "issues": parsed.issues.strip(),
                                "judge_keigo": nfkc(parsed.suggested_keigo) if verdict == "fix" else "",
                                "english_casual": parsed.english_casual.strip(),
                                "english_keigo": parsed.english_keigo.strip(), "model": a.model,
                                "tokens": rec["input_tokens"] + rec["output_tokens"]}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"  [{i}/{len(todo)}] {r['register']:9} {verdict}", flush=True)
            time.sleep(a.sleep)
    print("verdicts this run:", counts)
    print("Next: python -m scripts.generate_pairs export --refresh   (re-sorts the review CSVs into buckets)")


def review_rows(rows, spot_n=30, seed=7):
    """Rows for a review CSV: rule flags + judge verdict -> bucket, a random spot-check sample, sorted by priority."""
    judged = {j["sid"]: j for j in read_jsonl(JUDGED)}
    out = []
    for r in rows:
        j = judged.get(r["sid"], {})
        flags = auto_flags(r["register"], r["casual"], r["agent_keigo"])
        v = j.get("verdict", "")
        bucket = "3-drop" if v == "bad" else "1-likely-fine" if (v == "ok" and not flags) else "2-check"
        out.append({**r, "bucket": bucket, "spot_check": "", "auto_flags": flags, "judge_verdict": v or "not judged",
                    "judge_issues": j.get("issues", ""), "judge_keigo": j.get("judge_keigo", ""),
                    "english_casual": j.get("english_casual", ""), "english_keigo": j.get("english_keigo", ""),
                    "keep": "", "corrected_keigo": "", "corrected_register": "", "notes": ""})
    fine = [r for r in out if r["bucket"] == "1-likely-fine"]
    for r in random.Random(seed).sample(fine, min(spot_n, len(fine))):
        r["spot_check"] = "CHECK"
    out.sort(key=lambda r: (BUCKET_ORDER[r["bucket"]], r["spot_check"] != "CHECK", r["register"], r["sid"]))
    return out


def write_review(path, rows):
    # utf-8-sig so Excel shows the Japanese correctly when the file is double-clicked.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    c = {b: sum(r["bucket"] == b for r in rows) for b in BUCKET_ORDER}
    print(f"  {path.name}: {len(rows)} rows | to review: {c['2-check']} check + "
          f"{sum(r['spot_check'] == 'CHECK' for r in rows)} spot-check | likely fine: {c['1-likely-fine']} | "
          f"drop: {c['3-drop']}")


def refresh_unreviewed():
    """Rewrite review CSVs whose keep/corrected columns are all still empty, with the current columns and flags."""
    conv = {r["sid"]: r for r in read_jsonl(CONVERTED)}
    for p in sorted(OUT.glob("review_*.csv")):
        with p.open(encoding="utf-8-sig", newline="") as f:
            old = list(csv.DictReader(f))
        if any((row.get(k) or "").strip() for row in old for k in ("keep", "corrected_keigo", "corrected_register", "notes")):
            print(f"  {p.name}: already being reviewed, left unchanged")
            continue
        write_review(p, review_rows([conv.get(row["sid"], row) for row in old]))


def cmd_export(a):
    if a.refresh:
        refresh_unreviewed()
        return
    exported = set()
    files = sorted(OUT.glob("review_*.csv"))
    for p in files:
        with p.open(encoding="utf-8-sig", newline="") as f:
            exported |= {row["sid"] for row in csv.DictReader(f)}
    rows = [r for r in read_jsonl(CONVERTED) if r["sid"] not in exported
            and r["agent_status"] != "llm_failed" and r["register"] in REGISTERS
            and nfkc(r["agent_keigo"]) != nfkc(r["casual"])]
    if not rows:
        print("nothing new to export.")
        return
    path = OUT / f"review_{len(files) + 1:03d}.csv"
    write_review(path, review_rows(rows))
    print("See training/generated/REVIEW_GUIDE.md for how to fill it in.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sentences", help="write casual sentences (LLM, or --from-file)")
    p.add_argument("--n", type=int, default=60, help="total sentences, split evenly over the 3 registers")
    p.add_argument("--from-file", help="txt file: one sentence per line, optionally 'register<TAB>sentence'")
    p.add_argument("--temperature", type=float, default=0.9)
    p = sub.add_parser("convert", help="convert the sentences to keigo with the agent")
    p.add_argument("--condition", default="A2", choices=["A2", "A3"], help="A2 = no reviewer (cheaper)")
    p.add_argument("--limit", type=int, help="convert at most this many now")
    p.add_argument("--sleep", type=float, default=2.0)
    p = sub.add_parser("judge", help="pre-screen the converted pairs with a second model")
    p.add_argument("--model", default=JUDGE_MODEL)
    p.add_argument("--limit", type=int, help="judge at most this many now")
    p.add_argument("--sleep", type=float, default=1.0)
    p = sub.add_parser("export", help="write a new review CSV for the team")
    p.add_argument("--refresh", action="store_true",
                   help="rebuild the existing review CSVs that nobody has filled in yet (e.g. to add new columns)")
    a = ap.parse_args()
    {"sentences": cmd_sentences, "convert": cmd_convert, "judge": cmd_judge, "export": cmd_export}[a.cmd](a)
