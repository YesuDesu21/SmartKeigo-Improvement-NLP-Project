"""Generate new casual -> keigo training pairs for T5 (Track B), for HUMAN REVIEW before training.

Run from my-agent-app/. Three steps, each safe to re-run:

  1. Write casual sentences (cheap: about 1.5K Groq tokens per 20 sentences):
       python -m scripts.generate_pairs sentences --n 100
     Or add your own: put one sentence per line in a .txt file (optionally "register<TAB>sentence") and run
       python -m scripts.generate_pairs sentences --from-file my_sentences.txt

  2. Convert them with the agent (resumable; stops cleanly at the Groq daily limit, re-run to continue):
       python -m scripts.generate_pairs convert
     --condition A2 (default) skips the LLM reviewer, about half the tokens of A3; your team is the reviewer.

  3. Export the converted pairs that have not been exported yet to a new CSV for the team:
       python -m scripts.generate_pairs export
     -> ../training/generated/review_001.csv, review_002.csv, ...  (open in Excel / Google Sheets)
     Reviewers fill in: keep (y/n), corrected_keigo (only if the agent was wrong), corrected_register, notes.
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
SPLITS = ROOT / "corpus-dataset" / "splits"
REGISTERS = ("teineigo", "sonkeigo", "kenjogo")
CSV_COLUMNS = ["sid", "register", "casual", "agent_keigo", "agent_status", "auto_flags", "keep", "corrected_keigo",
               "corrected_register", "notes"]

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
    if re.search(r"(上司|部下|同僚|社員|お客)様", keigo):
        flags.append("odd 様 on a role word?")
    if register == "teineigo" and any(h in keigo for h in _SONKEIGO_HINTS + _KENJOGO_HINTS):
        flags.append("teineigo row uses sonkeigo/kenjogo?")
    if register == "sonkeigo" and not any(h in keigo for h in _SONKEIGO_HINTS) and not re.search(r"お[一-龠]+に", keigo):
        flags.append("no respectful verb found?")
    if register == "kenjogo" and not any(h in keigo for h in _KENJOGO_HINTS) and not re.search(r"(お|ご)[一-龠]+(し|いた)", keigo):
        flags.append("no humble verb found?")
    if len(keigo) > 2.2 * len(casual) + 10:
        flags.append("much longer than the original: meaning added?")
    return "; ".join(flags)


def refresh_unreviewed():
    """Rewrite review CSVs whose keep/corrected columns are all still empty, with the current columns and flags."""
    conv = {r["sid"]: r for r in read_jsonl(CONVERTED)}
    for p in sorted(OUT.glob("review_*.csv")):
        with p.open(encoding="utf-8-sig", newline="") as f:
            old = list(csv.DictReader(f))
        if any((row.get(k) or "").strip() for row in old for k in ("keep", "corrected_keigo", "corrected_register", "notes")):
            print(f"  {p.name}: already being reviewed, left unchanged")
            continue
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            for row in old:
                r = conv.get(row["sid"], row)
                w.writerow({**r, "auto_flags": auto_flags(r["register"], r["casual"], r["agent_keigo"]), "keep": "",
                            "corrected_keigo": "", "corrected_register": "", "notes": ""})
        print(f"  {p.name}: refreshed ({len(old)} rows)")


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
    # utf-8-sig so Excel shows the Japanese correctly when the file is double-clicked.
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["register"], r["sid"])):
            w.writerow({**r, "auto_flags": auto_flags(r["register"], r["casual"], r["agent_keigo"]), "keep": "", "corrected_keigo": "", "corrected_register": "", "notes": ""})
    print(f"wrote {len(rows)} pairs -> {path}")
    print("Reviewers: keep = y or n; corrected_keigo only if the agent's keigo is wrong; corrected_register only if "
          "the register is wrong. Rows with agent_status 'fallback' need the most careful check.")


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
    p = sub.add_parser("export", help="write a new review CSV for the team")
    p.add_argument("--refresh", action="store_true",
                   help="rebuild the existing review CSVs that nobody has filled in yet (e.g. to add new columns)")
    a = ap.parse_args()
    {"sentences": cmd_sentences, "convert": cmd_convert, "export": cmd_export}[a.cmd](a)
