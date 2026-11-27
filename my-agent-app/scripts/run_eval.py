"""Run paper conditions over a split and write predictions + usage.
    python -m scripts.run_eval --split ../corpus-dataset/splits/val.jsonl --conditions B1 A1 A2 A3 A4 --trials 3
Rows with category in teineigo/sonkeigo/kenjogo/no_change are used; the gold register is supplied
(controlled track) but the gold target is never passed to any system.
Add --resume to continue an interrupted run: rows that hit a Groq rate limit are re-run, all others kept."""
import argparse, json, sys, time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
from src.agent.config import CONDITIONS
from src.agent.graph import graph

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--conditions", nargs="+", default=["B1", "A1", "A2", "A3", "A4"])
ap.add_argument("--trials", type=int, default=3)
ap.add_argument("--limit", type=int)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--per-register", type=int, help="balanced sample: N items per register (seed 42)")
ap.add_argument("--out", default="predictions")
ap.add_argument("--resume", action="store_true",
                help="keep finished rows in the output file and only run missing or rate-limited ones")
ap.add_argument("--sleep", type=float, default=2.0, help="seconds to pause between sentences (eases rate limits)")
ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait after a per-minute rate limit")
ap.add_argument("--max-waits", type=int, default=3, help="per-minute waits before skipping a sentence")
a = ap.parse_args()

rows = [json.loads(l) for l in open(a.split, encoding="utf-8")]
import random
seen, uniq = set(), []
for r in rows:  # one run per (source, register); alternate targets are references, not extra items
    k = (r["source_text"], r["register"])
    if r["register"] and k not in seen:
        seen.add(k); uniq.append(r)
if a.per_register:
    by = {}
    for r in uniq:
        by.setdefault(r["register"], []).append(r)
    rng = random.Random(a.seed)
    uniq = [x for reg in sorted(by) for x in rng.sample(by[reg], min(a.per_register, len(by[reg])))]
rows = uniq[: a.limit]
Path(a.out).mkdir(exist_ok=True)


def run_one(r, cond, t):
    try:
        s = graph.invoke(
            {
                "casual_input": r["source_text"],
                "raw_input": r["source_text"],
                "requested_register": r["register"],
                "drafts": [],
                "usage": [],
                "execution_trace": [],
            },
            config={"configurable": CONDITIONS[cond]},
        )
        pred = s.get("final_output") or s["drafts"][-1]
        status = s["status"]
        rev = s.get("iteration_count", s.get("revision_count", 0))
        usage = s.get("usage", [])
        fallback = s.get("fallback_triggered", False)
        errors = [f'{u["node"]}: {u["error"]}' for u in usage if u.get("error")]
        critiques = [c for c in s.get("critique_history", []) if any(w in c.lower() for w in ("reject", "error", "critique", "fallback"))]
    except Exception as e:  # keep failures in the denominator
        pred, status, rev, usage, fallback = "", f"error:{type(e).__name__}", 0, [], False
        errors = [f"{type(e).__name__}: {e}"[:300]]
        critiques = []
    return {
        "id": r["id"],
        "condition": cond,
        "trial": t,
        "source_text": r["source_text"],
        "register": r["register"],
        "prediction": pred,
        "status": status,
        "revisions": rev,
        "fallback_triggered": fallback,
        "critiques": critiques,
        "calls": sum(u["calls"] for u in usage),
        "seconds": sum(u.get("seconds", 0.0) for u in usage),
        "tokens": sum(u.get("input_tokens", 0) + u.get("output_tokens", 0) for u in usage),
        "errors": errors,
    }


def rate_limited(rec):
    """'day' / 'minute' if any LLM call in this run hit a Groq rate limit, 'auth' if the API key was
    rejected, else None. Such rows say nothing about the system, so they are retried instead of scored.
    Other errors (e.g. the model returning malformed output) are real failures and are kept."""
    errs = " ".join(rec["errors"])
    if "invalid_api_key" in errs or "AuthenticationError" in errs:
        return "auth"
    if "RateLimit" not in errs and "429" not in errs:
        return None
    return "day" if "per day" in errs else "minute"


for cond in a.conditions:
    path = Path(f"{a.out}/{cond}.jsonl")
    done = {}
    if a.resume and path.exists():
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            if not rate_limited(rec):
                done[(rec["id"], rec["trial"])] = rec
    todo = [(r, t) for r in rows for t in range(a.trials) if (r["id"], t) not in done]
    print(f"{cond}: {len(done)} done, {len(todo)} to run", flush=True)
    # Rewrite the file with the kept rows, then append each new row as soon as it finishes,
    # so an interrupted run loses nothing.
    with path.open("w", encoding="utf-8") as f:
        for rec in done.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stop = False
    with path.open("a", encoding="utf-8") as f:
        for i, (r, t) in enumerate(todo, 1):
            for attempt in range(a.max_waits + 1):
                rec = run_one(r, cond, t)
                limit = rate_limited(rec)
                if limit != "minute" or attempt == a.max_waits:
                    break
                print(f"  per-minute rate limit, waiting {a.wait}s...", flush=True)
                time.sleep(a.wait)
            if limit == "auth":
                print(f"Groq rejected the API key (Invalid API Key) after {i - 1} new rows. Fix GROQ_API_KEY "
                      f"in .env, then run the same command again with --resume.", flush=True)
                stop = True
                break
            if limit == "day":
                print(f"Daily token limit reached after {i - 1} new rows. Run the same command again "
                      f"with --resume once the limit resets.", flush=True)
                stop = True
                break
            if limit is None:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{len(done) + i}/{len(done) + len(todo)}] {rec['id']} {rec['status']}", flush=True)
            else:
                print(f"  still rate-limited after {a.max_waits} waits; skipped {rec['id']} (retry with --resume)", flush=True)
            time.sleep(a.sleep)
    if stop:
        sys.exit(1)
    print("finished", cond)
