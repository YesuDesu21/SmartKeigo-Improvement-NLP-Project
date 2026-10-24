"""Run paper conditions over a split and write predictions + usage.
    python -m scripts.run_eval --split ../corpus-dataset/splits/val.jsonl --conditions B1 A1 A2 A3 A4 --trials 3
Rows with category in teineigo/sonkeigo/kenjogo/no_change are used; the gold register is supplied
(controlled track) but the gold target is never passed to any system."""
import argparse, json
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
for cond in a.conditions:
    with open(f"{a.out}/{cond}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            for t in range(a.trials):
                try:
                    s = graph.invoke({"raw_input": r["source_text"], "requested_register": r["register"], "drafts": [], "usage": []},
                                     config={"configurable": CONDITIONS[cond]})
                    pred, status, rev, usage = s["drafts"][-1], s["status"], s["revision_count"], s["usage"]
                    errors = [f'{u["node"]}: {u["error"]}' for u in usage if u.get("error")]
                except Exception as e:  # keep failures in the denominator
                    pred, status, rev, usage = "", f"error:{type(e).__name__}", 0, []
                    errors = [f"{type(e).__name__}: {e}"[:300]]
                f.write(json.dumps({"id": r["id"], "condition": cond, "trial": t, "source_text": r["source_text"], "register": r["register"],
                                    "prediction": pred, "status": status, "revisions": rev,
                                    "calls": sum(u["calls"] for u in usage), "seconds": sum(u["seconds"] for u in usage),
                                    "tokens": sum(u["input_tokens"] + u["output_tokens"] for u in usage),
                                    "errors": errors}, ensure_ascii=False) + "\n")
    print("finished", cond)
