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
                    critiques = [c for c in s.get("critique_history", []) if "reject" in c.lower() or "error" in c.lower()]
                except Exception as e:  # keep failures in the denominator
                    pred, status, rev, usage, fallback = "", f"error:{type(e).__name__}", 0, [], False
                    errors = [f"{type(e).__name__}: {e}"[:300]]
                    critiques = []
                f.write(
                    json.dumps(
                        {
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
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print("finished", cond)
