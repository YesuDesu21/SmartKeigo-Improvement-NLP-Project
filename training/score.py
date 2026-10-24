"""Score prediction files against a split. Items are (source_text, register); alternate annotator
targets for the same item are multi-references.
    python score.py --split ../corpus-dataset/splits/val.jsonl predictions/B2.jsonl ../my-agent-app/predictions/A3.jsonl
Reports, per register and overall: n, exact match (any reference), copy rate, char-BLEU (secondary),
failed/empty outputs, needs_review count, mean calls / seconds / tokens. Failures stay in the denominator."""
import argparse, json
from collections import defaultdict
from statistics import mean

from sacrebleu.metrics import BLEU

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--bertscore", action="store_true")
ap.add_argument("files", nargs="+")
a = ap.parse_args()

gold = defaultdict(list)
for r in map(json.loads, open(a.split, encoding="utf-8")):
    if r["register"]:
        gold[(r["source_text"], r["register"])].append(r["target_text"])
bleu = BLEU(tokenize="char")


def refs_for(keys):
    n = max(len(gold[k]) for k in keys)
    return [[(gold[k] * n)[i] for k in keys] for i in range(n)]


def summarize(ps):
    keys = [(p["source_text"], p["register"]) for p in ps]
    hyp = [p["prediction"] for p in ps]
    res = {
        "n": len(ps),
        "exact": round(mean(h in gold[k] for h, k in zip(hyp, keys)), 3),
        "copy_rate": round(mean(h == k[0] for h, k in zip(hyp, keys)), 3),
        "BLEU(char)": round(bleu.corpus_score(hyp, refs_for(keys)).score, 1),
        "failed/empty": sum(1 for p in ps if not p["prediction"].strip() or str(p["status"]).startswith("error")),
        "needs_review": sum(1 for p in ps if p["status"] == "needs_review"),
        "avg_calls": round(mean(p["calls"] for p in ps), 2),
        "avg_sec": round(mean(p["seconds"] for p in ps), 1),
        "avg_tokens": round(mean(p.get("tokens", 0) for p in ps)),
    }
    return res, keys, hyp


for fp in a.files:
    preds = [json.loads(l) for l in open(fp, encoding="utf-8")]
    if any("register" not in p for p in preds):
        raise SystemExit(f"{fp}: predictions lack a 'register' field; regenerate with the updated run_eval/predict_t5")
    preds = [p for p in preds if (p["source_text"], p["register"]) in gold]
    groups = [("ALL", preds)] + [(reg, [p for p in preds if p["register"] == reg])
                                  for reg in ("teineigo", "sonkeigo", "kenjogo", "no_change")]
    for name, ps in groups:
        if not ps:
            continue
        res, keys, hyp = summarize(ps)
        if a.bertscore:
            from bert_score import score as bs
            res["BERTScore_F1"] = round(float(bs(hyp, [gold[k][0] for k in keys], lang="ja")[2].mean()), 3)
        print(f"{fp.split('/')[-1].split(chr(92))[-1]:12s} {name:9s}", json.dumps(res, ensure_ascii=False))
