"""Generate B2 predictions in the same format as my-agent-app/scripts/run_eval.py."""
import argparse, json, time
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from train_t5 import fmt

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="checkpoints/t5-keigo/best")
ap.add_argument("--split", default="../corpus-dataset/splits/test.jsonl")
ap.add_argument("--out", default="predictions/B2.jsonl")
a = ap.parse_args()
tok = T5Tokenizer.from_pretrained(a.ckpt, is_fast=False)
model = T5ForConditionalGeneration.from_pretrained(a.ckpt).eval()
rows, seen = [], set()
for r in map(json.loads, open(a.split, encoding="utf-8")):
    k = (r["source_text"], r["register"])
    if r["register"] and k not in seen:
        seen.add(k); rows.append(r)
import os; os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows:
        t0 = time.perf_counter()
        ids = tok(fmt(r["register"], r["source_text"]), return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model.generate(**ids, num_beams=4, max_length=128)
        f.write(json.dumps({"id": r["id"], "condition": "B2", "trial": 0, "source_text": r["source_text"], "register": r["register"],
                            "prediction": tok.decode(out[0], skip_special_tokens=True), "status": "n/a", "revisions": 0,
                            "calls": 1, "seconds": time.perf_counter() - t0, "tokens": 0}, ensure_ascii=False) + "\n")
