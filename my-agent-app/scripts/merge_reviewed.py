"""Turn the team-reviewed CSVs (training/generated/review_*.csv) into extra T5 training data.

    python -m scripts.merge_reviewed

A row is used when keep is y/yes/1/ok/o/○/✓, or when corrected_keigo is filled in (and keep is not n).
The target is corrected_keigo if given, else agent_keigo; the register is corrected_register if given.
Writes ../training/generated/generated_train.jsonl in the same format as corpus-dataset/splits/train.jsonl.
The corpus splits are never modified, so the original Track B result stays reproducible. Train with it via
    python train_t5.py --extra generated/generated_train.jsonl --out checkpoints/t5-keigo-aug   (from training/)
"""
import csv, json, sys, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training" / "generated"
SPLITS = ROOT / "corpus-dataset" / "splits"
REGISTERS = ("teineigo", "sonkeigo", "kenjogo")
YES = {"y", "yes", "1", "ok", "o", "○", "◯", "✓", "true", "keep"}
NO = {"n", "no", "0", "x", "×", "false", "drop"}


def nfkc(s):
    return unicodedata.normalize("NFKC", s or "").strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    corpus = set()
    for split in ("train", "val", "test"):
        p = SPLITS / f"{split}.jsonl"
        if p.exists():
            corpus |= {nfkc(json.loads(l)["source_text"]) for l in p.open(encoding="utf-8")}
    files = sorted(OUT.glob("review_*.csv"))
    if not files:
        sys.exit(f"no review_*.csv files in {OUT}. Run: python -m scripts.generate_pairs export")

    rows, why = {}, Counter()
    for p in files:
        with p.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                keep, fixed = nfkc(r.get("keep")).lower(), nfkc(r.get("corrected_keigo"))
                if keep in NO or not (keep in YES or fixed):
                    why["not kept / not reviewed yet"] += 1
                    continue
                reg = nfkc(r.get("corrected_register")).lower() or nfkc(r.get("register")).lower()
                src, tgt = nfkc(r.get("casual")), fixed or nfkc(r.get("agent_keigo"))
                if reg not in REGISTERS:
                    why[f"bad register '{reg}'"] += 1
                    continue
                if not tgt or tgt == src:
                    why["empty or unchanged keigo"] += 1
                    continue
                if src in corpus:
                    why["already in the corpus"] += 1
                    continue
                rows[r["sid"]] = {  # a later review file overrides an earlier one for the same sentence
                    "id": r["sid"], "source_text": src, "target_text": tgt, "label_raw": reg, "category": reg,
                    "register": reg, "optional": False, "group_id": r["sid"], "is_identity": False, "sft_core": True,
                    "split": "train", "origin": "agent_generated_human_reviewed", "corrected": bool(fixed),
                    "review_file": p.name}

    out = OUT / "generated_train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} reviewed pairs -> {out}")
    print("  by register:", dict(Counter(r["register"] for r in rows.values())))
    print("  corrected by reviewers:", sum(r["corrected"] for r in rows.values()))
    for k, v in why.items():
        print(f"  skipped {v}: {k}")


if __name__ == "__main__":
    main()
