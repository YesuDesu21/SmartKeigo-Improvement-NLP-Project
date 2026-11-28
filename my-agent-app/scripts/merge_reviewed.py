"""Turn the team-reviewed CSVs (training/generated/review_*.csv) into extra T5 training data.

    python -m scripts.merge_reviewed                        # only rows a reviewer decided on
    python -m scripts.merge_reviewed --accept-likely-fine   # also keep unreviewed '1-likely-fine' rows

keep values:  y = use agent_keigo  |  j = use the judge's judge_keigo  |  n = drop
              (filling in corrected_keigo also keeps the row, with your correction, unless keep is n)
The register is corrected_register if given. Blank rows are skipped, except '1-likely-fine' rows with
--accept-likely-fine, which is only safe once the spot_check=CHECK rows were reviewed and nearly all were correct;
this script prints that spot-check result.

Several people may each send their own copy (review_002_ana.csv, review_002_ken.csv, ...): explicit decisions from
any file win over blanks, and for the same row filled in twice, the file that sorts last wins.
Writes ../training/generated/generated_train.jsonl in the format of corpus-dataset/splits/train.jsonl; the corpus
splits are never modified. Then train from training/:
    python train_t5.py --extra generated/generated_train.jsonl --out checkpoints/t5-keigo-aug
"""
import argparse, csv, json, sys, unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "training" / "generated"
SPLITS = ROOT / "corpus-dataset" / "splits"
REGISTERS = ("teineigo", "sonkeigo", "kenjogo")
YES = {"y", "yes", "1", "ok", "o", "○", "◯", "✓", "true", "keep"}
JUDGE = {"j", "judge"}
NO = {"n", "no", "0", "x", "×", "false", "drop"}


def nfkc(s):
    return unicodedata.normalize("NFKC", s or "").strip()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--accept-likely-fine", action="store_true",
                    help="keep unreviewed rows in bucket 1-likely-fine (only after a good spot-check)")
    a = ap.parse_args()

    corpus = set()
    for split in ("train", "val", "test"):
        p = SPLITS / f"{split}.jsonl"
        if p.exists():
            corpus |= {nfkc(json.loads(l)["source_text"]) for l in p.open(encoding="utf-8")}
    files = sorted(OUT.glob("review_*.csv"))
    if not files:
        sys.exit(f"no review_*.csv files in {OUT}. Run: python -m scripts.generate_pairs export")

    content, decided = {}, {}  # sid -> row (any file) / sid -> (row, file) with an explicit decision
    for p in files:
        with p.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                content.setdefault(r["sid"], r)
                keep = nfkc(r.get("keep")).lower()
                if keep or nfkc(r.get("corrected_keigo")):
                    decided[r["sid"]] = (r, p.name)

    # Spot-check report: how often the reviewer had to change or drop a "likely fine" row.
    spot = [decided[s][0] for s, r in content.items() if r.get("spot_check") == "CHECK" and s in decided]
    spot_total = sum(r.get("spot_check") == "CHECK" for r in content.values())
    spot_bad = sum(nfkc(r.get("keep")).lower() in NO | JUDGE or bool(nfkc(r.get("corrected_keigo"))) for r in spot)
    if spot_total:
        rate = spot_bad / len(spot) if spot else 0
        print(f"spot-check: {len(spot)}/{spot_total} reviewed, {spot_bad} needed a change or were dropped"
              + (f" ({rate:.0%})" if spot else ""))
        if a.accept_likely_fine and (len(spot) < min(20, spot_total) or rate > 0.10):
            sys.exit("Not accepting the likely-fine rows: review at least 20 spot_check=CHECK rows first, and the "
                     "error rate must be 10% or less. Otherwise review the likely-fine rows too.")

    rows, why = {}, Counter()
    for sid, r in content.items():
        if sid in decided:
            r, src_file = decided[sid]
            keep, fixed = nfkc(r.get("keep")).lower(), nfkc(r.get("corrected_keigo"))
            how = "reviewed"
        elif a.accept_likely_fine and r.get("bucket") == "1-likely-fine":
            keep, fixed, src_file, how = "y", "", "", "accepted_after_spot_check"
        else:
            why["not reviewed (blank)"] += 1
            continue
        if keep in NO:
            why["marked n"] += 1
            continue
        if keep in JUDGE and not fixed:
            fixed = nfkc(r.get("judge_keigo"))
            if not fixed:
                why["keep=j but no judge_keigo"] += 1
                continue
        elif not (keep in YES or fixed):
            why[f"unknown keep value '{keep}'"] += 1
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
        rows[sid] = {
            "id": sid, "source_text": src, "target_text": tgt, "label_raw": reg, "category": reg, "register": reg,
            "optional": False, "group_id": sid, "is_identity": False, "sft_core": True, "split": "train",
            "origin": "agent_generated", "review": how, "corrected": bool(fixed),
            "used_judge_fix": keep in JUDGE, "bucket": r.get("bucket", ""), "review_file": src_file}

    out = OUT / "generated_train.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} pairs -> {out}")
    print("  by register:", dict(Counter(r["register"] for r in rows.values())))
    print("  by review:", dict(Counter(r["review"] for r in rows.values())))
    print("  corrected:", sum(r["corrected"] for r in rows.values()),
          "(of which judge fixes:", sum(r["used_judge_fix"] for r in rows.values()), ")")
    for k, v in why.items():
        print(f"  skipped {v}: {k}")


if __name__ == "__main__":
    main()
