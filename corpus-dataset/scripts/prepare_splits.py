"""Clean integrated_csv.csv, normalise labels, and write leakage-free grouped splits.

Run from corpus-dataset/:
    python scripts/prepare_splits.py
Outputs:
    processed/full_keigo_corpus.jsonl
    splits/{train,val,test}.jsonl
    splits/checksums.json, splits/report.json
"""
import argparse, hashlib, json, random, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

KEIGO = ("teineigo", "sonkeigo", "kenjogo")
JP_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
PUNCT_RE = re.compile(r"[\s、。，．,.!！?？「」『』（）()\[\]・~〜ー-]+")


def clean_text(s) -> str:
    s = str(s).replace("\ufeff", "")
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def group_key(s: str) -> str:
    """Conservative key for grouping near-identical casual inputs (NFKC, no spaces/punct)."""
    return PUNCT_RE.sub("", unicodedata.normalize("NFKC", s)).lower()


def parse_label(raw):
    """Map the ~33 raw label strings to (category, register, optional_flag, raw_tokens)."""
    s = unicodedata.normalize("NFKC", str(raw)).lower().replace("\r", "").replace("\n", "")
    s = s.replace("身化語", "美化語")  # typo present in the source data
    optional = s.startswith("optional") or "どちらでもよい" in s
    toks = {k for k in (*KEIGO, "no_change", "other", "unclear") if k in s}
    if "美化語" in s:
        toks.add("bikago")
    keigo = [k for k in KEIGO if k in toks]
    if not toks or "unclear" in toks:
        cat = "unclear"
    elif optional:
        cat = "optional"
    elif toks == {"no_change"}:
        cat = "no_change"
    elif len(keigo) == 1 and not (toks - {keigo[0]}):
        cat = keigo[0]
    elif keigo:
        cat = "mixed"
    else:
        cat = "other"
    register = cat if cat in (*KEIGO, "no_change") else None
    return cat, register, optional, sorted(toks)


def assign_splits(groups, fracs, seed):
    """Greedy, category-aware assignment of whole source-groups to splits."""
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    assigned = {sp: Counter() for sp in fracs}
    seen = Counter()
    out = {}
    for k in keys:
        rows = groups[k]
        cat = Counter(r["category"] for r in rows).most_common(1)[0][0]
        n = len(rows)
        best = max(fracs, key=lambda sp: fracs[sp] * (seen[cat] + n) - assigned[sp][cat] + rng.random() * 1e-6)
        out[k] = best
        assigned[best][cat] += n
        seen[cat] += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="processed/integrated_csv.csv")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    out = Path(a.outdir)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "processed").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(a.input, dtype=str, encoding="utf-8-sig").dropna(subset=["source_text", "transformed_text"])
    n_raw = len(df)
    recs, seen, dropped = [], set(), Counter()
    for r in df.itertuples(index=False):
        src, tgt = clean_text(r.source_text), clean_text(r.transformed_text)
        cat, reg, opt, toks = parse_label(r.transformation_label)
        if not JP_RE.search(src):
            dropped["non_japanese_source"] += 1
            continue
        if cat == "unclear":
            dropped["unclear_label"] += 1
            continue
        dk = (src, tgt, cat)
        if dk in seen:
            dropped["exact_duplicate_after_cleaning"] += 1
            continue
        seen.add(dk)
        recs.append({
            "id": hashlib.sha1(f"{src}\t{tgt}".encode()).hexdigest()[:12],
            "source_text": src, "target_text": tgt,
            "label_raw": str(r.transformation_label).replace("\r", "").replace("\n", ""),
            "category": cat, "register": reg, "optional": opt,
            "group_id": hashlib.sha1(group_key(src).encode()).hexdigest()[:12],
            "is_identity": src == tgt,
            # clean single-class rows whose target changed, or true no_change controls
            "sft_core": (cat in KEIGO and src != tgt) or (cat == "no_change" and src == tgt),
        })

    groups = defaultdict(list)
    for x in recs:
        groups[x["group_id"]].append(x)
    fracs = {"train": 0.70, "val": 0.15, "test": 0.15}
    assign = assign_splits(groups, fracs, a.seed)
    splits = defaultdict(list)
    for x in recs:
        x["split"] = assign[x["group_id"]]
        splits[x["split"]].append(x)

    keys = {sp: {x["group_id"] for x in rows} for sp, rows in splits.items()}
    for p, q in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (keys[p] & keys[q]), f"source leakage between {p} and {q}"

    def dump(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    dump(out / "processed/full_keigo_corpus.jsonl", recs)
    sums = {}
    for sp, rows in splits.items():
        p = out / f"splits/{sp}.jsonl"
        dump(p, sorted(rows, key=lambda x: x["id"]))
        sums[f"{sp}.jsonl"] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "splits/checksums.json").write_text(json.dumps(sums, indent=2))

    report = {
        "seed": a.seed, "raw_rows": n_raw, "kept_rows": len(recs), "dropped": dict(dropped),
        "unique_source_groups": len(groups),
        "per_split": {sp: {"rows": len(rows), "groups": len(keys[sp]),
                           "categories": dict(Counter(x["category"] for x in rows)),
                           "sft_core": sum(x["sft_core"] for x in rows)} for sp, rows in splits.items()},
    }
    (out / "splits/report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
