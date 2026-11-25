"""Utility functions to compute BLEU (character‑level) and BERTScore.

These helpers are thin wrappers around the existing ``training/score.py``
logic but return the numeric values so they can be consumed by a UI or
other scripts.
"""

from pathlib import Path
from typing import Tuple, Dict, Any

import sacrebleu

# ``bert_score`` is an optional dependency – import lazily so that the
# module can be used even when the package is not installed (e.g. during
# early development of the fine‑tuned baseline).

def _maybe_import_bert_score():
    try:
        import bert_score
        return bert_score
    except Exception as exc:  # pragma: no cover – defensive
        raise ImportError(
            "bert_score is not installed. Install it via `pip install bert-score`"
        ) from exc


def _read_jsonl(file_path: Path) -> list:
    import json

    with file_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_pairs(pred_file: str, ref_file: str) -> Tuple[list, list]:
    """Pair each prediction with its gold sentence by ``id``.

    Gold files (the corpus splits) store the reference under ``target_text``;
    ``target`` is accepted as a fallback. Predictions whose ``id`` is not in
    the gold file are an error, since that means the wrong split was given.
    Returns parallel lists ``(hypotheses, references)``.
    """
    gold = {}
    for row in _read_jsonl(Path(ref_file)):
        ref = row.get("target_text", row.get("target"))
        if ref is not None:
            gold[row["id"]] = ref
    hyps, refs, missing = [], [], []
    for row in _read_jsonl(Path(pred_file)):
        if row["id"] not in gold:
            missing.append(row["id"])
            continue
        hyps.append(row.get("prediction") or "")
        refs.append(gold[row["id"]])
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(missing) + len(hyps)} predictions in {Path(pred_file).name} "
            f"have no gold sentence in {Path(ref_file).name} (wrong split?)"
        )
    if not hyps:
        raise ValueError(f"{Path(pred_file).name} contains no predictions")
    return hyps, refs


def compute_bleu(pred_file: str, ref_file: str) -> float:
    """Return the character BLEU score (sacreBLEU) for *pred_file*.

    Predictions are matched to gold sentences in *ref_file* by ``id``.
    """
    hyps, refs = load_pairs(pred_file, ref_file)
    # sacrebleu expects a list of reference lists – we have a single
    # reference set, so we nest it.
    bleu = sacrebleu.corpus_bleu(hyps, [refs], force=True, lowercase=False, tokenize="char")
    return bleu.score


def compute_bertscore(pred_file: str, ref_file: str, model_type: str = "bert-base-multilingual-cased", batch_size: int = 64) -> Dict[str, float]:
    """Return a dictionary with precision, recall and F1 BERTScore.

    The function lazily imports ``bert_score``; if the package is missing an
    informative ``ImportError`` is raised.
    """
    bert_score = _maybe_import_bert_score()
    hyps, refs = load_pairs(pred_file, ref_file)
    P, R, F1 = bert_score.score(hyps, refs, model_type=model_type, batch_size=batch_size, lang="ja", verbose=False)
    # Return average values as simple floats.
    return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}

def sentence_bleu(hyp: str, refs: list) -> float:
    """Character BLEU for one sentence against one or more references."""
    return sacrebleu.sentence_bleu(hyp, list(refs), tokenize="char").score


def make_bertscorer(model_type: str = "bert-base-multilingual-cased"):
    """Load the BERTScore model once; reuse it via ``bertscore_one``."""
    return _maybe_import_bert_score().BERTScorer(model_type=model_type, lang="ja")


def bertscore_one(scorer, hyp: str, refs: list) -> Dict[str, float]:
    """BERTScore of one sentence against one or more references (best-matching reference)."""
    P, R, F1 = scorer.score([hyp], [list(refs)])
    return {"precision": float(P[0]), "recall": float(R[0]), "f1": float(F1[0])}


if __name__ == "__main__":  # pragma: no cover – helper for manual testing
    import argparse, json

    parser = argparse.ArgumentParser(description="Compute BLEU and BERTScore for a prediction file.")
    parser.add_argument("--pred", required=True, help="Path to the prediction JSONL file.")
    parser.add_argument("--ref", required=True, help="Path to the gold‑standard JSONL file.")
    parser.add_argument("--bertscore", action="store_true", help="Also compute BERTScore.")
    args = parser.parse_args()

    bleu = compute_bleu(args.pred, args.ref)
    out: Dict[str, Any] = {"bleu": bleu}
    if args.bertscore:
        out.update(compute_bertscore(args.pred, args.ref))
    print(json.dumps(out, ensure_ascii=False, indent=2))
