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


def _load_jsonl(file_path: Path) -> Tuple[list, list]:
    """Read a JSONL file containing ``id`` and ``prediction`` fields.

    Returns two parallel lists: ``references`` and ``hypotheses``. The
    reference texts are expected to be stored under the key ``target`` in
    the same file (the gold standard) or can be supplied separately –
    for simplicity we assume the file already contains the reference in
    ``target``.
    """
    import json

    references: list[str] = []
    hypotheses: list[str] = []
    with file_path.open(encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            # ``target`` holds the gold sentence; ``prediction`` the model output.
            references.append(data.get("target", ""))
            hypotheses.append(data.get("prediction", ""))
    return references, hypotheses


def compute_bleu(pred_file: str, ref_file: str) -> float:
    """Return the character BLEU score (sacreBLEU) for *pred_file*.

    ``pred_file`` and ``ref_file`` are paths to JSONL files with the same
    ordering of examples.  The function loads the ``prediction`` column
    from *pred_file* and the ``target`` column from *ref_file* and
    computes a character‑level BLEU using ``sacrebleu.corpus_bleu``.
    """
    pred_path = Path(pred_file)
    ref_path = Path(ref_file)
    _, hyps = _load_jsonl(pred_path)
    refs, _ = _load_jsonl(ref_path)
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
    pred_path = Path(pred_file)
    ref_path = Path(ref_file)
    refs, hyps = _load_jsonl(ref_path)[0], _load_jsonl(pred_path)[1]
    P, R, F1 = bert_score.score(hyps, refs, model_type=model_type, batch_size=batch_size, lang="ja", verbose=False)
    # Return average values as simple floats.
    return {"precision": float(P.mean()), "recall": float(R.mean()), "f1": float(F1.mean())}

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
