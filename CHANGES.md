# SmartKeigo update – what changed and why

Drop these folders over your repo (same paths). Old `utils.py` is replaced by `retrieval.py`.

## Bugs fixed in the original agent
| Problem | Fix |
|---|---|
| Rejection loop had no cap (would hit LangGraph's recursion limit) | `max_revisions=2`, ends with `status="needs_review"` |
| Specialist never saw the reviewer's feedback or prior draft on retry | Both are injected into the revision prompt (tested) |
| Approval by `"OK" in text` | Pydantic `ReviewResult(approved: bool, issues[])` |
| `audience` was a free-text paragraph | Structured `AnalysisResult` (addressee, per-actor action + needed register, unknowns) |
| Pinecone index held 1 dummy template; `initialize_pinecone_index()` ran on every Streamlit rerun | `scripts/ingest_train.py` embeds **train.jsonl only**; guard refuses non-train files |
| `embeddings.float[0]` vs `float_[0]` inconsistency | single `_vecs()` helper |
| `langgraph.json` pointed at `graph`, code exported `keigo_app` | exports `graph` (+ alias) |
| No logging for the cost question (RQ4) | every call logs seconds / calls / tokens / failures in `state["usage"]` |
| Explanations promised in README never shown | specialist returns `explanation`, shown in UI |

## Ablations without code changes
`config.CONDITIONS` maps B1, A1–A4 to flags passed via `config={"configurable": ...}`.
`scripts/run_eval.py` runs them over a split (3 trials, failures kept in the denominator).

## Data
`corpus-dataset/scripts/prepare_splits.py` (already run, seed 42): 922 → 909 rows (9 `unclear`, 4 post-cleaning duplicates dropped),
33 raw label strings → 8 categories, grouped 70/15/15 split with zero source overlap, checksums + report.

Test split keigo items: teineigo 28, sonkeigo 13, kenjogo 11 (+47 no_change). That is small; report confidence intervals.

## Metric warning (found while testing score.py)
Copying the input unchanged gets **char-BLEU 82** on the items that *should* change, because keigo edits are only a few characters.
Report exact-match / edit-level accuracy and the human scores as primary; treat BLEU as secondary.

## Verified vs not verified
- Verified here: data prep run, graph logic (4 offline tests with a scripted fake LLM), scorer on the copy baseline, syntax of all scripts.
- NOT run here (no Groq/Cohere/Pinecone/HuggingFace access): live LLM calls, structured-output behaviour of the Groq model (now `openai/gpt-oss-120b`; Llama 3.3 70B is no longer offered), T5 training, BERTScore.
  First thing to try: `python -m scripts.run_eval --split ../corpus-dataset/splits/val.jsonl --conditions A3 --trials 1 --limit 5`.

## Known data issues still to fix by hand
- Some targets are wrong (e.g. 食事を食べる → ご食事を食べます). Review the ~450 `sft_core` train rows before training.
- Tohoku rows are learner-written sentences, not business casual; consider tagging domain.
- Alternate targets for one source are kept as multi-references in scoring; training sees them as separate pairs.
