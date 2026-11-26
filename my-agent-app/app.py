import streamlit as st
import sys
import pathlib

# Add project root to sys.path so sibling packages (e.g., training) are importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

from src.agent.config import CONDITIONS
from src.agent.graph import graph
from training.compute_metrics import (bertscore_one, compute_bleu, compute_bertscore, load_pairs,
                                      make_bertscorer, sentence_bleu)

st.set_page_config(page_title="Keigo Master", page_icon="🎌", layout="wide")
st.title("🎌 Keigo Master (ビジネス日本語コーチ)")
st.write("Enter casual Japanese below. The agent (Track A) and the fine-tuned T5 model (Track B) "
         "formalize it side by side.")

SPLITS_DIR = PROJECT_ROOT / "corpus-dataset" / "splits"


@st.cache_resource(show_spinner="Loading fine-tuned T5 model...")
def load_t5():
    from training.t5_infer import T5Keigo
    return T5Keigo()


@st.cache_resource(show_spinner="Loading BERTScore model...")
def load_bertscorer():
    return make_bertscorer()


@st.cache_data
def load_corpus_refs():
    """NFKC source sentence -> list of (register, gold keigo) from all corpus splits."""
    refs = {}
    for split in ("train", "val", "test"):
        p = SPLITS_DIR / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("target_text"):
                refs.setdefault(unicodedata.normalize("NFKC", r["source_text"]).strip(), []).append(
                    (r.get("register"), r["target_text"]))
    return refs


def run_agent(text, scenario, register, cond_key):
    t0 = time.perf_counter()
    s = graph.invoke(
        {"raw_input": text, "scenario": scenario, "requested_register": register, "drafts": [], "usage": []},
        config={"configurable": CONDITIONS[cond_key]})
    return s, time.perf_counter() - t0


def run_t5(model, text, register):
    t0 = time.perf_counter()
    return model.formalize(text, register), time.perf_counter() - t0


with st.sidebar:
    st.header("Pipeline")
    cond = st.selectbox("Agent preset", ["A3 (full)", "A2 (no reviewer)", "A1 (retrieval only)", "B1 (single prompt)", "A4 (no retrieval)"])
    reg = st.selectbox("Requested register", ["auto", "teineigo", "sonkeigo", "kenjogo"],
                       help="'auto' lets the agent decide. T5 always needs a register: it starts with teineigo "
                            "and is re-run with the agent's choice if that differs.")
    scenario = st.text_input("Scenario (optional)", placeholder="例: 取引先の部長へのメール")

text = st.text_area("Your Japanese input:", placeholder="おい、こんにちは先生。今週の宿題の期限はいつだっけ？")
reference = st.text_input(
    "Reference keigo (optional)",
    help="Correct keigo version to score both outputs against. Left empty, the app uses the corpus "
         "reference if this exact sentence is in the dataset; otherwise it compares the two outputs to each other.")

if st.button("Formalize", type="primary"):
    if not text.strip():
        st.warning("Please type something first!")
        st.stop()
    register = None if reg == "auto" else reg
    t5_model, scorer = load_t5(), load_bertscorer()

    # Run the agent (network-bound) and T5 (local CPU) at the same time.
    with st.spinner("Running agent and fine-tuned model in parallel..."):
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            agent_future = pool.submit(run_agent, text, scenario, register, cond.split()[0])
            t5_reg = register or "teineigo"
            t5_future = pool.submit(run_t5, t5_model, text, t5_reg)
            t5_out, t5_secs = t5_future.result()
            try:
                s, agent_secs = agent_future.result()
                agent_err = None
            except Exception as e:
                s, agent_secs, agent_err = None, 0.0, e
        # In auto mode, give T5 the register the agent chose so both target the same politeness level.
        agent_reg = ((s or {}).get("context") or {}).get("overall_register")
        if register is None and agent_reg in ("sonkeigo", "kenjogo") and agent_reg != t5_reg:
            t5_reg = agent_reg
            t5_out, extra = run_t5(t5_model, text, t5_reg)
            t5_secs += extra
        wall = time.perf_counter() - t_start

    agent_out = None if s is None else (s.get("final_output") or s["drafts"][-1])

    # ---- Pick what to score against ----
    if reference.strip():
        refs, ref_label = [reference.strip()], "your reference"
    else:
        hits = load_corpus_refs().get(unicodedata.normalize("NFKC", text).strip(), [])
        matched = [t for r, t in hits if register is None or r == register] or [t for _, t in hits]
        refs, ref_label = matched, f"corpus reference ({len(matched)})" if matched else None

    st.caption(f"Total {wall:.1f}s (agent {agent_secs:.1f}s, T5 {t5_secs:.1f}s, run in parallel)")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("🤖 Agent AI (Track A)")
        if agent_err is not None:
            st.error(f"Agent failed: {type(agent_err).__name__}. Check API keys / Groq rate limits.")
        else:
            st.code(agent_out, language="markdown")
            if agent_reg:
                st.caption(f"Register: {agent_reg}")
            llm_errors = [f'{u["node"]}: {u["error"]}' for u in s.get("usage", []) if u.get("error")]
            if s.get("fallback_triggered") and llm_errors:
                st.error("The language model call failed, so this is only the rule-based fallback applied to your "
                         "input, not a real agent result. Try again; if it keeps failing, check Groq rate limits.")
                with st.expander("Error details"):
                    st.write(llm_errors)
            elif s.get("fallback_triggered"):
                st.info("Conservative Teineigo fallback applied after maximum review iterations.")
            elif s.get("status") == "needs_review":
                st.warning("The reviewer still had concerns after the maximum revisions. Please check this output manually.")
            elif s.get("status") == "approved":
                st.success(f"Approved after {s['revision_count']} revision(s).")

    with col_b:
        st.subheader("🧠 Fine-tuned T5 (Track B)")
        st.code(t5_out, language="markdown")
        st.caption(f"Register: {t5_reg}")

    # ---- Scores ----
    st.subheader("📏 BLEU & BERTScore")
    if refs:
        st.caption(f"Each output scored against {ref_label}: " + " / ".join(refs))
        cols = st.columns(2)
        for col, name, out in ((cols[0], "Agent", agent_out), (cols[1], "Fine-tuned T5", t5_out)):
            with col:
                if out is None:
                    st.write(f"{name}: no output")
                    continue
                b = bertscore_one(scorer, out, refs)
                m1, m2 = st.columns(2)
                m1.metric(f"{name} BLEU (char)", f"{sentence_bleu(out, refs):.2f}")
                m2.metric(f"{name} BERTScore F1", f"{b['f1']:.4f}")
    elif agent_out is not None:
        st.caption("No reference for this sentence, so the two outputs are compared to each other "
                   "(agent output as reference). This shows how similar they are, not which is correct. "
                   "Add a reference keigo above to score them.")
        b = bertscore_one(scorer, t5_out, [agent_out])
        m1, m2 = st.columns(2)
        m1.metric("Agreement BLEU (char)", f"{sentence_bleu(t5_out, [agent_out]):.2f}")
        m2.metric("Agreement BERTScore F1", f"{b['f1']:.4f}")
    else:
        st.write("No reference and no agent output, so there is nothing to score against.")

    if s is not None:
        if s.get("explanation"):
            st.subheader("Why these forms? (agent)")
            st.write(s["explanation"])
        if s.get("context"):
            with st.expander("Context analysis"):
                st.json(s["context"])
        if s.get("feedback"):
            with st.expander("Outstanding reviewer issues"):
                st.json(s["feedback"])
        with st.expander("Run details"):
            u = s["usage"]
            st.write({"calls": sum(x["calls"] for x in u), "seconds": round(sum(x["seconds"] for x in u), 2),
                      "tokens": sum(x["input_tokens"] + x["output_tokens"] for x in u), "drafts": s["drafts"],
                      "retrieved_ids": s.get("retrieved_ids", [])})

# ----- Metrics UI (Agent vs Fine-tuned placeholder) -----
st.divider()
st.header("📊 Evaluation Metrics: Agent AI vs Fine-tuned Model")

# Fix #4: correct prediction path (run_eval.py writes here)
agent_pred_path = PROJECT_ROOT / "my-agent-app" / "predictions" / "test" / "A3.jsonl"
gold_path       = PROJECT_ROOT / "corpus-dataset" / "splits" / "test.jsonl"
# Track B: predict_t5.py writes here, from the test split by default
baseline_pred_path = PROJECT_ROOT / "training" / "predictions" / "B2.jsonl"
baseline_gold_path = PROJECT_ROOT / "corpus-dataset" / "splits" / "test.jsonl"
st.caption("Both tracks are scored on the test split, matched to gold sentences by id. "
           "T5 has 2 extra test sentences (99 vs 97), so the totals differ very slightly.")

if st.button("Compute BLEU & BERTScore"):
    col1, col2 = st.columns(2)

    # --- Track A: Agent AI ---
    with col1:
        st.subheader("🤖 Agent AI (Track A)")
        if not agent_pred_path.exists() or not gold_path.exists():
            st.error("Agent predictions not found. Run `scripts/run_eval.py` first.")
        else:
            with st.spinner("Computing Agent metrics..."):
                try:
                    bleu = compute_bleu(str(agent_pred_path), str(gold_path))
                    bert = compute_bertscore(str(agent_pred_path), str(gold_path))
                    st.caption(f"{len(load_pairs(str(agent_pred_path), str(gold_path))[0])} sentences · {gold_path.name}")
                    st.metric("BLEU (char)", f"{bleu:.2f}")
                    st.metric("BERTScore F1", f"{bert['f1']:.4f}")
                    st.metric("BERTScore P",  f"{bert['precision']:.4f}")
                    st.metric("BERTScore R",  f"{bert['recall']:.4f}")
                except Exception as e:
                    st.error(f"Error computing Agent metrics: {e}")

    # --- Track B: Fine-tuned Model (placeholder) ---
    with col2:
        st.subheader("🧠 Fine-tuned Model (Track B)")
        if not baseline_pred_path.exists():
            st.info("⏳ Fine-tuned model predictions not yet available.\n\n"
                    "Your coworker should write predictions to:\n"
                    "`training/predictions/B2.jsonl`\n\n"
                    "Schema required:\n"
                    "```json\n"
                    "{\"id\": \"...\", \"source_text\": \"...\", \"register\": \"teineigo\",\n"
                    " \"prediction\": \"...\", \"target\": \"...\", \"status\": \"ok\",\n"
                    " \"calls\": 0, \"seconds\": 0, \"tokens\": 0}\n"
                    "```")
        else:
            with st.spinner("Computing Fine-tuned model metrics..."):
                try:
                    baseline_bleu = compute_bleu(str(baseline_pred_path), str(baseline_gold_path))
                    baseline_bert = compute_bertscore(str(baseline_pred_path), str(baseline_gold_path))
                    st.caption(f"{len(load_pairs(str(baseline_pred_path), str(baseline_gold_path))[0])} sentences · {baseline_gold_path.name}")
                    st.metric("BLEU (char)", f"{baseline_bleu:.2f}")
                    st.metric("BERTScore F1", f"{baseline_bert['f1']:.4f}")
                    st.metric("BERTScore P",  f"{baseline_bert['precision']:.4f}")
                    st.metric("BERTScore R",  f"{baseline_bert['recall']:.4f}")
                except Exception as e:
                    st.error(f"Error computing Fine-tuned model metrics: {e}")
