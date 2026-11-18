import streamlit as st
import sys
import pathlib

# Add project root to sys.path so sibling packages (e.g., training) are importable
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.agent.config import CONDITIONS
from src.agent.graph import graph
from training.compute_metrics import compute_bleu, compute_bertscore

st.set_page_config(page_title="Keigo Master", page_icon="🎌")
st.title("🎌 Keigo Master (ビジネス日本語コーチ)")
st.write("Enter casual Japanese below to convert it into business keigo.")

with st.sidebar:
    st.header("Pipeline")
    cond = st.selectbox("Preset", ["A3 (full)", "A2 (no reviewer)", "A1 (retrieval only)", "B1 (single prompt)", "A4 (no retrieval)"])
    reg = st.selectbox("Requested register", ["auto", "teineigo", "sonkeigo", "kenjogo"])
    scenario = st.text_input("Scenario (optional)", placeholder="例: 取引先の部長へのメール")

text = st.text_area("Your Japanese input:", placeholder="明日、ミーティングに行けないです。")

if st.button("Refine Communication"):
    if not text.strip():
        st.warning("Please type something first!")
    else:
        with st.spinner("Analyzing and generating..."):
            try:
                s = graph.invoke(
                    {"raw_input": text, "scenario": scenario, "requested_register": None if reg == "auto" else reg,
                     "drafts": [], "usage": []},
                    config={"configurable": CONDITIONS[cond.split()[0]]})
            except Exception as e:
                st.error(f"Pipeline failed: {type(e).__name__}. Check API keys / rate limits.")
                st.stop()
        st.subheader("Refined Keigo")
        output_text = s.get("final_output") or s["drafts"][-1]
        st.code(output_text, language="markdown")
        if s.get("fallback_triggered"):
            st.info("Conservative Teineigo fallback applied after maximum review iterations.")
        elif s.get("status") == "needs_review":
            st.warning("The reviewer still had concerns after the maximum revisions. Please check this output manually.")
        elif s.get("status") == "approved":
            st.success(f"Approved after {s['revision_count']} revision(s).")
        if s.get("explanation"):
            st.subheader("Why these forms?")
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
agent_pred_path = PROJECT_ROOT / "my-agent-app" / "predictions" / "A3.jsonl"
gold_path       = PROJECT_ROOT / "corpus-dataset" / "splits" / "val.jsonl"
# Track B placeholder – your coworker writes predictions here
baseline_pred_path = PROJECT_ROOT / "training" / "predictions" / "B2.jsonl"

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
                    baseline_bleu = compute_bleu(str(baseline_pred_path), str(gold_path))
                    baseline_bert = compute_bertscore(str(baseline_pred_path), str(gold_path))
                    st.metric("BLEU (char)", f"{baseline_bleu:.2f}")
                    st.metric("BERTScore F1", f"{baseline_bert['f1']:.4f}")
                    st.metric("BERTScore P",  f"{baseline_bert['precision']:.4f}")
                    st.metric("BERTScore R",  f"{baseline_bert['recall']:.4f}")
                except Exception as e:
                    st.error(f"Error computing Fine-tuned model metrics: {e}")
