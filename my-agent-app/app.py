import streamlit as st

from src.agent.config import CONDITIONS
from src.agent.graph import graph

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
        st.code(s["drafts"][-1], language="markdown")
        if s["status"] == "needs_review":
            st.warning("The reviewer still had concerns after the maximum revisions. Please check this output manually.")
        elif s["status"] == "approved":
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
