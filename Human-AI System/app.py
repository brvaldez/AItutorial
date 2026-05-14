"""
app.py — Human-AI Text Classification Triage System
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="Human-AI Text Classification Triage", layout="wide")

# ── Load data and models ──────────────────────────────────────────────────────

from data import load_data, build_models

with st.spinner("Loading dataset and training models (first run may take ~60 s)…"):
    data_train, data_test = load_data()
    baseline, gs, best_pipeline, X_train, X_val, y_train, y_val = build_models(data_train)

X_test, y_test = data_test.data, data_test.target

# ── Header ────────────────────────────────────────────────────────────────────

st.title("Human-AI Text Classification Triage System")
st.markdown(
    "Automatically categorises 20-Newsgroups posts into one of **four topic groups** "
    "and routes uncertain predictions to a human reviewer."
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "EDA",
    "Baseline Pipeline",
    "Grid Search",
    "Human-in-the-Loop",
    "Live Demo",
])

with tab1:
    import part1
    part1.render(data_train, data_test)

with tab2:
    import part2
    part2.render(baseline, X_val, y_val)

with tab3:
    import part3
    part3.render(gs, best_pipeline, X_test, y_test)

with tab4:
    import part4
    threshold = part4.render(best_pipeline, X_test, y_test)

with tab5:
    import part5
    part5.render(best_pipeline, threshold)
