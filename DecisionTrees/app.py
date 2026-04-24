"""
app.py — ClearBank Underwriting Model Dashboard
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="ClearBank Underwriting Model",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Header
st.title("ClearBank — Loan Default Prediction Model")
st.markdown(
    "A full end-to-end ML pipeline for credit risk scoring, "
    "built on the Titanic dataset. Each tab corresponds to one part of the assignment."
)
st.divider()

#  Load data once 
from data import load_data
df, X, y, X_train, X_test, y_train, y_test = load_data()

#  Tabs 
tab2, tab3, tab4, tab5 = st.tabs([
    "EDA",
    "Pipeline",
    "Tuning",
    "Deploy",
])

with tab2:
    import part2
    part2.render(df, X, y)

with tab3:
    import part3
    baseline_pipeline = part3.render(X_train, X_test, y_train, y_test)

with tab4:
    import part4
    best_pipeline = part4.render(baseline_pipeline, X_train, X_test, y_train, y_test)

with tab5:
    import part5
    part5.render(best_pipeline, X_test, y_test)