"""
app.py — HR Attrition SVM Dashboard
Run with: streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="HR Attrition — SVM Model",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("HR Attrition — SVM Prediction Model")
st.markdown(
    "End-to-end SVM pipeline for employee attrition prediction, "
    "built on the IBM HR Analytics dataset. Each tab corresponds to one part of the assignment."
)
st.divider()

from data import load_data
df, X, y, X_train, X_test, y_train, y_test = load_data()

tab_eda, tab_pipeline, tab_tuning, tab_deploy = st.tabs([
    "EDA",
    "Pipeline",
    "Tuning",
    "Deploy",
])

with tab_eda:
    import eda
    eda.render(df, X, y)

with tab_pipeline:
    import pipeline as pipeline_tab
    baseline_pipeline = pipeline_tab.render(X_train, X_test, y_train, y_test)

with tab_tuning:
    import tuning
    best_pipeline = tuning.render(baseline_pipeline, X_train, X_test, y_train, y_test)

with tab_deploy:
    import deploy
    deploy.render(best_pipeline, X, X_test, y_test)
