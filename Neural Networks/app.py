"""
app.py — Palmer Penguins MLP Dashboard
Run with: streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="Palmer Penguins — MLP Classifier",
    page_icon="🐧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Palmer Penguins — MLP Species Classifier")
st.markdown("A full end-to-end neural network pipeline for species prediction. Each tab corresponds to one part of the assignment.")
st.divider()

# Load data once
from data import load_data
df, X, y, X_train, X_test, y_train, y_test = load_data()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA",
    "⚙️ Pipeline",
    "🔍 Grid Search",
    "⚖️ Bias-Variance",
    "🚀 Deploy & Predict",
])

with tab1:
    import part1
    part1.render(df, X, y)

with tab2:
    import part2
    baseline_pipeline = part2.render(X_train, X_test, y_train, y_test)

with tab3:
    import part3
    best_pipeline = part3.render(baseline_pipeline, X_train, X_test, y_train, y_test)

with tab4:
    import part4
    part4.render(df)

with tab5:
    import part5
    part5.render(best_pipeline, X_test, y_test)
