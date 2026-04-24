"""
part5.py — Deploying the Underwriting Model
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.metrics import accuracy_score

NUM_FEATURES = ["age", "fare", "sibsp", "parch"]
CAT_FEATURES = ["sex", "embarked", "pclass"]


def render(best_pipeline, X_test, y_test):
    st.header("Deploying the Underwriting Model")
    st.caption("The model is approved. Now we make it production-ready: interpret it, persist it, and score new applicants automatically.")

    # Feature Importances
    st.subheader("Feature Importances")

    ohe_cats = (best_pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["ohe"]
                .get_feature_names_out(CAT_FEATURES))
    all_features = NUM_FEATURES + list(ohe_cats)
    importances  = best_pipeline.named_steps["decisiontreeclassifier"].feature_importances_

    fi = (pd.Series(importances, index=all_features)
            .sort_values(ascending=False)
            .pipe(lambda s: s[s > 0]))

    fi_plot = fi.sort_values(ascending=True)
    cmap    = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(fi_plot)))

    fig, ax = plt.subplots(figsize=(9, max(4, len(fi_plot) * 0.42)))
    bars = ax.barh(fi_plot.index, fi_plot.values, color=cmap, edgecolor="white")
    for bar, v in zip(bars, fi_plot.values):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("Importance (Gini reduction)")
    ax.set_title("Feature Importances — Tuned Decision Tree", fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    fig.patch.set_facecolor("#f9f9f9")
    st.pyplot(fig, width='stretch')

    # Top 3 table
    top3 = fi.head(3).reset_index()
    top3.columns = ["Feature","Importance"]
    top3["Banking Context"] = [
        "Sex (female) — strongest predictor by far. Female passengers survived at ~74%. As a banking proxy it's powerful, but sex is a legally protected characteristic and would be excluded from a real credit model.",
        "Pclass 1 — first-class passengers represent the highest-income, most-resourced applicants, analogous to a prime credit tier. High importance confirms wealth is a fundamental risk separator.",
        "Pclass 3 — third-class passengers represent the highest-risk segment. Its symmetric importance with Pclass 1 shows the model learned to differentiate socioeconomic extremes — a sensible credit stratification.",
    ]
    top3["Importance"] = top3["Importance"].round(4)
    st.dataframe(top3, hide_index=True, width='stretch')

    # Save & Reload
    st.subheader("Saving & Reloading the Pipeline")

    buf = io.BytesIO()
    joblib.dump(best_pipeline, buf)
    buf.seek(0)
    reloaded = joblib.load(buf)

    acc_orig     = accuracy_score(y_test, best_pipeline.predict(X_test))
    acc_reloaded = accuracy_score(y_test, reloaded.predict(X_test))
    identical    = np.array_equal(best_pipeline.predict(X_test), reloaded.predict(X_test))

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Accuracy",  f"{acc_orig:.4f}")
    col2.metric("Reloaded Accuracy",  f"{acc_reloaded:.4f}")
    col3.metric("Predictions Match",  " Yes" if identical else " No")

    buf.seek(0)
    st.download_button(
        label="️  Download clearbank_pipeline.joblib",
        data=buf,
        file_name="clearbank_pipeline.joblib",
        mime="application/octet-stream",
    )

    st.code("""
# How to use it in production:
import joblib

pipeline = joblib.load("clearbank_pipeline.joblib")
predictions = pipeline.predict(new_applicants_df)
probabilities = pipeline.predict_proba(new_applicants_df)
    """, language="python")

    # Morning Batch Scoring
    st.subheader("Batch Scoring New Applicants")
    st.caption("Simulate the daily applicant feed. Edit values and click Score.")

    st.markdown("**Preset applicants (from the assignment):**")
    default_applicants = pd.DataFrame({
        "pclass":   [1, 3],
        "sex":      ["female", "male"],
        "age":      [34.0, 28.0],
        "sibsp":    [1, 0],
        "parch":    [0, 0],
        "fare":     [83.0, 7.9],
        "embarked": ["S", "Q"],
    })

    edited = st.data_editor(
        default_applicants,
        width='stretch',
        num_rows="dynamic",
        column_config={
            "pclass":   st.column_config.SelectboxColumn("Pclass", options=[1,2,3]),
            "sex":      st.column_config.SelectboxColumn("Sex", options=["female","male"]),
            "embarked": st.column_config.SelectboxColumn("Embarked", options=["S","C","Q"]),
            "age":      st.column_config.NumberColumn("Age", min_value=0.0, max_value=80.0),
            "fare":     st.column_config.NumberColumn("Fare (£)", min_value=0.0),
            "sibsp":    st.column_config.NumberColumn("Siblings/Spouse", min_value=0),
            "parch":    st.column_config.NumberColumn("Parents/Children", min_value=0),
        }
    )

    if st.button("Score Applicants", type="primary"):
        proba = reloaded.predict_proba(edited)
        preds = reloaded.predict(edited)

        st.markdown("**Underwriting Decisions:**")
        for i, (pred, prob) in enumerate(zip(preds, proba)):
            if pred == 1:
                st.success(f"**Applicant {i+1} —  APPROVED** &nbsp;&nbsp; Repay confidence: **{prob[1]:.0%}**")
            else:
                st.error(f"**Applicant {i+1} —  DECLINED** &nbsp;&nbsp; Default risk: **{prob[0]:.0%}**")

        results = edited.copy()
        results["Decision"]    = ["APPROVED" if p==1 else "DECLINED" for p in preds]
        results["Repay Prob"]  = [f"{p[1]:.1%}" for p in proba]
        results["Default Risk"] = [f"{p[0]:.1%}" for p in proba]
        st.dataframe(results, width='stretch', hide_index=True)

    # How It Works
    st.subheader("How It Works")

    st.info("""
    Every morning, the system receives a spreadsheet of new applicants with their demographics 
    and financial details — exactly as they come from the application form, raw and unprocessed. 
    The Pipeline automatically handles all the behind-the-scenes data preparation: missing ages 
    are filled using the historical median, text fields like "male/female" or "Southampton" are 
    converted into numbers the model can read, and all figures are rescaled so that a £500 fare 
    and a 2-sibling count are treated on comparable terms. 
    
    The underwriting team never needs to touch or transform the data manually — they simply 
    pass the raw applicant file in, and within seconds the model outputs a scored decision 
    and confidence level for each person. This means a consistent, auditable process every 
    day with no room for human preprocessing error or inconsistent manual judgement.
    """)