"""
tabs/deploy.py — Deploying the Attrition Model
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.metrics import accuracy_score

CAT_FEATURES = ["BusinessTravel", "Department", "EducationField", "Gender",
                "JobRole", "MaritalStatus", "OverTime"]


def render(best_pipeline, X, X_test, y_test):
    st.header("Deploying the Attrition Model")
    st.caption("The model is approved. Now we interpret it, persist it, and score new employees.")

    # Feature importances via linear SVM coefficients
    st.subheader("Feature Importance (Linear SVM Coefficients)")

    ohe_cats = (best_pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .get_feature_names_out(CAT_FEATURES))

    num_features = [
        "Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
        "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
        "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
        "YearsSinceLastPromotion", "YearsWithCurrManager",
    ]
    all_features = num_features + list(ohe_cats)

    clf = best_pipeline.named_steps["classifier"]
    if hasattr(clf, "coef_"):
        coefficients = clf.coef_[0]
    else:
        # RBF doesn't expose coef_ — fall back to permutation importance approximation
        st.info("RBF kernel does not expose linear coefficients. Showing absolute dual coefficient magnitude as a proxy.")
        coefficients = np.abs(clf.dual_coef_).sum(axis=0)
        all_features = [f"SV {i}" for i in range(len(coefficients))]

    importance = pd.Series(np.abs(coefficients), index=all_features).sort_values(ascending=False).head(15)
    importance_plot = importance.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(3.5, len(importance_plot) * 0.38)))
    cmap = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(importance_plot)))
    bars = ax.barh(importance_plot.index, importance_plot.values, color=cmap, edgecolor="white")
    for bar, v in zip(bars, importance_plot.values):
        ax.text(v + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("|Coefficient| (linear SVM)")
    ax.set_title("Top Feature Importances — Tuned SVM", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("#f9f9f9")
    st.pyplot(fig)

    # Save & reload
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
    col3.metric("Predictions Match",  "Yes" if identical else "No")

    buf.seek(0)
    st.download_button(
        label="Download attrition_pipeline.joblib",
        data=buf,
        file_name="attrition_pipeline.joblib",
        mime="application/octet-stream",
    )

    st.code("""
import joblib

pipeline = joblib.load("attrition_pipeline.joblib")
predictions  = pipeline.predict(new_employees_df)
probabilities = pipeline.predict_proba(new_employees_df)
    """, language="python")

    # Scoring form
    st.subheader("Score a New Employee")
    st.caption("Fill in the values below and click Score. The pipeline handles all preprocessing internally.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        age      = st.number_input("Age",           18, 60,  30)
        gender   = st.selectbox("Gender",           sorted(X["Gender"].unique()))
        marital  = st.selectbox("Marital Status",   sorted(X["MaritalStatus"].unique()))
        edu_field = st.selectbox("Education Field", sorted(X["EducationField"].unique()))

    with col2:
        st.markdown("**Job & Compensation**")
        dept         = st.selectbox("Department",      sorted(X["Department"].unique()))
        job_role     = st.selectbox("Job Role",        sorted(X["JobRole"].unique()))
        monthly_inc  = st.number_input("Monthly Income ($)", 1009, 19999, 5000, 500)
        stock_opt    = st.selectbox("Stock Option Level", sorted(X["StockOptionLevel"].unique()))
        overtime     = st.selectbox("Overtime",        sorted(X["OverTime"].unique()))
        travel       = st.selectbox("Business Travel", sorted(X["BusinessTravel"].unique()))

    with col3:
        st.markdown("**Tenure & Satisfaction**")
        yrs_company  = st.slider("Years at Company",       0, 40,  5)
        job_sat      = st.select_slider("Job Satisfaction (1–4)",
                                        options=sorted(X["JobSatisfaction"].unique()))
        wlb          = st.select_slider("Work-Life Balance (1–4)",
                                        options=sorted(X["WorkLifeBalance"].unique()))
        env_sat      = st.select_slider("Environment Satisfaction (1–4)",
                                        options=sorted(X["EnvironmentSatisfaction"].unique()))

    # Build row with all required features, defaulting to medians/modes
    employee = {}
    for col in X.columns:
        if X[col].dtype == object:
            employee[col] = X[col].mode()[0]
        else:
            employee[col] = int(X[col].median())

    employee.update({
        "Age": age, "Gender": gender, "MaritalStatus": marital,
        "EducationField": edu_field, "Department": dept, "JobRole": job_role,
        "MonthlyIncome": monthly_inc, "StockOptionLevel": stock_opt,
        "OverTime": overtime, "BusinessTravel": travel,
        "YearsAtCompany": yrs_company, "JobSatisfaction": job_sat,
        "WorkLifeBalance": wlb, "EnvironmentSatisfaction": env_sat,
    })

    if st.button("Score Employee", type="primary"):
        proba = reloaded.predict_proba(pd.DataFrame([employee]))[0]
        pred  = reloaded.predict(pd.DataFrame([employee]))[0]

        st.divider()
        if pred == 1:
            st.error(f"**ATTRITION RISK — Likely to Leave** &nbsp;&nbsp; Attrition probability: **{proba[1]:.0%}**")
        else:
            st.success(f"**LOW RISK — Likely to Stay** &nbsp;&nbsp; Retention probability: **{proba[0]:.0%}**")

        col1, col2 = st.columns(2)
        col1.metric("P(Leave)", f"{proba[1]:.1%}")
        col2.metric("P(Stay)",  f"{proba[0]:.1%}")
        st.progress(float(proba[1]), text=f"Attrition probability: {proba[1]:.1%}")

    # How it works
    st.subheader("How It Works")
    st.info("""
    Every day, HR receives a file of employee records — exactly as they come from the HR system, 
    raw and unprocessed. The Pipeline automatically handles all data preparation: categorical fields 
    like Department and JobRole are one-hot encoded, numeric fields are scaled so that Monthly Income 
    and Years at Company are treated on comparable terms, and any unknown categories are handled 
    gracefully without errors.

    HR never needs to transform the data manually. They pass in the raw employee file and within 
    seconds the model outputs an attrition risk score and decision for each person — a consistent, 
    auditable process every day.
    """)