"""
tabs/pipeline.py — Building the SVM Pipeline
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

NUM_FEATURES = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome",
                "MonthlyRate", "NumCompaniesWorked", "PercentSalaryHike", "TotalWorkingYears",
                "TrainingTimesLastYear", "YearsAtCompany", "YearsInCurrentRole",
                "YearsSinceLastPromotion", "YearsWithCurrManager"]
CAT_FEATURES = ["BusinessTravel", "Department", "EducationField", "Gender",
                "JobRole", "MaritalStatus", "OverTime"]


@st.cache_resource
def build_and_train(_X_train, _y_train):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   SVC(kernel="linear", probability=True,
                             class_weight="balanced", random_state=42)),
    ])
    pipeline.fit(_X_train, _y_train)
    return pipeline


def render(X_train, X_test, y_train, y_test):
    st.header("Building the Pipeline")
    st.caption("A Pipeline chains preprocessing and the SVM classifier into one object — preventing data leakage and making deployment clean.")

    pipeline = build_and_train(X_train, y_train)

    # Architecture
    st.subheader("Pipeline Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric sub-pipeline**")
        st.code("""
ColumnTransformer → StandardScaler
Features: Age, MonthlyIncome,
          YearsAtCompany, Fare, ...
        """)
        st.caption("SVMs are sensitive to feature scale. StandardScaler ensures no single feature dominates the margin.")
    with col2:
        st.markdown("**Categorical sub-pipeline**")
        st.code("""
ColumnTransformer → OneHotEncoder
Features: Department, JobRole,
          MaritalStatus, OverTime, ...
        """)
        st.caption("`handle_unknown='ignore'` prevents errors if unseen categories appear in production.")

    st.markdown("Both sub-pipelines are combined with a `ColumnTransformer`, then an `SVC(kernel='linear')` is attached as the final step. `class_weight='balanced'` corrects for the 84/16 class imbalance.")

    # Baseline performance
    st.subheader("Baseline Performance (Linear SVM)")

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc  = accuracy_score(y_test,  pipeline.predict(X_test))
    gap       = train_acc - test_acc

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Accuracy", f"{train_acc:.2%}")
    col2.metric("Test Accuracy",     f"{test_acc:.2%}")
    col3.metric("Gap (Overfitting)", f"{gap:.2%}")

    # Confusion matrix & report
    st.subheader("Confusion Matrix & Classification Report")

    y_pred = pipeline.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    cr     = classification_report(y_test, y_pred,
                                   target_names=["Stayed", "Left"],
                                   output_dict=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Left"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title("Baseline Confusion Matrix", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig)

    with col2:
        cr_df = pd.DataFrame(cr).T.drop("accuracy", errors="ignore")
        cr_df = cr_df[["precision", "recall", "f1-score", "support"]].round(3)
        st.dataframe(cr_df)

    st.info("**Why accuracy is insufficient:** With 84% of employees staying, always predicting 'Stay' scores 84% accuracy. Recall for the 'Left' class tells us what fraction of actual leavers we caught — the metric the HR team actually cares about.")

    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**True Negative ({tn})** — correctly predicted 'Stay'. No unnecessary intervention.")
        st.error(f"**False Negative ({fn})** — missed a leaver. Employee left without retention action.")
    with col2:
        st.error(f"**False Positive ({fp})** — flagged someone who stayed. Wastes retention budget.")
        st.success(f"**True Positive ({tp})** — correctly flagged a leaver. Retention action can be taken.")

    return pipeline