"""
part3.py — Building the Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

NUM_FEATURES = ["age", "fare", "sibsp", "parch"]
CAT_FEATURES = ["sex", "embarked", "pclass"]


@st.cache_resource
def build_and_train(_X_train, _y_train):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, NUM_FEATURES),
        ("cat", cat_pipe, CAT_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor",           preprocessor),
        ("decisiontreeclassifier", DecisionTreeClassifier(random_state=42)),
    ])
    pipeline.fit(_X_train, _y_train)
    return pipeline


def get_feature_names(pipeline):
    ohe_cats = (pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .named_steps["ohe"]
                .get_feature_names_out(CAT_FEATURES))
    return NUM_FEATURES + list(ohe_cats)


def render(X_train, X_test, y_train, y_test):
    st.header("Building the Pipeline")
    st.caption("A Pipeline bundles preprocessing and modelling into one object — preventing data leakage and making deployment clean.")

    pipeline = build_and_train(X_train, y_train)

    # Architecture
    st.subheader("Pipeline Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric sub-pipeline** (`age`, `fare`, `sibsp`, `parch`)")
        st.code("""
Pipeline([
  ("imputer", SimpleImputer(strategy="median")),
  ("scaler",  StandardScaler()),
])
        """, language="python")
    with col2:
        st.markdown("**Categorical sub-pipeline** (`sex`, `embarked`, `pclass`)")
        st.code("""
Pipeline([
  ("imputer", SimpleImputer(strategy="most_frequent")),
  ("ohe",     OneHotEncoder(handle_unknown="ignore")),
])
        """, language="python")

    st.markdown("Both sub-pipelines are combined with a `ColumnTransformer`, then a `DecisionTreeClassifier` is attached as the final step.")

    # Baseline Accuracy
    st.subheader("Baseline Performance")

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc  = accuracy_score(y_test,  pipeline.predict(X_test))
    gap       = train_acc - test_acc

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Accuracy", f"{train_acc:.2%}")
    col2.metric("Test Accuracy",     f"{test_acc:.2%}")
    col3.metric("Gap (Overfitting)", f"{gap:.2%}",
                delta=f"{'️ Overfitting' if gap > 0.1 else ' Acceptable'}",
                delta_color="inverse")

    if gap > 0.1:
        st.warning(f"A **{gap:.1%} gap** between train and test accuracy is a clear sign of overfitting. The untuned tree memorises all 712 training rows — including noise — and fails to generalise. GridSearchCV in the next step will fix this.")

    # Confusion Matrix & Report
    st.subheader("Confusion Matrix & Classification Report")

    y_pred = pipeline.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    cr     = classification_report(y_test, y_pred,
                                   target_names=["Default","Repay"],
                                   output_dict=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ConfusionMatrixDisplay(cm, display_labels=["Default","Repay"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title("Confusion Matrix", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width='stretch')

    with col2:
        cr_df = pd.DataFrame(cr).T.drop("accuracy", errors="ignore")
        cr_df = cr_df[["precision","recall","f1-score","support"]].round(3)
        cr_df.index.name = "Class"
        st.dataframe(cr_df.style.highlight_max(axis=0, color="#d4edda"), width='stretch')

        recall_default = cm[0,0] / (cm[0,0] + cm[0,1])
        st.metric("Recall — Default (class 0)", f"{recall_default:.2%}",
                  help="Of all actual defaulters, what % did we correctly flag?")

    st.markdown("**Reading the confusion matrix — all 4 cells:**")
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Actual Default → Predicted Default** ({tn} applicants)\n\n"
                   "True Negative — the model correctly identified a high-risk applicant and declined them. "
                   "This is the best outcome for the bank: a bad loan was caught before it was approved.")
        st.error(f"**Actual Repay → Predicted Default** ({fn} applicants)\n\n"
                 "False Negative — the model incorrectly flagged a creditworthy customer as risky and declined them. "
                 "The bank loses no money but turns away a good customer — a missed revenue opportunity.")
    with col2:
        st.error(f"**Actual Default → Predicted Repay** ({fp} applicants)\n\n"
                 "False Positive — the model missed a defaulter and approved their loan. "
                 "This is the most costly error: the bank lends money it will not get back. "
                 "The risk team wants to minimise this cell above all others.")
        st.success(f"**Actual Repay → Predicted Repay** ({tp} applicants)\n\n"
                   "True Positive — the model correctly identified a low-risk applicant and approved them. "
                   "Good loan approved, interest earned, no loss.")

    st.info("**Recall for Default** is the most important metric for the risk team. Missing a defaulter (false positive) means approving a loan that will be lost — the direct financial cost. Tuning for F1 in the next step improves this.")

    # Tree Visualisation
    st.subheader("Decision Tree Structure")
    st.caption("Showing only the first 3 levels for readability. The full untuned tree is far deeper and has memorised the training set.")

    feat_names = get_feature_names(pipeline)

    fig, ax = plt.subplots(figsize=(20, 7))
    plot_tree(
        pipeline.named_steps["decisiontreeclassifier"],
        max_depth=3,
        feature_names=feat_names,
        class_names=["Default","Repay"],
        filled=True, rounded=True, fontsize=8, ax=ax,
    )
    ax.set_title("Decision Tree — first 3 levels shown", fontweight="bold", fontsize=12)
    fig.patch.set_facecolor("#f9f9f9")
    st.pyplot(fig, width='stretch')

    st.success("Each node shows: the split condition, Gini impurity, sample count, and predicted class. Any rejected applicant can be traced from the root to their leaf — providing a full, auditable explanation for the compliance team.")

    return pipeline