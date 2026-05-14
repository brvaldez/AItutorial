"""
part2.py — Building the Pipeline
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

NUM_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CAT_FEATURES = ["island", "sex"]


@st.cache_resource
def build_and_train(_X_train, _y_train):
    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CAT_FEATURES),
        ("num", StandardScaler(),                                  NUM_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   MLPClassifier(
            hidden_layer_sizes=(64,), activation="relu",
            max_iter=500, random_state=42,
        )),
    ])
    pipeline.fit(_X_train, _y_train)
    return pipeline


def render(X_train, X_test, y_train, y_test):
    st.header("Building the Pipeline")
    st.caption("A Pipeline bundles preprocessing and modelling into one object — preventing data leakage and making deployment clean.")

    pipeline = build_and_train(X_train, y_train)

    # ── Architecture ──────────────────────────────────────────────────────────
    st.subheader("Pipeline Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Categorical sub-pipeline** (`island`, `sex`)")
        st.code('OneHotEncoder(drop="first", sparse_output=False)', language="python")
        st.caption("Converts text categories to binary columns. `drop='first'` removes one column to avoid multicollinearity.")

    with col2:
        st.markdown("**Numeric sub-pipeline** (`bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g`)")
        st.code("StandardScaler()", language="python")
        st.caption("Rescales each feature to mean=0, std=1 so no single measurement dominates gradient descent.")

    st.markdown("Both sub-pipelines are combined with a `ColumnTransformer`, then an `MLPClassifier` is attached as the final step.")

    input_count = pipeline.named_steps["preprocessor"].transform(X_train.iloc[:1]).shape[1]
    st.code(f"""
Pipeline([
  ("preprocessor", ColumnTransformer([
      ("cat", OneHotEncoder(drop="first"), ["island", "sex"]),
      ("num", StandardScaler(),            ["bill_length_mm", "bill_depth_mm",
                                            "flipper_length_mm", "body_mass_g"]),
  ])),
  ("classifier", MLPClassifier(
      hidden_layer_sizes=(64,),  # one hidden layer, 64 neurons
      activation="relu",
      max_iter=500,
      random_state=42,
  )),
])
# Input layer: {input_count} features after encoding
    """, language="python")

    st.info("**Why a Pipeline?** Calling `pipeline.fit(X_train)` makes the scaler and encoder learn from training data *only*. The same learned statistics are applied to X_test automatically — preventing **data leakage**.")

    # ── Baseline Performance ──────────────────────────────────────────────────
    st.subheader("Baseline Performance")

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc  = accuracy_score(y_test,  pipeline.predict(X_test))

    col1, col2, col3 = st.columns(3)
    col1.metric("Training Accuracy", f"{train_acc:.2%}")
    col2.metric("Test Accuracy",     f"{test_acc:.2%}")
    col3.metric("Gap",               f"{train_acc - test_acc:.2%}")

    # ── Confusion matrix + report ─────────────────────────────────────────────
    st.subheader("Confusion Matrix & Classification Report")

    y_pred = pipeline.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    cr     = classification_report(y_test, y_pred, output_dict=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Baseline MLP — Confusion Matrix", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")

    with col2:
        cr_df = (pd.DataFrame(cr).T
                   .drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")
                   [["precision", "recall", "f1-score", "support"]]
                   .round(3))
        st.dataframe(cr_df, width="stretch")

    st.success("The baseline already achieves near-perfect results. The Palmer Penguins dataset is clean and well-separated, so even a small MLP with one hidden layer handles it easily.")

    return pipeline
