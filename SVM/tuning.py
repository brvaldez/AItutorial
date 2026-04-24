"""
tabs/tuning.py — Tuning the SVM with GridSearchCV
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

PARAM_GRID = {
    "classifier__kernel": ["linear", "rbf"],
    "classifier__C":      [0.1, 1, 10],
    "classifier__gamma":  ["scale", "auto"],
}


@st.cache_resource
def run_grid_search(_baseline_pipeline, _X_train, _y_train):
    gs = GridSearchCV(
        _baseline_pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    gs.fit(_X_train, _y_train)
    return gs


def render(baseline_pipeline, X_train, X_test, y_train, y_test):
    st.header("Tuning the Model with GridSearchCV")
    st.caption("GridSearchCV tests every combination of hyperparameters using 5-fold cross-validation, optimising for F1 to catch as many leavers as possible.")

    # Parameter grid
    st.subheader("Parameter Grid")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**`kernel`**")
        st.code('["linear", "rbf"]')
        st.caption("Linear draws a flat boundary. RBF can curve around non-linear patterns.")
    with col2:
        st.markdown("**`C`**")
        st.code("[0.1, 1, 10]")
        st.caption("Low C = wide margin (high bias). High C = narrow margin (high variance).")
    with col3:
        st.markdown("**`gamma`** (RBF only)")
        st.code('["scale", "auto"]')
        st.caption("Controls influence radius of each training point. High gamma = overfitting risk.")

    total = 2 * 3 * 2
    st.info(f"**{total} combinations × 5 folds = {total * 5} model fits.** Scored by F1 — the harmonic mean of precision and recall — which penalises both missing leavers and false alarms equally.")

    # Run
    with st.spinner("Running GridSearchCV..."):
        gs = run_grid_search(baseline_pipeline, X_train, y_train)

    best = gs.best_estimator_

    # Results
    st.subheader("Best Parameters")

    bp = gs.best_params_
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best kernel", bp["classifier__kernel"])
    col2.metric("Best C",      str(bp["classifier__C"]))
    col3.metric("Best gamma",  bp["classifier__gamma"])
    col4.metric("Best CV F1",  f"{gs.best_score_:.4f}")

    # Confusion matrices side by side
    y_pred_base  = baseline_pipeline.predict(X_test)
    y_pred_tuned = best.predict(X_test)
    cm_base      = confusion_matrix(y_test, y_pred_base)
    cm_tuned     = confusion_matrix(y_test, y_pred_tuned)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(cm_base, display_labels=["Stayed", "Left"]).plot(
            ax=ax, colorbar=False, cmap="Oranges"
        )
        ax.set_title("Baseline", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig)
    with col2:
        st.markdown("**Tuned Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(cm_tuned, display_labels=["Stayed", "Left"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title("Tuned (GridSearchCV)", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig)

    # Classification reports
    cr_base  = classification_report(y_test, y_pred_base,  target_names=["Stayed","Left"], output_dict=True)
    cr_tuned = classification_report(y_test, y_pred_tuned, target_names=["Stayed","Left"], output_dict=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Report**")
        df_b = pd.DataFrame(cr_base).T.drop("accuracy", errors="ignore")[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(df_b)
    with col2:
        st.markdown("**Tuned Report**")
        df_t = pd.DataFrame(cr_tuned).T.drop("accuracy", errors="ignore")[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(df_t.style.highlight_max(axis=0, color="#d4edda"))

    # Head-to-head
    st.subheader("Baseline vs Tuned: Head-to-Head")

    f1_b = f1_score(y_test, y_pred_base)
    f1_t = f1_score(y_test, y_pred_tuned)
    gap_b = accuracy_score(y_train, baseline_pipeline.predict(X_train)) - accuracy_score(y_test, y_pred_base)
    gap_t = accuracy_score(y_train, best.predict(X_train)) - accuracy_score(y_test, y_pred_tuned)

    comp = pd.DataFrame({
        "Model":        ["Baseline", "Tuned"],
        "Test Accuracy":[f"{accuracy_score(y_test, y_pred_base):.2%}", f"{accuracy_score(y_test, y_pred_tuned):.2%}"],
        "F1 Score":     [f"{f1_b:.4f}", f"{f1_t:.4f}"],
        "Overfit Gap":  [f"{gap_b:.2%}", f"{gap_t:.2%}"],
    })
    st.dataframe(comp, hide_index=True)

    col1, col2 = st.columns(2)
    col1.metric("F1 Improvement",       f"{f1_t - f1_b:+.4f}")
    col2.metric("Overfitting Reduction", f"{gap_b - gap_t:+.2%}")

    # C trade-off explanation
    st.subheader("C — Margin vs Variance Trade-off")
    st.markdown("""
    **Low C (e.g. 0.1):** The optimiser allows more margin violations. The boundary is wide and smooth — high bias, may underfit complex attrition patterns.

    **High C (e.g. 10):** The optimiser penalises violations heavily and narrows the margin to fit training points closely — high variance, risks overfitting to noise in HR data.

    **Best C found:** the value that balances these extremes on the held-out folds.
    """)

    # C sweep chart
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import StratifiedKFold

    c_vals = [0.01, 0.1, 1, 10, 100]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    means, stds = [], []

    for c in c_vals:
        pipe = Pipeline([
            ("preprocessor", baseline_pipeline.named_steps["preprocessor"]),
            ("classifier", SVC(kernel="rbf", C=c, gamma="scale",
                               probability=True, class_weight="balanced", random_state=42)),
        ])
        scores = [
            f1_score(y_train.iloc[va],
                     pipe.fit(X_train.iloc[tr], y_train.iloc[tr]).predict(X_train.iloc[va]),
                     zero_division=0)
            for tr, va in cv.split(X_train, y_train)
        ]
        means.append(np.mean(scores))
        stds.append(np.std(scores))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(len(c_vals)), means, marker="o", color="#4A90D9", linewidth=2)
    ax.fill_between(range(len(c_vals)),
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="#4A90D9")
    ax.set_xticks(range(len(c_vals)))
    ax.set_xticklabels([str(c) for c in c_vals])
    ax.set_xlabel("C value")
    ax.set_ylabel("Mean CV F1")
    ax.set_title("C — Margin vs Variance Trade-off (RBF kernel)", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    st.pyplot(fig)

    return best