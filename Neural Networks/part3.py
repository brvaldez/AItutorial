"""
part3.py — Tuning the Model with GridSearchCV
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
)

PARAM_GRID = {
    "classifier__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
    "classifier__alpha":              [0.0001, 0.001, 0.01],
    "classifier__learning_rate_init": [0.001, 0.01],
}


@st.cache_resource
def run_grid_search(_baseline_pipeline, _X_train, _y_train):
    gs = GridSearchCV(
        _baseline_pipeline,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
    )
    gs.fit(_X_train, _y_train)
    return gs


def render(baseline_pipeline, X_train, X_test, y_train, y_test):
    st.header("Tuning the Model with GridSearchCV")
    st.caption("GridSearchCV tests every combination of hyperparameters using 5-fold cross-validation, optimising for weighted F1 to handle class imbalance.")

    # ── Parameter grid ────────────────────────────────────────────────────────
    st.subheader("Parameter Grid")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**`hidden_layer_sizes`**")
        st.code("[(32,), (64,), (128,), (64, 32)]")
        st.caption("Network size. (64, 32) adds a second hidden layer.")
    with col2:
        st.markdown("**`alpha`**")
        st.code("[0.0001, 0.001, 0.01]")
        st.caption("L2 regularisation strength. Higher = smoother boundary, less overfitting.")
    with col3:
        st.markdown("**`learning_rate_init`**")
        st.code("[0.001, 0.01]")
        st.caption("Step size for gradient descent. Too large = unstable; too small = slow.")

    total = 4 * 3 * 2
    st.info(f"**{total} combinations × 5 folds = {total * 5} model fits** evaluated in total.")

    # ── Run ───────────────────────────────────────────────────────────────────
    with st.spinner("Running GridSearchCV…"):
        gs = run_grid_search(baseline_pipeline, X_train, y_train)

    best = gs.best_estimator_

    # ── Best parameters ───────────────────────────────────────────────────────
    st.subheader("Best Parameters & Test Evaluation")

    bp = gs.best_params_
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("hidden_layer_sizes",  str(bp["classifier__hidden_layer_sizes"]))
    col2.metric("alpha",               str(bp["classifier__alpha"]))
    col3.metric("learning_rate_init",  str(bp["classifier__learning_rate_init"]))
    col4.metric("Best CV F1",          f"{gs.best_score_:.4f}")

    # ── Confusion matrices ────────────────────────────────────────────────────
    y_pred_base  = baseline_pipeline.predict(X_test)
    y_pred_tuned = best.predict(X_test)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred_base,  labels=baseline_pipeline.classes_),
            display_labels=baseline_pipeline.classes_,
        ).plot(ax=ax, colorbar=False, cmap="Oranges")
        ax.set_title("Baseline", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")

    with col2:
        st.markdown("**Tuned Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, y_pred_tuned, labels=best.classes_),
            display_labels=best.classes_,
        ).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title("Tuned (GridSearchCV)", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")

    # ── Classification reports ────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Classification Report**")
        cr_b = pd.DataFrame(classification_report(y_test, y_pred_base,  output_dict=True)).T
        cr_b = cr_b.drop(["accuracy","macro avg","weighted avg"], errors="ignore")[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(cr_b, width="stretch")

    with col2:
        st.markdown("**Tuned Classification Report**")
        cr_t = pd.DataFrame(classification_report(y_test, y_pred_tuned, output_dict=True)).T
        cr_t = cr_t.drop(["accuracy","macro avg","weighted avg"], errors="ignore")[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(cr_t.style.highlight_max(axis=0, color="#d4edda"), width="stretch")

    # ── Head-to-head comparison ───────────────────────────────────────────────
    st.subheader("Baseline vs Tuned: Head-to-Head")

    tr_acc_b = accuracy_score(y_train, baseline_pipeline.predict(X_train))
    te_acc_b = accuracy_score(y_test,  y_pred_base)
    f1_b     = f1_score(y_test, y_pred_base, average="weighted")
    gap_b    = tr_acc_b - te_acc_b

    tr_acc_t = accuracy_score(y_train, best.predict(X_train))
    te_acc_t = accuracy_score(y_test,  y_pred_tuned)
    f1_t     = f1_score(y_test, y_pred_tuned, average="weighted")
    gap_t    = tr_acc_t - te_acc_t

    st.dataframe(pd.DataFrame({
        "Model":          ["Baseline", "Tuned"],
        "Train Accuracy": [f"{tr_acc_b:.2%}", f"{tr_acc_t:.2%}"],
        "Test Accuracy":  [f"{te_acc_b:.2%}", f"{te_acc_t:.2%}"],
        "F1 (weighted)":  [f"{f1_b:.4f}",     f"{f1_t:.4f}"],
        "Overfit Gap":    [f"{gap_b:.2%}",     f"{gap_t:.2%}"],
    }), hide_index=True, width="stretch")

    col1, col2 = st.columns(2)
    col1.metric("F1 Change",          f"{f1_t - f1_b:+.4f}")
    col2.metric("Overfitting Change", f"{gap_b - gap_t:+.2%}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    st.subheader("Training Loss Curve — Best Model")

    loss = best.named_steps["classifier"].loss_curve_
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(loss, color="#3b82f6", linewidth=2)
    ax.fill_between(range(len(loss)), loss, alpha=0.12, color="#3b82f6")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve — Best MLP", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.patch.set_facecolor("#f9f9f9")
    st.pyplot(fig, width="stretch")

    # ── Cross-validation on training set (k=5) ───────────────────────────────
    st.subheader("5-Fold Cross-Validation on Training Set")
    st.caption("cross_val_score re-fits the best pipeline 5 times on different training folds, giving a more robust estimate of generalisation than a single train/test split.")

    cv_scores = cross_val_score(
        best,
        X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="f1_weighted",
        n_jobs=-1,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean F1 (CV)",   f"{cv_scores.mean():.4f}")
    col2.metric("Std F1 (CV)",    f"{cv_scores.std():.4f}")
    col3.metric("Min / Max",      f"{cv_scores.min():.4f} / {cv_scores.max():.4f}")

    # Per-fold bar chart
    fig, ax = plt.subplots(figsize=(7, 3))
    colours = ["#22c55e" if s >= cv_scores.mean() else "#f59e0b" for s in cv_scores]
    bars = ax.bar([f"Fold {i+1}" for i in range(5)], cv_scores,
                  color=colours, edgecolor="white", linewidth=1.2, width=0.5)
    ax.axhline(cv_scores.mean(), color="#3b82f6", linewidth=1.8,
               linestyle="--", label=f"Mean = {cv_scores.mean():.4f}")
    ax.fill_between([-0.5, 4.5],
                    cv_scores.mean() - cv_scores.std(),
                    cv_scores.mean() + cv_scores.std(),
                    alpha=0.1, color="#3b82f6", label=f"±1 std = {cv_scores.std():.4f}")
    for bar, score in zip(bars, cv_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{score:.4f}", ha="center", va="bottom", fontsize=9, fontweight="600")
    ax.set_ylim(max(0, cv_scores.min() - 0.05), min(1.05, cv_scores.max() + 0.05))
    ax.set_ylabel("F1 (weighted)")
    ax.set_title("F1 Score per Fold — Best Pipeline on X_train", fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.patch.set_facecolor("#f9f9f9")
    st.pyplot(fig, width="stretch")

    cv_df = pd.DataFrame({
        "Fold":        [f"Fold {i+1}" for i in range(5)],
        "F1 (weighted)": cv_scores.round(4),
        "vs Mean":     [f"{s - cv_scores.mean():+.4f}" for s in cv_scores],
    })
    st.dataframe(cv_df, hide_index=True, width="stretch")

    st.info(
        f"A standard deviation of **{cv_scores.std():.4f}** across folds tells us how stable "
        "the model is. A low std means the pipeline generalises consistently regardless of which "
        "training examples it sees — a key requirement before deployment."
    )

    # ── Full CV results ───────────────────────────────────────────────────────
    st.subheader("All GridSearchCV Results")

    cv = pd.DataFrame(gs.cv_results_)[[
        "param_classifier__hidden_layer_sizes",
        "param_classifier__alpha",
        "param_classifier__learning_rate_init",
        "mean_test_score", "std_test_score", "rank_test_score",
    ]].copy()
    cv.columns = ["hidden_layers", "alpha", "lr_init", "mean_F1", "std_F1", "rank"]
    cv["mean_F1"] = cv["mean_F1"].round(4)
    cv["std_F1"]  = cv["std_F1"].round(4)
    st.dataframe(cv.sort_values("rank"), hide_index=True, width="stretch", height=280)

    return best
