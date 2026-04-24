"""
part4.py — Tuning the Model with GridSearchCV
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)


PARAM_GRID = {
    "decisiontreeclassifier__max_depth":         [3, 5, 7, 10, None],
    "decisiontreeclassifier__min_samples_split":  [2, 10, 20, 50],
    "decisiontreeclassifier__criterion":          ["gini", "entropy"],
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
    st.header("Tuning the Model")
    st.caption("GridSearchCV exhaustively tests every combination of hyperparameters using 5-fold cross-validation, optimising for F1 score to catch as many defaulters as possible.")

    # Parameter Grid
    st.subheader("Parameter Grid")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**`max_depth`**")
        st.code("[3, 5, 7, 10, None]")
        st.caption("Controls tree depth. `None` = unlimited (baseline).")
    with col2:
        st.markdown("**`min_samples_split`**")
        st.code("[2, 10, 20, 50]")
        st.caption("Min samples needed before a node can split.")
    with col3:
        st.markdown("**`criterion`**")
        st.code('["gini", "entropy"]')
        st.caption("Impurity measure used to evaluate splits.")

    total_combos = 5 * 4 * 2
    st.info(f"**{total_combos} combinations × 5 folds = {total_combos*5} model fits** evaluated in total. Scored by F1 — the harmonic mean of precision and recall, which balances catching defaulters against false rejections.")

    #  Run GridSearch 
    with st.spinner("Running GridSearchCV (5-fold, 40 combinations)..."):
        gs = run_grid_search(baseline_pipeline, X_train, y_train)

    best = gs.best_estimator_

    # Results
    st.subheader("Best Parameters & Test Evaluation")

    bp = gs.best_params_
    col1, col2, col3 = st.columns(3)
    col1.metric("Best max_depth",         str(bp["decisiontreeclassifier__max_depth"]))
    col2.metric("Best min_samples_split",  str(bp["decisiontreeclassifier__min_samples_split"]))
    col3.metric("Best criterion",          bp["decisiontreeclassifier__criterion"])

    st.metric("Best Cross-Validated F1", f"{gs.best_score_:.4f}")

    y_pred_base  = baseline_pipeline.predict(X_test)
    y_pred_tuned = best.predict(X_test)
    cm_base      = confusion_matrix(y_test, y_pred_base)
    cm_tuned     = confusion_matrix(y_test, y_pred_tuned)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(cm_base, display_labels=["Default","Repay"]).plot(
            ax=ax, colorbar=False, cmap="Oranges"
        )
        ax.set_title("Baseline", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width='stretch')

    with col2:
        st.markdown("**Tuned Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ConfusionMatrixDisplay(cm_tuned, display_labels=["Default","Repay"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title("Tuned (GridSearchCV)", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width='stretch')

    # Confusion matrix legend (uses tuned numbers)
    st.markdown("**Reading the confusion matrix — all 4 cells:**")
    tn, fp, fn, tp = cm_tuned[0,0], cm_tuned[0,1], cm_tuned[1,0], cm_tuned[1,1]
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Actual Default → Predicted Default** ({tn} applicants)\n\n"
                   "True Negative — correctly identified a high-risk applicant and declined them. "
                   "Best outcome for the bank: a bad loan was caught before approval.")
        st.error(f"**Actual Repay → Predicted Default** ({fn} applicants)\n\n"
                 "False Negative — wrongly flagged a creditworthy customer as risky and declined them. "
                 "No financial loss, but the bank turns away a good customer and misses revenue.")
    with col2:
        st.error(f"**Actual Default → Predicted Repay** ({fp} applicants)\n\n"
                 "False Positive — missed a defaulter and approved their loan. "
                 "The most costly error: the bank lends money it will not recover. "
                 "The risk team wants to minimise this cell above all others.")
        st.success(f"**Actual Repay → Predicted Repay** ({tp} applicants)\n\n"
                   "True Positive — correctly identified a low-risk applicant and approved them. "
                   "Good loan approved, interest earned, no loss.")

    # Classification reports side by side
    cr_base  = classification_report(y_test, y_pred_base,  target_names=["Default","Repay"], output_dict=True)
    cr_tuned = classification_report(y_test, y_pred_tuned, target_names=["Default","Repay"], output_dict=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Baseline Classification Report**")
        df_b = pd.DataFrame(cr_base).T.drop("accuracy",errors="ignore")[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(df_b, width='stretch')
    with col2:
        st.markdown("**Tuned Classification Report**")
        df_t = pd.DataFrame(cr_tuned).T.drop("accuracy",errors="ignore")[["precision","recall","f1-score","support"]].round(3)
        st.dataframe(df_t.style.highlight_max(axis=0,color="#d4edda"), width='stretch')

    # Comparison
    st.subheader("Baseline vs Tuned: Head-to-Head")

    tr_acc_b = accuracy_score(y_train, baseline_pipeline.predict(X_train))
    te_acc_b = accuracy_score(y_test,  y_pred_base)
    f1_b     = f1_score(y_test, y_pred_base)
    gap_b    = tr_acc_b - te_acc_b

    tr_acc_t = accuracy_score(y_train, best.predict(X_train))
    te_acc_t = accuracy_score(y_test,  y_pred_tuned)
    f1_t     = f1_score(y_test, y_pred_tuned)
    gap_t    = tr_acc_t - te_acc_t

    comp = pd.DataFrame({
        "Model":          ["Baseline", "Tuned"],
        "Train Accuracy": [f"{tr_acc_b:.2%}", f"{tr_acc_t:.2%}"],
        "Test Accuracy":  [f"{te_acc_b:.2%}", f"{te_acc_t:.2%}"],
        "F1 Score":       [f"{f1_b:.4f}", f"{f1_t:.4f}"],
        "Overfit Gap":    [f"{gap_b:.2%}", f"{gap_t:.2%}"],
    })
    st.dataframe(comp, hide_index=True, width='stretch')

    col1, col2 = st.columns(2)
    col1.metric("F1 Improvement",        f"{f1_t - f1_b:+.4f}", delta_color="normal")
    col2.metric("Overfitting Reduction",  f"{gap_b - gap_t:+.2%}", delta_color="normal")

    st.markdown(f"""
    **What to tell the Head of Risk:**

    Tuning the decision tree delivered a **+{f1_t - f1_b:.1%} improvement in F1 score** 
    and collapsed the train-test gap from **{gap_b:.1%} down to {gap_t:.1%}**, 
    confirming the model now generalises rather than memorises. 
    On the test set, the tuned model catches a higher proportion of actual defaulters 
    (recall for class 0 improved from {cm_base[0,0]/(cm_base[0,0]+cm_base[0,1]):.1%} 
    to {cm_tuned[0,0]/(cm_tuned[0,0]+cm_tuned[0,1]):.1%}), 
    meaning fewer bad loans slip through to the repayment book. 
    The trade-off is a modest increase in false positives — some creditworthy customers 
    will be incorrectly declined — but this is the conservative position the risk team should prefer. 
    For the bank, a model that flags the majority of likely defaulters before approval has concrete 
    balance-sheet value versus one that fails the moment it sees new applicants.
    """)

    return best