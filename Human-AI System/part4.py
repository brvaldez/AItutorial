"""
part4.py — Human-in-the-Loop Triage Layer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from data import SHORT_LABELS, THRESHOLD


def render(best_pipeline, X_test, y_test):
    st.header("Human-in-the-Loop Triage Layer")
    st.markdown(
        f"Predictions with `max(predict_proba) ≥ threshold` are auto-classified.  \n"
        "Predictions below are escalated to a human reviewer."
    )

    threshold = THRESHOLD
    st.info(f"Confidence threshold fixed at **{THRESHOLD}** as defined in the assignment.")

    proba     = best_pipeline.predict_proba(X_test)
    max_conf  = proba.max(axis=1)
    auto_mask = max_conf >= threshold
    y_arr     = np.array(y_test)

    n_auto   = auto_mask.sum()
    n_review = (~auto_mask).sum()
    n_total  = len(y_test)
    pct_auto   = n_auto   / n_total * 100
    pct_review = n_review / n_total * 100
    auto_acc   = (best_pipeline.predict(X_test)[auto_mask] == y_arr[auto_mask]).mean()

    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.metric("Auto-Classified",      f"{n_auto} ({pct_auto:.1f}%)")
    col_h2.metric("Sent to Human",        f"{n_review} ({pct_review:.1f}%)")
    col_h3.metric("Auto-Class Accuracy",  f"{auto_acc:.3f}")

    col_pie, col_conf = st.columns(2)

    with col_pie:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            [n_auto, n_review],
            labels=[f'Auto ({pct_auto:.1f}%)', f'Human Review ({pct_review:.1f}%)'],
            colors=['#4CAF50', '#FF9800'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11}
        )
        ax.set_title('Test-Set Triage Breakdown')
        st.pyplot(fig)

    with col_conf:
        st.subheader("Confidence Score Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.hist(max_conf, bins=40, color='#4C72B0', edgecolor='white', alpha=0.85)
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold = {threshold:.2f}')
        ax2.set_xlabel('Max Class Probability')
        ax2.set_ylabel('Number of Posts')
        ax2.set_title('Prediction Confidence Distribution')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Triage Breakdown by True Category")
    rows = []
    for i, label in enumerate(SHORT_LABELS):
        cat_mask    = y_arr == i
        cat_auto    = (max_conf[cat_mask] >= threshold).sum()
        joint_mask  = cat_mask & auto_mask
        cat_acc     = (
            (best_pipeline.predict(X_test)[joint_mask] == y_arr[joint_mask]).mean()
            if joint_mask.sum() > 0 else float('nan')
        )
        rows.append({
            'Category':       label,
            'Total':          cat_mask.sum(),
            'Auto-Classified': cat_auto,
            'Human Review':   cat_mask.sum() - cat_auto,
            'Auto %':         f"{cat_auto / cat_mask.sum() * 100:.1f}%",
            'Auto Accuracy':  f"{cat_acc:.3f}" if not np.isnan(cat_acc) else "—",
        })
    st.dataframe(pd.DataFrame(rows).set_index('Category'))

    return threshold
