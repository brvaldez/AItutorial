"""
part2.py — Baseline Pipeline: TF-IDF + Logistic Regression
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from data import SHORT_LABELS


def render(baseline, X_val, y_val):
    st.header("Baseline Pipeline")
    st.markdown(
        "**Pipeline:** `TfidfVectorizer(max_features=5000)` → `LogisticRegression`  \n"
        "Vectorizer fitted **only** on the 80 % training split to prevent data leakage."
    )

    y_val_pred  = baseline.predict(X_val)
    report_dict = classification_report(y_val, y_val_pred, target_names=SHORT_LABELS, output_dict=True)
    report_df   = pd.DataFrame(report_dict).T.round(3)

    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.subheader("Classification Report (Validation Set)")
        st.dataframe(
            report_df.style.format("{:.3f}").background_gradient(cmap='Blues', subset=['f1-score'])
        )

    with col_b2:
        st.subheader("Confusion Matrix (Validation Set)")
        cm = confusion_matrix(y_val, y_val_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=ax)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title('Baseline Confusion Matrix')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
