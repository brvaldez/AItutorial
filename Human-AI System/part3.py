"""
part3.py — Hyperparameter Tuning with GridSearchCV
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from data import SHORT_LABELS


def render(gs, best_pipeline, X_test, y_test):
    st.header("Hyperparameter Tuning")
    st.caption("GridSearchCV with 5-fold cross-validation over max_features, ngram_range, and C — scored by macro F1.")

    best_params = gs.best_params_
    col_g1, col_g2, col_g3 = st.columns(3)
    col_g1.metric("Best max_features", best_params['tfidf__max_features'])
    col_g2.metric("Best ngram_range",  str(best_params['tfidf__ngram_range']))
    col_g3.metric("Best C",            best_params['lr__C'])
    st.metric("Best CV Macro F1", f"{gs.best_score_:.4f}")

    st.subheader("CV Results Heatmap (mean macro F1 by C × max_features, ngram=(1,1))")
    cv_df = pd.DataFrame(gs.cv_results_)
    pivot = cv_df[
        cv_df['param_tfidf__ngram_range'].astype(str) == "(1, 1)"
    ].pivot_table(
        index='param_lr__C',
        columns='param_tfidf__max_features',
        values='mean_test_score'
    )
    fig, ax = plt.subplots(figsize=(4, 2.8))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax)
    ax.set_title('Mean Macro F1 (ngram=(1,1))')
    ax.set_xlabel('max_features')
    ax.set_ylabel('C')
    plt.tight_layout()
    st.pyplot(fig, width='content')

    st.subheader("Best Pipeline — Test-Set Evaluation")
    y_pred      = best_pipeline.predict(X_test)
    best_report = classification_report(y_test, y_pred, target_names=SHORT_LABELS, output_dict=True)
    best_rep_df = pd.DataFrame(best_report).T.round(3)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.dataframe(
            best_rep_df.style.format("{:.3f}").background_gradient(cmap='Greens', subset=['f1-score'])
        )
    with col_t2:
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=SHORT_LABELS, yticklabels=SHORT_LABELS, ax=ax2)
        ax2.set_xlabel('Predicted', fontsize=11)
        ax2.set_ylabel('Actual', fontsize=11)
        ax2.set_title('Best Pipeline Confusion Matrix')
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig2)
