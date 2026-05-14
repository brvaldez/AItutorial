"""
part1.py — EDA: Exploring the 20 Newsgroups Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from data import SHORT_LABELS


def render(data_train, data_test):
    st.header("Exploratory Data Analysis")
    st.caption("Before building any model, we inspect class balance, document length, and raw post samples.")

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.subheader("Class Distribution")
        counts_train = pd.Series(data_train.target).value_counts().sort_index()
        counts_test  = pd.Series(data_test.target ).value_counts().sort_index()
        dist_df = pd.DataFrame(
            {'Train': counts_train.values, 'Test': counts_test.values},
            index=SHORT_LABELS
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        dist_df.plot(kind='bar', ax=ax, color=['#4C72B0', '#DD8452'], edgecolor='white')
        ax.set_xticklabels(SHORT_LABELS, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('Number of Posts')
        ax.set_title('Posts per Category')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("The dataset is roughly balanced across all four categories in both splits.")

    with col_eda2:
        st.subheader("Document Length Distribution")
        lengths = [len(doc.split()) for doc in data_train.data]
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(lengths, bins=60, color='#4C72B0', edgecolor='white', alpha=0.85)
        ax2.axvline(np.mean(lengths),   color='red',    linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(lengths):.0f}')
        ax2.axvline(np.median(lengths), color='orange', linestyle='--', linewidth=1.5, label=f'Median = {np.median(lengths):.0f}')
        ax2.set_xlabel('Words per Post')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Training Post Lengths')
        ax2.legend(fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    st.subheader("Document Length Statistics (Training Set)")
    lengths = [len(doc.split()) for doc in data_train.data]
    len_stats = {
        'Mean words':   f"{np.mean(lengths):.1f}",
        'Median words': f"{np.median(lengths):.1f}",
        'Min words':    f"{np.min(lengths)}",
        'Max words':    f"{np.max(lengths)}",
        'Std dev':      f"{np.std(lengths):.1f}",
    }
    st.table(pd.DataFrame(len_stats, index=['Value']).T)

    st.subheader("Sample Posts")
    targets = np.array(data_train.target)
    for class_idx, label in enumerate(SHORT_LABELS):
        positions = np.where(targets == class_idx)[0]
        idx     = positions[0]
        doc     = data_train.data[idx]
        preview = doc.strip()[:300].replace('\n', ' ')
        with st.expander(f"**{label}**"):
            st.write(preview + ("…" if len(doc) > 300 else ""))
