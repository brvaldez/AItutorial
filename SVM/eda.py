"""
tabs/eda.py — EDA: Understanding the Data
"""

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def render(df, X, y):
    st.header("Understanding the Data")
    st.caption("Before building any model, we explore the dataset to understand class imbalance, feature distributions, and which variables are most predictive of attrition.")

    # Overview
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Employees", f"{len(df):,}")
    col2.metric("Features", X.shape[1])
    col3.metric("Left (Yes)", int(y.sum()))
    col4.metric("Stayed (No)", int((y == 0).sum()))

    # Class imbalance — half-width so it doesn't stretch
    st.subheader("Class Imbalance")

    counts = df["Attrition"].value_counts()
    fig, ax = plt.subplots(figsize=(4, 3.5))
    bars = ax.bar(counts.index, counts.values,
                  color=["#4A90D9", "#E8735A"], edgecolor="white", width=0.45)
    ax.set_ylabel("Count")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5,
                f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontsize=11, fontweight="bold")

    col_chart, col_note = st.columns([1, 1])
    with col_chart:
        st.pyplot(fig)
    with col_note:
        st.warning(
            f"**{counts['Yes']} employees left vs {counts['No']} stayed — an ~84/16 split.** "
            "Accuracy alone is misleading: a model that always predicts 'Stay' scores 84% "
            "while catching zero leavers. We use F1 and ROC-AUC as primary metrics."
        )

    # Attrition rate by categorical feature
    st.subheader("Attrition Rate by Categorical Feature")

    cat_cols = X.select_dtypes("object").columns.tolist()
    selected = st.selectbox("Select feature", cat_cols)

    rate = (df.groupby(selected)["Attrition"]
              .apply(lambda s: (s == "Yes").mean() * 100)
              .reset_index(name="Attrition %")
              .sort_values("Attrition %", ascending=False))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(rate[selected], rate["Attrition %"], color="#E8735A", edgecolor="white")
    ax.set_ylabel("Attrition %")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    plt.xticks(rotation=25, ha="right", fontsize=9)
    for i, (_, row) in enumerate(rate.iterrows()):
        ax.text(i, row["Attrition %"] + 0.4, f"{row['Attrition %']:.1f}%",
                ha="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    # Numeric distribution
    st.subheader("Numeric Feature Distribution")

    num_cols = X.select_dtypes("number").columns.tolist()
    selected_num = st.selectbox("Select feature", num_cols)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    for val, label, color in [(0, "Stayed", "#4A90D9"), (1, "Left", "#E8735A")]:
        ax.hist(df[y == val][selected_num], bins=30, alpha=0.65,
                label=label, color=color, edgecolor="white", density=True)
    ax.set_xlabel(selected_num)
    ax.set_ylabel("Density")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor("#f9f9f9")
    fig.patch.set_facecolor("#f9f9f9")
    plt.tight_layout()
    st.pyplot(fig)

    # Encoding note
    st.subheader("Handling Categorical Features")

    col1, col2 = st.columns(2)
    with col1:
        st.error("**Without encoding:** sklearn cannot handle string values like 'Male', 'Sales', or 'Single' — they must be converted to numbers before training.")
    with col2:
        st.success("**Fix: `OneHotEncoder`** — creates one binary column per category level. `handle_unknown='ignore'` prevents errors if production data contains unseen values.")

    st.info("**Why `StandardScaler` matters for SVMs:** The SVM margin is measured in Euclidean distance. Without scaling, MonthlyIncome (range: 1k–20k) would dominate over StockOptionLevel (0–3). Tree-based models don't have this requirement — SVMs do.")