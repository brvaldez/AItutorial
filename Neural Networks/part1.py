"""
part1.py — EDA: Understanding the Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from scipy.stats import gaussian_kde

NUM_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CAT_FEATURES = ["island", "sex"]
SPECIES      = ["Adelie", "Chinstrap", "Gentoo"]
PALETTE      = {"Adelie": "#3b82f6", "Chinstrap": "#f59e0b", "Gentoo": "#10b981"}


def render(df, X, y):
    st.header("Understanding the Data")
    st.caption("Before building any model, we explore the dataset thoroughly. A model built on misunderstood data is unreliable.")

    # ── Overview ──────────────────────────────────────────────────────────────
    st.subheader("Dataset Overview")

    counts = df["species"].value_counts()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Samples", len(df))
    col2.metric("Adelie",    counts["Adelie"],    f"{counts['Adelie']/len(df)*100:.1f}%")
    col3.metric("Chinstrap", counts["Chinstrap"], f"{counts['Chinstrap']/len(df)*100:.1f}%")
    col4.metric("Gentoo",    counts["Gentoo"],    f"{counts['Gentoo']/len(df)*100:.1f}%")
    col5.metric("Missing Values", "0", "after dropna()")

    st.info("**Categorical features:** `island`, `sex` → converted to 0/1 columns with **OneHotEncoder**.  \n"
            "**Numeric features:** `bill_length_mm`, `bill_depth_mm`, `flipper_length_mm`, `body_mass_g` → rescaled with **StandardScaler**.")

    # ── Class distribution + species per island ───────────────────────────────
    st.subheader("Class Distribution")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bars = ax.bar(SPECIES, [counts[s] for s in SPECIES],
                      color=[PALETTE[s] for s in SPECIES],
                      edgecolor="white", linewidth=1.2, width=0.55)
        for bar, s in zip(bars, SPECIES):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{counts[s]}  ({counts[s]/len(df)*100:.0f}%)",
                    ha="center", va="bottom", fontsize=9, fontweight="600")
        ax.set_title("Species Count", fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, counts.max() + 25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")

    with col2:
        pivot  = df.groupby(["island", "species"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bottom  = np.zeros(len(pivot))
        for sp in SPECIES:
            if sp in pivot.columns:
                vals = pivot[sp].values
                ax.bar(pivot.index, vals, bottom=bottom, label=sp,
                       color=PALETTE[sp], edgecolor="white", linewidth=0.8)
                bottom += vals
        ax.set_title("Species per Island", fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")

    st.caption("Adelie is the only species found on all three islands. Gentoo is exclusive to Biscoe, Chinstrap to Dream. Island alone is a strong predictor.")

    # ── Numeric distributions ─────────────────────────────────────────────────
    st.subheader("Feature Distributions by Species")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, col in zip(axes, NUM_FEATURES):
        for sp in SPECIES:
            data = df.loc[df["species"] == sp, col].dropna()
            ax.hist(data, bins=18, alpha=0.55, color=PALETTE[sp], edgecolor="white")
            xs = np.linspace(data.min(), data.max(), 200)
            ax.plot(xs, gaussian_kde(data)(xs) * len(data) * (data.max() - data.min()) / 18,
                    color=PALETTE[sp], linewidth=1.8)
        ax.set_title(col.replace("_mm", "").replace("_", " ").title(), fontsize=9, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    axes[-1].legend(
        handles=[mpatches.Patch(color=PALETTE[s], label=s) for s in SPECIES],
        fontsize=7, loc="upper right"
    )
    fig.patch.set_facecolor("#f9f9f9")
    st.pyplot(fig, width="stretch")

    st.caption("Gentoo penguins are clearly heavier with longer flippers. Adelie and Chinstrap overlap more, especially in bill length, making those features harder to separate without a model.")

    # ── Correlation heat-map ──────────────────────────────────────────────────
    st.subheader("Correlation Matrix")

    col_hm, col_note = st.columns([1, 1])
    with col_hm:
        fig, ax = plt.subplots(figsize=(5, 3.8))
        sns.heatmap(df[NUM_FEATURES].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.5, ax=ax, vmin=-1, vmax=1,
                    annot_kws={"size": 9})
        ax.set_title("Pearson Correlation — Numeric Features", fontweight="bold")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")
    with col_note:
        st.info("**flipper_length_mm** and **body_mass_g** are strongly correlated (≈ 0.87). Larger penguins have longer flippers. The model will receive both, but high correlation means they carry somewhat redundant information.")

    # ── Train/Test split ──────────────────────────────────────────────────────
    st.subheader("Train / Test Split")

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    col1, col2, col3 = st.columns(3)
    col1.metric("Train Set", f"{len(X_tr)} rows")
    col2.metric("Test Set",  f"{len(X_te)} rows")
    col3.metric("Split", "80 / 20", "stratified")

    train_counts = y_tr.value_counts()
    test_counts  = y_te.value_counts()
    st.dataframe(pd.DataFrame({
        "Species":  SPECIES,
        "Train n":  [train_counts.get(s, 0) for s in SPECIES],
        "Train %":  [f"{train_counts.get(s,0)/len(y_tr)*100:.1f}%" for s in SPECIES],
        "Test n":   [test_counts.get(s, 0)  for s in SPECIES],
        "Test %":   [f"{test_counts.get(s,0)/len(y_te)*100:.1f}%"  for s in SPECIES],
    }), hide_index=True, width="stretch")

    st.success("`stratify=y` keeps class proportions equal in both sets. Without it, random chance could leave Chinstrap — the smallest class — underrepresented in the test set, giving a misleading picture of model performance.")
