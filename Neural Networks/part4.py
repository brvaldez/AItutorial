"""
part4.py — Bias-Variance Trade-off & Why StandardScaler Matters
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler

NUM_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def render(df):
    st.header("Bias-Variance Trade-off & StandardScaler")
    st.caption("Two fundamental concepts every ML practitioner needs to understand before tuning or deploying a model.")

    col1, col2 = st.columns(2)

    # ── Bias-Variance ─────────────────────────────────────────────────────────
    with col1:
        st.subheader("Bias-Variance Trade-off")

        st.error("**Bias** = how wrong the model is on average. A network too small cannot learn the true boundary between species — it fails on both train and test. This is **underfitting**.")
        st.warning("**Variance** = how much the model changes with different training data. A very large network memorises training examples and fails on unseen data. This is **overfitting**.")
        st.success("**The goal:** find the network size and regularisation strength (alpha) that balance both — the sweet spot GridSearchCV was looking for.")

        x     = np.linspace(0, 10, 200)
        bias  = 10 / (x + 1)
        var   = 0.15 * x ** 1.4
        total = bias + var + 1.5
        opt_x = x[np.argmin(total)]

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        ax.plot(x, bias,  label="Bias²",      color="#ef4444", linewidth=2)
        ax.plot(x, var,   label="Variance",    color="#3b82f6", linewidth=2)
        ax.plot(x, total, label="Total Error", color="#7c3aed", linewidth=2.5, linestyle="--")
        ax.axvline(opt_x, color="#22c55e", linestyle=":", linewidth=1.5, label="Optimal")
        ax.set_xlabel("Model Complexity  (bigger network / smaller alpha)", fontsize=8)
        ax.set_ylabel("Error")
        ax.set_title("Bias-Variance Trade-off", fontweight="bold")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")

        st.markdown("""
        **Effect of `alpha` on the decision boundary:**
        - **Small alpha** → large weights allowed → sharp, complex boundary → higher variance.
        - **Large alpha** → weights penalised → smoother boundary → lower variance, slightly higher bias.
        - **Too large alpha** → model too simple → underfitting.

        `alpha` adds `alpha × Σ(weights²)` to the loss, penalising large weights and pushing the network toward simpler, more generalisable solutions.
        """)

    # ── StandardScaler ────────────────────────────────────────────────────────
    with col2:
        st.subheader("Why StandardScaler is Necessary")

        st.warning("Without scaling, features with large numeric ranges dominate gradient descent. **body_mass_g** (2 700–6 300) vs **bill_depth_mm** (13–21): the gradient for body_mass_g is hundreds of times larger, so the network almost ignores the smaller features.")

        raw    = df[NUM_FEATURES].dropna()
        scaled = pd.DataFrame(StandardScaler().fit_transform(raw), columns=NUM_FEATURES)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3.5))
        raw.boxplot(ax=axes[0], patch_artist=True,
                    boxprops=dict(facecolor="#bfdbfe"),
                    medianprops=dict(color="#1d4ed8", linewidth=2))
        axes[0].set_title("Before Scaling", fontweight="bold", fontsize=9)
        axes[0].tick_params(axis="x", rotation=30, labelsize=7)

        scaled.boxplot(ax=axes[1], patch_artist=True,
                       boxprops=dict(facecolor="#bbf7d0"),
                       medianprops=dict(color="#15803d", linewidth=2))
        axes[1].set_title("After StandardScaler", fontweight="bold", fontsize=9)
        axes[1].tick_params(axis="x", rotation=30, labelsize=7)
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")

        st.success("After scaling, every feature has mean = 0 and std = 1. All features now contribute equally to the gradient — training becomes faster and more stable.")

        st.markdown("""
        **Formula:** `x_scaled = (x − mean) / std`

        **Critical rule:** the scaler is *fitted only on X_train* (inside the Pipeline).  
        The learned mean and std are then applied to X_test — never the other way around.  
        Fitting on the full dataset before splitting leaks test-set information into training (**data leakage**), giving falsely optimistic results.
        """)

        st.markdown("**Raw feature ranges (before scaling):**")
        st.dataframe(raw.agg(["min", "max", "mean", "std"]).round(2).T, width="stretch")
