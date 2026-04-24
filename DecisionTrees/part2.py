"""
part2.py — EDA: Understanding the Data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def render(df, X, y):
    st.header("Understanding the Data")
    st.caption("Before building any model, we explore the dataset thoroughly. A model built on misunderstood data is a liability.")

    # Shape & Missing Values
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Features", df.shape[1] - 1)
    col3.metric("Missing Cells", int(df.isnull().sum().sum()))

    st.markdown("**Missing values per column:**")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Count"]
    missing["% Missing"] = (missing["Missing Count"] / len(df) * 100).round(1).astype(str) + "%"
    missing["Status"] = missing["Missing Count"].apply(
        lambda x: "Needs imputation" if x > 0 else "Complete"
    )
    st.dataframe(missing, width='stretch', hide_index=True)

    st.info("**Pipeline fix:** `SimpleImputer(strategy='median')` fills missing ages; `SimpleImputer(strategy='most_frequent')` fills missing embarked — both learned only from training data to prevent leakage.")

    # Survival Rates
    st.subheader("Survival Rates by Credit Tier")

    overall = y.mean()
    by_class = df.groupby("pclass")["survived"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Overall Rate", f"{overall:.1%}")
    col2.metric("Pclass 1 (Prime)", f"{by_class[1]:.1%}", delta=f"{by_class[1]-overall:+.1%} vs avg")
    col3.metric("Pclass 2 (Near-prime)", f"{by_class[2]:.1%}", delta=f"{by_class[2]-overall:+.1%} vs avg")
    col4.metric("Pclass 3 (Subprime)", f"{by_class[3]:.1%}", delta=f"{by_class[3]-overall:+.1%} vs avg")

    st.markdown("""
    **Banking interpretation:** Passenger class is a direct proxy for socioeconomic status — 
    first-class passengers paid substantially higher fares and had greater financial resources, 
    analogous to a **prime credit tier** with stable income. Third-class passengers, who default 
    at nearly 3× the rate of first-class, represent the **subprime segment**: limited liquidity, 
    less ability to absorb financial shocks. Pclass is one of the strongest risk stratifiers in 
    the dataset and should be heavily weighted by underwriters.
    """)

    # Visualisations
    st.subheader("Survival Patterns")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Repayment Rate by Sex**")
        fig, ax = plt.subplots(figsize=(5, 4))
        ss = df.groupby("sex")["survived"].mean()
        colors = ["#4A90D9", "#E8735A"]
        bars = ax.bar(ss.index, ss.values, color=colors, edgecolor="white", linewidth=2, width=0.45)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Repayment Rate")
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f9f9f9")
        fig.patch.set_facecolor("#f9f9f9")
        for bar, v in zip(bars, ss.values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.03,
                    f"{v:.1%}", ha="center", fontsize=13, fontweight="bold")
        st.pyplot(fig, width='stretch')
        st.caption("Female passengers survived at much higher rates (~74% vs ~19%). In a real lending model, sex is a legally protected characteristic and would be excluded to prevent discriminatory outcomes.")

    with col_right:
        st.markdown("**Age Distribution by Outcome**")
        fig, ax = plt.subplots(figsize=(5, 4))
        for val, label, color in [(0,"Default","#E8735A"), (1,"Repay","#4A90D9")]:
            ax.hist(df[df.survived==val]["age"].dropna(), bins=28,
                    alpha=0.65, label=label, color=color, edgecolor="white")
        ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.legend()
        ax.spines[["top","right"]].set_visible(False)
        ax.set_facecolor("#f9f9f9")
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width='stretch')
        st.caption("Children (age < 10) show higher survival; the distributions overlap heavily in the 20–50 range, making age a moderate but not standalone risk signal.")

    # Encoding
    st.subheader("Handling Categorical Features")

    col1, col2 = st.columns(2)
    with col1:
        st.error("**Without encoding:** `ValueError: could not convert string to float` — sklearn's internal math cannot handle string values like 'male' or 'S'.")
    with col2:
        st.success("**Fix: `OneHotEncoder`** — creates one binary column per category level. `sex` → `sex_female`, `sex_male`. `handle_unknown='ignore'` prevents errors on unseen values in production.")

    st.markdown("**Example transformation:**")
    example = pd.DataFrame({
        "Original sex": ["female","male","female"],
        "→ sex_female": [1, 0, 1],
        "→ sex_male":   [0, 1, 0],
        "Original embarked": ["S","C","Q"],
        "→ embarked_S": [1,0,0],
        "→ embarked_C": [0,1,0],
        "→ embarked_Q": [0,0,1],
    })
    st.dataframe(example, hide_index=True, width='stretch')

    # Train/Test Split
    st.subheader("Train / Test Split")

    from sklearn.model_selection import train_test_split
    X = df.drop(columns="survived")
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Train Set", f"{len(X_train)} rows", f"Survival: {y_train.mean():.2%}")
    col2.metric("Test Set",  f"{len(X_test)} rows",  f"Survival: {y_test.mean():.2%}")
    col3.metric("Rate Difference", f"{abs(y_train.mean()-y_test.mean()):.3%}", "Stratification worked")

    st.success("`stratify=y` ensures both splits preserve the same class ratio (~40% survived). Without this, the test set could accidentally be 60% defaulters — giving the risk team a misleading picture of model performance on real applicants.")