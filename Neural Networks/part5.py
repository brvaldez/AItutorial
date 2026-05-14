"""
part5.py — Deploying the Model: Predict New Penguins
"""

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from sklearn.metrics import accuracy_score

SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
PALETTE = {"Adelie": "#3b82f6", "Chinstrap": "#f59e0b", "Gentoo": "#10b981"}


def render(best_pipeline, X_test, y_test):
    st.header("Deploying the Model")
    st.caption("The model is trained and tuned. Now we make it production-ready: save it, reload it, and score new penguins from raw input.")

    # ── Save & Reload ─────────────────────────────────────────────────────────
    st.subheader("Saving & Reloading the Pipeline")

    buf = io.BytesIO()
    joblib.dump(best_pipeline, buf)
    buf.seek(0)
    reloaded = joblib.load(buf)

    acc_orig     = accuracy_score(y_test, best_pipeline.predict(X_test))
    acc_reloaded = accuracy_score(y_test, reloaded.predict(X_test))
    identical    = np.array_equal(best_pipeline.predict(X_test), reloaded.predict(X_test))

    col1, col2, col3 = st.columns(3)
    col1.metric("Original Accuracy",  f"{acc_orig:.4f}")
    col2.metric("Reloaded Accuracy",  f"{acc_reloaded:.4f}")
    col3.metric("Predictions Match",  "✅ Yes" if identical else "❌ No")

    buf.seek(0)
    st.download_button(
        label="⬇️  Download penguin_pipeline.joblib",
        data=buf,
        file_name="penguin_pipeline.joblib",
        mime="application/octet-stream",
    )

    st.code("""
import joblib, pandas as pd

pipeline = joblib.load("penguin_pipeline.joblib")
predictions   = pipeline.predict(new_penguins_df)
probabilities = pipeline.predict_proba(new_penguins_df)
    """, language="python")

    # ── Predict new penguins ──────────────────────────────────────────────────
    st.subheader("Predict a New Penguin")
    st.caption("Fill in the measurements and click Predict. The pipeline handles all encoding and scaling automatically.")

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Categorical**")
            island = st.selectbox("Island", ["Biscoe", "Dream", "Torgersen"])
            sex    = st.selectbox("Sex",    ["Male", "Female"])

        with col2:
            st.markdown("**Bill**")
            bill_length = st.number_input("Bill Length (mm)", min_value=30.0, max_value=65.0, value=48.5, step=0.1, format="%.1f")
            bill_depth  = st.number_input("Bill Depth (mm)",  min_value=13.0, max_value=22.0, value=14.8, step=0.1, format="%.1f")

        with col3:
            st.markdown("**Body**")
            flipper   = st.number_input("Flipper Length (mm)", min_value=170.0, max_value=235.0, value=210.0, step=1.0, format="%.0f")
            body_mass = st.number_input("Body Mass (g)",        min_value=2500.0, max_value=6500.0, value=5200.0, step=50.0, format="%.0f")

        st.form_submit_button("🐧 Predict Species", width="stretch", type="primary")

    row = pd.DataFrame([{
        "island": island, "sex": sex,
        "bill_length_mm": bill_length, "bill_depth_mm": bill_depth,
        "flipper_length_mm": flipper, "body_mass_g": body_mass,
    }])

    prediction = reloaded.predict(row)[0]
    proba      = reloaded.predict_proba(row)[0]
    classes    = list(reloaded.classes_)
    confidence = proba[classes.index(prediction)]
    colour     = PALETTE[prediction]

    st.markdown("---")

    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown(f"""
        <div style="background:{colour}20; border: 2px solid {colour}; color:{colour};
                    border-radius:14px; padding:1.2rem; text-align:center;
                    font-weight:700; font-size:1.4rem; margin-bottom:1rem">
          🐧 {prediction}
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        col_a.metric("Confidence", f"{confidence*100:.1f}%")
        col_b.metric("Species",    prediction)
        st.dataframe(row, hide_index=True, width="stretch")

    with col2:
        probs_ordered = [proba[classes.index(s)] for s in SPECIES]
        fig, ax = plt.subplots(figsize=(5, 2.8))
        bars = ax.barh(SPECIES, probs_ordered,
                       color=[PALETTE[s] for s in SPECIES],
                       edgecolor="white", height=0.45)
        for bar, prob in zip(bars, probs_ordered):
            ax.text(prob + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{prob*100:.1f}%", va="center", fontsize=10, fontweight="600")
        ax.set_xlim(0, 1.18)
        ax.set_xlabel("Probability")
        ax.set_title("Probability per Species", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("#f9f9f9")
        st.pyplot(fig, width="stretch")
