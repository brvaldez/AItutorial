"""
part5.py — Live Classifier Demo
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from data import SHORT_LABELS, MODEL_PATH

DEMO_SNIPPETS = [
    (
        "Snippet 1 — Space",
        "NASA has confirmed that the Artemis mission will launch next month, carrying "
        "astronauts to lunar orbit for the first time since Apollo. Scientists are "
        "particularly excited about studying the moon's south pole ice deposits."
    ),
    (
        "Snippet 2 — Hockey",
        "The Stanley Cup playoffs are heating up. The overtime goal last night was "
        "incredible — the puck deflected off two defenders before finding the back "
        "of the net. The goalie had no chance."
    ),
]


def render(best_pipeline, threshold):
    st.header("Live Classifier Demo")
    st.markdown("Enter any text below to classify it using the saved pipeline.")

    for title, default_text in DEMO_SNIPPETS:
        st.subheader(title)
        user_text = st.text_area("Text", value=default_text, height=100, key=title)

        if user_text.strip():
            proba_live = best_pipeline.predict_proba([user_text])[0]
            pred_idx   = int(np.argmax(proba_live))
            confidence = proba_live[pred_idx]
            pred_label = SHORT_LABELS[pred_idx]
            routed_to  = "Auto-classified ✅" if confidence >= threshold else "Human review 🔍"

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Category", pred_label)
            c2.metric("Confidence",         f"{confidence:.3f}")
            c3.metric("Routing Decision",   routed_to)

            fig, ax = plt.subplots(figsize=(5, 2.5))
            colors = ['#4CAF50' if i == pred_idx else '#90CAF9' for i in range(4)]
            ax.barh(SHORT_LABELS, proba_live, color=colors, edgecolor='white')
            ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5,
                       label=f'Threshold = {threshold:.2f}')
            ax.set_xlabel('Probability')
            ax.set_title('Class Probabilities')
            ax.legend(fontsize=9)
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        st.divider()

    st.caption(f"Pipeline saved to `{MODEL_PATH}` using joblib.")
