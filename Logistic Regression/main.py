import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# CONFIGURATION
st.set_page_config(page_title="Logistic Regression Visualization")
st.title("Amazon Reviews")


# LOAD DATA 
@st.cache_data
def load_data():
    df = pd.read_csv('amazon_reviews.csv')
    features = ['review_length', 'rating', 'num_exclamation_marks',
                'contains_question_mark', 'uppercase_ratio', 'helpful_votes']
    target = 'is_helpful'
    df = df[features + [target]]
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[target] = pd.to_numeric(df[target], errors='coerce')
    df = df.dropna()
    df[target] = df[target].astype(int)
    return df

df = load_data()
features = ['review_length', 'rating', 'num_exclamation_marks',
            'contains_question_mark', 'uppercase_ratio', 'helpful_votes']
X = df[features].astype(np.float64)
y = df['is_helpful'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, max_iter=1000))
])

K_FOLDS = 5
y_cv_pred = cross_val_predict(pipe, X_train, y_train, cv=K_FOLDS)
pipe.fit(X_train, y_train)
y_test_pred = pipe.predict(X_test)

def show_scenario_cards(cm):
    tn, fp, fn, tp = cm.ravel()
    total_actual_helpful     = tp + fn
    total_actual_not_helpful = tn + fp
    tp_rate = tp / total_actual_helpful     if total_actual_helpful     > 0 else 0
    tn_rate = tn / total_actual_not_helpful if total_actual_not_helpful > 0 else 0
    fp_rate = fp / total_actual_not_helpful if total_actual_not_helpful > 0 else 0
    fn_rate = fn / total_actual_helpful     if total_actual_helpful     > 0 else 0

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**True Positive**\n\nOf reviews that were helpful, the model predicted helpful {tp_rate*100:.1f}% of the time. ({tp} out of {total_actual_helpful})")
    with c2:
        st.markdown(f"**True Negative**\n\nOf reviews that were not helpful, the model predicted not helpful {tn_rate*100:.1f}% of the time. ({tn} out of {total_actual_not_helpful})")
    with c1:
        st.markdown(f"**False Negative**\n\nOf reviews that were helpful, the model wrongly predicted not helpful {fn_rate*100:.1f}% of the time. ({fn} out of {total_actual_helpful})")
    with c2:
        st.markdown(f"**False Positive**\n\nOf reviews that were not helpful, the model wrongly predicted helpful {fp_rate*100:.1f}% of the time. ({fp} out of {total_actual_not_helpful})")

def make_confusion_heatmap(cm, title):
    tn, fp, fn, tp = cm.ravel()
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        text=[[f"{tn}\nTrue Negative\n(Not helpful)", f"{fp}\nFalse Positive"],
              [f"{fn}\nFalse Negative", f"{tp}\nTrue Positive\n(Helpful)"]],
        texttemplate="%{text}",
        textfont={"size": 11},
        colorscale='Blues',
        showscale=False
    ))
    fig.update_layout(title=title, xaxis_title='Predicted', yaxis_title='Actual', height=380)
    return fig

# Cross-validation results (80% train set)
st.subheader("Cross-Validation Results (80% Train Set)")
st.caption(f"{K_FOLDS}-fold cross-validation on the training data.")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{accuracy_score(y_train, y_cv_pred):.3f}")
with col2:
    st.metric("Precision", f"{precision_score(y_train, y_cv_pred, zero_division=0):.3f}")
with col3:
    st.metric("Recall", f"{recall_score(y_train, y_cv_pred, zero_division=0):.3f}")

cm_cv = confusion_matrix(y_train, y_cv_pred)
show_scenario_cards(cm_cv)
st.write("**Confusion Matrix — Cross-Validation**")
st.plotly_chart(make_confusion_heatmap(cm_cv, "CV Confusion Matrix"), use_container_width=True)

# Test set results (held-out 20%)
st.subheader("Test Set Results (Held-Out 20%)")
st.caption("Model trained on 80% of data, evaluated on the unseen 20%.")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, y_test_pred):.3f}")
with col2:
    st.metric("Precision", f"{precision_score(y_test, y_test_pred, zero_division=0):.3f}")
with col3:
    st.metric("Recall", f"{recall_score(y_test, y_test_pred, zero_division=0):.3f}")

cm_test = confusion_matrix(y_test, y_test_pred)
show_scenario_cards(cm_test)
st.write("**Confusion Matrix — Test Set**")
st.plotly_chart(make_confusion_heatmap(cm_test, "Test Set Confusion Matrix"), use_container_width=True)

# --- 5. VISUALIZATION ---
st.subheader("Probability vs Feature")
st.write("As **review length** (or selected feature) increases, the predicted probability of being helpful changes. All other variables are held constant at their median.")

selected_feature = st.selectbox("Feature:", features, index=0)

min_val = float(X_train[selected_feature].min())
max_val = float(X_train[selected_feature].max())
span = max(max_val - min_val, 1.0)
x_min_plot = max(0, min_val - 1.2 * span)
x_max_plot = max_val + 1.2 * span
n_pts = 300
x_range = np.linspace(x_min_plot, x_max_plot, n_pts)

other_constant = X_train.median()
temp = pd.DataFrame({
    col: (x_range if col == selected_feature else np.full(n_pts, other_constant[col]))
    for col in features
})[features]
y_probs = pipe.predict_proba(temp)[:, 1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_range, y_probs, color='#2e86ab', linewidth=2.5, label='Sigmoid (other vars constant)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='Decision boundary (0.5)')

ax.set_title(f"The Sigmoid Curve: P(helpful) vs {selected_feature.replace('_', ' ').title()}", fontsize=14)
ax.set_xlabel(selected_feature.replace('_', ' ').title(), fontsize=12)
ax.set_ylabel("Predicted probability (helpful = 1)", fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
plt.close(fig)