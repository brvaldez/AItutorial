import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, make_scorer, recall_score, precision_score

# PAGE CONFIGURATION
st.set_page_config(page_title="Breast Cancer Diagnosis via Naive Bayes", layout="wide")


@st.cache_data
def load_and_train():
    # Load Data
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # Split (Keep this for the specific confusion matrix visualization)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the main model
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Predict on Test Set (for CM visualization only)
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- CROSS VALIDATION METRICS ---
    k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Custom scorers for Malignant (0) as Positive Class
    # Precision: TP / (TP + FP)
    # Recall (Sensitivity): TP / (TP + FN)
    scorer_precision = make_scorer(precision_score, pos_label=0)
    scorer_recall = make_scorer(recall_score, pos_label=0)

    # Specificity is Recall of the Negative Class (1)
    scorer_specificity = make_scorer(recall_score, pos_label=1)

    # Calculate Cross-Val Scores
    cv_acc = cross_val_score(model, X, y, cv=k_fold, scoring='accuracy').mean()
    cv_sensitivity = cross_val_score(model, X, y, cv=k_fold, scoring=scorer_recall).mean()
    cv_precision = cross_val_score(model, X, y, cv=k_fold, scoring=scorer_precision).mean()
    cv_specificity = cross_val_score(model, X, y, cv=k_fold, scoring=scorer_specificity).mean()


    return X, y, feature_names, target_names, test_accuracy, cv_acc, cv_sensitivity, cv_precision, cv_specificity, cm, y_test, y_pred


# Load the data and model results
X, y, feature_names, target_names, test_acc, cv_acc, cv_sens, cv_prec, cv_spec, cm, y_test, y_pred = load_and_train()

# MODEL PERFORMANCE SECTION
st.header("1. Model Performance")
st.markdown("""
We evaluate the model using **5-Fold Cross Validation** to ensure robustness. 
*Metrics below represent the average performance across 5 different data splits.*
""")

col_metrics, col_cm = st.columns([1, 2])

with col_metrics:
    st.subheader("Key Metrics (5-Fold CV)")

    st.metric("Overall Accuracy", f"{cv_acc:.1%}", help="Average accuracy across 5 folds.")

    st.markdown("### Conditional Proportions")
    st.markdown("*(Based on Cross Validation Averages)*")

    st.markdown("**Given tumor is Malignant** → % predicted Malignant:")
    st.metric("Sensitivity (Recall)", f"{cv_sens:.1%}", label_visibility="collapsed")

    st.markdown("**Given tumor is Benign** → % predicted Benign:")
    st.metric("Specificity (Recall of Benign)", f"{cv_spec:.1%}", label_visibility="collapsed")

    st.markdown("**Given model predicts Malignant** → % actually Malignant:")
    st.metric("Precision", f"{cv_prec:.1%}", label_visibility="collapsed")

with col_cm:
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Malignant (Pred)', 'Benign (Pred)'],
                yticklabels=['Malignant (True)', 'Benign (True)'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    st.pyplot(fig_cm)

# Calculate Separation Scores
feature_scores = []
for i, name in enumerate(feature_names):
    malignant_values = X[y == 0][:, i]
    benign_values = X[y == 1][:, i]

    m_std = malignant_values.std()
    b_std = benign_values.std()

    if m_std + b_std == 0:
        separation = 0
    else:
        separation = abs(malignant_values.mean() - benign_values.mean()) / (m_std + b_std)

    feature_scores.append((name, separation, i))

# Sort and get top 3
feature_scores.sort(key=lambda x: x[1], reverse=True)
top_3_features = feature_scores[:3]

# --- 4. VISUALIZATION OF TOP 3 FEATURES ---
st.subheader("2. Top 3 Discriminative Features")
st.markdown(
    "These graphs show *why* the model works: the Benign (Blue) and Malignant (Red) populations are clearly distinct.")

# Create 3 columns for the 3 graphs
cols = st.columns(3)

for idx, (name, score, feature_idx) in enumerate(top_3_features):
    with cols[idx]:
        # Data for this feature
        mal_vals = X[y == 0][:, feature_idx]
        ben_vals = X[y == 1][:, feature_idx]

        fig, ax = plt.subplots(figsize=(5, 4))

        # Histograms
        ax.hist(mal_vals, bins=25, alpha=0.5, color='#c0392b', label='Malignant', density=True)
        ax.hist(ben_vals, bins=25, alpha=0.5, color='#2980b9', label='Benign', density=True)

        # Gaussian Fits
        x_range = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 200)
        m_mean, m_std = mal_vals.mean(), mal_vals.std()
        b_mean, b_std = ben_vals.mean(), ben_vals.std()

        g_mal = (1 / (m_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - m_mean) / m_std) ** 2)
        g_ben = (1 / (b_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - b_mean) / b_std) ** 2)

        ax.plot(x_range, g_mal, color='#922b21', linewidth=2)
        ax.plot(x_range, g_ben, color='#1a5276', linewidth=2)

        ax.set_title(f"#{idx + 1}: {name}\nScore: {score:.2f}", fontsize=10)
        ax.get_yaxis().set_visible(False)  # Hide y-axis for cleaner look
        ax.legend(fontsize=8)

        st.pyplot(fig)