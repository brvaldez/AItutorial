import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="KNN Diabetes Predictor", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('knn_data.csv')
    return df

df = load_data()

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test (80% training, 20% held-out test set)
# the test portion is preserved and never used during cross-validation
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# X_test and y_test represent the 20% hold‑out set used only for final evaluation


# Streamlit app
st.title("KNN Diabetes Prediction Model")
st.markdown("---")

# Sidebar for user input
st.sidebar.header("Configuration")
k_value = st.sidebar.slider("Select K value", min_value=1, max_value=25, value=5)

# Train model with selected K
knn = KNeighborsClassifier(n_neighbors=k_value)

# 5-fold cross‑validation on training set only (the reserved 20% is not touched)
cv_acc_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
cv_prec_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='precision')
cv_rec_scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='recall')
cv_acc = np.mean(cv_acc_scores)
cv_prec = np.mean(cv_prec_scores)
cv_rec = np.mean(cv_rec_scores)

# now retrain on the full training portion before evaluating on hold‑out set
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display metrics with smaller font in two rows: CV metrics first, hold‑out second
st.subheader(f"Model Performance with K = {k_value}")

st.write("*Note: first row shows cross‑validation results (5 folds on the training set); "
         "second row shows evaluation on the 20% hold‑out test set.*")

st.markdown("<div style='font-size:90%'>", unsafe_allow_html=True)
# first row: CV metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("CV Accuracy (train)", f"{cv_acc:.2%}")
with col2:
    st.metric("CV Precision (train)", f"{cv_prec:.2%}")
with col3:
    st.metric("CV Recall (train)", f"{cv_rec:.2%}")

st.markdown("<br>", unsafe_allow_html=True)

# second row: hold‑out/test metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Test Accuracy", f"{accuracy:.2%}")
with col2:
    st.metric("Test Precision", f"{precision:.2%}")
with col3:
    st.metric("Test Recall", f"{recall:.2%}")

# brief definitions below test metrics
st.caption(
    "Accuracy = overall correctness; "
    "Precision = of predicted positive, how many correct; "
    "Recall = of actual positive, how many found"
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# Confusion Matrix
st.subheader("Confusion Matrix")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', aspect='auto')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['No Diabetes', 'Diabetes'])
    ax.set_yticklabels(['No Diabetes', 'Diabetes'])
    
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", 
                          color="white" if cm[i, j] > cm.max() / 2 else "black", 
                          fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.write("**Confusion Matrix Breakdown:**")
    st.write("")
    
    st.write("**True Negative (TN):** " + str(cm[0,0]))
    st.write("→ Model predicted 'No Diabetes' AND patient actually does not have diabetes")
    
    st.write("")
    st.write("**False Positive (FP):** " + str(cm[0,1]))
    st.write("→ Model predicted 'Diabetes' BUT patient does NOT have diabetes")
    
    st.write("")
    st.write("**False Negative (FN):** " + str(cm[1,0]))
    st.write("→ Model predicted 'No Diabetes' BUT patient DOES have diabetes")
    
    st.write("")
    st.write("**True Positive (TP):** " + str(cm[1,1]))
    st.write("→ Model predicted 'Diabetes' AND patient actually has diabetes")

st.markdown("---")

# K Value Analysis
st.subheader("Effect of K Value on Model Performance")

st.write("*Blue line = training accuracy (80% train set); orange line = testing accuracy (20% hold‑out set)*")

# Calculate accuracies for different K values
k_values = list(range(1, 26))
train_accs = []
test_accs = []
    
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, knn.predict(X_train)))
    test_accs.append(accuracy_score(y_test, knn.predict(X_test)))
    
# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_values, train_accs, marker='o', label='Training Accuracy', linewidth=2, markersize=6)
ax.plot(k_values, test_accs, marker='s', label='Testing Accuracy', linewidth=2, markersize=6)

best_k = k_values[np.argmax(test_accs)]
best_test_acc = max(test_accs)

ax.set_xlabel('K Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Effect of K on KNN Model Performance', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(k_values[::2])

st.pyplot(fig)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Optimal K Value", best_k)
with col2:
    st.metric("Best Hold‑out Accuracy", f"{best_test_acc:.4f}")
with col3:
    st.metric("Training Accuracy at Best K", f"{train_accs[best_k-1]:.4f}")
