"""
data.py — Loads and caches the IBM HR Attrition dataset.
All other modules import from here.
"""

import ssl
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context

URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
DROP = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]


@st.cache_data
def load_data():
    df = pd.read_csv(URL)
    df.drop(columns=DROP, inplace=True)

    y = (df["Attrition"] == "Yes").astype(int)
    X = df.drop(columns=["Attrition"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return df, X, y, X_train, X_test, y_train, y_test
