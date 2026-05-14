"""
data.py — Loads and caches the Palmer Penguins dataset.
All other modules import from here.
"""

import streamlit as st
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split


@st.cache_data
def load_data():
    df = load_penguins()
    df.dropna(inplace=True)
    df["sex"] = df["sex"].str.capitalize()

    X = df[["island", "sex", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return df, X, y, X_train, X_test, y_train, y_test
