"""
data.py — Loads and caches the 20 Newsgroups dataset and trains both pipelines.
All other modules import constants and cached objects from here.
"""

import os
import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

CATEGORIES   = ['rec.sport.hockey', 'sci.space', 'talk.politics.guns', 'comp.graphics']
SHORT_LABELS = ['Comp. Graphics', 'Hockey', 'Space', 'Guns / Politics']
THRESHOLD    = 0.85
MODEL_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_classifier.joblib")


@st.cache_data
def load_data():
    data_train = fetch_20newsgroups(
        subset='train', categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes')
    )
    data_test = fetch_20newsgroups(
        subset='test', categories=CATEGORIES,
        remove=('headers', 'footers', 'quotes')
    )
    return data_train, data_test


@st.cache_resource
def build_models(_data_train):
    X_all, y_all = _data_train.data, _data_train.target

    # Stratified split — vectorizer fitted only on X_train to prevent leakage
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    baseline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('lr', LogisticRegression(max_iter=1000, random_state=42))
    ])
    baseline.fit(X_train, y_train)

    param_grid = {
        'tfidf__max_features': [5000, 10000],
        'tfidf__ngram_range':  [(1, 1), (1, 2)],
        'lr__C':               [0.1, 1, 10],
    }
    gs = GridSearchCV(
        Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('lr', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        param_grid, cv=5, scoring='f1_macro', n_jobs=-1
    )
    gs.fit(X_all, y_all)
    best = gs.best_estimator_

    joblib.dump(best, MODEL_PATH)
    return baseline, gs, best, X_train, X_val, y_train, y_val
