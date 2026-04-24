"""
data.py — Generates and caches the synthetic Titanic/ClearBank dataset.
All other modules import from here to ensure a single source of truth.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split


@st.cache_data
def load_data():
    np.random.seed(42)

    records = []
    specs = [
        # (pclass, sex, n, age_mu, age_sd, sibsp_dist, sibsp_p, parch_dist, parch_p, fare_mu, fare_sd, emb_opts, emb_p, surv_p)
        (1,"female", 94, 37,14, [0,1,2,3],[.55,.30,.10,.05], [0,1,2],[.65,.25,.10], 106,75, ["S","C","Q"],[.45,.50,.05], 0.968),
        (1,"male",  122, 41,15, [0,1,2],  [.70,.20,.10],     [0,1,2],[.75,.15,.10],  69,62, ["S","C","Q"],[.45,.52,.03], 0.369),
        (2,"female", 76, 28,12, [0,1,2],  [.60,.30,.10],     [0,1,2],[.55,.30,.15],  22,10, ["S","C","Q"],[.65,.30,.05], 0.921),
        (2,"male",  108, 30,14, [0,1,2],  [.70,.22,.08],     [0,1,2],[.65,.25,.10],  19,10, ["S","C","Q"],[.70,.26,.04], 0.157),
        (3,"female",144, 22,12, [0,1,2,3,4],[.45,.25,.15,.10,.05],[0,1,2,3],[.45,.25,.20,.10],15,10,["S","C","Q"],[.55,.20,.25],0.500),
        (3,"male",  347, 26,14, [0,1,2,3,4,5],[.55,.20,.12,.07,.04,.02],[0,1,2,3,4],[.65,.15,.10,.06,.04],13,10,["S","C","Q"],[.60,.15,.25],0.135),
    ]

    for pclass,sex,n,a_mu,a_sd,sib_d,sib_p,par_d,par_p,f_mu,f_sd,emb_opts,emb_p,sp in specs:
        for _ in range(n):
            records.append({
                "pclass":   pclass,
                "sex":      sex,
                "age":      np.random.normal(a_mu, a_sd),
                "sibsp":    np.random.choice(sib_d, p=sib_p),
                "parch":    np.random.choice(par_d, p=par_p),
                "fare":     abs(np.random.normal(f_mu, f_sd)),
                "embarked": np.random.choice(emb_opts, p=emb_p),
                "survived": int(np.random.random() < sp),
            })

    df = pd.DataFrame(records)

    # Inject missing values
    age_mask = np.random.random(len(df)) < 0.20
    df.loc[age_mask, "age"] = np.nan
    df.loc[np.random.choice(df.index, 2, replace=False), "embarked"] = np.nan

    # Clip to realistic ranges
    df["age"]   = df["age"].clip(0.5, 80)
    df["fare"]  = df["fare"].clip(0, 512)
    df["sibsp"] = df["sibsp"].clip(0, 8)
    df["parch"] = df["parch"].clip(0, 6)

    y = df["survived"]
    X = df.drop(columns="survived")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return df, X, y, X_train, X_test, y_train, y_test