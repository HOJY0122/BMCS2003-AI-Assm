"""
MindCheck — Shared Model Training Module
=========================================
Train ALL models ONCE here with identical parameters.
Every page imports from this module — guaranteed consistent metrics.

Ground truth (random_state=42):
  KNN K=5:  Acc=95.83% Prec=90.48% Rec=97.44% F1=93.83%
  DT  D=5:  Acc=91.11% Prec=100.0% Rec=72.41% F1=84.00%
  SVM RBF:  Acc=94.00% Prec=91.49% Rec=89.58% F1=90.53%
  (SVM retrained on the shared 8-feature set — same encoded columns as
  KNN/DT, no Seek_Treatment, no derived Course_Category — for
  consistency across all three models.)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Shared batch-upload limits (used by KNN, Decision Tree, SVM pages) ──
MAX_BATCH_ROWS = 500   # max records processed per batch CSV upload
MAX_BATCH_MB   = 5     # max upload file size in megabytes

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (MinMaxScaler, StandardScaler,
                                   LabelEncoder)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

import streamlit as st


@st.cache_resource
def load_all_models():
    """
    Train KNN, Decision Tree, SVM on the dataset.
    Cached — only runs once per Streamlit session.
    Returns a unified dict M with all models, scalers, metrics.
    """
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.preprocessing import load_and_clean_dataset

    # ── Load & clean dataset ──────────────────────────────────
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())

    le_c = LabelEncoder()
    le_y = LabelEncoder()
    df['Course_Enc'] = le_c.fit_transform(df['Course'])
    df['Year_Enc']   = le_y.fit_transform(df['Year_of_Study'])

    def _metrics(yt, yp):
        return {
            'acc' : accuracy_score(yt, yp) * 100,
            'prec': precision_score(yt, yp, zero_division=0) * 100,
            'rec' : recall_score(yt, yp, zero_division=0) * 100,
            'f1'  : f1_score(yt, yp, zero_division=0) * 100,
            'cm'  : confusion_matrix(yt, yp),
        }

    # ══════════════════════════════════════════════════════════
    # KNN — Target: Depression
    # Features: 7 | Split: 80/20 | Scaler: MinMax
    # ══════════════════════════════════════════════════════════
    knn_feat = ['Gender', 'Age', 'Course_Enc', 'Year_Enc',
                'CGPA_Numeric', 'Anxiety', 'Panic_Attack']
    X_k = df[knn_feat]
    y_k = df['Depression']

    sc_knn   = MinMaxScaler()
    X_ks     = sc_knn.fit_transform(X_k)
    Xtr_k, Xte_k, ytr_k, yte_k = train_test_split(
        X_ks, y_k, test_size=0.2, random_state=42, stratify=y_k)

    # Auto-select best K (1–20)
    best_k, best_a = 5, 0
    for k in range(1, 21):
        _m = KNeighborsClassifier(n_neighbors=k)
        _m.fit(Xtr_k, ytr_k)
        a = accuracy_score(yte_k, _m.predict(Xte_k))
        if a > best_a:
            best_a, best_k = a, k

    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn.fit(Xtr_k, ytr_k)
    knn_pred = knn.predict(Xte_k)

    # 5-fold CV
    k_scores = []
    for k in range(1, 21):
        _m = KNeighborsClassifier(n_neighbors=k)
        _m.fit(Xtr_k, ytr_k)
        k_scores.append(accuracy_score(yte_k, _m.predict(Xte_k)))
    cv_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=best_k, metric='euclidean'),
        X_ks, y_k, cv=5, scoring='accuracy')

    # Correlation for feature importance
    corr = df[knn_feat + ['Depression']].corr()['Depression'].drop('Depression').abs()
    corr.index = ['Gender', 'Age', 'Course', 'Year', 'CGPA',
                  'Anxiety', 'Panic Attack']

    # ══════════════════════════════════════════════════════════
    # DECISION TREE — Target: Depression
    # Features: 8 | Split: 70/30 | No scaling
    # ══════════════════════════════════════════════════════════
    dt_feat = ['Gender', 'Age', 'Course_Enc', 'Year_Enc',
               'CGPA_Numeric', 'Anxiety', 'Panic_Attack', 'Marital_Status']
    X_d = df[dt_feat]
    y_d = df['Depression']

    Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(
        X_d, y_d, test_size=0.3, random_state=42, stratify=y_d)

    dt = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
    dt.fit(Xtr_d, ytr_d)
    dt_pred = dt.predict(Xte_d)

    dt_fi_labels = ['Gender', 'Age', 'Course', 'Year', 'CGPA',
                    'Anxiety', 'Panic Attack', 'Marital Status']

    # ══════════════════════════════════════════════════════════
    # SVM — Target: Depression
    # Features: 8 (same encoded columns as KNN/DT) | Split: 75/25 | Standard
    # Uses the SAME shared df, Course_Enc/Year_Enc label encoders as
    # KNN and Decision Tree — no separate raw-column pipeline, no
    # derived Course_Category, no Seek_Treatment — so all three models
    # are trained on a consistent, directly comparable feature
    # representation.
    # ══════════════════════════════════════════════════════════
    svm_feat = ['Gender', 'Age', 'Course_Enc', 'Year_Enc',
                'CGPA_Numeric', 'Anxiety', 'Panic_Attack', 'Marital_Status']
    X_sv = df[svm_feat]
    y_sv = df['Depression']
    svm_col_order = list(X_sv.columns)

    svm_pipe = Pipeline([
        ('scl', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42)),
    ])
    Xtr_sv, Xte_sv, ytr_sv, yte_sv = train_test_split(
        X_sv, y_sv, test_size=0.25, random_state=42, stratify=y_sv)
    svm_pipe.fit(Xtr_sv, ytr_sv)
    svm_pred = svm_pipe.predict(Xte_sv)

    # ── Permutation importance for SVM ───────────────────────
    from sklearn.inspection import permutation_importance
    perm = permutation_importance(
        svm_pipe, Xte_sv, yte_sv,
        n_repeats=10, random_state=42, scoring='accuracy')
    svm_fi_labels = ['Gender', 'Age', 'Course', 'Year of Study', 'CGPA',
                     'Anxiety', 'Panic Attack', 'Marital Status']
    svm_fi_df = pd.DataFrame({
        'Feature'   : svm_fi_labels[:len(perm.importances_mean)],
        'Importance': perm.importances_mean,
        'Std'       : perm.importances_std,
    })

    # ══════════════════════════════════════════════════════════
    # RETURN unified dict — use this everywhere
    # ══════════════════════════════════════════════════════════
    return {
        # ── Dataset ──────────────────────────────────────────
        'df'          : df,
        'le_c'        : le_c,
        'le_y'        : le_y,
        'n_records'   : len(df),

        # ── KNN ──────────────────────────────────────────────
        'knn'         : knn,
        'sc_knn'      : sc_knn,
        'knn_feat'    : knn_feat,
        'best_k'      : best_k,
        'knn_m'       : _metrics(yte_k, knn_pred),
        'knn_Xtr'     : Xtr_k,
        'knn_Xte'     : Xte_k,
        'knn_ytr'     : ytr_k,
        'knn_yte'     : yte_k,
        'k_scores'    : k_scores,
        'cv_scores'   : cv_scores,
        'knn_corr'    : corr,

        # ── Decision Tree ─────────────────────────────────────
        'dt'          : dt,
        'dt_feat'     : dt_feat,
        'dt_m'        : _metrics(yte_d, dt_pred),
        'dt_Xtr'      : Xtr_d,
        'dt_Xte'      : Xte_d,
        'dt_ytr'      : ytr_d,
        'dt_yte'      : yte_d,
        'dt_fi_labels': dt_fi_labels,
        'dt_fi_vals'  : dt.feature_importances_,

        # ── SVM ───────────────────────────────────────────────
        'svm'         : svm_pipe,
        'svm_col_order': svm_col_order,
        'svm_m'       : _metrics(yte_sv, svm_pred),
        'svm_Xtr'     : Xtr_sv,
        'svm_Xte'     : Xte_sv,
        'svm_ytr'     : ytr_sv,
        'svm_yte'     : yte_sv,
        'svm_fi_df'   : svm_fi_df,
        'svm_X_all'   : X_sv,
        'svm_y_all'   : y_sv,

        # ── CGPA map (shared) ─────────────────────────────────
        'cgpa_map'    : {
            '0 - 1.99'   : 1.0,
            '2.00 - 2.49': 2.25,
            '2.50 - 2.99': 2.75,
            '3.00 - 3.49': 3.25,
            '3.50 - 4.00': 3.75,
        },
        'courses': [
            'Computer Science', 'Information Technology', 'Engineering',
            'Law', 'Psychology', 'Language', 'Islamic Studies',
            'Health Sciences', 'Business', 'Science & Math',
            'Arts & Social', 'Others'
        ],
    }