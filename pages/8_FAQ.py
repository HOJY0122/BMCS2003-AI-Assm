import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar
from utils.models import load_all_models

st.set_page_config(page_title="FAQ — MindCheck", page_icon="❓",
                   layout="wide", initial_sidebar_state="expanded")
sidebar("faq")

M = load_all_models()

# ── Live metrics used throughout this page — no hardcoded numbers,
# so this always matches whatever's shown on the predictor/comparison
# pages, even if the dataset changes. ──────────────────────────────
_knn = M['knn_m']; _dt = M['dt_m']; _svm = M['svm_m']
_knn_tn, _knn_fp, _knn_fn, _knn_tp = _knn['cm'].ravel()
_knn_test_n   = int(_knn['cm'].sum())
_knn_wrong_n  = int(_knn_fp + _knn_fn)
_cv = M['cv_scores']

# ── PAGE ──────────────────────────────────────────────────────
st.title("❓ Frequently Asked Questions")
st.caption("Learn about the algorithms, metrics, and system used in this project")
st.divider()

# Section: Algorithms
st.subheader("Algorithms")

with st.expander("What is KNN (K-Nearest Neighbor)?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(
            "KNN classifies a student's mental health status by finding the **K most similar "
            "students** in the training data and taking a majority vote on their class label. "
            "Distance is measured using **Euclidean distance** after MinMax scaling to normalize "
            "all features to the range [0, 1]."
        )
        st.write(
            f"In this project, K={M['best_k']} was selected as optimal by systematically testing K "
            "values from 1 to 20 and selecting the K with the highest test accuracy."
        )
    with c2:
        st.metric("Best K", f"{M['best_k']}")
        st.metric("Accuracy", f"{_knn['acc']:.2f}%")
        st.metric("Recall", f"{_knn['rec']:.2f}%")

with st.expander("What is Decision Tree (CART)?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(
            "Decision Tree CART recursively splits data based on the feature that most reduces "
            "**Gini impurity** at each node. It builds an interpretable tree structure where each "
            "path from root to leaf represents a decision rule."
        )
        st.write(
            "Our tree has max depth 5, with **Marital Status** as the root split — the most "
            "discriminative feature identified by the algorithm. The tree structure can be "
            "visualized to understand how decisions are made."
        )
    with c2:
        st.metric("Max Depth", "5")
        st.metric("Accuracy", f"{_dt['acc']:.2f}%")
        st.metric("Root Feature", "Marital Status")

with st.expander("What is SVM (Support Vector Machine)?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(
            "SVM finds the **optimal hyperplane** that maximally separates two classes in "
            "high-dimensional feature space. The distance from the hyperplane to the nearest "
            "data points (support vectors) is maximized — this is called the margin."
        )
        st.write(
            "The **RBF (Radial Basis Function) kernel** is used to handle non-linear data "
            "by implicitly mapping features to a higher-dimensional space. "
            "Standard Scaling is applied before training."
        )
    with c2:
        st.metric("Kernel", "RBF")
        st.metric("Accuracy", f"{_svm['acc']:.2f}%")
        st.metric("Target", "Depression")

st.divider()

# Section: Metrics
st.subheader("Evaluation Metrics")

with st.expander("What is Accuracy?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Accuracy** measures the overall proportion of correctly classified instances out of all instances.")
        st.code("Accuracy = (TP + TN) / (TP + TN + FP + FN)", language="text")
        st.write(
            f"KNN achieved **{_knn['acc']:.2f}%** accuracy on the {_knn_test_n}-record test set — "
            f"only {_knn_wrong_n} wrong prediction{'s' if _knn_wrong_n != 1 else ''} out of {_knn_test_n}."
        )
    with c2:
        st.metric("KNN", f"{_knn['acc']:.2f}%")
        st.metric("Decision Tree", f"{_dt['acc']:.2f}%")
        st.metric("SVM", f"{_svm['acc']:.2f}%")

with st.expander("What is Precision?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Precision** measures of all students predicted as depressed, how many actually are depressed.")
        st.code("Precision = TP / (TP + FP)", language="text")
        st.write("High precision means fewer false alarms — students predicted as depressed who are not actually depressed.")
    with c2:
        st.metric("KNN", f"{_knn['prec']:.2f}%")
        st.metric("Decision Tree", f"{_dt['prec']:.2f}%")
        st.metric("SVM", f"{_svm['prec']:.2f}%")

with st.expander("What is Recall (Sensitivity)?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**Recall** measures of all students who actually have depression, how many the model correctly identified.")
        st.code("Recall = TP / (TP + FN)", language="text")
        st.write(
            "Recall is the **most important metric** for mental health screening. "
            "Missing a depressed student (false negative) is more serious than "
            f"a false alarm (false positive). KNN achieved **{_knn['rec']:.2f}% recall** — "
            f"it correctly identified {_knn['rec']:.2f}% of all depressed students."
        )
    with c2:
        st.metric("KNN", f"{_knn['rec']:.2f}%")
        st.metric("Decision Tree", f"{_dt['rec']:.2f}%")
        st.metric("SVM", f"{_svm['rec']:.2f}%")

with st.expander("What is F1 Score?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write("**F1 Score** is the harmonic mean of Precision and Recall, balancing both metrics into a single number.")
        st.code("F1 = 2 × (Precision × Recall) / (Precision + Recall)", language="text")
        st.write("It is useful when you need to balance between false positives and false negatives.")
    with c2:
        st.metric("KNN", f"{_knn['f1']:.2f}%")
        st.metric("Decision Tree", f"{_dt['f1']:.2f}%")
        st.metric("SVM", f"{_svm['f1']:.2f}%")

with st.expander("What is a Confusion Matrix?", expanded=False):
    st.write("A **Confusion Matrix** shows the full breakdown of predictions vs actual values.")
    st.write("")
    cm1, cm2, cm3 = st.columns([1, 2, 1])
    with cm2:
        st.markdown("""
        |  | **Predicted: No** | **Predicted: Yes** |
        |--|--|--|
        | **Actual: No** | True Negative (TN) | False Positive (FP) |
        | **Actual: Yes** | False Negative (FN) | True Positive (TP) |
        """)
    st.write("")
    st.info(
        f"**KNN Confusion Matrix (Test Set: {_knn_test_n} records):**\n\n"
        f"TN = {int(_knn_tn)} (correctly predicted No Depression) | "
        f"FP = {int(_knn_fp)} (wrongly predicted Depression)\n\n"
        f"FN = {int(_knn_fn)} (missed {int(_knn_fn)} depressed student{'s' if _knn_fn != 1 else ''}) | "
        f"TP = {int(_knn_tp)} (correctly predicted Depression)"
    )

with st.expander("What is 5-Fold Cross Validation?", expanded=False):
    c1, c2 = st.columns([2, 1])
    with c1:
        st.write(
            "Instead of testing the model once, **5-Fold CV** splits training data into "
            "5 equal parts. It trains and tests 5 times — each time using a different "
            "part as the test set — then averages the results."
        )
        st.write("KNN 5-Fold CV Results:")
        fold_data = {
            "Fold": [f"Fold {i+1}" for i in range(len(_cv))],
            "Accuracy": [f"{s*100:.1f}%" for s in _cv],
        }
        st.dataframe(pd.DataFrame(fold_data), width='stretch', hide_index=True)
    with c2:
        st.metric("CV Mean", f"{_cv.mean()*100:.2f}%")
        st.metric("CV Std Dev", f"{_cv.std()*100:.2f}%")
        st.metric("CV Max", f"{_cv.max()*100:.2f}%")

st.divider()

# Section: General
st.subheader("General Questions")

with st.expander("Why three different algorithms?", expanded=False):
    st.write(
        "Each member independently implements a different algorithm to enable a "
        "**rigorous side-by-side comparison** on the same dataset. By keeping the "
        "dataset, preprocessing, and evaluation metrics consistent, any performance "
        "difference can be attributed purely to the algorithm choice — "
        "this is good experimental design in ML research."
    )

with st.expander("Why is the dataset from IIUM Malaysia?", expanded=False):
    st.write(
        "The dataset was collected by Shariful07 (2020) through an online survey "
        "from students at the **International Islamic University Malaysia (IIUM)**. "
        "It is publicly available on Kaggle and is one of the few mental health "
        "datasets specifically targeting **Malaysian university students**, "
        "making it highly relevant for this project context."
    )

with st.expander("What does Panic_Attack target mean for SVM?", expanded=False):
    st.write(
        "For Member 3 (SVM), the prediction target is whether a student has **panic attacks** "
        "(Yes = 1 / No = 0). This is different from Members 1 and 2 who predict **depression**. "
        "Each member chose a different target variable to demonstrate that the system can "
        "predict multiple mental health conditions, not just one."
    )

with st.expander("How do I use the prediction form?", expanded=False):
    st.write(
        "Navigate to any model page (KNN, Decision Tree, or SVM) using the sidebar. "
        "Scroll down to **Step 5: Prediction**. Fill in the student information form with:"
    )
    st.write("- Name, Gender, Age, Course, Year of Study, CGPA")
    st.write("- Whether the student has Anxiety and/or Panic Attack")
    st.write("Then click **Predict** — the model will instantly show the predicted result and confidence level.")

st.divider()
st.caption("MindCheck · BMCS2003 Artificial Intelligence · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")