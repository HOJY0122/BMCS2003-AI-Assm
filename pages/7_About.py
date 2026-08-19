import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar

st.set_page_config(page_title="About — MindCheck", page_icon="ℹ️",
                   layout="wide", initial_sidebar_state="expanded")
sidebar("about")



# ── PAGE ───────────────────────────────────────────────────────
st.title("ℹ️ About MindCheck")
st.caption("Student Mental Health Prediction System — BMCS2003 Artificial Intelligence")
st.divider()

# Problem, Approach, Impact
st.subheader("Overview")
c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("**🎯 Problem Statement**")
        st.write(
            "Mental health issues among university students are increasing globally. "
            "Many students do not seek help early due to stigma and limited access "
            "to screening tools. There is a need for an intelligent, data-driven system "
            "that can identify at-risk students early and encourage timely intervention."
        )
with c2:
    with st.container(border=True):
        st.markdown("**📊 Our Approach**")
        st.write(
            "We implement and compare three supervised classification algorithms — "
            "KNN, Decision Tree (CART), and SVM — on 600 IIUM student records from Kaggle. "
            "Each member independently builds and evaluates a different algorithm "
            "using the same dataset and evaluation metrics."
        )
with c3:
    with st.container(border=True):
        st.markdown("**🚀 Significance**")
        st.write(
            "The system gives university counsellors an early-warning tool "
            "to flag at-risk students using academic and demographic features. "
            "The comparative analysis of three algorithms provides insights into "
            "the most effective method for this mental health classification task."
        )

st.divider()

# Objectives
st.subheader("Objectives")
o1, o2 = st.columns(2)
with o1:
    with st.container(border=True):
        st.markdown("**1. Predict Mental Health Conditions**")
        st.write("Develop a supervised ML system to predict depression, anxiety, and panic attack among university students using demographic and academic features.")
    st.write("")
    with st.container(border=True):
        st.markdown("**3. Evaluate Performance**")
        st.write("Assess each model using Accuracy, Precision, Recall, F1 Score, Confusion Matrix and 5-Fold Cross Validation.")
with o2:
    with st.container(border=True):
        st.markdown("**2. Compare Three Algorithms**")
        st.write("Implement and compare KNN, Decision Tree (CART), and SVM — each by a different group member — on the same dataset.")
    st.write("")
    with st.container(border=True):
        st.markdown("**4. Deploy Interactive Prototype**")
        st.write("Build and deploy a multi-page Streamlit web app with live prediction forms, EDA visualizations, and model comparison dashboard.")

st.divider()

# Pipeline
st.subheader("System Pipeline")
st.caption("Five stages from raw data to real-time prediction")
st.write("")

p1, p2, p3, p4, p5 = st.columns(5)
for col, icon, title, desc in zip(
    [p1, p2, p3, p4, p5],
    ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"],
    ["Data Collection", "Preprocessing", "Model Training", "Evaluation", "Deployment"],
    [
        "600 records from Kaggle IIUM student mental health survey",
        "Clean, encode, scale and engineer new features from raw data",
        "Train KNN (K=5), Decision Tree (depth 5), SVM (RBF) independently",
        "Accuracy, Precision, Recall, F1, Confusion Matrix, 5-Fold CV",
        "Interactive Streamlit multi-page web application"
    ]
):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)

st.divider()

# Team
st.subheader("Group Members")
st.caption("Tutorial Group 3 | Tutor: Dr Goh | BMCS2003 AI | 202605 Session")
st.write("")

t1, t2, t3 = st.columns(3)
with t1:
    with st.container(border=True):
        st.markdown("#### Member 1")
        st.markdown("**Ho Jun Yon**")
        st.caption("Student ID: 2612634")
        st.divider()
        st.write("**Algorithm:** K-Nearest Neighbor (KNN)")
        st.write("**Target:** Depression Prediction")
        ca, cb = st.columns(2)
        ca.metric("Accuracy", "95.83%")
        cb.metric("Recall", "97.44%")
        st.caption("Best K=5 · MinMax Scaler · 80/20 Split · CV Mean: 86.67%")
        if st.button("View KNN Page", key="tm_knn", use_container_width=True):
            st.switch_page("pages/2_KNN.py")

with t2:
    with st.container(border=True):
        st.markdown("#### Member 2")
        st.markdown("**Irvin Tan Wei Shen**")
        st.caption("Student ID: 2612638")
        st.divider()
        st.write("**Algorithm:** Decision Tree (CART)")
        st.write("**Target:** Depression Prediction")
        ca, cb = st.columns(2)
        ca.metric("Accuracy", "85.50%")
        cb.metric("Recall", "89.18%")
        st.caption("Max Depth=5 · Gini Criterion · Root: Marital Status")
        if st.button("View DT Page", key="tm_dt", use_container_width=True):
            st.switch_page("pages/3_Decision_Tree.py")

with t3:
    with st.container(border=True):
        st.markdown("#### Member 3")
        st.markdown("**Chiang Jun Hang**")
        st.caption("Student ID: 2612610")
        st.divider()
        st.write("**Algorithm:** Support Vector Machine (SVM)")
        st.write("**Target:** Panic Attack Prediction")
        ca, cb = st.columns(2)
        ca.metric("Accuracy", "TBD")
        cb.metric("Recall", "TBD")
        st.caption("RBF Kernel · Standard Scaler · 75/25 Split")
        if st.button("View SVM Page", key="tm_svm", use_container_width=True):
            st.switch_page("pages/4_SVM.py")

st.divider()
st.caption("MindCheck · BMCS2003 Artificial Intelligence · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")
