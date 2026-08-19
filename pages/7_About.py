import streamlit as st

st.set_page_config(page_title="About — MindCheck", page_icon="ℹ️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] { background: #12152A !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06) !important; color: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    text-align: left !important; transition: all 0.18s !important; margin-bottom: 4px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(61,82,255,0.25) !important; border-color: #3D52FF !important; color: white !important;
}
div[data-testid="metric-container"] {
    background: #F8F9FF; border: 1px solid #E2E8FF; border-radius: 12px;
    padding: 18px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🧠 MindCheck")
    st.caption("BMCS2003 Artificial Intelligence")
    st.divider()
    st.markdown("**MAIN**")
    if st.button("🏠  Home", key="sb_home", use_container_width=True): st.switch_page("Home.py")
    st.divider()
    st.markdown("**ANALYSIS**")
    if st.button("📊  EDA", key="sb_eda", use_container_width=True): st.switch_page("pages/1_EDA.py")
    if st.button("📋  Dataset", key="sb_ds", use_container_width=True): st.switch_page("pages/6_Dataset.py")
    st.divider()
    st.markdown("**MODELS**")
    if st.button("🔵  KNN — Ho Jun Yon", key="sb_knn", use_container_width=True): st.switch_page("pages/2_KNN.py")
    if st.button("🌳  Decision Tree — Irvin", key="sb_dt", use_container_width=True): st.switch_page("pages/3_Decision_Tree.py")
    if st.button("🔴  SVM — Chiang Jun Hang", key="sb_svm", use_container_width=True): st.switch_page("pages/4_SVM.py")
    st.divider()
    st.markdown("**RESULTS**")
    if st.button("📈  Compare All Models", key="sb_cmp", use_container_width=True): st.switch_page("pages/5_Comparison.py")
    st.divider()
    st.markdown("**INFO**")
    if st.button("ℹ️  About", key="sb_about", use_container_width=True): st.switch_page("pages/7_About.py")
    if st.button("❓  FAQ", key="sb_faq", use_container_width=True): st.switch_page("pages/8_FAQ.py")
    st.divider()
    st.caption("Tutorial Group 3 · Tutor: Dr Goh\n202605 Session · TARUMT")

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
