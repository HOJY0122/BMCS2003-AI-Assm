import streamlit as st

st.set_page_config(
    page_title="MindCheck — Student Mental Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS (minimal, safe) ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] { background: #12152A !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.8) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    text-align: left !important; transition: all 0.18s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(61,82,255,0.2) !important;
    border-color: #3D52FF !important; color: white !important;
}
div[data-testid="metric-container"] {
    background: white; border: 1px solid #E2E8FF;
    border-radius: 12px; padding: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 MindCheck")
    st.caption("BMCS2003 Artificial Intelligence")
    st.divider()

    st.markdown("**MAIN**")
    if st.button("🏠  Home", key="sb_home", use_container_width=True):
        st.switch_page("Home.py")

    st.markdown("**ANALYSIS**")
    if st.button("📊  Exploratory Data Analysis", key="sb_eda", use_container_width=True):
        st.switch_page("pages/1_EDA.py")
    if st.button("📋  Dataset Overview", key="sb_ds", use_container_width=True):
        st.switch_page("pages/6_Dataset.py")

    st.markdown("**MODELS**")
    if st.button("🔵  KNN — Ho Jun Yon", key="sb_knn", use_container_width=True):
        st.switch_page("pages/2_KNN.py")
    if st.button("🌳  Decision Tree — Irvin", key="sb_dt", use_container_width=True):
        st.switch_page("pages/3_Decision_Tree.py")
    if st.button("🔴  SVM — Chiang Jun Hang", key="sb_svm", use_container_width=True):
        st.switch_page("pages/4_SVM.py")

    st.markdown("**COMPARE**")
    if st.button("📈  Compare All Models", key="sb_cmp", use_container_width=True):
        st.switch_page("pages/5_Comparison.py")

    st.divider()
    st.caption("Tutorial Group 3 · Tutor: Dr Goh\n202605 Session · TARUMT")

# ══════════════════════════════════════════════════════════════
# HERO SECTION
# ══════════════════════════════════════════════════════════════
st.markdown("---")

col_hero, col_space = st.columns([3, 1])
with col_hero:
    st.markdown("##### 🎓 SUPERVISED MACHINE LEARNING · BMCS2003 · TARUMT")
    st.title("Student Mental Health\nPrediction System")
    st.markdown(
        "An AI-powered system that analyses student demographics and academic data "
        "to predict **depression** and **panic attack** risk — "
        "enabling early detection and timely support."
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# STATS ROW
# ══════════════════════════════════════════════════════════════
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Student Records", "600", help="Total records in dataset")
s2.metric("Best Accuracy", "95.83%", help="KNN model accuracy")
s3.metric("Best Recall", "97.44%", help="KNN model recall")
s4.metric("ML Algorithms", "3", help="KNN, Decision Tree, SVM")
s5.metric("Features", "11", help="Original dataset features")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# NAVIGATION CARDS — using Streamlit columns + containers
# ══════════════════════════════════════════════════════════════
st.subheader("Navigate to a Section")
st.caption("Click any button below to explore the system")

st.markdown("<br>", unsafe_allow_html=True)

# Row 1
c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("### 📊 EDA")
        st.caption("Exploratory Data Analysis")
        st.write("Explore dataset visualizations, distributions, and correlations.")
        if st.button("Open EDA", key="nc_eda", use_container_width=True):
            st.switch_page("pages/1_EDA.py")

with c2:
    with st.container(border=True):
        st.markdown("### 🔵 KNN Model")
        st.caption("Member 1 — Ho Jun Yon")
        st.write("K-Nearest Neighbor · **95.83% accuracy** · K = 5 · Target: Depression")
        if st.button("Open KNN", key="nc_knn", use_container_width=True):
            st.switch_page("pages/2_KNN.py")

with c3:
    with st.container(border=True):
        st.markdown("### 🌳 Decision Tree")
        st.caption("Member 2 — Irvin Tan Wei Shen")
        st.write("CART Algorithm · **85.50% accuracy** · Depth 5 · Target: Depression")
        if st.button("Open Decision Tree", key="nc_dt", use_container_width=True):
            st.switch_page("pages/3_Decision_Tree.py")

st.markdown("<br>", unsafe_allow_html=True)

# Row 2
c4, c5, c6 = st.columns(3)

with c4:
    with st.container(border=True):
        st.markdown("### 🔴 SVM Model")
        st.caption("Member 3 — Chiang Jun Hang")
        st.write("Support Vector Machine · RBF Kernel · Target: Panic Attack")
        if st.button("Open SVM", key="nc_svm", use_container_width=True):
            st.switch_page("pages/4_SVM.py")

with c5:
    with st.container(border=True):
        st.markdown("### 📈 Compare Models")
        st.caption("All 3 Algorithms")
        st.write("Side-by-side performance comparison of KNN, Decision Tree, and SVM.")
        if st.button("Open Comparison", key="nc_cmp", use_container_width=True):
            st.switch_page("pages/5_Comparison.py")

with c6:
    with st.container(border=True):
        st.markdown("### 📋 Dataset")
        st.caption("Kaggle — IIUM Malaysia")
        st.write("600 student records · 11 features · Preprocessing steps and summary.")
        if st.button("Open Dataset", key="nc_ds", use_container_width=True):
            st.switch_page("pages/6_Dataset.py")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# ABOUT SECTION
# ══════════════════════════════════════════════════════════════
st.subheader("About This Project")

ab1, ab2, ab3 = st.columns(3)

with ab1:
    with st.container(border=True):
        st.markdown("**🎯 Problem Statement**")
        st.write(
            "Mental health issues among university students are increasing globally. "
            "Many students do not seek help early due to stigma and limited screening tools. "
            "This system uses AI to enable early detection."
        )

with ab2:
    with st.container(border=True):
        st.markdown("**📊 Our Approach**")
        st.write(
            "We implement and compare three supervised classification algorithms — "
            "KNN, Decision Tree, and SVM — on 600 IIUM student records. "
            "Each member independently builds and evaluates a different algorithm."
        )

with ab3:
    with st.container(border=True):
        st.markdown("**🚀 Impact**")
        st.write(
            "The system gives university counsellors an early-warning tool "
            "to flag at-risk students using academic and demographic features, "
            "enabling timely mental health support and intervention."
        )

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SYSTEM PIPELINE
# ══════════════════════════════════════════════════════════════
st.subheader("System Pipeline")
st.caption("Five stages from raw data to real-time prediction")

p1, p2, p3, p4, p5 = st.columns(5)

steps = [
    ("1️⃣", "Data Collection", "600 records from Kaggle IIUM student survey"),
    ("2️⃣", "Preprocessing", "Clean, encode, scale and engineer features"),
    ("3️⃣", "Model Training", "KNN (K=5), Decision Tree (depth 5), SVM (RBF)"),
    ("4️⃣", "Evaluation", "Accuracy, Precision, Recall, F1, Confusion Matrix, CV"),
    ("5️⃣", "Deployment", "Interactive Streamlit multi-page web application"),
]

for col, (icon, title, desc) in zip([p1, p2, p3, p4, p5], steps):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{title}**")
            st.caption(desc)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# TEAM SECTION
# ══════════════════════════════════════════════════════════════
st.subheader("Group Members")
st.caption("Tutorial Group 3 | Tutor: Dr Goh | BMCS2003 AI | 202605 Session")

st.markdown("<br>", unsafe_allow_html=True)

t1, t2, t3 = st.columns(3)

with t1:
    with st.container(border=True):
        st.markdown("#### 🔵 Member 1")
        st.markdown("**Ho Jun Yon**")
        st.caption("Student ID: 2612634")
        st.divider()
        st.write("**Algorithm:** K-Nearest Neighbor (KNN)")
        st.write("**Target:** Depression Prediction")
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", "95.83%")
        col_b.metric("Recall", "97.44%")
        st.write("**Best K:** 5 | **Scaling:** MinMax | **Split:** 80/20")
        if st.button("View KNN Page", key="tm_knn", use_container_width=True):
            st.switch_page("pages/2_KNN.py")

with t2:
    with st.container(border=True):
        st.markdown("#### 🌳 Member 2")
        st.markdown("**Irvin Tan Wei Shen**")
        st.caption("Student ID: 2612638")
        st.divider()
        st.write("**Algorithm:** Decision Tree (CART)")
        st.write("**Target:** Depression Prediction")
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", "85.50%")
        col_b.metric("Recall", "89.18%")
        st.write("**Max Depth:** 5 | **Criterion:** Gini | **Root:** Marital Status")
        if st.button("View DT Page", key="tm_dt", use_container_width=True):
            st.switch_page("pages/3_Decision_Tree.py")

with t3:
    with st.container(border=True):
        st.markdown("#### 🔴 Member 3")
        st.markdown("**Chiang Jun Hang**")
        st.caption("Student ID: 2612610")
        st.divider()
        st.write("**Algorithm:** Support Vector Machine (SVM)")
        st.write("**Target:** Panic Attack Prediction")
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", "TBD")
        col_b.metric("Recall", "TBD")
        st.write("**Kernel:** RBF | **Scaling:** Standard | **Split:** 75/25")
        if st.button("View SVM Page", key="tm_svm", use_container_width=True):
            st.switch_page("pages/4_SVM.py")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# FAQ
# ══════════════════════════════════════════════════════════════
st.subheader("Frequently Asked Questions")

with st.expander("What is KNN (K-Nearest Neighbor)?"):
    st.write(
        "KNN classifies a student's mental health status by finding the K most similar "
        "students in the training data and taking a majority vote. K=5 was selected as "
        "optimal through testing K=1 to K=20. Distance is measured using Euclidean "
        "distance after MinMax scaling to normalize all features to [0,1]."
    )

with st.expander("What is Decision Tree (CART)?"):
    st.write(
        "Decision Tree CART recursively splits data based on the feature with the highest "
        "Gini impurity reduction at each node. It builds an interpretable tree where each "
        "path from root to leaf represents a decision rule. Our tree has max depth 5, "
        "with Marital Status as the root split — the most discriminative feature."
    )

with st.expander("What is SVM (Support Vector Machine)?"):
    st.write(
        "SVM finds the optimal hyperplane that maximally separates two classes in "
        "high-dimensional space. The RBF (Radial Basis Function) kernel is used to "
        "handle non-linear data. Standard Scaling is applied before training. "
        "It targets Panic Attack prediction."
    )

with st.expander("Why does Recall matter most for mental health screening?"):
    st.write(
        "In mental health screening, missing a depressed student (false negative) is more "
        "serious than a false alarm (false positive). KNN achieved 97.44% recall — "
        "meaning it correctly identified 97.44% of all depressed students in the test set. "
        "A high recall minimizes the risk of failing to identify at-risk students."
    )

with st.expander("What is 5-Fold Cross Validation?"):
    st.write(
        "Instead of testing the model once, 5-Fold CV splits training data into 5 equal "
        "parts. It trains and tests 5 times — each time using a different part as the test "
        "set — then averages the accuracy. KNN achieved 86.67% CV mean with only 2.12% "
        "standard deviation, confirming the model is stable and not overfitting."
    )

with st.expander("Why three different algorithms for the same project?"):
    st.write(
        "Each member independently implements a different algorithm to enable a rigorous "
        "side-by-side comparison on the same dataset. By keeping dataset, preprocessing, "
        "and evaluation metrics consistent, any performance difference can be attributed "
        "purely to the algorithm choice — which is good experimental design in ML research."
    )

st.markdown("---")
st.caption("MindCheck · BMCS2003 Artificial Intelligence · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")
