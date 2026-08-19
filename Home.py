import streamlit as st

st.set_page_config(
    page_title="MindCheck — Student Mental Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
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
    margin-bottom: 4px !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(61,82,255,0.25) !important;
    border-color: #3D52FF !important; color: white !important;
}
div[data-testid="metric-container"] {
    background: #F8F9FF; border: 1px solid #E2E8FF;
    border-radius: 12px; padding: 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 MindCheck")
    st.caption("BMCS2003 Artificial Intelligence")
    st.divider()

    st.markdown("**MAIN**")
    if st.button("🏠  Home", key="sb_home", use_container_width=True):
        st.switch_page("Home.py")

    st.divider()
    st.markdown("**ANALYSIS**")
    if st.button("📊  EDA", key="sb_eda", use_container_width=True):
        st.switch_page("pages/1_EDA.py")
    if st.button("📋  Dataset", key="sb_ds", use_container_width=True):
        st.switch_page("pages/6_Dataset.py")

    st.divider()
    st.markdown("**MODELS**")
    if st.button("🔵  KNN — Ho Jun Yon", key="sb_knn", use_container_width=True):
        st.switch_page("pages/2_KNN.py")
    if st.button("🌳  Decision Tree — Irvin", key="sb_dt", use_container_width=True):
        st.switch_page("pages/3_Decision_Tree.py")
    if st.button("🔴  SVM — Chiang Jun Hang", key="sb_svm", use_container_width=True):
        st.switch_page("pages/4_SVM.py")

    st.divider()
    st.markdown("**RESULTS**")
    if st.button("📈  Compare All Models", key="sb_cmp", use_container_width=True):
        st.switch_page("pages/5_Comparison.py")

    st.divider()
    st.markdown("**INFO**")
    if st.button("ℹ️  About", key="sb_about", use_container_width=True):
        st.switch_page("pages/7_About.py")
    if st.button("❓  FAQ", key="sb_faq", use_container_width=True):
        st.switch_page("pages/8_FAQ.py")

    st.divider()
    st.caption("Tutorial Group 3 · Tutor: Dr Goh\n202605 Session · TARUMT")

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("##### 🎓 SUPERVISED MACHINE LEARNING · BMCS2003 · TARUMT")
st.title("Student Mental Health Prediction System")
st.write(
    "An AI-powered system that analyses student demographics and academic data "
    "to predict **depression** and **panic attack** risk — "
    "enabling early detection and timely support."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Student Records",  "600",    help="Total records in dataset")
s2.metric("Best Accuracy",    "95.83%", help="KNN model accuracy")
s3.metric("Best Recall",      "97.44%", help="KNN model recall")
s4.metric("ML Algorithms",    "3",      help="KNN, Decision Tree, SVM")
s5.metric("Features",         "11",     help="Original dataset features")

st.divider()

# ══════════════════════════════════════════════════════════════
# NAVIGATION CARDS
# ══════════════════════════════════════════════════════════════
st.subheader("Navigate")
st.caption("Select a section to explore")
st.write("")

r1c1, r1c2, r1c3 = st.columns(3)

with r1c1:
    with st.container(border=True):
        st.markdown("### 📊 EDA")
        st.caption("Exploratory Data Analysis")
        st.write("Dataset distributions, correlations and visualizations.")
        if st.button("Open EDA →", key="nc_eda", use_container_width=True):
            st.switch_page("pages/1_EDA.py")

with r1c2:
    with st.container(border=True):
        st.markdown("### 🔵 KNN Model")
        st.caption("Member 1 — Ho Jun Yon")
        st.write("K-Nearest Neighbor · **95.83%** accuracy · K = 5")
        if st.button("Open KNN →", key="nc_knn", use_container_width=True):
            st.switch_page("pages/2_KNN.py")

with r1c3:
    with st.container(border=True):
        st.markdown("### 🌳 Decision Tree")
        st.caption("Member 2 — Irvin Tan Wei Shen")
        st.write("CART Algorithm · **85.50%** accuracy · Depth 5")
        if st.button("Open Decision Tree →", key="nc_dt", use_container_width=True):
            st.switch_page("pages/3_Decision_Tree.py")

st.write("")

r2c1, r2c2, r2c3 = st.columns(3)

with r2c1:
    with st.container(border=True):
        st.markdown("### 🔴 SVM Model")
        st.caption("Member 3 — Chiang Jun Hang")
        st.write("Support Vector Machine · RBF Kernel · Panic Attack")
        if st.button("Open SVM →", key="nc_svm", use_container_width=True):
            st.switch_page("pages/4_SVM.py")

with r2c2:
    with st.container(border=True):
        st.markdown("### 📈 Compare Models")
        st.caption("All 3 Algorithms")
        st.write("Side-by-side performance comparison of all models.")
        if st.button("Open Comparison →", key="nc_cmp", use_container_width=True):
            st.switch_page("pages/5_Comparison.py")

with r2c3:
    with st.container(border=True):
        st.markdown("### 📋 Dataset")
        st.caption("Kaggle — IIUM Malaysia")
        st.write("600 student records · 11 features · Preprocessing details.")
        if st.button("Open Dataset →", key="nc_ds", use_container_width=True):
            st.switch_page("pages/6_Dataset.py")

st.divider()
st.caption("MindCheck · BMCS2003 Artificial Intelligence · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")
