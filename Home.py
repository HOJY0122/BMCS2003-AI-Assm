import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.sidebar import sidebar
from utils.models import load_all_models

st.set_page_config(
    page_title="MindCheck — Student Mental Health",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

sidebar("home")

M = load_all_models()

# Determine the best accuracy/recall live, across all 3 models, instead
# of hardcoding "KNN is best" — stays correct even if the dataset changes
# which model wins.
_accs = {'KNN': M['knn_m']['acc'], 'Decision Tree': M['dt_m']['acc'], 'SVM': M['svm_m']['acc']}
_recs = {'KNN': M['knn_m']['rec'], 'Decision Tree': M['dt_m']['rec'], 'SVM': M['svm_m']['rec']}
_best_acc_model = max(_accs, key=_accs.get)
_best_rec_model = max(_recs, key=_recs.get)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("##### 🎓  SUPERVISED MACHINE LEARNING · BMCS2003 · TARUMT")
st.title("Student Mental Health\nPrediction System")
st.write(
    "An AI-powered system that analyses student demographics and academic data "
    "to predict **depression** and **panic attack** risk — "
    "enabling early detection and timely support."
)
st.divider()

# ── STATS ─────────────────────────────────────────────────────
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Student Records", f"{M['n_records']}", help="Total records in dataset")
s2.metric("Best Accuracy",   f"{_accs[_best_acc_model]:.2f}%", help=f"{_best_acc_model} model on test set")
s3.metric("Best Recall",     f"{_recs[_best_rec_model]:.2f}%", help=f"{_best_rec_model} correctly found {_recs[_best_rec_model]:.2f}% of depressed students")
s4.metric("ML Algorithms",   "3",      help="KNN · Decision Tree · SVM")
s5.metric("Features",        "11",     help="Original dataset features")
st.divider()

# ── NAV CARDS ─────────────────────────────────────────────────
st.subheader("Navigate")
st.caption("Click any card to open that section")
st.write("")

c1, c2, c3 = st.columns(3)
with c1:
    with st.container(border=True):
        st.markdown("### EDA")
        st.caption("Exploratory Data Analysis")
        st.write("Visualize distributions, correlations and feature analysis.")
        if st.button("Open EDA →", key="nc_eda", width='stretch'):
            st.switch_page("pages/1_EDA.py")
with c2:
    with st.container(border=True):
        st.markdown("### KNN Model")
        st.caption("Member 1 — Ho Jun Yon")
        st.write(f"K-Nearest Neighbor · **{M['knn_m']['acc']:.2f}%** accuracy · K = {M['best_k']} · Depression")
        if st.button("Open KNN →", key="nc_knn", width='stretch'):
            st.switch_page("pages/2_KNN.py")
with c3:
    with st.container(border=True):
        st.markdown("### Decision Tree")
        st.caption("Member 2 — Irvin Tan Wei Shen")
        st.write(f"CART Algorithm · **{M['dt_m']['acc']:.2f}%** accuracy · Depth 5 · Depression")
        if st.button("Open Decision Tree →", key="nc_dt", width='stretch'):
            st.switch_page("pages/3_Decision_Tree.py")

st.write("")

c4, c5, c6 = st.columns(3)
with c4:
    with st.container(border=True):
        st.markdown("### SVM Model")
        st.caption("Member 3 — Chiang Jun Hang")
        st.write(f"Support Vector Machine · RBF Kernel · **{M['svm_m']['acc']:.2f}%** accuracy · Panic Attack")
        if st.button("Open SVM →", key="nc_svm", width='stretch'):
            st.switch_page("pages/4_SVM.py")
with c5:
    with st.container(border=True):
        st.markdown("### Compare Models")
        st.caption("All 3 Algorithms Side by Side")
        st.write("Performance comparison of KNN, Decision Tree and SVM.")
        if st.button("Open Comparison →", key="nc_cmp", width='stretch'):
            st.switch_page("pages/5_Comparison.py")
with c6:
    with st.container(border=True):
        st.markdown("### Dataset")
        st.caption("Kaggle — IIUM Malaysia")
        st.write(f"{M['n_records']} records · 11 features · Preprocessing steps and summary.")
        if st.button("Open Dataset →", key="nc_ds", width='stretch'):
            st.switch_page("pages/6_Dataset.py")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")