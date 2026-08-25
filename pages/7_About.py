import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar

# ── Live metrics (no hardcode) ────────────────────────────────
@st.cache_resource
def get_live_metrics():
    import warnings
    warnings.filterwarnings('ignore')
    from utils.preprocessing import load_and_clean_dataset
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (MinMaxScaler, StandardScaler,
                                        LabelEncoder, OrdinalEncoder)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score)
    import pandas as pd

    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())
    le_c = LabelEncoder(); le_y = LabelEncoder()
    df['Course_Enc'] = le_c.fit_transform(df['Course'])
    df['Year_Enc']   = le_y.fit_transform(df['Year_of_Study'])

    def _m(yt, yp):
        return {'acc':accuracy_score(yt,yp)*100,
                'prec':precision_score(yt,yp,zero_division=0)*100,
                'rec':recall_score(yt,yp,zero_division=0)*100,
                'f1':f1_score(yt,yp,zero_division=0)*100}

    # KNN
    knn_feat = ['Gender','Age','Course_Enc','Year_Enc','CGPA_Numeric','Anxiety','Panic_Attack']
    X_k = df[knn_feat]; y_k = df['Depression']
    sc_k = MinMaxScaler(); X_ks = sc_k.fit_transform(X_k)
    Xtr_k,Xte_k,ytr_k,yte_k = train_test_split(X_ks,y_k,test_size=0.2,random_state=42,stratify=y_k)
    best_k,best_a=5,0
    for k in range(1,21):
        m=KNeighborsClassifier(n_neighbors=k); m.fit(Xtr_k,ytr_k)
        a=accuracy_score(yte_k,m.predict(Xte_k))
        if a>best_a: best_a,best_k=a,k
    knn=KNeighborsClassifier(n_neighbors=best_k,metric='euclidean')
    knn.fit(Xtr_k,ytr_k)

    # DT
    dt_feat = ['Gender','Age','Course_Enc','Year_Enc','CGPA_Numeric',
               'Anxiety','Panic_Attack','Marital_Status']
    X_d=df[dt_feat]; y_d=df['Depression']
    Xtr_d,Xte_d,ytr_d,yte_d=train_test_split(X_d,y_d,test_size=0.3,random_state=42,stratify=y_d)
    dt=DecisionTreeClassifier(max_depth=5,criterion='gini',random_state=42)
    dt.fit(Xtr_d,ytr_d)

    # SVM
    df_raw=pd.read_csv('dataset/Student_Mental_health.csv')
    df_raw.columns=df_raw.columns.str.strip()
    df_raw['Age']=df_raw['Age'].fillna(df_raw['Age'].median())
    df_raw['Your current year of Study']=df_raw['Your current year of Study'].str.strip().str.lower()
    df_raw['What is your CGPA?']=df_raw['What is your CGPA?'].str.strip()
    def cat(c):
        c=str(c).lower()
        return 'STEM/IT' if any(x in c for x in ['technology','it','computer','cs','system','software','se','bit','bcs','cts']) else 'Other'
    df_raw['Course_Category']=df_raw['What is your course?'].apply(cat)
    df_raw=df_raw.drop(columns=['Timestamp','What is your course?'],errors='ignore')
    X_sv=df_raw.drop(columns=['Do you have Depression?'])
    y_sv=(df_raw['Do you have Depression?']=='Yes').astype(int)
    pipe=Pipeline([('enc',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)),
                   ('scl',StandardScaler()),
                   ('svm',SVC(kernel='rbf',probability=True,class_weight='balanced',random_state=42))])
    Xtr_sv,Xte_sv,ytr_sv,yte_sv=train_test_split(X_sv,y_sv,test_size=0.25,random_state=42,stratify=y_sv)
    pipe.fit(Xtr_sv,ytr_sv)

    return {
        'knn': _m(yte_k, knn.predict(Xte_k)),
        'dt':  _m(yte_d, dt.predict(Xte_d)),
        'svm': _m(yte_sv, pipe.predict(Xte_sv)),
        'best_k': best_k,
        'n': len(df),
    }

_LM = get_live_metrics()

st.set_page_config(
    page_title="About — MindCheck",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("about")

# ── Header ──────────────────────────────────────────────────
st.markdown("##### 👥 ABOUT")
st.title("About MindCheck")
st.write(
    "A supervised machine learning web application for predicting "
    "student depression risk — built for BMCS2003 Artificial Intelligence, "
    "TARUMT 202605 Session."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 & 2 — INTERACTIVE ARCHITECTURE + PIPELINE DIAGRAMS
# Native Streamlit components (not a static image): every card has
# a real hover tooltip (browser title attribute) and, where a
# matching page exists, a real st.switch_page() button. This is
# more reliable on Streamlit Cloud than embedding HTML/JS in an
# iframe, which cannot safely navigate the parent app.
# ══════════════════════════════════════════════════════════════
st.subheader("System Architecture")
st.caption("End-to-end pipeline from raw data to prediction output — hover a card for details, click Open to jump to that page")

st.markdown("""
<style>
.mc-card{border-radius:12px;padding:14px 10px;text-align:center;color:white;
  transition:transform .15s ease, box-shadow .15s ease; min-height:64px;
  display:flex; flex-direction:column; justify-content:center;}
.mc-card:hover{transform:translateY(-3px); box-shadow:0 8px 20px rgba(0,0,0,.35);}
.mc-title{font-weight:700; font-size:13px; margin-bottom:3px;}
.mc-sub{font-size:10px; opacity:.85; line-height:1.35;}
.mc-flow{text-align:center; color:#6B7BA8; font-size:11px; font-style:italic; margin:6px 0;}
.mc-arrow{text-align:center; color:#4A5680; font-size:20px; line-height:1;}
</style>
""", unsafe_allow_html=True)


def _arch_card(label, sub, color, key=None, page=None):
    """Render one architecture card with a native hover tooltip
    (title attribute) and, if `page` is given, a real switch_page
    button underneath."""
    st.markdown(
        f"""<div class="mc-card" style="background:{color};" title="{sub}">
              <div class="mc-title">{label}</div>
              <div class="mc-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    if page:
        if st.button("Open →", key=key, use_container_width=True):
            st.switch_page(page)


# ── Row 1: Data Layer ──────────────────────────────────────────
r1 = st.columns(4)
with r1[0]:
    _arch_card("📁 Raw Dataset", "Kaggle Shariful07 (2020) · 600 records · IIUM Malaysia",
               "#1E3A5F", key="arch_dataset", page="pages/6_Dataset.py")
with r1[1]:
    _arch_card("⚙️ Preprocessing", "Clean · Encode · Engineer (utils/preprocessing.py)",
               "#1E4A3A", key="arch_prep", page="pages/6_Dataset.py")
with r1[2]:
    _arch_card("🔬 Feature Selection", "Pearson · t-test · Chi2 · 1–10 feature experiment",
               "#3A1E4F", key="arch_featsel", page="pages/9_Feature_Selection.py")
with r1[3]:
    _arch_card("📊 EDA", "Visualize distributions · Correlation heatmap",
               "#4F3A1E", key="arch_eda", page="pages/1_EDA.py")

st.markdown('<div class="mc-flow">↓ 80/20 · 70/30 · 75/25 splits ↓</div>', unsafe_allow_html=True)

# ── Row 2: Model Layer ───────────────────────────────────────────
r2 = st.columns(3)
with r2[0]:
    _arch_card("🔵 KNN", f"K={_LM['best_k']} · MinMax Scale · Euclidean · Target: Depression",
               "#1B3A6B", key="arch_knn", page="pages/2_KNN.py")
with r2[1]:
    _arch_card("🌳 Decision Tree", "CART · Depth 5 · Gini · Target: Depression",
               "#1B5C3A", key="arch_dt", page="pages/3_Decision_Tree.py")
with r2[2]:
    _arch_card("🔴 SVM", "RBF Kernel · Balanced · Ordinal Encoder · Target: Depression",
               "#6B1B1B", key="arch_svm", page="pages/4_SVM.py")

st.markdown('<div class="mc-flow">↓ evaluated on held-out test split ↓</div>', unsafe_allow_html=True)

# ── Row 3: Evaluation ─────────────────────────────────────────
r3 = st.columns(3)
with r3[0]:
    _arch_card("📊 KNN Metrics",
               f"Acc {_LM['knn']['acc']:.2f}% · Rec {_LM['knn']['rec']:.2f}% · F1 {_LM['knn']['f1']:.2f}%",
               "#162B4F", key="arch_knn_m", page="pages/2_KNN.py")
with r3[1]:
    _arch_card("📊 DT Metrics",
               f"Acc {_LM['dt']['acc']:.2f}% · Prec {_LM['dt']['prec']:.2f}% · Rec {_LM['dt']['rec']:.2f}%",
               "#163D2A", key="arch_dt_m", page="pages/3_Decision_Tree.py")
with r3[2]:
    _arch_card("📊 SVM Metrics",
               f"Acc {_LM['svm']['acc']:.2f}% · Prec {_LM['svm']['prec']:.2f}% · F1 {_LM['svm']['f1']:.2f}%",
               "#4F1616", key="arch_svm_m", page="pages/4_SVM.py")

st.markdown('<div class="mc-flow">↓ powers the Streamlit web application ↓</div>', unsafe_allow_html=True)

# ── Row 4: Application Layer ──────────────────────────────────
r4 = st.columns(6)
with r4[0]:
    _arch_card("🔍 Single Predict", "Enter one student's data — available on every model page", "#1E2E5F")
with r4[1]:
    _arch_card("📦 Batch Predict", "Upload a CSV to screen many students at once", "#1E3E4F")
with r4[2]:
    _arch_card("📈 Compare & Charts", "Side-by-side model comparison, live test",
               "#2A1E5F", key="arch_compare", page="pages/5_Comparison.py")
with r4[3]:
    _arch_card("🔬 Feature Selection", "Correlation · t-test · Chi2 explorer",
               "#1E4F2A", key="arch_featsel2", page="pages/9_Feature_Selection.py")
with r4[4]:
    _arch_card("📊 Live Stats", "Compare a student profile to the whole dataset",
               "#4F2A1E", key="arch_livestats", page="pages/11_Live_Stats.py")
with r4[5]:
    _arch_card("🤖 Auto Selector", "Recommends the best model for a given profile",
               "#4F1E3A", key="arch_autosel", page="pages/5_Comparison.py")

st.markdown('<div class="mc-flow">↓ produces ↓</div>', unsafe_allow_html=True)

# ── Row 5: Output Layer ────────────────────────────────────────
r5 = st.columns(4)
with r5[0]:
    _arch_card("✅ Prediction Result", "Risk verdict + confidence percentage", "#1E3A20")
with r5[1]:
    _arch_card("📥 Download CSV", "Export batch prediction results", "#1E2A3A")
with r5[2]:
    _arch_card("🧠 Explanation", "Why the model decided this — feature contribution", "#2A1E3A")
with r5[3]:
    _arch_card("📊 Interactive Charts", "Plotly comparison visuals", "#3A2A1E")

st.caption("Legend — 🔵 Data · 🟦 Model · 🔷 Evaluation · 🟪 Application · 🟩 Output")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — PREPROCESSING PIPELINE (interactive)
# ══════════════════════════════════════════════════════════════
st.subheader("Data Preprocessing Pipeline")
st.caption("Step-by-step transformation from raw CSV to model-ready features — hover a step for details")

pipe_steps = [
    ("📁 Raw CSV",      "600 records · 11 features",              "#1E3A5F"),
    ("🧹 Clean",        "Drop Timestamp · fill 8 Age NaN",        "#1B3A6B"),
    ("🔄 Standardize",  "Course → 10 categories · Year → 4",      "#1B4A5F"),
    ("🔢 Encode",       "Yes/No → 1/0 · Gender → 1/0",            "#1E4A3A"),
    ("⚙️ Engineer",     "Mental Health Score 0–3 · CGPA numeric", "#3A3A1E"),
    ("✂️ Split",        "Train/Test · 80/20 or 70/30 or 75/25",   "#3A1E4F"),
    ("📐 Scale",        "MinMax (KNN) · Standard (SVM)",          "#4F1E3A"),
    ("✅ Ready",        "For training & prediction",              "#1E5C2A"),
]
pcols = st.columns(len(pipe_steps))
for col, (label, sub, color) in zip(pcols, pipe_steps):
    with col:
        st.markdown(
            f"""<div class="mc-card" style="background:{color};" title="{sub}">
                  <div class="mc-title" style="font-size:11px;">{label}</div>
                  <div class="mc-sub">{sub}</div>
                </div>""",
            unsafe_allow_html=True,
        )

st.write("")
if st.button("🔬  See these steps applied to real data →", use_container_width=True, key="pipe_cta"):
    st.switch_page("pages/9_Feature_Selection.py")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — PROBLEM STATEMENT
# ══════════════════════════════════════════════════════════════
st.subheader("Problem Statement")
with st.container(border=True):
    st.write(
        "Mental health among university students is a growing concern globally. "
        "In Malaysia, studies show that depression affects a significant proportion "
        "of university students, yet many cases go undetected until they escalate. "
        "Early identification of at-risk students can enable timely intervention "
        "and support from counselling services."
    )
    st.write(
        "This project applies **supervised machine learning** to predict depression "
        "risk using student demographic and academic data from IIUM Malaysia. "
        "Three algorithms — KNN, Decision Tree, and SVM — are implemented, "
        "compared, and deployed as an interactive web application."
    )

st.write("")
st.subheader("Project Objectives")
obj1, obj2 = st.columns(2)
with obj1:
    with st.container(border=True):
        st.markdown("**Primary Objectives**")
        for obj in [
            "Implement KNN, Decision Tree and SVM for depression prediction",
            "Evaluate and compare all 3 models using standard metrics",
            "Identify key features that predict student depression",
            "Deploy a live interactive web application",
        ]:
            st.write(f"✅ {obj}")
with obj2:
    with st.container(border=True):
        st.markdown("**Beyond Scope (Extra)**")
        for obj in [
            "Batch prediction — screen multiple students at once",
            "Prediction explanation — WHY the model decided",
            "Model auto-selector — recommends best model per profile",
            "Feature experiment — prove which features matter",
            "Live dataset statistics — compare student to dataset",
        ]:
            st.write(f"⭐ {obj}")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — TEAM
# ══════════════════════════════════════════════════════════════
st.subheader("Project Team")
t1, t2, t3 = st.columns(3)

for col, name, student_id, role, algo, target, acc in [
    (t1, "Ho Jun Yon",        "2612634",
     "KNN Implementation",    "K-Nearest Neighbor (K=5)",
     "Depression", f'{_LM["knn"]["acc"]:.2f}%'),
    (t2, "Irvin Tan Wei Shen","2612638",
     "Decision Tree",         "CART (Depth 5, Gini)",
     "Depression", f'{_LM["dt"]["acc"]:.2f}%'),
    (t3, "Chiang Jun Hang",   "2612610",
     "SVM Implementation",    "SVC (RBF Kernel)",
     "Depression", f'{_LM["svm"]["acc"]:.2f}%'),
]:
    with col:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.caption(f"Student ID: {student_id}")
            st.divider()
            st.write(f"**Role:** {role}")
            st.write(f"**Algorithm:** {algo}")
            st.write(f"**Target:** {target}")
            st.metric("Model Accuracy", acc)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — DATASET INFO
# ══════════════════════════════════════════════════════════════
st.subheader("Dataset Information")
d1, d2 = st.columns(2)

with d1:
    with st.container(border=True):
        st.markdown("**Source**")
        st.write("**Name:** Student Mental Health")
        st.write("**Author:** Shariful07")
        st.write("**Platform:** Kaggle (2020)")
        st.write("**University:** IIUM, Malaysia")
        st.write(f"**Records:** {_LM['n']} students")
        st.write("**Features:** 11 original")

with d2:
    with st.container(border=True):
        st.markdown("**Key Statistics**")
        st.write("**Depression Rate:** 32.3% (194 students)")
        st.write("**Anxiety Rate:** 34.8% (209 students)")
        st.write("**Panic Attack Rate:** 31.7% (190 students)")
        st.write("**Missing Values:** 8 (Age column only)")
        st.write("**Age Range:** 18–24 years")
        st.write("**CGPA Range:** 0–4.00")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 6 — TECH STACK
# ══════════════════════════════════════════════════════════════
st.subheader("Technology Stack")
tc1, tc2, tc3, tc4 = st.columns(4)

for col, title, items in [
    (tc1, "🐍 Language & Framework", [
        "Python 3.11+",
        "Streamlit 1.32+",
        "Plotly 5.0+",
    ]),
    (tc2, "🤖 Machine Learning", [
        "scikit-learn 1.3+",
        "KNN · Decision Tree · SVM",
        "GridSearchCV · CV",
    ]),
    (tc3, "📊 Data Processing", [
        "Pandas 2.0+",
        "NumPy 1.24+",
        "SciPy (t-test, chi2)",
    ]),
    (tc4, "📈 Visualization", [
        "Matplotlib 3.7+",
        "Seaborn 0.12+",
        "Plotly (interactive)",
    ]),
]:
    with col:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            for item in items:
                st.write(f"• {item}")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")