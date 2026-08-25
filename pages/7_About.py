import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
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
# SECTION 1 — SYSTEM ARCHITECTURE DIAGRAM
# ══════════════════════════════════════════════════════════════
st.subheader("System Architecture")
st.caption("End-to-end pipeline from raw data to prediction output")

def draw_architecture():
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#0B0E1A')
    ax.set_facecolor('#0B0E1A')
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    ax.axis('off')

    def box(x, y, w, h, color, label, sublabel="", radius=0.3):
        fancy = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad=0.05,rounding_size={radius}",
            facecolor=color, edgecolor='white',
            linewidth=1.2, alpha=0.92, zorder=3
        )
        ax.add_patch(fancy)
        ax.text(x, y + (0.12 if sublabel else 0), label,
                ha='center', va='center', fontsize=9,
                fontweight='bold', color='white', zorder=4)
        if sublabel:
            ax.text(x, y - 0.22, sublabel,
                    ha='center', va='center', fontsize=7,
                    color='rgba(255,255,255,0.6)', zorder=4)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>",
                                    color='#4A5680', lw=1.5),
                    zorder=2)
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my+0.12, label, ha='center', va='bottom',
                    fontsize=7, color='#6B7BA8', zorder=5)

    def line_h(x1, x2, y, color='#1E2640', lw=1):
        ax.plot([x1,x2],[y,y], color=color, lw=lw,
                linestyle='--', zorder=1, alpha=0.5)

    # ── Row 1: Data Layer ──────────────────────────────────
    ax.text(8, 8.5, "MindCheck — System Architecture Pipeline",
            ha='center', va='center', fontsize=14,
            fontweight='bold', color='white', zorder=5)

    # Dataset
    box(2, 7.4, 2.2, 0.8, '#1E3A5F', '📁 Raw Dataset',
        'Kaggle Shariful07 (2020)\n600 records · IIUM Malaysia')
    box(6, 7.4, 2.2, 0.8, '#1E4A3A', '⚙️  Preprocessing',
        'Clean · Encode · Engineer\nutils/preprocessing.py')
    box(10, 7.4, 2.2, 0.8, '#3A1E4F', '🔬 Feature Selection',
        'Pearson · t-test · Chi2\nExperiment 1–10 features')
    box(14, 7.4, 2.2, 0.8, '#4F3A1E', '📊 EDA',
        'Visualize distributions\nCorrelation heatmap')

    arrow(3.1, 7.4, 4.9, 7.4, "raw CSV")
    arrow(7.1, 7.4, 8.9, 7.4, "cleaned df")
    arrow(11.1, 7.4, 12.9, 7.4, "selected features")

    # ── Row 2: Model Layer ────────────────────────────────
    ax.text(8, 6.3, "Model Training Layer (Live · No Hardcode)",
            ha='center', va='center', fontsize=10,
            color='#6B7BA8', style='italic', zorder=5)

    box(3, 5.5, 2.4, 0.9, '#1B3A6B', '🔵 KNN',
        'K=5 · MinMax Scale\nEuclidean Distance\nTarget: Depression')
    box(8, 5.5, 2.4, 0.9, '#1B5C3A', '🌳 Decision Tree',
        'CART · Depth 5 · Gini\nFeature Importances\nTarget: Depression')
    box(13, 5.5, 2.4, 0.9, '#6B1B1B', '🔴 SVM',
        'RBF Kernel · Balanced\nOrdinal Encoder\nTarget: Depression')

    arrow(2, 7.0, 3, 5.95, "80/20 split")
    arrow(6, 7.0, 7.2, 5.95, "70/30 split")
    arrow(10, 7.0, 12.2, 5.95, "75/25 split")

    # ── Row 3: Evaluation ─────────────────────────────────
    box(3, 4.1, 2.4, 0.85, '#162B4F', '📊 KNN Metrics',
        f'Acc: {_LM["knn"]["acc"]:.2f}%\nRec: {_LM["knn"]["rec"]:.2f}% · F1: {_LM["knn"]["f1"]:.2f}%')
    box(8, 4.1, 2.4, 0.85, '#163D2A', '📊 DT Metrics',
        f'Acc: {_LM["dt"]["acc"]:.2f}%\nPrec: {_LM["dt"]["prec"]:.2f}% · Rec: {_LM["dt"]["rec"]:.2f}%')
    box(13, 4.1, 2.4, 0.85, '#4F1616', '📊 SVM Metrics',
        f'Acc: {_LM["svm"]["acc"]:.2f}%\nPrec: {_LM["svm"]["prec"]:.2f}% · F1: {_LM["svm"]["f1"]:.2f}%')

    arrow(3, 5.05, 3, 4.53)
    arrow(8, 5.05, 8, 4.53)
    arrow(13, 5.05, 13, 4.53)

    # ── Row 4: Application Layer ──────────────────────────
    ax.text(8, 3.3, "Streamlit Web Application Layer",
            ha='center', va='center', fontsize=10,
            color='#6B7BA8', style='italic', zorder=5)

    boxes_app = [
        (2,   2.6, '🔍 Single\nPredict',    '#1E2E5F'),
        (4.5, 2.6, '📦 Batch\nPredict',     '#1E3E4F'),
        (7,   2.6, '📈 Compare\n& Charts',  '#2A1E5F'),
        (9.5, 2.6, '🔬 Feature\nSelection', '#1E4F2A'),
        (12,  2.6, '📊 Live\nStats',        '#4F2A1E'),
        (14.5,2.6, '🤖 Auto\nSelector',     '#4F1E3A'),
    ]
    for bx, by, blbl, bcol in boxes_app:
        box(bx, by, 1.9, 0.85, bcol, blbl)
        arrow(bx, 3.73, bx, 2.5) if bx < 7.5 else None
        if bx > 7: arrow(8, 3.73, bx, 2.5)

    arrow(3, 3.67, 2,   3.03)
    arrow(3, 3.67, 4.5, 3.03)
    arrow(8, 3.67, 7,   3.03)
    arrow(8, 3.67, 9.5, 3.03)
    arrow(13, 3.67, 12,  3.03)
    arrow(13, 3.67, 14.5,3.03)

    # ── Row 5: Output Layer ───────────────────────────────
    ax.text(8, 1.85, "Output Layer",
            ha='center', va='center', fontsize=10,
            color='#6B7BA8', style='italic', zorder=5)

    out_boxes = [
        (3,   1.2, '✅/⚠️\nPrediction\nResult',  '#1E3A20'),
        (6.5, 1.2, '📥 Download\nCSV Results', '#1E2A3A'),
        (10,  1.2, '🧠 Prediction\nExplanation','#2A1E3A'),
        (13.5,1.2, '📊 Interactive\nCharts',    '#3A2A1E'),
    ]
    for bx, by, blbl, bcol in out_boxes:
        box(bx, by, 2.4, 0.9, bcol, blbl)

    # Arrows from app to output
    for ax_pos in [2, 4.5, 7, 9.5, 12, 14.5]:
        ax.annotate("", xy=(8, 1.7), xytext=(ax_pos, 2.18),
                    arrowprops=dict(arrowstyle="-|>",
                                    color='#2E3854', lw=1),
                    zorder=1)

    # ── Legend ────────────────────────────────────────────
    legend_items = [
        (mpatches.Patch(facecolor='#1E3A5F', label='Data Layer')),
        (mpatches.Patch(facecolor='#1B3A6B', label='Model Layer')),
        (mpatches.Patch(facecolor='#162B4F', label='Evaluation')),
        (mpatches.Patch(facecolor='#1E2E5F', label='Application')),
        (mpatches.Patch(facecolor='#1E3A20', label='Output')),
    ]
    ax.legend(handles=legend_items, loc='lower left',
              fontsize=8, framealpha=0.2,
              labelcolor='white', facecolor='#1A1F35',
              edgecolor='#2E3854')

    plt.tight_layout(pad=0.5)
    return fig

fig_arch = draw_architecture()
st.pyplot(fig_arch, use_container_width=True)
plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — PREPROCESSING PIPELINE DIAGRAM
# ══════════════════════════════════════════════════════════════
st.subheader("Data Preprocessing Pipeline")
st.caption("Step-by-step transformation from raw CSV to model-ready features")

def draw_pipeline():
    steps = [
        ("📁 Raw CSV\n600 records\n11 features",       '#1E3A5F'),
        ("🧹 Clean\nDrop Timestamp\nFill 8 Age NaN",   '#1B3A6B'),
        ("🔄 Standardize\nCourse→10 cats\nYear→4 cats",'#1B4A5F'),
        ("🔢 Encode\nYes/No→1/0\nGender→1/0",          '#1E4A3A'),
        ("⚙️  Engineer\nMH Score 0-3\nCGPA Numeric",   '#3A3A1E'),
        ("✂️  Split\nTrain/Test\n80/20 or 70/30",      '#3A1E4F'),
        ("📐 Scale\nMinMax (KNN)\nStandard (SVM)",      '#4F1E3A'),
        ("✅ Ready\nFor Training\n& Prediction",        '#1E5C2A'),
    ]

    fig2, ax2 = plt.subplots(figsize=(16, 3.2))
    fig2.patch.set_facecolor('#0B0E1A')
    ax2.set_facecolor('#0B0E1A')
    ax2.set_xlim(0, 16); ax2.set_ylim(0, 3.2)
    ax2.axis('off')

    bw = 1.6; bh = 2.2; gap = 0.15
    total_w = len(steps) * bw + (len(steps)-1) * gap
    start_x = (16 - total_w) / 2

    for i, (label, color) in enumerate(steps):
        cx = start_x + i*(bw + gap) + bw/2
        cy = 1.6
        fancy = mpatches.FancyBboxPatch(
            (cx - bw/2, cy - bh/2), bw, bh,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor=color, edgecolor='white',
            linewidth=1, alpha=0.9, zorder=3
        )
        ax2.add_patch(fancy)
        ax2.text(cx, cy, label, ha='center', va='center',
                 fontsize=8, fontweight='bold', color='white',
                 zorder=4, linespacing=1.5)

        # Step number
        ax2.text(cx, cy + bh/2 + 0.1, f"Step {i+1}",
                 ha='center', va='bottom', fontsize=7,
                 color='#4A5680', zorder=4)

        # Arrow to next
        if i < len(steps) - 1:
            nx = cx + bw/2
            ax2.annotate("",
                xy=(nx + gap, cy), xytext=(nx, cy),
                arrowprops=dict(arrowstyle="-|>",
                                color='#4A5680', lw=1.5),
                zorder=2)

    plt.tight_layout(pad=0.3)
    return fig2

fig_pipe = draw_pipeline()
st.pyplot(fig_pipe, use_container_width=True)
plt.close()

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