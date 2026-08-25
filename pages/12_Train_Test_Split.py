import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys, os, warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar
from utils.models import load_all_models

st.set_page_config(
    page_title="Train/Test Split — MindCheck",
    page_icon="✂️", layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("split")

M = load_all_models()

# ── Header ─────────────────────────────────────────────────────
st.markdown("##### ✂️ TRAIN / TEST SPLIT")
st.title("How We Split the Dataset for Training & Testing")
st.write(
    "Before training, the dataset is split into two parts: "
    "**Training set** (model learns from this) and "
    "**Test set** (model is evaluated on this — never seen during training). "
    "This prevents overfitting and gives a fair accuracy measure."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — VISUAL OVERVIEW ALL 3 MODELS
# ══════════════════════════════════════════════════════════════
st.subheader("1. Split Overview — All 3 Models")
st.caption("Live computed from the actual training process — not hardcoded")

total = M['n_records']

splits = {
    'KNN':           {'train': len(M['knn_Xtr']), 'test': len(M['knn_Xte']),
                      'ratio': '80 / 20', 'color_tr': '#3B82F6', 'color_te': '#93C5FD',
                      'member': 'Ho Jun Yon', 'icon': '🔵',
                      'train_dep': int(M['knn_ytr'].sum()), 'test_dep': int(M['knn_yte'].sum())},
    'Decision Tree': {'train': len(M['dt_Xtr']),  'test': len(M['dt_Xte']),
                      'ratio': '70 / 30', 'color_tr': '#10B981', 'color_te': '#6EE7B7',
                      'member': 'Irvin Tan', 'icon': '🌳',
                      'train_dep': int(M['dt_ytr'].sum()), 'test_dep': int(M['dt_yte'].sum())},
    'SVM':           {'train': len(M['svm_Xtr']), 'test': len(M['svm_Xte']),
                      'ratio': '75 / 25', 'color_tr': '#EF4444', 'color_te': '#FCA5A5',
                      'member': 'Chiang Jun Hang', 'icon': '🔴',
                      'train_dep': int(M['svm_ytr'].sum()), 'test_dep': int(M['svm_yte'].sum())},
}

# Summary metric row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Dataset",   f"{total} records")
m2.metric("KNN Split",       "480 train / 120 test")
m3.metric("DT Split",        "420 train / 180 test")
m4.metric("SVM Split",       "450 train / 150 test")

st.write("")

# Visual bar chart showing split proportions
fig_ov, ax_ov = plt.subplots(figsize=(12, 3.5))
models = list(splits.keys())
train_sizes = [splits[m]['train'] for m in models]
test_sizes  = [splits[m]['test']  for m in models]
x = np.arange(len(models))

b1 = ax_ov.barh(x, train_sizes, color=['#3B82F6','#10B981','#EF4444'],
                 alpha=0.85, label='Training Set', height=0.5)
b2 = ax_ov.barh(x, test_sizes, left=train_sizes,
                 color=['#93C5FD','#6EE7B7','#FCA5A5'],
                 alpha=0.85, label='Test Set', height=0.5)

for i, (tr, te) in enumerate(zip(train_sizes, test_sizes)):
    ax_ov.text(tr/2,       i, f'Train: {tr}', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
    ax_ov.text(tr + te/2,  i, f'Test: {te}',  ha='center', va='center',
               fontsize=11, fontweight='bold', color='#1E293B')

ax_ov.set_yticks(x)
ax_ov.set_yticklabels([f"{splits[m]['icon']} {m}" for m in models], fontsize=12)
ax_ov.set_xlabel('Number of Records', fontsize=11)
ax_ov.set_title(f'Train / Test Split — {total} Total Records',
                fontsize=13, fontweight='bold')
ax_ov.legend(fontsize=10, loc='lower right')
ax_ov.set_xlim(0, total + 30)
plt.tight_layout()
st.pyplot(fig_ov, use_container_width=True)
plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — PER MODEL DETAIL
# ══════════════════════════════════════════════════════════════
st.subheader("2. Detailed Split per Model")
st.caption("Click each tab to explore each model's train/test breakdown")

tab_knn, tab_dt, tab_svm = st.tabs(["🔵 KNN", "🌳 Decision Tree", "🔴 SVM"])

def draw_split_detail(tab, name, info, ytr, yte):
    with tab:
        total_tr = info['train']; total_te = info['test']
        dep_tr   = info['train_dep']; nodep_tr = total_tr - dep_tr
        dep_te   = info['test_dep'];  nodep_te = total_te - dep_te
        pct_tr   = total_tr / total * 100
        pct_te   = total_te / total * 100

        st.markdown(f"### {info['icon']} {name} — {info['ratio']} Split")
        st.caption(f"Member: {info['member']} · random_state=42 · stratify=y")

        # Metrics row
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("Total Records",   str(total))
        c2.metric("Training Set",    str(total_tr), delta=f"{pct_tr:.0f}% of total")
        c3.metric("Test Set",        str(total_te), delta=f"{pct_te:.0f}% of total")
        c4.metric("Train — Dep",     str(dep_tr),   delta=f"{dep_tr/total_tr*100:.1f}%")
        c5.metric("Train — No Dep",  str(nodep_tr), delta=f"{nodep_tr/total_tr*100:.1f}%")
        c6.metric("Test — Dep",      str(dep_te),   delta=f"{dep_te/total_te*100:.1f}%")

        st.write("")
        col1, col2, col3 = st.columns(3)

        # Chart 1 — Pie: overall split
        with col1:
            fig1, ax1 = plt.subplots(figsize=(4, 3.5))
            ax1.pie(
                [total_tr, total_te],
                labels=[f'Training\n{total_tr} records\n({pct_tr:.0f}%)',
                        f'Testing\n{total_te} records\n({pct_te:.0f}%)'],
                colors=[info['color_tr'], info['color_te']],
                autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2},
                textprops={'fontsize':10,'fontweight':'bold'},
            )
            ax1.set_title(f'{name}\nTrain / Test Split', fontweight='bold', fontsize=11)
            plt.tight_layout()
            st.pyplot(fig1, use_container_width=True); plt.close()

        # Chart 2 — Class distribution in train set
        with col2:
            fig2, ax2 = plt.subplots(figsize=(4, 3.5))
            bars = ax2.bar(['No Depression','Depression'],
                           [nodep_tr, dep_tr],
                           color=['#10B981','#EF4444'],
                           edgecolor='white', width=0.5, alpha=0.9)
            for bar, val in zip(bars, [nodep_tr, dep_tr]):
                ax2.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+2,
                         str(val), ha='center', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Number of Students')
            ax2.set_title(f'Training Set\n{total_tr} records — Class Distribution',
                          fontweight='bold', fontsize=11)
            ax2.set_ylim(0, max(nodep_tr, dep_tr)*1.2)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True); plt.close()

        # Chart 3 — Class distribution in test set
        with col3:
            fig3, ax3 = plt.subplots(figsize=(4, 3.5))
            bars3 = ax3.bar(['No Depression','Depression'],
                            [nodep_te, dep_te],
                            color=['#10B981','#EF4444'],
                            edgecolor='white', width=0.5, alpha=0.9)
            for bar, val in zip(bars3, [nodep_te, dep_te]):
                ax3.text(bar.get_x()+bar.get_width()/2,
                         bar.get_height()+0.5,
                         str(val), ha='center', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Number of Students')
            ax3.set_title(f'Test Set\n{total_te} records — Class Distribution',
                          fontweight='bold', fontsize=11)
            ax3.set_ylim(0, max(nodep_te, dep_te)*1.2)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True); plt.close()

        # Why stratify?
        with st.container(border=True):
            st.markdown("**Why `stratify=y`?**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"✅ **With stratify** — class ratio preserved:")
                st.write(f"  Train: {dep_tr/total_tr*100:.1f}% Depression")
                st.write(f"  Test:  {dep_te/total_te*100:.1f}% Depression")
            with col_b:
                overall_dep_pct = (int(M['knn_ytr'].sum()) + int(M['knn_yte'].sum())) / total * 100
                st.write(f"📊 **Overall dataset:** {overall_dep_pct:.1f}% Depression")
                st.write("Stratify ensures test set is representative —")
                st.write("prevents accidentally biased evaluation.")

        # Feature columns used
        st.write("")
        st.markdown("**Features Used for Training**")
        feat_labels = {
            'KNN':           ['Gender','Age','Course','Year','CGPA','Anxiety','Panic Attack'],
            'Decision Tree': ['Gender','Age','Course','Year','CGPA','Anxiety','Panic Attack','Marital Status'],
            'SVM':           ['Gender','Age','Year of Study','CGPA','Marital Status',
                              'Anxiety','Panic Attack','Seek Treatment','Course Category'],
        }
        feat_df = pd.DataFrame({
            'Feature': feat_labels[name],
            'Used in Training': ['✅'] * len(feat_labels[name]),
        })
        st.dataframe(feat_df.set_index('Feature'), use_container_width=True)
        st.caption(f"Target variable: **Depression** (0 = No, 1 = Yes) · "
                   f"{len(feat_labels[name])} features used")

draw_split_detail(tab_knn, 'KNN',           splits['KNN'],           M['knn_ytr'], M['knn_yte'])
draw_split_detail(tab_dt,  'Decision Tree', splits['Decision Tree'], M['dt_ytr'],  M['dt_yte'])
draw_split_detail(tab_svm, 'SVM',           splits['SVM'],           M['svm_ytr'], M['svm_yte'])

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — WHY DIFFERENT SPLITS?
# ══════════════════════════════════════════════════════════════
st.subheader("3. Why Different Split Ratios?")

r1, r2, r3 = st.columns(3)
with r1:
    with st.container(border=True):
        st.markdown("### 🔵 KNN — 80/20")
        st.write("**480 train | 120 test**")
        st.write("")
        st.write("KNN is a **distance-based** algorithm.")
        st.write("More training data → more neighbours to compare against → better accuracy.")
        st.write("80% gives the model the most reference points for finding similar students.")
        st.write("")
        st.success(f"Accuracy: {M['knn_m']['acc']:.2f}%")

with r2:
    with st.container(border=True):
        st.markdown("### 🌳 Decision Tree — 70/30")
        st.write("**420 train | 180 test**")
        st.write("")
        st.write("DT builds **explicit rules** from training data.")
        st.write("A larger test set (30%) gives a more reliable evaluation of whether the rules generalise.")
        st.write("Too small test set could give misleadingly high accuracy.")
        st.write("")
        st.success(f"Accuracy: {M['dt_m']['acc']:.2f}%")

with r3:
    with st.container(border=True):
        st.markdown("### 🔴 SVM — 75/25")
        st.write("**450 train | 150 test**")
        st.write("")
        st.write("SVM finds the **optimal hyperplane** separating classes.")
        st.write("75/25 is a balanced trade-off — enough training data for the RBF kernel, enough test data for reliable metrics.")
        st.write("")
        st.success(f"Accuracy: {M['svm_m']['acc']:.2f}%")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — HOW STRATIFY WORKS
# ══════════════════════════════════════════════════════════════
st.subheader("4. How Stratified Split Works")

fig_s, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, (name, info) in zip(axes, splits.items()):
    tr = info['train']; te = info['test']
    dep_tr = info['train_dep']; dep_te = info['test_dep']
    nodep_tr = tr - dep_tr; nodep_te = te - dep_te

    x = np.arange(2)
    ax.bar(x - 0.2, [nodep_tr, dep_tr], 0.35,
           label='Training', color=info['color_tr'], alpha=0.9, edgecolor='white')
    ax.bar(x + 0.2, [nodep_te, dep_te], 0.35,
           label='Test',     color=info['color_te'], alpha=0.9, edgecolor='white')

    # Show percentages
    for val, xp, side in [(nodep_tr, -0.2, 'tr'), (dep_tr, -0.2, 'tr'),
                           (nodep_te, +0.2, 'te'), (dep_te, +0.2, 'te')]:
        i = 0 if val in [nodep_tr, nodep_te] else 1
        ax.text(i + xp, val + 1, str(val),
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['No Depression', 'Depression'], fontsize=9)
    ax.set_ylabel('Students')
    ax.set_title(f'{info["icon"]} {name}\n{info["ratio"]} — stratify=y',
                 fontweight='bold', fontsize=10)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle('Class Distribution Preserved in Both Train & Test Sets',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
st.pyplot(fig_s, use_container_width=True)
plt.close()

st.info(
    "**Key Insight:** `stratify=y` ensures the **same proportion** of depressed/non-depressed "
    "students appears in both training and test sets. Without stratify, the test set might "
    "accidentally have too many or too few depressed students, giving misleading accuracy."
)

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")