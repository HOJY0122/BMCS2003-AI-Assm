import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar

st.set_page_config(
    page_title="Compare Models — MindCheck",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

sidebar("compare")

# ══════════════════════════════════════════════════════════════
# RESULTS DATA
# ══════════════════════════════════════════════════════════════
results = {
    "Model":     ["KNN (K=5)",        "Decision Tree (CART)", "SVM (RBF)"],
    "Member":    ["Ho Jun Yon",        "Irvin Tan Wei Shen",   "Chiang Jun Hang"],
    "Target":    ["Depression",        "Depression",           "Panic Attack"],
    "Accuracy":  [95.83,               85.50,                  None],
    "Precision": [90.48,               72.38,                  None],
    "Recall":    [97.44,               89.18,                  None],
    "F1 Score":  [93.83,               79.91,                  None],
}

df_results = pd.DataFrame(results)

# ── Page Header ────────────────────────────────────────────────
st.markdown("##### RESULTS")
st.title("Model Comparison")
st.write(
    "Side-by-side performance comparison of all three supervised learning algorithms. "
    "Each model was independently implemented by a different group member."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1: SUMMARY METRICS CARDS
# ══════════════════════════════════════════════════════════════
st.subheader("Performance at a Glance")
st.write("")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("#### 🔵 KNN")
        st.caption("Member 1 — Ho Jun Yon | Target: Depression")
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",  "95.83%")
        m2.metric("Precision", "90.48%")
        m3, m4 = st.columns(2)
        m3.metric("Recall",    "97.44%")
        m4.metric("F1 Score",  "93.83%")
        st.divider()
        st.caption("K=5 · MinMax Scaler · 80/20 Split · CV Mean: 86.67%")
        if st.button("View Full KNN Page", key="cmp_knn",
                     use_container_width=True):
            st.switch_page("pages/2_KNN.py")

with c2:
    with st.container(border=True):
        st.markdown("#### 🌳 Decision Tree")
        st.caption("Member 2 — Irvin Tan Wei Shen | Target: Depression")
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",  "85.50%")
        m2.metric("Precision", "72.38%")
        m3, m4 = st.columns(2)
        m3.metric("Recall",    "89.18%")
        m4.metric("F1 Score",  "79.91%")
        st.divider()
        st.caption("CART · Depth 5 · Gini · Root: Marital Status")
        if st.button("View Full DT Page", key="cmp_dt",
                     use_container_width=True):
            st.switch_page("pages/3_Decision_Tree.py")

with c3:
    with st.container(border=True):
        st.markdown("#### 🔴 SVM")
        st.caption("Member 3 — Chiang Jun Hang | Target: Panic Attack")
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",  "TBD")
        m2.metric("Precision", "TBD")
        m3, m4 = st.columns(2)
        m3.metric("Recall",    "TBD")
        m4.metric("F1 Score",  "TBD")
        st.divider()
        st.caption("RBF Kernel · Standard Scaler · 75/25 Split")
        if st.button("View Full SVM Page", key="cmp_svm",
                     use_container_width=True):
            st.switch_page("pages/4_SVM.py")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2: COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
st.subheader("Comparison Table")

table_data = {
    "":          ["Algorithm", "Member", "Target Variable",
                  "Accuracy", "Precision", "Recall", "F1 Score",
                  "Encoding", "Scaling", "Train/Test Split", "Extra"],
    "🔵 KNN":    ["K-Nearest Neighbor", "Ho Jun Yon", "Depression",
                  "95.83%", "90.48%", "97.44%", "93.83%",
                  "Label Encoding", "MinMax Scaler", "80% / 20%",
                  "CV Mean: 86.67% · K=5"],
    "🌳 Decision Tree": ["CART", "Irvin Tan Wei Shen", "Depression",
                         "85.50%", "72.38%", "89.18%", "79.91%",
                         "One-Hot Encoding", "No Scaling", "Full Dataset",
                         "Max Depth: 5 · Root: Marital Status"],
    "🔴 SVM":    ["Support Vector Machine", "Chiang Jun Hang", "Panic Attack",
                  "TBD", "TBD", "TBD", "TBD",
                  "Label Encoding", "Standard Scaler", "75% / 25%",
                  "RBF Kernel · C & gamma tuned"],
}

st.dataframe(
    pd.DataFrame(table_data).set_index(""),
    use_container_width=True,
    height=420
)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3: BAR CHART COMPARISON
# ══════════════════════════════════════════════════════════════
st.subheader("Metric Comparison Chart")
st.caption("Based on KNN and Decision Tree results. SVM to be updated.")

metrics   = ["Accuracy", "Precision", "Recall", "F1 Score"]
knn_vals  = [95.83, 90.48, 97.44, 93.83]
dt_vals   = [85.50, 72.38, 89.18, 79.91]
svm_vals  = [0, 0, 0, 0]  # TBD

x     = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor('#0B0E1A')
ax.set_facecolor('#0B0E1A')

bars1 = ax.bar(x - width, knn_vals, width, label='KNN',
               color='#5B7FFF', alpha=0.9, edgecolor='#0B0E1A', linewidth=1.2)
bars2 = ax.bar(x,          dt_vals,  width, label='Decision Tree',
               color='#10B981', alpha=0.9, edgecolor='#0B0E1A', linewidth=1.2)
bars3 = ax.bar(x + width, svm_vals, width, label='SVM (TBD)',
               color='#EF4444', alpha=0.4, edgecolor='#0B0E1A', linewidth=1.2)

# Labels on bars
for bars, vals in [(bars1, knn_vals), (bars2, dt_vals)]:
    for bar, val in zip(bars, vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color='white')

ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12, color='#9CA3AF')
ax.set_ylabel('Score (%)', color='#9CA3AF', fontsize=11)
ax.set_ylim(0, 115)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.yaxis.set_tick_params(labelcolor='#9CA3AF')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#1E2640')
ax.spines['bottom'].set_color('#1E2640')
ax.set_title('Algorithm Performance Comparison', color='white',
             fontsize=14, fontweight='bold', pad=16)

legend = ax.legend(fontsize=11, framealpha=0.2,
                   labelcolor='white', facecolor='#1E2640',
                   edgecolor='#2E3854')

ax.grid(axis='y', color='#1E2640', linewidth=0.8, alpha=0.6)
ax.set_axisbelow(True)

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4: RADAR CHART
# ══════════════════════════════════════════════════════════════
st.subheader("Radar Chart — KNN vs Decision Tree")
st.caption("Visualizing all 4 metrics on a radar / spider chart.")

categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

knn_r = [95.83, 90.48, 97.44, 93.83]
dt_r  = [85.50, 72.38, 89.18, 79.91]
knn_r += knn_r[:1]
dt_r  += dt_r[:1]

fig2, ax2 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
fig2.patch.set_facecolor('#0B0E1A')
ax2.set_facecolor('#0B0E1A')

ax2.plot(angles, knn_r, 'o-', linewidth=2, color='#5B7FFF', label='KNN')
ax2.fill(angles, knn_r, alpha=0.15, color='#5B7FFF')
ax2.plot(angles, dt_r, 'o-', linewidth=2, color='#10B981', label='Decision Tree')
ax2.fill(angles, dt_r, alpha=0.15, color='#10B981')

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(categories, color='#9CA3AF', fontsize=11)
ax2.set_ylim(0, 110)
ax2.set_yticks([20, 40, 60, 80, 100])
ax2.yaxis.set_tick_params(labelcolor='#4A5060')
ax2.tick_params(axis='y', labelsize=8)
ax2.grid(color='#1E2640', linewidth=0.8)
ax2.spines['polar'].set_color('#1E2640')
ax2.set_title('KNN vs Decision Tree', color='white',
              fontsize=13, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15),
           fontsize=10, labelcolor='white',
           facecolor='#1E2640', edgecolor='#2E3854',
           framealpha=0.8)

st.pyplot(fig2)
plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5: CONFUSION MATRIX SIDE BY SIDE
# ══════════════════════════════════════════════════════════════
st.subheader("Confusion Matrix Comparison")

cm1, cm2 = st.columns(2)

with cm1:
    st.markdown("**🔵 KNN Confusion Matrix** (Test Set: 120 records)")
    knn_cm = np.array([[77, 4], [1, 38]])
    fig_k, ax_k = plt.subplots(figsize=(4, 3))
    fig_k.patch.set_facecolor('#0B0E1A')
    ax_k.set_facecolor('#0B0E1A')
    sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues',
                ax=ax_k, linewidths=0.5, linecolor='#0B0E1A',
                xticklabels=['No Dep', 'Depression'],
                yticklabels=['No Dep', 'Depression'],
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
    ax_k.set_xlabel('Predicted', color='#9CA3AF', fontsize=10)
    ax_k.set_ylabel('Actual',    color='#9CA3AF', fontsize=10)
    ax_k.tick_params(colors='#9CA3AF')
    ax_k.set_title('KNN Confusion Matrix', color='white',
                   fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    st.pyplot(fig_k)
    plt.close()

    c_a, c_b, c_c, c_d = st.columns(4)
    c_a.metric("TN", "77")
    c_b.metric("FP", "4")
    c_c.metric("FN", "1")
    c_d.metric("TP", "38")

with cm2:
    st.markdown("**🌳 Decision Tree Confusion Matrix** (Full: 600 records)")
    dt_cm = np.array([[340, 66], [21, 173]])
    fig_d, ax_d = plt.subplots(figsize=(4, 3))
    fig_d.patch.set_facecolor('#0B0E1A')
    ax_d.set_facecolor('#0B0E1A')
    sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Greens',
                ax=ax_d, linewidths=0.5, linecolor='#0B0E1A',
                xticklabels=['No Dep', 'Depression'],
                yticklabels=['No Dep', 'Depression'],
                annot_kws={'size': 14, 'weight': 'bold', 'color': 'white'})
    ax_d.set_xlabel('Predicted', color='#9CA3AF', fontsize=10)
    ax_d.set_ylabel('Actual',    color='#9CA3AF', fontsize=10)
    ax_d.tick_params(colors='#9CA3AF')
    ax_d.set_title('Decision Tree Confusion Matrix', color='white',
                   fontsize=11, fontweight='bold', pad=10)
    plt.tight_layout()
    st.pyplot(fig_d)
    plt.close()

    c_a, c_b, c_c, c_d = st.columns(4)
    c_a.metric("TN", "340")
    c_b.metric("FP", "66")
    c_c.metric("FN", "21")
    c_d.metric("TP", "173")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 6: KEY FINDINGS
# ══════════════════════════════════════════════════════════════
st.subheader("Key Findings")

f1, f2, f3 = st.columns(3)

with f1:
    with st.container(border=True):
        st.markdown("**🏆 Best Overall: KNN**")
        st.write(
            "KNN (K=5) achieved the highest accuracy of **95.83%** and recall of **97.44%**. "
            "It correctly identified 38 out of 39 depressed students in the test set, "
            "missing only 1."
        )

with f2:
    with st.container(border=True):
        st.markdown("**📖 Most Interpretable: Decision Tree**")
        st.write(
            "Decision Tree CART provides a visual decision path that can be explained "
            "to non-technical users. Marital Status is the most discriminative root feature, "
            "followed by Anxiety and Treatment-seeking behaviour."
        )

with f3:
    with st.container(border=True):
        st.markdown("**⚠️ Why Recall Matters Most**")
        st.write(
            "In mental health screening, missing a depressed student (false negative) "
            "is more serious than a false alarm. KNN's 97.44% recall means it misses "
            "almost no at-risk students."
        )

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")
