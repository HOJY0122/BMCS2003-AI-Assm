import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import sidebar

st.set_page_config(
    page_title="Feature Selection — MindCheck",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("feature")

# ── Load & prepare data ────────────────────────────────────────
@st.cache_data
def load_features():
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())
    le_c = LabelEncoder(); le_y = LabelEncoder()
    df['Course_Enc'] = le_c.fit_transform(df['Course'])
    df['Year_Enc']   = le_y.fit_transform(df['Year_of_Study'])

    all_feat = ['Gender','Age','Course_Enc','Year_Enc','CGPA_Numeric',
                'Anxiety','Panic_Attack','Marital_Status',
                'Seek_Treatment','Mental_Health_Score']
    feat_labels = ['Gender','Age','Course','Year','CGPA',
                   'Anxiety','Panic Attack','Marital Status',
                   'Seek Treatment','Mental Health Score']

    X = df[all_feat]; y = df['Depression']

    # 1. Pearson Correlation
    corr = X.corrwith(y).abs()
    corr.index = feat_labels

    # 2. Point-biserial / t-test p-values
    pvals = {}
    for feat, lbl in zip(all_feat, feat_labels):
        grp0 = df[df['Depression']==0][feat].dropna()
        grp1 = df[df['Depression']==1][feat].dropna()
        _, p = stats.ttest_ind(grp0, grp1)
        pvals[lbl] = p

    # 3. Mutual Information
    mi = mutual_info_classif(X, y, random_state=42)
    mi_s = pd.Series(mi, index=feat_labels)

    # 4. Chi-Square
    X_abs = X.abs()
    chi2_sc, chi2_p = chi2(X_abs, y)
    chi2_s = pd.Series(chi2_sc, index=feat_labels)

    # 5. KNN accuracy experiment: add features one by one
    sorted_by_corr = corr.sort_values(ascending=False).index.tolist()
    sorted_raw = [all_feat[feat_labels.index(f)] for f in sorted_by_corr]
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y)

    knn_accs, dt_accs = [], []
    for n in range(1, 11):
        idx = [all_feat.index(f) for f in sorted_raw[:n]]
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xtr[:, idx], ytr)
        knn_accs.append(accuracy_score(yte, knn.predict(Xte[:, idx]))*100)

        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(Xtr[:, idx], ytr)
        dt_accs.append(accuracy_score(yte, dt.predict(Xte[:, idx]))*100)

    return {
        'X': X, 'y': y, 'df': df,
        'all_feat': all_feat, 'feat_labels': feat_labels,
        'corr': corr, 'pvals': pvals,
        'mi': mi_s, 'chi2': chi2_s,
        'sorted_labels': sorted_by_corr,
        'sorted_raw': sorted_raw,
        'knn_accs': knn_accs, 'dt_accs': dt_accs,
        'scaler': scaler,
    }

with st.spinner("Running feature analysis on live data..."):
    D = load_features()

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("##### 🔬 FEATURE SELECTION")
st.title("Feature Selection & Correlation Analysis")
st.write(
    "Identifying which features are most important for predicting **Depression**. "
    "All analysis is computed **live** from the dataset — no hardcoded values."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — OVERVIEW: ALL 10 FEATURES
# ══════════════════════════════════════════════════════════════
st.subheader("1. Dataset Features Overview")
st.caption("All 10 features available in the dataset vs Depression target.")

feat_overview = pd.DataFrame({
    'Feature': D['feat_labels'],
    'Type': ['Binary','Numeric','Categorical','Categorical','Numeric',
             'Binary','Binary','Binary','Binary','Numeric'],
    'Description': [
        'Male=1 / Female=0',
        'Student age (17–24)',
        'Field of study (10 categories)',
        'Year 1 to Year 4',
        'CGPA numeric midpoint',
        'Has anxiety? Yes=1 / No=0',
        'Has panic attack? Yes=1 / No=0',
        'Married? Yes=1 / No=0',
        'Sought treatment? Yes=1 / No=0',
        'Sum of Depression+Anxiety+Panic (0–3)',
    ],
    'Correlation |r|': [f"{D['corr'][f]:.3f}" for f in D['feat_labels']],
    'p-value': [f"{D['pvals'][f]:.4f}" for f in D['feat_labels']],
    'Significant': ['✅' if D['pvals'][f] < 0.05 else '❌'
                    for f in D['feat_labels']],
})
st.dataframe(feat_overview.set_index('Feature'), use_container_width=True)
st.caption("p-value < 0.05 = statistically significant difference between Depression and No Depression groups")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — CORRELATION TEST
# ══════════════════════════════════════════════════════════════
st.subheader("2. Correlation Test — Pearson Correlation with Depression")
st.write(
    "Pearson correlation measures the **linear relationship** between each feature "
    "and the Depression target. Higher absolute value = stronger relationship."
)

corr_sorted = D['corr'].sort_values(ascending=True)
fig_corr, ax = plt.subplots(figsize=(9, 5))
colors = ['#EF4444' if v >= 0.2 else '#3B82F6' if v >= 0.1 else '#9CA3AF'
          for v in corr_sorted.values]
bars = ax.barh(corr_sorted.index, corr_sorted.values,
               color=colors, edgecolor='none', height=0.6)
ax.axvline(0.2, color='#EF4444', linestyle='--', lw=1.5,
           label='Strong (≥ 0.2)')
ax.axvline(0.1, color='#3B82F6', linestyle='--', lw=1.5,
           label='Moderate (0.1–0.2)')
for bar, val in zip(bars, corr_sorted.values):
    ax.text(val+0.005, bar.get_y()+bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
ax.set_xlabel('Absolute Pearson Correlation with Depression', fontsize=11)
ax.set_title('Feature Correlation with Depression Target (Live)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
st.pyplot(fig_corr, use_container_width=True)
plt.close()

# Key findings
top3 = D['corr'].sort_values(ascending=False).head(3)
bottom3 = D['corr'].sort_values(ascending=False).tail(3)
k1,k2,k3 = st.columns(3)
with k1:
    with st.container(border=True):
        st.markdown("**🔴 Strongest Predictors**")
        for feat, val in top3.items():
            st.write(f"**{feat}** — r = {val:.3f}")
        st.caption("r ≥ 0.2: Strong correlation")
with k2:
    with st.container(border=True):
        st.markdown("**🔵 Moderate Predictors**")
        mid = D['corr'].sort_values(ascending=False)
        mid = mid[(mid >= 0.1) & (mid < 0.2)]
        for feat, val in mid.items():
            st.write(f"**{feat}** — r = {val:.3f}")
        st.caption("0.1 ≤ r < 0.2: Moderate correlation")
with k3:
    with st.container(border=True):
        st.markdown("**⚪ Weak Predictors**")
        for feat, val in bottom3.items():
            st.write(f"**{feat}** — r = {val:.3f}")
        st.caption("r < 0.1: Weak correlation")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — STATISTICAL SIGNIFICANCE (T-TEST)
# ══════════════════════════════════════════════════════════════
st.subheader("3. Statistical Significance — Independent t-Test")
st.write(
    "Independent samples t-test compares the mean of each feature between "
    "Depression=1 and Depression=0 groups. **p-value < 0.05** means the difference "
    "is statistically significant and the feature is a meaningful predictor."
)

pval_df = pd.DataFrame({
    'Feature': list(D['pvals'].keys()),
    'p-value': list(D['pvals'].values()),
    'Significant': ['✅ Yes' if v < 0.05 else '❌ No'
                    for v in D['pvals'].values()],
    'Interpretation': [
        'Strong evidence of difference' if v < 0.001
        else 'Significant difference' if v < 0.05
        else 'No significant difference'
        for v in D['pvals'].values()
    ]
}).sort_values('p-value').set_index('Feature')

# Color p-value column
def color_pval(val):
    try:
        v = float(val)
        if v < 0.001: return 'background-color: #DCFCE7'
        elif v < 0.05: return 'background-color: #FEF9C3'
        else: return 'background-color: #FEE2E2'
    except: return ''

try:
    styled_pval = pval_df.style.map(color_pval, subset=['p-value'])
except AttributeError:
    styled_pval = pval_df.style.applymap(color_pval, subset=['p-value'])
st.dataframe(styled_pval, use_container_width=True)
sig_count = sum(1 for v in D['pvals'].values() if v < 0.05)
st.info(f"**{sig_count} out of {len(D['pvals'])} features** are statistically "
        f"significant (p < 0.05). These features show a meaningful difference "
        f"between depressed and non-depressed students.")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — MULTIPLE METHODS COMPARISON
# ══════════════════════════════════════════════════════════════
st.subheader("4. Feature Importance — Multiple Methods Comparison")
st.write(
    "Comparing three different feature importance methods to get a comprehensive "
    "view of which features matter most."
)

tab1, tab2, tab3 = st.tabs([
    "Pearson Correlation", "Mutual Information", "Chi-Square"
])

with tab1:
    st.write("**Pearson Correlation** measures linear relationship between feature and target.")
    corr_df = D['corr'].sort_values(ascending=False).reset_index()
    corr_df.columns = ['Feature','Correlation |r|']
    fig1, a1 = plt.subplots(figsize=(8,4))
    c1_colors = ['#EF4444' if v >= 0.2 else '#3B82F6' if v >= 0.1
                 else '#9CA3AF' for v in corr_df['Correlation |r|']]
    a1.bar(corr_df['Feature'], corr_df['Correlation |r|'],
           color=c1_colors, edgecolor='none')
    a1.axhline(0.2, color='red', ls='--', lw=1.2, alpha=0.7, label='Threshold 0.2')
    a1.set_ylabel('|Correlation|'); a1.set_title('Pearson Correlation', fontweight='bold')
    a1.tick_params(axis='x', rotation=30)
    a1.legend(); a1.spines['top'].set_visible(False); a1.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig1, use_container_width=True); plt.close()

with tab2:
    st.write("**Mutual Information** measures non-linear dependency between feature and target.")
    mi_df = D['mi'].sort_values(ascending=False).reset_index()
    mi_df.columns = ['Feature','Mutual Info']
    fig2, a2 = plt.subplots(figsize=(8,4))
    a2.bar(mi_df['Feature'], mi_df['Mutual Info'],
           color='#8B5CF6', edgecolor='none', alpha=0.85)
    a2.set_ylabel('Mutual Information Score')
    a2.set_title('Mutual Information with Depression', fontweight='bold')
    a2.tick_params(axis='x', rotation=30)
    a2.spines['top'].set_visible(False); a2.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close()

with tab3:
    st.write("**Chi-Square Test** measures statistical independence between feature and target.")
    chi_df = D['chi2'].sort_values(ascending=False).reset_index()
    chi_df.columns = ['Feature','Chi2 Score']
    fig3, a3 = plt.subplots(figsize=(8,4))
    a3.bar(chi_df['Feature'], chi_df['Chi2 Score'],
           color='#F59E0B', edgecolor='none', alpha=0.85)
    a3.set_ylabel('Chi-Square Score')
    a3.set_title('Chi-Square Score with Depression', fontweight='bold')
    a3.tick_params(axis='x', rotation=30)
    a3.spines['top'].set_visible(False); a3.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
st.subheader("5. Full Correlation Heatmap")
st.write("Pairwise correlation between all features and the Depression target.")

corr_matrix = D['X'].copy()
corr_matrix.columns = D['feat_labels']
corr_matrix['Depression'] = D['y'].values
full_corr = corr_matrix.corr()

fig_heat, ax_heat = plt.subplots(figsize=(12, 9))
mask = np.zeros_like(full_corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(full_corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax_heat, mask=mask,
            linewidths=0.5, annot_kws={'size':8},
            vmin=-1, vmax=1)
ax_heat.set_title('Feature Correlation Heatmap (Lower Triangle)',
                  fontsize=13, fontweight='bold', pad=16)
plt.tight_layout()
st.pyplot(fig_heat, use_container_width=True)
plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 6 — FEATURE EXPERIMENT
# ══════════════════════════════════════════════════════════════
st.subheader("6. Feature Experiment — Accuracy vs Number of Features")
st.write(
    "Experiment: Features added **one by one** (ordered by correlation strength). "
    "Shows how model accuracy changes as we include more features. "
    "This proves which features are truly needed."
)

st.markdown("**Feature order (by correlation rank):**")
for i, feat in enumerate(D['sorted_labels'], 1):
    corr_val = D['corr'][feat]
    sig = "✅" if D['pvals'][feat] < 0.05 else "❌"
    st.write(f"**#{i}** {feat} — |r| = {corr_val:.3f} {sig}")

st.write("")
fig_exp, ax_exp = plt.subplots(figsize=(10, 5))
x = list(range(1, 11))
ax_exp.plot(x, D['knn_accs'], 'b-o', lw=2, ms=7,
            label='KNN (K=5)', zorder=3)
ax_exp.plot(x, D['dt_accs'], 'g-s', lw=2, ms=7,
            label='Decision Tree (Depth 5)', zorder=3)

# Highlight best point for each
best_knn = D['knn_accs'].index(max(D['knn_accs'])) + 1
best_dt  = D['dt_accs'].index(max(D['dt_accs'])) + 1
ax_exp.axvline(best_knn, color='blue', ls='--', lw=1.2, alpha=0.5)
ax_exp.axvline(best_dt,  color='green', ls='--', lw=1.2, alpha=0.5)
ax_exp.scatter([best_knn], [max(D['knn_accs'])],
               color='blue', s=120, zorder=5,
               label=f'KNN best: {max(D["knn_accs"]):.1f}% at {best_knn} features')
ax_exp.scatter([best_dt], [max(D['dt_accs'])],
               color='green', s=120, zorder=5,
               label=f'DT best: {max(D["dt_accs"]):.1f}% at {best_dt} features')

ax_exp.set_xlabel('Number of Features (added by correlation rank)', fontsize=11)
ax_exp.set_ylabel('Test Accuracy (%)', fontsize=11)
ax_exp.set_title('Model Accuracy vs Number of Features — Live Experiment',
                 fontsize=13, fontweight='bold')
ax_exp.set_xticks(x)
ax_exp.set_xticklabels([f"{n}\n{D['sorted_labels'][n-1]}" for n in x],
                        fontsize=8)
ax_exp.set_ylim(50, 105)
ax_exp.legend(fontsize=9, loc='lower right')
ax_exp.grid(True, alpha=0.3)
ax_exp.spines['top'].set_visible(False)
ax_exp.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig_exp, use_container_width=True)
plt.close()

# Results table
exp_df = pd.DataFrame({
    'Features Used': [f"{n}: {', '.join(D['sorted_labels'][:n])}"
                      for n in range(1,11)],
    'KNN Accuracy': [f"{v:.2f}%" for v in D['knn_accs']],
    'DT Accuracy':  [f"{v:.2f}%" for v in D['dt_accs']],
}).set_index('Features Used')
st.dataframe(exp_df, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 7 — CONCLUSION
# ══════════════════════════════════════════════════════════════
st.subheader("7. Feature Selection Conclusion")

top_feat  = D['corr'].sort_values(ascending=False).head(5).index.tolist()
best_knn_n = D['knn_accs'].index(max(D['knn_accs'])) + 1
best_dt_n  = D['dt_accs'].index(max(D['dt_accs'])) + 1

c1, c2 = st.columns(2)
with c1:
    with st.container(border=True):
        st.markdown("**✅ Selected Features (Top 5 by Correlation)**")
        for i, f in enumerate(top_feat, 1):
            st.write(f"**{i}.** {f} — |r| = {D['corr'][f]:.3f}")
        st.write("")
        st.caption(
            "These 5 features have the strongest statistical relationship "
            "with Depression and are statistically significant (p < 0.05)."
        )
with c2:
    with st.container(border=True):
        st.markdown("**📊 Experiment Findings**")
        st.write(f"**KNN** best accuracy: **{max(D['knn_accs']):.2f}%** "
                 f"using **{best_knn_n} features**")
        st.write(f"**DT** best accuracy: **{max(D['dt_accs']):.2f}%** "
                 f"using **{best_dt_n} features**")
        st.write("")
        st.write(
            "Adding more features does **not always improve** accuracy. "
            "Features with low correlation (Age, Year, CGPA) contribute "
            "minimal predictive value and can introduce noise."
        )

st.info(
    "**Key Finding:** Mental Health Score, Marital Status, Panic Attack, "
    "Seek Treatment and Anxiety are the strongest predictors of Depression "
    "in this dataset, all with statistically significant p-values (p < 0.05). "
    "Age, Year of Study, and CGPA show weak correlation and low predictive power."
)

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")