from utils.preprocessing import load_and_clean_dataset
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar

st.set_page_config(
    page_title="Exploratory Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("eda")

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    h1 { color: #1a1a2e; font-family: Arial, sans-serif; }
    h2 { color: #16213e; font-family: Arial, sans-serif; }
    h3 { color: #0f3460; font-family: Arial, sans-serif; }
    .section-title {
        font-size: 18px;
        font-weight: bold;
        color: #0f3460;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .summary-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 4px solid #0f3460;
        margin-bottom: 10px;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────
st.title("Exploratory Data Analysis (EDA)")
st.markdown("Student Mental Health Dataset — Overview and Visualization")
st.markdown("---")

# ── Load Data ──────────────────────────────────────────────────


@st.cache_data
def get_data():
    return load_and_clean_dataset('dataset/Student_Mental_health.csv')


df = get_data()

# ══════════════════════════════════════════════════════════════
# SECTION 1: DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">1. Dataset Overview</div>',
            unsafe_allow_html=True)

o1, o2, o3, o4 = st.columns(4)
o1.metric("Total Records",    f"{df.shape[0]}")
o2.metric("Total Features",   f"{df.shape[1]}")
o3.metric("Missing Values",   f"{df.isnull().sum().sum()}")
o4.metric("Target Variables", "3")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Dataset Preview (First 50 Rows)**")
st.dataframe(df.head(50), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Basic Statistical Description**")
st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Column Info ────────────────────────────────────────────────
st.markdown("**Column Data Types and Non-Null Count**")
col_info = pd.DataFrame({
    'Column': df.columns,
    'Data Type': df.dtypes.values.astype(str),
    'Non-Null Count': df.notnull().sum().values,
    'Null Count': df.isnull().sum().values,
    'Unique Values': [df[c].nunique() for c in df.columns],
})
st.dataframe(col_info, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 2: TARGET VARIABLE DISTRIBUTION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">2. Target Variable Distribution</div>',
            unsafe_allow_html=True)
st.write("Distribution of the three mental health conditions used as prediction targets.")

t1, t2, t3 = st.columns(3)

targets = [
    ('Depression',   'Member 1 - KNN Target',         '#2874A6', '#E74C3C'),
    ('Anxiety',      'Member 2 - Decision Tree Target', '#1E8449', '#F39C12'),
    ('Panic_Attack', 'Member 3 - SVM Target',          '#7D3C98', '#E74C3C'),
]

for col, (target, subtitle, c_no, c_yes) in zip([t1, t2, t3], targets):
    with col:
        counts = df[target].value_counts().sort_index()
        no_count = counts.get(0, 0)
        yes_count = counts.get(1, 0)
        total = no_count + yes_count

        st.markdown(f"**{target}**")
        st.caption(subtitle)

        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(['No', 'Yes'], [no_count, yes_count],
                      color=[c_no, c_yes], width=0.5, edgecolor='white')
        ax.set_title(f'{target} Distribution', fontsize=11, fontweight='bold')
        ax.set_ylabel('Count')
        ax.set_ylim(0, max(no_count, yes_count) + 10)
        for bar, val in zip(bars, [no_count, yes_count]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1,
                    f'{val} ({val/total*100:.1f}%)',
                    ha='center', fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

        c_a, c_b = st.columns(2)
        c_a.metric("No",  f"{no_count}  ({no_count/total*100:.1f}%)")
        c_b.metric("Yes", f"{yes_count} ({yes_count/total*100:.1f}%)")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 3: DEMOGRAPHIC ANALYSIS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">3. Demographic Analysis</div>',
            unsafe_allow_html=True)

d1, d2 = st.columns(2)

with d1:
    st.markdown("**Gender Distribution**")
    gender_counts = df['Gender'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#2874A6', '#E74C3C']
    wedges, texts, autotexts = ax.pie(
        gender_counts.values,
        labels=['Female', 'Male'],
        colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax.set_title('Gender Distribution', fontsize=12, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with d2:
    st.markdown("**Age Distribution**")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(df['Age'], bins=range(17, 26), color='#2874A6',
            edgecolor='white', linewidth=1.2, rwidth=0.85)
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    ax.set_title('Age Distribution of Students',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

st.markdown("<br>", unsafe_allow_html=True)
d3, d4 = st.columns(2)

with d3:
    st.markdown("**Year of Study Distribution**")
    year_counts = df['Year_of_Study'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ['#2874A6', '#1E8449', '#7D3C98', '#E67E22']
    bars = ax.bar(year_counts.index, year_counts.values,
                  color=colors, edgecolor='white', linewidth=1.2)
    ax.set_ylabel('Count')
    ax.set_title('Students by Year of Study', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, year_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                str(val), ha='center', fontsize=10, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with d4:
    st.markdown("**CGPA Distribution**")
    cgpa_counts = df['CGPA'].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(cgpa_counts.index, cgpa_counts.values,
                   color='#2874A6', edgecolor='white', linewidth=1.2)
    ax.set_xlabel('Count')
    ax.set_title('Students by CGPA Range', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, cgpa_counts.values):
        ax.text(bar.get_width() + 0.3,
                bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=10, fontweight='bold')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 4: COURSE DISTRIBUTION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">4. Course Distribution</div>',
            unsafe_allow_html=True)

course_counts = df['Course'].value_counts()
fig, ax = plt.subplots(figsize=(10, 4))
colors = plt.cm.Blues(
    [0.4 + 0.06*i for i in range(len(course_counts))]
)[::-1]
bars = ax.barh(course_counts.index, course_counts.values,
               color=colors, edgecolor='white', linewidth=1.2)
ax.set_xlabel('Number of Students')
ax.set_title('Number of Students by Course', fontsize=13, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, val in zip(bars, course_counts.values):
    ax.text(bar.get_width() + 0.2,
            bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=10, fontweight='bold')
st.pyplot(fig)
plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 5: MENTAL HEALTH vs DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">5. Mental Health vs Demographics</div>',
            unsafe_allow_html=True)
st.write("Comparison of mental health conditions across gender and year of study.")

g1, g2, g3 = st.columns(3)

conditions = ['Depression', 'Anxiety', 'Panic_Attack']
col_list = [g1, g2, g3]
colors_list = [
    ['#2874A6', '#E74C3C'],
    ['#1E8449', '#F39C12'],
    ['#7D3C98', '#E74C3C'],
]

for col, cond, colors in zip(col_list, conditions, colors_list):
    with col:
        st.markdown(f"**{cond} by Gender**")
        gender_cond = df.groupby(
            'Gender')[cond].value_counts().unstack().fillna(0)
        gender_cond.index = ['Female', 'Male']
        gender_cond.columns = ['No', 'Yes']
        fig, ax = plt.subplots(figsize=(4, 3))
        gender_cond.plot(kind='bar', ax=ax, color=colors,
                         edgecolor='white', linewidth=1.2, rot=0)
        ax.set_title(f'{cond} by Gender', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count')
        ax.legend(['No', 'Yes'], loc='upper right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

st.markdown("<br>", unsafe_allow_html=True)

y1, y2, y3 = st.columns(3)
for col, cond, colors in zip([y1, y2, y3], conditions, colors_list):
    with col:
        st.markdown(f"**{cond} by Year of Study**")
        year_cond = df.groupby('Year_of_Study')[
            cond].value_counts().unstack().fillna(0)
        year_cond.columns = ['No', 'Yes']
        fig, ax = plt.subplots(figsize=(4, 3))
        year_cond.plot(kind='bar', ax=ax, color=colors,
                       edgecolor='white', linewidth=1.2, rot=0)
        ax.set_title(f'{cond} by Year', fontsize=10, fontweight='bold')
        ax.set_ylabel('Count')
        ax.legend(['No', 'Yes'], loc='upper right', fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
        plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 6: MENTAL HEALTH SCORE
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">6. Mental Health Score Distribution</div>',
            unsafe_allow_html=True)
st.write("Mental Health Score = Depression + Anxiety + Panic Attack (Range: 0 to 3)")

mhs_counts = df['Mental_Health_Score'].value_counts().sort_index()
labels_mhs = ['0 - Healthy', '1 - Mild', '2 - Moderate', '3 - Severe']
colors_mhs = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

ms1, ms2 = st.columns(2)
with ms1:
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels_mhs, mhs_counts.values,
                  color=colors_mhs, edgecolor='white', linewidth=1.5)
    ax.set_ylabel('Number of Students')
    ax.set_title('Mental Health Score Distribution',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar, val in zip(bars, mhs_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                str(val), ha='center', fontsize=11, fontweight='bold')
    st.pyplot(fig)
    plt.close()

with ms2:
    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        mhs_counts.values, labels=labels_mhs,
        colors=colors_mhs, autopct='%1.1f%%',
        startangle=90, pctdistance=0.78,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight('bold')
    ax.set_title('Mental Health Score Proportion',
                 fontsize=12, fontweight='bold')
    st.pyplot(fig)
    plt.close()

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, label, val, color in zip(
        [sc1, sc2, sc3, sc4], ['Healthy', 'Mild', 'Moderate', 'Severe'],
        mhs_counts.values, colors_mhs
    ):
        col.metric(label, str(val))

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 7: CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">7. Correlation Heatmap</div>',
            unsafe_allow_html=True)
st.write("Correlation between all numerical features. Values close to 1 or -1 indicate strong correlation.")

numeric_df = df.select_dtypes(include='number')
fig, ax = plt.subplots(figsize=(11, 7))
mask = (numeric_df.corr() == 0)
sns.heatmap(
    numeric_df.corr(), annot=True, fmt='.2f',
    cmap='coolwarm', ax=ax, linewidths=0.5,
    linecolor='white', annot_kws={'size': 9},
    cbar_kws={'shrink': 0.8}
)
ax.set_title('Feature Correlation Matrix',
             fontsize=13, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("""
**How to read the heatmap:**
- Values close to **+1.0** indicate strong positive correlation.
- Values close to **-1.0** indicate strong negative correlation.
- Values close to **0** indicate little to no linear relationship.
""")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:13px;'>"
    "BMCS2003 Artificial Intelligence | 202605 Session | Tutorial Group 3 | Tutor: Dr Goh"
    "</p>",
    unsafe_allow_html=True
)
