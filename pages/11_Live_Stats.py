import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import sidebar

st.set_page_config(
    page_title="Live Dataset Statistics — MindCheck",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("stats")

# ── Load dataset ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())
    return df

df = load_data()

# ── Session state for predictions log ─────────────────────────
if 'live_predictions' not in st.session_state:
    st.session_state.live_predictions = []

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("##### 📊 LIVE DATASET STATISTICS")
st.title("Live Dataset Statistics")
st.write(
    "Explore the dataset in real-time and see how **your student profile compares** "
    f"to the {len(df)} IIUM students in the dataset. All charts update live."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1 — LIVE OVERVIEW METRICS
# ══════════════════════════════════════════════════════════════
st.subheader("Dataset Overview")
st.caption(f"Live computed from {len(df)} student records — no hardcode")

m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Total Students",    str(len(df)))
m2.metric("Depression Rate",   f"{df['Depression'].mean()*100:.1f}%")
m3.metric("Anxiety Rate",      f"{df['Anxiety'].mean()*100:.1f}%")
m4.metric("Panic Attack Rate", f"{df['Panic_Attack'].mean()*100:.1f}%")
m5.metric("Avg Age",           f"{df['Age'].mean():.1f} yrs")
m6.metric("Avg CGPA",          f"{df['CGPA_Numeric'].mean():.2f}")
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — STUDENT PROFILE COMPARATOR
# ══════════════════════════════════════════════════════════════
st.subheader("Compare Your Student to the Dataset")
st.write(
    "Enter a student's details to see **exactly where they stand** "
    f"relative to all {len(df)} students in the dataset."
)

CGPA_MAP = {'0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
            '3.00 - 3.49':3.25,'3.50 - 4.00':3.75}

with st.container(border=True):
    p1, p2, p3 = st.columns(3)
    with p1:
        s_name    = st.text_input("Name", placeholder="e.g. Ahmad", key="ls_name")
        s_gender  = st.selectbox("Gender", ["Female","Male"], key="ls_gender")
        s_age     = st.slider("Age", 17, 30, 20, key="ls_age")
    with p2:
        s_year    = st.selectbox("Year of Study",
                      ["Year 1","Year 2","Year 3","Year 4"], key="ls_year")
        s_cgpa    = st.selectbox("CGPA Range", list(CGPA_MAP.keys()), key="ls_cgpa")
        s_anxiety = st.selectbox("Has Anxiety?",      ["No","Yes"], key="ls_anxiety")
    with p3:
        s_panic   = st.selectbox("Has Panic Attack?", ["No","Yes"], key="ls_panic")
        s_marital = st.selectbox("Marital Status",    ["No","Yes"], key="ls_marital")
        s_treat   = st.selectbox("Sought Treatment?", ["No","Yes"], key="ls_treat")
    compare_btn = st.button("📊  Compare to Dataset",
                            width='stretch', type="primary", key="ls_compare")

if compare_btn:
    s_name_lbl = s_name.strip() or "Student"
    s_cgpa_num = CGPA_MAP[s_cgpa]
    s_gender_n = 1 if s_gender == "Male" else 0
    s_anx_n    = 1 if s_anxiety == "Yes" else 0
    s_pan_n    = 1 if s_panic   == "Yes" else 0

    # Add to predictions log
    st.session_state.live_predictions.append({
        'Name': s_name_lbl, 'Age': s_age, 'CGPA': s_cgpa_num,
        'Anxiety': s_anx_n, 'Panic': s_pan_n,
        'Gender': s_gender_n,
    })

    st.divider()
    st.subheader(f"Profile Analysis for {s_name_lbl}")

    # Row 1: Age + CGPA percentile
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)

    age_pct = (df['Age'] < s_age).mean() * 100
    cgpa_pct = (df['CGPA_Numeric'] < s_cgpa_num).mean() * 100
    dep_same_gender = df[df['Gender']==s_gender_n]['Depression'].mean()*100
    mhs_same = df['Mental_Health_Score'].mean()

    r1c1.metric("Age Percentile",
                f"{age_pct:.0f}th",
                help=f"Older than {age_pct:.0f}% of students in dataset")
    r1c2.metric("CGPA Percentile",
                f"{cgpa_pct:.0f}th",
                help=f"Higher CGPA than {cgpa_pct:.0f}% of students")
    r1c3.metric(f"Depression Rate ({s_gender})",
                f"{dep_same_gender:.1f}%",
                help=f"Depression rate among {s_gender} students in dataset")
    r1c4.metric("Dataset Avg Mental Score",
                f"{mhs_same:.2f}/3",
                help="Average mental health score in dataset (0=healthy, 3=severe)")

    st.write("")

    # Row 2: Visual comparisons
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Age distribution with student marker
    axes[0].hist(df['Age'], bins=8, color='#3B82F6', alpha=0.7,
                 edgecolor='white', label='All Students')
    axes[0].axvline(s_age, color='#EF4444', lw=2.5, ls='--',
                    label=f'{s_name_lbl} (Age {s_age})')
    axes[0].set_xlabel('Age'); axes[0].set_ylabel('Count')
    axes[0].set_title('Age Distribution', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Plot 2: CGPA distribution with marker
    axes[1].hist(df['CGPA_Numeric'], bins=8, color='#10B981', alpha=0.7,
                 edgecolor='white', label='All Students')
    axes[1].axvline(s_cgpa_num, color='#EF4444', lw=2.5, ls='--',
                    label=f'{s_name_lbl} (CGPA ~{s_cgpa_num})')
    axes[1].set_xlabel('CGPA (Numeric)'); axes[1].set_ylabel('Count')
    axes[1].set_title('CGPA Distribution', fontweight='bold')
    axes[1].legend(fontsize=8)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    # Plot 3: Mental health indicator radar
    categories = ['Depression\nRate', 'Anxiety\nRate', 'Panic\nRate', 'Seek\nTreatment']
    dataset_vals = [
        df['Depression'].mean()*100,
        df['Anxiety'].mean()*100,
        df['Panic_Attack'].mean()*100,
        df['Seek_Treatment'].mean()*100,
    ]
    student_vals = [
        0 if s_anx_n == 0 and s_pan_n == 0 else 60,
        s_anx_n * 100,
        s_pan_n * 100,
        100 if s_treat == "Yes" else 0,
    ]
    x = np.arange(len(categories))
    w = 0.35
    bars1 = axes[2].bar(x - w/2, dataset_vals, w, label='Dataset Average',
                         color='#3B82F6', alpha=0.8, edgecolor='none')
    bars2 = axes[2].bar(x + w/2, student_vals, w, label=f'{s_name_lbl}',
                         color='#EF4444', alpha=0.8, edgecolor='none')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(categories, fontsize=8)
    axes[2].set_ylabel('Rate (%)')
    axes[2].set_title('Student vs Dataset\nMental Health Indicators', fontweight='bold')
    axes[2].legend(fontsize=8)
    axes[2].set_ylim(0, 120)
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close()

    st.write("")

    # Similar students in dataset
    st.markdown("**Similar Students in Dataset**")
    st.caption("Students with matching gender and similar characteristics")

    similar = df[
        (df['Gender'] == s_gender_n) &
        (df['CGPA_Numeric'].between(s_cgpa_num - 0.5, s_cgpa_num + 0.5)) &
        (df['Age'].between(s_age - 2, s_age + 2))
    ].copy()

    if len(similar) > 0:
        dep_rate_similar = similar['Depression'].mean() * 100
        anx_rate_similar = similar['Anxiety'].mean() * 100
        panic_rate_similar = similar['Panic_Attack'].mean() * 100

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Similar Students Found", str(len(similar)))
        sc2.metric("Their Depression Rate", f"{dep_rate_similar:.1f}%")
        sc3.metric("Their Anxiety Rate",    f"{anx_rate_similar:.1f}%")
        sc4.metric("Their Panic Rate",      f"{panic_rate_similar:.1f}%")

        st.write("")
        with st.expander(f"View {min(10,len(similar))} most similar students"):
            display_similar = similar[['Gender','Age','CGPA_Numeric','Anxiety',
                                       'Panic_Attack','Depression','Mental_Health_Score']
                                     ].head(10).copy()
            display_similar.columns = ['Gender','Age','CGPA','Anxiety',
                                        'Panic','Depression','MH Score']
            display_similar['Gender'] = display_similar['Gender'].map({1:'Male',0:'Female'})
            display_similar['Depression'] = display_similar['Depression'].map({1:'Yes',0:'No'})
            display_similar['Anxiety']    = display_similar['Anxiety'].map({1:'Yes',0:'No'})
            display_similar['Panic']      = display_similar['Panic'].map({1:'Yes',0:'No'})
            st.dataframe(display_similar.reset_index(drop=True),
                         width='stretch', hide_index=True)
    else:
        st.info("No similar students found with matching criteria. Try adjusting the filters.")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — LIVE INTERACTIVE CHARTS
# ══════════════════════════════════════════════════════════════
st.subheader("Interactive Dataset Explorer")
st.caption("Filter and explore the dataset live")

# Filters
fc1, fc2, fc3, fc4 = st.columns(4)
with fc1:
    f_gender = st.multiselect("Gender",
        options=["Male","Female"], default=["Male","Female"], key="f_gender")
with fc2:
    f_year = st.multiselect("Year of Study",
        options=["Year 1","Year 2","Year 3","Year 4"],
        default=["Year 1","Year 2","Year 3","Year 4"], key="f_year")
with fc3:
    f_dep = st.selectbox("Depression",
        ["All","Yes","No"], key="f_dep")
with fc4:
    f_anxiety = st.selectbox("Anxiety",
        ["All","Yes","No"], key="f_anxiety")

# Apply filters
fdf = df.copy()
if f_gender:
    g_map = {"Male":1,"Female":0}
    fdf = fdf[fdf['Gender'].isin([g_map[g] for g in f_gender])]
if "Year 1" not in f_year or "Year 2" not in f_year:
    yr_map = {"Year 1":"year 1","Year 2":"year 2","Year 3":"year 3","Year 4":"year 4"}
    selected_years = [yr_map[y] for y in f_year]
    if 'Year_of_Study' in fdf.columns:
        fdf = fdf[fdf['Year_of_Study'].str.lower().isin(selected_years)]
if f_dep != "All":
    fdf = fdf[fdf['Depression'] == (1 if f_dep=="Yes" else 0)]
if f_anxiety != "All":
    fdf = fdf[fdf['Anxiety'] == (1 if f_anxiety=="Yes" else 0)]

st.caption(f"Showing **{len(fdf)}** students after filtering")
st.write("")

# Charts row 1
ch1, ch2, ch3 = st.columns(3)

with ch1:
    fig1, ax1 = plt.subplots(figsize=(4,3.5))
    dep_counts = fdf['Depression'].value_counts()
    ax1.pie(dep_counts.values,
            labels=['No Depression','Depression'] if 0 in dep_counts.index else ['Depression'],
            colors=['#10B981','#EF4444'] if 0 in dep_counts.index else ['#EF4444'],
            autopct='%1.1f%%', startangle=90,
            wedgeprops={'edgecolor':'white','linewidth':2},
            textprops={'fontsize':10,'fontweight':'bold'})
    ax1.set_title('Depression Distribution\n(Filtered)', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig1, width='stretch'); plt.close()

with ch2:
    fig2, ax2 = plt.subplots(figsize=(4,3.5))
    mh_cats = ['Healthy\n(0)', 'Mild\n(1)', 'Moderate\n(2)', 'Severe\n(3)']
    mh_colors = ['#10B981','#F59E0B','#EF4444','#7C3AED']
    mh_counts = [
        (fdf['Mental_Health_Score']==0).sum(),
        (fdf['Mental_Health_Score']==1).sum(),
        (fdf['Mental_Health_Score']==2).sum(),
        (fdf['Mental_Health_Score']==3).sum(),
    ]
    bars2 = ax2.bar(mh_cats, mh_counts, color=mh_colors,
                    edgecolor='none', alpha=0.9)
    for bar, val in zip(bars2, mh_counts):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                 str(val), ha='center', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Number of Students')
    ax2.set_title('Mental Health Score\nDistribution', fontweight='bold')
    ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, width='stretch'); plt.close()

with ch3:
    fig3, ax3 = plt.subplots(figsize=(4,3.5))
    cgpa_dep0 = fdf[fdf['Depression']==0]['CGPA_Numeric']
    cgpa_dep1 = fdf[fdf['Depression']==1]['CGPA_Numeric']
    ax3.hist(cgpa_dep0, bins=8, alpha=0.7, color='#10B981',
             label='No Depression', edgecolor='white')
    ax3.hist(cgpa_dep1, bins=8, alpha=0.7, color='#EF4444',
             label='Depression', edgecolor='white')
    ax3.set_xlabel('CGPA (Numeric)')
    ax3.set_ylabel('Count')
    ax3.set_title('CGPA by Depression Status', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3, width='stretch'); plt.close()

# Charts row 2
ch4, ch5, ch6 = st.columns(3)

with ch4:
    fig4, ax4 = plt.subplots(figsize=(4,3.5))
    age_dep0 = fdf[fdf['Depression']==0]['Age']
    age_dep1 = fdf[fdf['Depression']==1]['Age']
    ax4.hist(age_dep0, bins=7, alpha=0.7, color='#10B981',
             label='No Depression', edgecolor='white')
    ax4.hist(age_dep1, bins=7, alpha=0.7, color='#EF4444',
             label='Depression', edgecolor='white')
    ax4.set_xlabel('Age'); ax4.set_ylabel('Count')
    ax4.set_title('Age by Depression Status', fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.spines['top'].set_visible(False); ax4.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig4, width='stretch'); plt.close()

with ch5:
    fig5, ax5 = plt.subplots(figsize=(4,3.5))
    conditions = ['Depression','Anxiety','Panic_Attack','Seek_Treatment']
    cond_labels = ['Depression','Anxiety','Panic\nAttack','Seek\nTreatment']
    rates_m = [fdf[fdf['Gender']==1][c].mean()*100 for c in conditions]
    rates_f = [fdf[fdf['Gender']==0][c].mean()*100 for c in conditions]
    x5 = np.arange(len(conditions)); w5 = 0.35
    ax5.bar(x5-w5/2, rates_m, w5, label='Male',   color='#3B82F6', alpha=0.85, edgecolor='none')
    ax5.bar(x5+w5/2, rates_f, w5, label='Female', color='#EC4899', alpha=0.85, edgecolor='none')
    ax5.set_xticks(x5); ax5.set_xticklabels(cond_labels, fontsize=8)
    ax5.set_ylabel('Rate (%)'); ax5.set_ylim(0,100)
    ax5.set_title('Conditions by Gender', fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.spines['top'].set_visible(False); ax5.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5, width='stretch'); plt.close()

with ch6:
    fig6, ax6 = plt.subplots(figsize=(4,3.5))
    corr_feats = ['Age','CGPA_Numeric','Gender','Anxiety','Panic_Attack',
                  'Marital_Status','Seek_Treatment']
    corr_labels = ['Age','CGPA','Gender','Anxiety','Panic','Marital','Treatment']
    corr_vals = [fdf[f].corr(fdf['Depression']) for f in corr_feats]
    colors6 = ['#EF4444' if v > 0 else '#10B981' for v in corr_vals]
    sorted_pairs = sorted(zip(corr_vals, corr_labels, colors6))
    ax6.barh([p[1] for p in sorted_pairs],
              [p[0] for p in sorted_pairs],
              color=[p[2] for p in sorted_pairs],
              edgecolor='none', height=0.6)
    ax6.axvline(0, color='black', lw=1)
    ax6.set_xlabel('Correlation with Depression')
    ax6.set_title('Feature Correlation\n(Filtered Data)', fontweight='bold')
    ax6.spines['top'].set_visible(False); ax6.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig6, width='stretch'); plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — LIVE PREDICTION TRACKER
# ══════════════════════════════════════════════════════════════
st.subheader("Live Prediction Session Tracker")
st.caption("Tracks all students you've compared this session")

if len(st.session_state.live_predictions) == 0:
    st.info("No comparisons yet. Use the form above to compare a student to the dataset.")
else:
    preds_df = pd.DataFrame(st.session_state.live_predictions)
    total_p  = len(preds_df)

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Students Compared", str(total_p))
    pm2.metric("Avg Age Compared",  f"{preds_df['Age'].mean():.1f}")
    pm3.metric("Avg CGPA Compared", f"{preds_df['CGPA'].mean():.2f}")
    pm4.metric("With Anxiety",
               f"{int(preds_df['Anxiety'].sum())} / {total_p}")

    st.write("")
    st.dataframe(preds_df.rename(columns={
        'Name':'Student','Age':'Age','CGPA':'CGPA',
        'Anxiety':'Anxiety (1=Yes)','Panic':'Panic (1=Yes)',
        'Gender':'Gender (1=Male)'
    }), width='stretch', hide_index=True)

    if st.button("Clear Session History", key="ls_clear"):
        st.session_state.live_predictions = []
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — RAW DATA EXPLORER
# ══════════════════════════════════════════════════════════════
st.subheader("Raw Dataset Explorer")
st.caption(f"Browse and filter the full {len(df)}-record dataset")

col_show = st.multiselect(
    "Select columns to display",
    options=list(df.columns),
    default=['Gender','Age','Course','Year_of_Study','CGPA_Numeric',
             'Depression','Anxiety','Panic_Attack','Mental_Health_Score'],
    key="raw_cols"
)

sort_col = st.selectbox("Sort by", options=col_show if col_show else ['Depression'],
                         key="raw_sort")
sort_asc = st.radio("Sort order", ["Ascending","Descending"],
                    horizontal=True, key="raw_asc")

if col_show:
    raw_display = df[col_show].sort_values(
        sort_col, ascending=(sort_asc=="Ascending"))
    st.dataframe(raw_display.reset_index(drop=True),
                 width='stretch', hide_index=True,
                 height=400)
    st.caption(f"Showing all {len(raw_display)} records")

    # Download
    st.download_button(
        "⬇️  Download Filtered Dataset",
        data=raw_display.to_csv(index=False).encode(),
        file_name="mindcheck_filtered_dataset.csv",
        mime="text/csv",
        width='stretch',
    )

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")