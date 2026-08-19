import streamlit as st
import pandas as pd
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import sidebar

st.set_page_config(
    page_title="Dataset — MindCheck",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("dataset")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
html, body, [data-testid="stAppViewContainer"] { background: #F0F4FF !important; font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] > .main { background: #F0F4FF !important; padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="collapsedControl"] { display: flex !important; visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] { display: flex !important; visibility: visible !important; }
[data-testid="stSidebar"] { background: #1A1D2E !important; }
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebarNav"] { display: none !important; }
.stButton > button {
    background: #3D52FF !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 12px 20px !important; width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #2A3ECC !important; transform: translateY(-2px) !important; }
div[data-testid="metric-container"] {
    background: white !important; border: 1px solid #E2E8FF !important;
    border-radius: 14px !important; padding: 20px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: all 0.2s !important;
}
div[data-testid="metric-container"]:hover { box-shadow: 0 8px 24px rgba(61,82,255,0.1) !important; transform: translateY(-2px) !important; }
div[data-testid="metric-container"] label { font-size: 11px !important; color: #9CA3AF !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
div[data-testid="metric-container"] [data-testid="metric-value"] { font-family: 'Space Grotesk', sans-serif !important; font-size: 24px !important; font-weight: 700 !important; color: #3D52FF !important; }
div[data-testid="stExpander"] { border: 1px solid #E2E8FF !important; border-radius: 12px !important; background: white !important; overflow: hidden !important; margin-bottom: 8px !important; }
div[data-testid="stExpander"] summary { font-weight: 600 !important; font-size: 14px !important; color: #1A1D2E !important; padding: 14px 18px !important; }
div[data-testid="stExpander"]:hover { box-shadow: 0 4px 16px rgba(61,82,255,0.08) !important; }
.page-header { background: white; padding: 40px 60px 32px; border-bottom: 1px solid #E2E8FF; }
.page-label { font-size: 11px; font-weight: 700; color: #3D52FF; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
.page-title { font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 700; color: #1A1D2E; letter-spacing: -0.8px; margin-bottom: 8px; }
.page-sub { font-size: 15px; color: #6B7280; line-height: 1.6; }
.content-wrap { padding: 40px 60px; }
.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 20px; font-weight: 700; color: #1A1D2E; margin-bottom: 16px; padding-bottom: 10px; border-bottom: 2px solid #E2E8FF; }
.ds-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.ds-table th { background: #1A1D2E; color: white; padding: 12px 16px; text-align: left; font-weight: 600; font-size: 12px; }
.ds-table th:first-child { border-radius: 8px 0 0 0; }
.ds-table th:last-child { border-radius: 0 8px 0 0; }
.ds-table td { padding: 11px 16px; border-bottom: 1px solid #F0F4FF; color: #374151; vertical-align: middle; }
.ds-table tr:hover td { background: #F8F9FF; }
.ds-table tr:last-child td { border-bottom: none; }
.tag-target { background: #DCFCE7; color: #15803D; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
.tag-input { background: #EEF1FF; color: #3D52FF; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
.info-box { background: white; border-radius: 14px; padding: 24px; border: 1px solid #E2E8FF; margin-bottom: 16px; }
.step-row { display: flex; gap: 16px; align-items: flex-start; padding: 14px 0; border-bottom: 1px solid #F0F4FF; }
.step-num { width: 28px; height: 28px; background: #3D52FF; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 700; color: white; flex-shrink: 0; }
.step-title { font-size: 14px; font-weight: 600; color: #1A1D2E; margin-bottom: 3px; }
.step-desc { font-size: 13px; color: #6B7280; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# Sidebar handled by shared sidebar.py

# Header
st.markdown("""
<div class="page-header">
    <div class="page-label">Dataset</div>
    <div class="page-title">Student Mental Health Dataset</div>
    <div class="page-sub">Source: Kaggle — Shariful07 (2020) | 600 student records | IIUM Malaysia | 11 original features</div>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def get_data():
    return load_and_clean_dataset('dataset/Student_Mental_health.csv')

df = get_data()

st.markdown("<div class='content-wrap'>", unsafe_allow_html=True)

# Metrics
m1,m2,m3,m4,m5,m6 = st.columns(6)
m1.metric("Total Records", "600")
m2.metric("Original Features", "11")
m3.metric("Depression (Yes)", "194 (32%)")
m4.metric("Anxiety (Yes)", "209 (35%)")
m5.metric("Panic Attack (Yes)", "190 (32%)")
m6.metric("Missing Values", "8 (Age)")

st.markdown("<br>", unsafe_allow_html=True)

# Feature table
st.markdown("<div class='section-title'>Feature Description</div>", unsafe_allow_html=True)
with st.expander("View All Features", expanded=True):
    st.markdown("""
    <table class="ds-table">
    <tr><th>#</th><th>Feature Name</th><th>Original Column</th><th>Description</th><th>Values</th><th>Role</th></tr>
    <tr><td>1</td><td><b>Gender</b></td><td>Choose your gender</td><td>Student gender</td><td>Male / Female → 1 / 0</td><td><span class="tag-input">Input</span></td></tr>
    <tr><td>2</td><td><b>Age</b></td><td>Age</td><td>Student age</td><td>17–24 (numeric)</td><td><span class="tag-input">Input</span></td></tr>
    <tr><td>3</td><td><b>Course</b></td><td>What is your course?</td><td>Field of study (grouped)</td><td>10 categories</td><td><span class="tag-input">Input</span></td></tr>
    <tr><td>4</td><td><b>Year_of_Study</b></td><td>Your current year of Study</td><td>Academic year</td><td>Year 1–4</td><td><span class="tag-input">Input</span></td></tr>
    <tr><td>5</td><td><b>CGPA</b></td><td>What is your CGPA?</td><td>GPA range</td><td>0–1.99 to 3.50–4.00</td><td><span class="tag-input">Input</span></td></tr>
    <tr><td>6</td><td><b>Marital_Status</b></td><td>Marital status</td><td>Marital status</td><td>Yes / No → 1 / 0</td><td><span class="tag-input">Input</span></td></tr>
    <tr><td>7</td><td><b>Depression</b></td><td>Do you have Depression?</td><td>Has depression?</td><td>Yes / No → 1 / 0</td><td><span class="tag-target">Target (KNN & DT)</span></td></tr>
    <tr><td>8</td><td><b>Anxiety</b></td><td>Do you have Anxiety?</td><td>Has anxiety?</td><td>Yes / No → 1 / 0</td><td><span class="tag-input">Input</span></td></tr>
    <tr><td>9</td><td><b>Panic_Attack</b></td><td>Do you have Panic attack?</td><td>Has panic attacks?</td><td>Yes / No → 1 / 0</td><td><span class="tag-target">Target (SVM)</span></td></tr>
    <tr><td>10</td><td><b>Seek_Treatment</b></td><td>Did you seek any specialist...</td><td>Sought specialist help?</td><td>Yes / No → 1 / 0</td><td><span class="tag-input">Input</span></td></tr>
    </table>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Engineered features
st.markdown("<div class='section-title'>Engineered Features</div>", unsafe_allow_html=True)
e1, e2, e3 = st.columns(3)
with e1:
    st.markdown("""
    <div class="info-box">
        <div style="font-size:13px;font-weight:700;color:#3D52FF;margin-bottom:10px;">Mental Health Score</div>
        <div style="font-size:13px;color:#6B7280;line-height:1.7;">
            <b>Formula:</b> Depression + Anxiety + Panic_Attack<br>
            <b>Range:</b> 0 to 3<br>
            <b>0</b> = Healthy<br>
            <b>1</b> = Mild<br>
            <b>2</b> = Moderate<br>
            <b>3</b> = Severe
        </div>
    </div>
    """, unsafe_allow_html=True)
with e2:
    st.markdown("""
    <div class="info-box">
        <div style="font-size:13px;font-weight:700;color:#10B981;margin-bottom:10px;">Age Group</div>
        <div style="font-size:13px;color:#6B7280;line-height:1.7;">
            Groups student age into categories:<br><br>
            <b>Early</b> = Age 18–19<br>
            <b>Mid</b> = Age 20–21<br>
            <b>Senior</b> = Age 22+
        </div>
    </div>
    """, unsafe_allow_html=True)
with e3:
    st.markdown("""
    <div class="info-box">
        <div style="font-size:13px;font-weight:700;color:#F59E0B;margin-bottom:10px;">CGPA Numeric</div>
        <div style="font-size:13px;color:#6B7280;line-height:1.7;">
            Converts CGPA range to midpoint:<br><br>
            0–1.99 → <b>1.00</b><br>
            2.00–2.49 → <b>2.25</b><br>
            2.50–2.99 → <b>2.75</b><br>
            3.00–3.49 → <b>3.25</b><br>
            3.50–4.00 → <b>3.75</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Preprocessing steps
st.markdown("<div class='section-title'>Preprocessing Steps Applied</div>", unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <div class="step-row">
        <div class="step-num">1</div>
        <div><div class="step-title">Column Renaming</div><div class="step-desc">All 11 column names standardized to short machine-readable identifiers. Example: "Choose your gender" → "Gender", "What is your CGPA?" → "CGPA".</div></div>
    </div>
    <div class="step-row">
        <div class="step-num">2</div>
        <div><div class="step-title">Missing Value Handling</div><div class="step-desc">8 missing values in the Age column were filled with the median age value (19 years) to preserve dataset integrity without removing rows.</div></div>
    </div>
    <div class="step-row">
        <div class="step-num">3</div>
        <div><div class="step-title">Inconsistency Correction</div><div class="step-desc">48 variations of course names were grouped into 10 standard categories (e.g., "KOE", "koe", "Koe" → "Engineering"). Year of study inconsistencies ("year 1" vs "Year 1") were standardized.</div></div>
    </div>
    <div class="step-row">
        <div class="step-num">4</div>
        <div><div class="step-title">Binary Encoding</div><div class="step-desc">All Yes/No columns (Depression, Anxiety, Panic Attack, Seek Treatment, Marital Status) and Gender were encoded as binary integers: 1 = Yes/Male, 0 = No/Female.</div></div>
    </div>
    <div class="step-row" style="border-bottom:none;">
        <div class="step-num">5</div>
        <div><div class="step-title">Feature Engineering</div><div class="step-desc">Three new derived features were created: Mental Health Score (sum of 3 conditions, range 0–3), Age Group (Early/Mid/Senior), and CGPA Numeric (midpoint value of CGPA range).</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Dataset preview
st.markdown("<div class='section-title'>Cleaned Dataset Preview</div>", unsafe_allow_html=True)
st.dataframe(df.head(20), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Basic stats
st.markdown("<div class='section-title'>Statistical Summary</div>", unsafe_allow_html=True)
st.dataframe(df.describe().style.format("{:.2f}"), use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# Navigate
n1, n2, _ = st.columns([1, 1, 4])
with n1:
    if st.button("Explore EDA Visualizations"):
        st.switch_page("pages/1_EDA.py")
with n2:
    if st.button("Back to Home"):
        st.switch_page("Home.py")

st.markdown("</div>", unsafe_allow_html=True)
