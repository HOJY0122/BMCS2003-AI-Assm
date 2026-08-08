import streamlit as st

st.set_page_config(
    page_title="Student Mental Health Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f9f9f9; }

    /* Title styling */
    h1 { color: #1a1a2e; font-family: Arial, sans-serif; }
    h2 { color: #16213e; font-family: Arial, sans-serif; }
    h3 { color: #0f3460; font-family: Arial, sans-serif; }

    /* Card styling */
    .card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 4px solid #0f3460;
        margin-bottom: 10px;
    }

    /* Member card */
    .member-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-top: 4px solid #0f3460;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Section divider */
    .section-title {
        font-size: 18px;
        font-weight: bold;
        color: #0f3460;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #1a1a2e; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }

    /* Table styling */
    .info-table {
        width: 100%;
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }
    .info-table th {
        background-color: #0f3460;
        color: white;
        padding: 10px 14px;
        text-align: left;
    }
    .info-table td {
        padding: 9px 14px;
        border-bottom: 1px solid #e0e0e0;
    }
    .info-table tr:nth-child(even) { background-color: #f5f7fa; }
    .info-table tr:hover { background-color: #eef2f7; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────
st.sidebar.markdown("## Student Mental Health")
st.sidebar.markdown("Supervised Machine Learning System")
st.sidebar.markdown("---")
st.sidebar.markdown("**Navigation**")
st.sidebar.markdown("""
- Home  
- Exploratory Data Analysis (EDA)  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Support Vector Machine (SVM)  
- Model Comparison  
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Subject:** BMCS2003 Artificial Intelligence")
st.sidebar.markdown("**Session:** 202605 / Year 2026/27")
st.sidebar.markdown("**Tutor:** Dr Goh")

# ── Page Title ─────────────────────────────────────────────────
st.title("Student Mental Health Prediction System")
st.markdown("Supervised Machine Learning | BMCS2003 Artificial Intelligence")
st.markdown("---")

# ── Section 1: Overview ────────────────────────────────────────
st.markdown('<div class="section-title">Project Overview</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
        <b>Problem Statement</b><br><br>
        Mental health issues among university students are increasing globally.
        Many students do not seek help early due to stigma and lack of awareness.
        This system uses Machine Learning to predict mental health conditions
        based on student demographic and academic data, enabling early detection
        and timely intervention.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
        <b>Objectives</b><br><br>
        1. Predict Depression, Anxiety, and Panic Attack among students.<br><br>
        2. Implement and compare KNN, Decision Tree, and SVM algorithms.<br><br>
        3. Evaluate each model using Accuracy, Precision, Recall, and F1 Score.<br><br>
        4. Deploy an interactive prediction system using Streamlit.
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
        <b>Dataset Information</b><br><br>
        Source: Kaggle (Shariful07, 2020)<br><br>
        Total Records: 101 students<br><br>
        Original Features: 11 columns<br><br>
        University: IIUM Malaysia<br><br>
        Type: Classification (Supervised Learning)
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 2: Dataset Summary ────────────────────────────────
st.markdown('<div class="section-title">Dataset Summary</div>', unsafe_allow_html=True)

st.markdown("""
<table class="info-table">
    <tr>
        <th>Feature</th>
        <th>Description</th>
        <th>Type</th>
    </tr>
    <tr><td>Gender</td><td>Student gender (Male / Female)</td><td>Categorical</td></tr>
    <tr><td>Age</td><td>Student age (18 - 24)</td><td>Numerical</td></tr>
    <tr><td>Course</td><td>Field of study</td><td>Categorical</td></tr>
    <tr><td>Year of Study</td><td>Current year in university (Year 1 - 4)</td><td>Categorical</td></tr>
    <tr><td>CGPA</td><td>Current cumulative GPA range</td><td>Categorical</td></tr>
    <tr><td>Marital Status</td><td>Marital status of student</td><td>Categorical</td></tr>
    <tr><td>Depression</td><td>Whether student has depression (Yes / No)</td><td>Target (KNN)</td></tr>
    <tr><td>Anxiety</td><td>Whether student has anxiety (Yes / No)</td><td>Target (Decision Tree)</td></tr>
    <tr><td>Panic Attack</td><td>Whether student has panic attacks (Yes / No)</td><td>Target (SVM)</td></tr>
    <tr><td>Seek Treatment</td><td>Whether student sought specialist help (Yes / No)</td><td>Categorical</td></tr>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 3: Group Members ──────────────────────────────────
st.markdown('<div class="section-title">Group Members</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="member-card">
        <b>Member 1</b><br>
        Ho Jun Yon<br>
        Student ID: 2612634<br><br>
        <hr style="border: 0.5px solid #e0e0e0;">
        Algorithm: K-Nearest Neighbor (KNN)<br><br>
        Target: Depression Prediction<br><br>
        Encoding: Label Encoding<br>
        Scaling: MinMax Scaler<br>
        Split: 80% Train / 20% Test
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="member-card">
        <b>Member 2</b><br>
        Irvin Tan Wei Shen<br>
        Student ID: 2612638<br><br>
        <hr style="border: 0.5px solid #e0e0e0;">
        Algorithm: Decision Tree<br><br>
        Target: Anxiety Prediction<br><br>
        Encoding: One-Hot Encoding<br>
        Scaling: No Scaling<br>
        Split: 70% Train / 30% Test
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="member-card">
        <b>Member 3</b><br>
        Chiang Jun Hang<br>
        Student ID: 2612610<br><br>
        <hr style="border: 0.5px solid #e0e0e0;">
        Algorithm: Support Vector Machine (SVM)<br><br>
        Target: Panic Attack Prediction<br><br>
        Encoding: Label Encoding<br>
        Scaling: Standard Scaler<br>
        Split: 75% Train / 25% Test
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 4: System Flow ────────────────────────────────────
st.markdown('<div class="section-title">System Pipeline</div>', unsafe_allow_html=True)

s1, s2, s3, s4, s5 = st.columns(5)

steps = [
    ("1. Data Collection",   "Obtain dataset from Kaggle (101 student records)"),
    ("2. Preprocessing",     "Clean, encode, scale and engineer features"),
    ("3. Model Training",    "Train KNN, Decision Tree, and SVM separately"),
    ("4. Evaluation",        "Measure Accuracy, Precision, Recall, F1 Score"),
    ("5. Deployment",        "Deploy prediction system via Streamlit"),
]

for col, (title, desc) in zip([s1, s2, s3, s4, s5], steps):
    with col:
        st.markdown(f"""
        <div style="background:#0f3460; color:white; border-radius:8px;
                    padding:14px; text-align:center; min-height:120px;">
            <b>{title}</b><br><br>
            <span style="font-size:13px;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Section 5: Algorithm Comparison Table ─────────────────────
st.markdown('<div class="section-title">Algorithm Overview</div>', unsafe_allow_html=True)

st.markdown("""
<table class="info-table">
    <tr>
        <th>Member</th>
        <th>Algorithm</th>
        <th>Target</th>
        <th>Encoding</th>
        <th>Scaling</th>
        <th>Train/Test Split</th>
    </tr>
    <tr>
        <td>Ho Jun Yon (Member 1)</td>
        <td>K-Nearest Neighbor (KNN)</td>
        <td>Depression</td>
        <td>Label Encoding</td>
        <td>MinMax Scaler</td>
        <td>80% / 20%</td>
    </tr>
    <tr>
        <td>Irvin Tan Wei Shen (Member 2)</td>
        <td>Decision Tree</td>
        <td>Anxiety</td>
        <td>One-Hot Encoding</td>
        <td>No Scaling</td>
        <td>70% / 30%</td>
    </tr>
    <tr>
        <td>Chiang Jun Hang (Member 3)</td>
        <td>Support Vector Machine (SVM)</td>
        <td>Panic Attack</td>
        <td>Label Encoding</td>
        <td>Standard Scaler</td>
        <td>75% / 25%</td>
    </tr>
</table>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:13px;'>"
    "BMCS2003 Artificial Intelligence | 202605 Session | Tutorial Group 3 | Tutor: Dr Goh"
    "</p>",
    unsafe_allow_html=True
)