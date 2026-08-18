import streamlit as st

st.set_page_config(
    page_title="MindCheck — Student Mental Health",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background: #F0F4FF !important;
    font-family: 'Inter', sans-serif; color: #1A1D2E;
}
[data-testid="stAppViewContainer"] > .main { background: #F0F4FF !important; padding: 0 !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stButton > button {
    background: #3D52FF !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 12px 24px !important;
    transition: all 0.2s !important; width: 100% !important;
}
.stButton > button:hover {
    background: #2A3ECC !important; transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(61,82,255,0.35) !important;
}
div[data-testid="metric-container"] {
    background: white !important; border: 1px solid #E2E8FF !important;
    border-radius: 14px !important; padding: 20px 24px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04) !important;
    transition: all 0.2s !important;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 8px 24px rgba(61,82,255,0.1) !important;
    transform: translateY(-2px) !important; border-color: #C7D0FF !important;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important; color: #9CA3AF !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 24px !important; font-weight: 700 !important; color: #3D52FF !important;
}
div[data-testid="stExpander"] {
    border: 1px solid #E2E8FF !important; border-radius: 12px !important;
    background: white !important; overflow: hidden !important;
    margin-bottom: 8px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
    transition: box-shadow 0.2s !important;
}
div[data-testid="stExpander"]:hover { box-shadow: 0 4px 16px rgba(61,82,255,0.08) !important; }
div[data-testid="stExpander"] summary {
    font-weight: 600 !important; font-size: 14px !important;
    color: #1A1D2E !important; padding: 14px 18px !important;
}
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #E8EDFF !important; border-radius: 12px !important;
    padding: 4px !important; gap: 4px !important; border: 1px solid #D0D9FF !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px !important; font-weight: 500 !important;
    font-size: 14px !important; color: #6B7280 !important;
    padding: 8px 22px !important; background: transparent !important; transition: all 0.2s !important;
}
div[data-testid="stTabs"] [aria-selected="true"] {
    background: white !important; color: #3D52FF !important;
    font-weight: 600 !important; box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
}
hr { border: none !important; border-top: 1px solid #E2E8FF !important; margin: 0 !important; }
.hero {
    background: linear-gradient(135deg, #1A1D2E 0%, #2D3278 60%, #1A1D2E 100%);
    padding: 90px 60px 32px; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute;
    top: -80px; right: -80px; width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(61,82,255,0.25) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-inner { position: relative; z-index: 1; }
.hero-tag {
    display: inline-block; background: rgba(61,82,255,0.18);
    color: #93A3FF; font-size: 11px; font-weight: 700;
    padding: 6px 14px; border-radius: 20px;
    border: 1px solid rgba(61,82,255,0.3); margin-bottom: 20px;
    letter-spacing: 1.5px; text-transform: uppercase;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 50px; font-weight: 700; color: #FFFFFF;
    line-height: 1.15; letter-spacing: -1.5px;
    max-width: 660px; margin-bottom: 20px;
}
.hero-title span {
    background: linear-gradient(135deg, #6B86FF, #93CFFF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    font-size: 16px; color: rgba(255,255,255,0.6);
    max-width: 500px; line-height: 1.75; margin-bottom: 28px;
}
.stats-bar {
    background: white; border-bottom: 1px solid #E2E8FF;
    padding: 22px 60px; display: flex; gap: 44px;
    align-items: center; flex-wrap: wrap;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.stat-item { text-align: center; min-width: 70px; }
.stat-number {
    font-family: 'Space Grotesk', sans-serif; font-size: 24px;
    font-weight: 700; color: #3D52FF; line-height: 1; margin-bottom: 3px;
}
.stat-label { font-size: 11px; color: #9CA3AF; }
.stat-divider { width: 1px; height: 32px; background: #E2E8FF; flex-shrink: 0; }
.section-wrap { padding: 64px 60px; }
.section-wrap-alt { padding: 64px 60px; background: white; }
.section-label {
    font-size: 11px; font-weight: 700; color: #3D52FF;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif; font-size: 32px;
    font-weight: 700; color: #1A1D2E; letter-spacing: -0.8px;
    margin-bottom: 10px; line-height: 1.2;
}
.section-sub { font-size: 15px; color: #6B7280; max-width: 520px; line-height: 1.7; }
.cards-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 32px; }
.info-card {
    background: white; border-radius: 14px; padding: 24px;
    border: 1px solid #E2E8FF; transition: all 0.25s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
}
.info-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #3D52FF, #6B86FF);
    transform: scaleX(0); transition: transform 0.3s; transform-origin: left;
}
.info-card:hover { box-shadow: 0 12px 40px rgba(61,82,255,0.1); transform: translateY(-4px); border-color: #C7D0FF; }
.info-card:hover::after { transform: scaleX(1); }
.card-icon { width: 40px; height: 40px; background: #EEF1FF; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 12px; }
.card-title { font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 600; color: #1A1D2E; margin-bottom: 7px; }
.card-text { font-size: 13px; color: #6B7280; line-height: 1.65; }
.pipeline { display: flex; gap: 0; margin-top: 36px; position: relative; }
.pipeline::before { content: ''; position: absolute; top: 26px; left: 44px; right: 44px; height: 2px; background: linear-gradient(90deg, #3D52FF, #93CFFF); z-index: 0; }
.pipe-step { flex: 1; text-align: center; position: relative; z-index: 1; padding: 0 10px; }
.pipe-num { width: 52px; height: 52px; background: linear-gradient(135deg, #3D52FF, #6B86FF); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-family: 'Space Grotesk', sans-serif; font-size: 16px; font-weight: 700; color: white; margin: 0 auto 12px; box-shadow: 0 4px 16px rgba(61,82,255,0.3); }
.pipe-title { font-size: 13px; font-weight: 600; color: #1A1D2E; margin-bottom: 4px; }
.pipe-desc { font-size: 11px; color: #9CA3AF; line-height: 1.5; }
.algo-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 32px; }
.algo-card { border-radius: 16px; padding: 24px; border: 1px solid transparent; transition: all 0.25s cubic-bezier(.4,0,.2,1); }
.algo-card:hover { transform: translateY(-5px); box-shadow: 0 16px 48px rgba(0,0,0,0.08); }
.algo-1 { background: linear-gradient(135deg, #EEF1FF, #F8F9FF); border-color: #C7D0FF; }
.algo-2 { background: linear-gradient(135deg, #ECFDF5, #F8FFFC); border-color: #A7F3D0; }
.algo-3 { background: linear-gradient(135deg, #FFFBEB, #FFFFF8); border-color: #FDE68A; }
.algo-name { font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700; margin-bottom: 2px; }
.algo-1 .algo-name { color: #3D52FF; }
.algo-2 .algo-name { color: #059669; }
.algo-3 .algo-name { color: #D97706; }
.algo-member { font-size: 12px; color: #9CA3AF; margin-bottom: 16px; }
.algo-acc { font-family: 'Space Grotesk', sans-serif; font-size: 36px; font-weight: 700; line-height: 1; margin: 0 0 2px; }
.algo-1 .algo-acc { color: #3D52FF; }
.algo-2 .algo-acc { color: #059669; }
.algo-3 .algo-acc { color: #D97706; }
.algo-acc-label { font-size: 11px; color: #9CA3AF; margin-bottom: 16px; }
.algo-metric { display: flex; justify-content: space-between; align-items: center; padding: 7px 0; border-bottom: 1px solid rgba(0,0,0,0.05); font-size: 13px; }
.algo-metric:last-child { border-bottom: none; }
.algo-metric-label { color: #6B7280; }
.algo-metric-value { font-weight: 700; color: #1A1D2E; font-size: 13px; }
.member-card { background: white; border-radius: 14px; overflow: hidden; border: 1px solid #E2E8FF; transition: all 0.25s cubic-bezier(.4,0,.2,1); }
.member-card:hover { box-shadow: 0 12px 40px rgba(61,82,255,0.1); transform: translateY(-4px); border-color: #C7D0FF; }
.member-header { padding: 22px 22px 16px; border-bottom: 1px solid #F3F4FF; }
.member-avatar { width: 48px; height: 48px; border-radius: 12px; font-family: 'Space Grotesk', sans-serif; font-size: 18px; font-weight: 700; color: white; display: flex; align-items: center; justify-content: center; margin-bottom: 12px; }
.avatar-1 { background: linear-gradient(135deg, #3D52FF, #6B86FF); }
.avatar-2 { background: linear-gradient(135deg, #10B981, #34D399); }
.avatar-3 { background: linear-gradient(135deg, #F59E0B, #FBBF24); }
.member-name { font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 600; color: #1A1D2E; margin-bottom: 2px; }
.member-id { font-size: 12px; color: #9CA3AF; }
.member-body { padding: 16px 22px 20px; }
.member-algo { display: inline-block; background: #EEF1FF; color: #3D52FF; font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 20px; margin-bottom: 12px; }
.member-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #F3F4FF; font-size: 13px; }
.member-row:last-child { border-bottom: none; }
.member-row-label { color: #9CA3AF; }
.member-row-value { font-weight: 500; color: #1A1D2E; }
.ds-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
.ds-table th { background: #1A1D2E; color: white; padding: 11px 14px; text-align: left; font-weight: 600; font-size: 12px; }
.ds-table th:first-child { border-radius: 8px 0 0 0; }
.ds-table th:last-child { border-radius: 0 8px 0 0; }
.ds-table td { padding: 10px 14px; border-bottom: 1px solid #F0F4FF; color: #374151; }
.ds-table tr:hover td { background: #F8F9FF; }
.ds-table tr:last-child td { border-bottom: none; }
.tag-target { background: #DCFCE7; color: #15803D; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
.tag-input { background: #EEF1FF; color: #3D52FF; font-size: 11px; font-weight: 600; padding: 2px 8px; border-radius: 4px; }
.footer { background: #1A1D2E; padding: 40px 60px; display: flex; justify-content: space-between; align-items: center; }
.footer-brand { font-family: 'Space Grotesk', sans-serif; font-size: 19px; font-weight: 700; color: white; }
.footer-brand span { color: #6B86FF; }
.footer-text { font-size: 12px; color: rgba(255,255,255,0.35); margin-top: 4px; }
.footer-right { text-align: right; font-size: 12px; color: rgba(255,255,255,0.35); line-height: 1.9; }
.members-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-top: 32px; }
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-tag">Supervised Machine Learning · BMCS2003 AI · TARUMT</div>
    <h1 class="hero-title">Predict Student<br><span>Mental Health</span><br>with AI</h1>
    <p class="hero-sub">An intelligent system that analyses student demographics and academic data to predict depression and panic attack risks — enabling early intervention and timely support.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Hero buttons
hb1, hb2, hb3, hb4, _ = st.columns([1.1, 1.1, 1.1, 1.1, 3])
with hb1:
    if st.button("Try KNN Prediction"):
        st.switch_page("pages/2_KNN.py")
with hb2:
    if st.button("Explore EDA"):
        st.switch_page("pages/1_EDA.py")
with hb3:
    if st.button("Decision Tree"):
        st.switch_page("pages/3_Decision_Tree.py")
with hb4:
    if st.button("Compare Models"):
        st.switch_page("pages/5_Comparison.py")

# Stats bar
st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><div class="stat-number">600</div><div class="stat-label">Student Records</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-number">95.83%</div><div class="stat-label">Best Accuracy</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-number">97.44%</div><div class="stat-label">Best Recall</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-number">3</div><div class="stat-label">ML Algorithms</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-number">11</div><div class="stat-label">Features</div></div>
    <div class="stat-divider"></div>
    <div class="stat-item"><div class="stat-number">IIUM</div><div class="stat-label">Dataset Source</div></div>
</div>
""", unsafe_allow_html=True)

# Main tabs
st.markdown("<div style='padding: 36px 60px 0;'>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["About", "Dataset", "Results", "Team"])
st.markdown("</div>", unsafe_allow_html=True)

with tab1:
    st.markdown("""
    <div class="section-wrap-alt">
        <div class="section-label">About This Project</div>
        <div class="section-title">What is MindCheck?</div>
        <div class="section-sub">A Supervised ML system that predicts student mental health conditions using demographic and academic data, deployed as a Streamlit web app.</div>
        <div class="cards-grid">
            <div class="info-card"><div class="card-icon">🎯</div><div class="card-title">Problem Statement</div><div class="card-text">Mental health issues among university students are increasing. Many students don't seek help early due to stigma. This system uses AI for early screening and detection.</div></div>
            <div class="info-card"><div class="card-icon">📊</div><div class="card-title">Our Approach</div><div class="card-text">We implement and compare KNN, Decision Tree, and SVM on 600 IIUM student records. Each member independently builds and evaluates a different algorithm.</div></div>
            <div class="info-card"><div class="card-icon">🚀</div><div class="card-title">Real-World Impact</div><div class="card-text">The system gives counsellors an early-warning tool to flag at-risk students using academic and demographic features, enabling timely mental health support.</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 0 60px 20px; background:white;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>Objectives</div>", unsafe_allow_html=True)
    for i, (t, d) in enumerate([
        ("Predict Mental Health Conditions", "Develop a system to predict depression, anxiety, and panic attack among university students using demographic and academic features."),
        ("Compare Three Algorithms", "Implement and compare KNN, Decision Tree (CART), and SVM — each by a different group member — on the same dataset."),
        ("Rigorous Evaluation", "Evaluate each model using Accuracy, Precision, Recall, F1 Score, Confusion Matrix and 5-Fold Cross Validation."),
        ("Deploy Interactive Prototype", "Build and deploy a multi-page Streamlit web app with live prediction forms and EDA visualizations."),
    ], 1):
        with st.expander(f"Objective {i}: {t}"):
            st.markdown(f"<p style='font-size:14px;color:#6B7280;line-height:1.7;'>{d}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="section-wrap">
        <div class="section-label">System Flow</div>
        <div class="section-title">How the System Works</div>
        <div class="section-sub">Five clear stages from raw data to real-time prediction.</div>
        <div class="pipeline">
            <div class="pipe-step"><div class="pipe-num">1</div><div class="pipe-title">Data Collection</div><div class="pipe-desc">600 records from Kaggle IIUM student survey</div></div>
            <div class="pipe-step"><div class="pipe-num">2</div><div class="pipe-title">Preprocessing</div><div class="pipe-desc">Clean, encode, scale and engineer features</div></div>
            <div class="pipe-step"><div class="pipe-num">3</div><div class="pipe-title">Model Training</div><div class="pipe-desc">KNN (K=5), Decision Tree (depth 5), SVM (RBF)</div></div>
            <div class="pipe-step"><div class="pipe-num">4</div><div class="pipe-title">Evaluation</div><div class="pipe-desc">Accuracy, Precision, Recall, F1, CV</div></div>
            <div class="pipe-step"><div class="pipe-num">5</div><div class="pipe-title">Deployment</div><div class="pipe-desc">Interactive Streamlit web app</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='padding: 0 60px 56px; background:white;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>FAQ</div><div class='section-title'>Common Questions</div><br>", unsafe_allow_html=True)
    for q, a in [
        ("What is KNN?", "KNN classifies a student by finding the K most similar students in training data and taking majority vote. K=5 was selected after testing K=1 to K=20. Distance uses Euclidean metric after MinMax scaling."),
        ("What is Decision Tree (CART)?", "CART splits data based on the feature with highest Gini impurity reduction at each node. Our tree has max depth 5, with Marital Status as root split — the most discriminative feature."),
        ("What is SVM?", "SVM finds the optimal hyperplane separating two classes in high-dimensional space. RBF kernel handles non-linear data. Standard Scaling applied before training."),
        ("Why does Recall matter most?", "Missing a depressed student (false negative) is more serious than a false alarm. KNN achieved 97.44% recall — it correctly identified 97.44% of all depressed students."),
        ("What is 5-Fold Cross Validation?", "Splits training data into 5 parts, trains/tests 5 times on different parts, then averages accuracy. KNN achieved 86.67% CV mean with 2.12% std dev — proving model stability."),
        ("Why three different algorithms?", "Each member independently implements a different algorithm for rigorous comparison on the same dataset. Any performance difference is purely due to algorithm choice."),
    ]:
        with st.expander(q):
            st.markdown(f"<p style='font-size:14px;color:#6B7280;line-height:1.7;'>{a}</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="section-wrap-alt">
        <div class="section-label">Dataset</div>
        <div class="section-title">Student Mental Health Dataset</div>
        <div class="section-sub">Source: Kaggle — Shariful07 (2020) | 600 records | 11 features | IIUM Malaysia</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='padding: 0 60px 48px; background:white;'>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Records", "600")
    c2.metric("Features", "11 original")
    c3.metric("Depression Yes", "194 (32%)")
    c4.metric("Anxiety Yes", "209 (35%)")
    c5.metric("Panic Attack Yes", "190 (32%)")
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("Dataset Feature Description", expanded=True):
        st.markdown("""
        <table class="ds-table">
        <tr><th>Feature</th><th>Description</th><th>Values</th><th>Role</th></tr>
        <tr><td><b>Gender</b></td><td>Student gender</td><td>Male / Female</td><td><span class="tag-input">Input</span></td></tr>
        <tr><td><b>Age</b></td><td>Student age</td><td>17–24</td><td><span class="tag-input">Input</span></td></tr>
        <tr><td><b>Course</b></td><td>Field of study</td><td>10 categories</td><td><span class="tag-input">Input</span></td></tr>
        <tr><td><b>Year of Study</b></td><td>Academic year</td><td>Year 1–4</td><td><span class="tag-input">Input</span></td></tr>
        <tr><td><b>CGPA</b></td><td>GPA range</td><td>0–1.99 to 3.50–4.00</td><td><span class="tag-input">Input</span></td></tr>
        <tr><td><b>Marital Status</b></td><td>Marital status</td><td>Yes / No</td><td><span class="tag-input">Input</span></td></tr>
        <tr><td><b>Depression</b></td><td>Has depression?</td><td>Yes/No → 1/0</td><td><span class="tag-target">Target (KNN & DT)</span></td></tr>
        <tr><td><b>Anxiety</b></td><td>Has anxiety?</td><td>Yes/No → 1/0</td><td><span class="tag-input">Input</span></td></tr>
        <tr><td><b>Panic Attack</b></td><td>Has panic attacks?</td><td>Yes/No → 1/0</td><td><span class="tag-target">Target (SVM)</span></td></tr>
        <tr><td><b>Seek Treatment</b></td><td>Sought specialist help?</td><td>Yes/No → 1/0</td><td><span class="tag-input">Input</span></td></tr>
        </table>""", unsafe_allow_html=True)
    with st.expander("Preprocessing Steps"):
        for t, d in [
            ("Column Renaming", "All column names standardized (e.g., 'Choose your gender' → 'Gender')."),
            ("Missing Values", "8 missing Age values filled with median (19)."),
            ("Inconsistency Fix", "48 course variations grouped into 10 categories. Year format standardized."),
            ("Binary Encoding", "Yes/No columns and Gender encoded as 1/0."),
            ("Feature Engineering", "Mental Health Score (0–3), Age Group, CGPA Numeric midpoint added."),
        ]:
            a, b = st.columns([1,3])
            with a: st.markdown(f"<div style='font-size:13px;font-weight:600;color:#3D52FF;padding:8px 0;'>{t}</div>", unsafe_allow_html=True)
            with b: st.markdown(f"<div style='font-size:13px;color:#6B7280;padding:8px 0;'>{d}</div>", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding: 12px 60px 40px;'>", unsafe_allow_html=True)
    if st.button("Explore Full EDA Visualizations"):
        st.switch_page("pages/1_EDA.py")
    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="section-wrap">
        <div class="section-label">Model Performance</div>
        <div class="section-title">Algorithm Results at a Glance</div>
        <div class="section-sub">Each member independently implemented a different algorithm. Click below to view full model details.</div>
        <div class="algo-grid">
            <div class="algo-card algo-1">
                <div class="algo-name">KNN</div>
                <div class="algo-member">Member 1 — Ho Jun Yon</div>
                <div class="algo-acc">95.83%</div>
                <div class="algo-acc-label">Accuracy | K = 5 | Target: Depression</div>
                <div class="algo-metric"><span class="algo-metric-label">Precision</span><span class="algo-metric-value">90.48%</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Recall</span><span class="algo-metric-value">97.44%</span></div>
                <div class="algo-metric"><span class="algo-metric-label">F1 Score</span><span class="algo-metric-value">93.83%</span></div>
                <div class="algo-metric"><span class="algo-metric-label">CV Mean</span><span class="algo-metric-value">86.67%</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Split</span><span class="algo-metric-value">80% / 20%</span></div>
            </div>
            <div class="algo-card algo-2">
                <div class="algo-name">Decision Tree</div>
                <div class="algo-member">Member 2 — Irvin Tan Wei Shen</div>
                <div class="algo-acc">85.50%</div>
                <div class="algo-acc-label">Accuracy | CART Depth 5 | Target: Depression</div>
                <div class="algo-metric"><span class="algo-metric-label">Precision</span><span class="algo-metric-value">72.38%</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Recall</span><span class="algo-metric-value">89.18%</span></div>
                <div class="algo-metric"><span class="algo-metric-label">F1 Score</span><span class="algo-metric-value">79.91%</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Root Feature</span><span class="algo-metric-value">Marital Status</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Dataset</span><span class="algo-metric-value">600 records (full)</span></div>
            </div>
            <div class="algo-card algo-3">
                <div class="algo-name">SVM</div>
                <div class="algo-member">Member 3 — Chiang Jun Hang</div>
                <div class="algo-acc">TBD</div>
                <div class="algo-acc-label">Accuracy | RBF Kernel | Target: Panic Attack</div>
                <div class="algo-metric"><span class="algo-metric-label">Precision</span><span class="algo-metric-value">TBD</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Recall</span><span class="algo-metric-value">TBD</span></div>
                <div class="algo-metric"><span class="algo-metric-label">F1 Score</span><span class="algo-metric-value">TBD</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Scaling</span><span class="algo-metric-value">Standard Scaler</span></div>
                <div class="algo-metric"><span class="algo-metric-label">Split</span><span class="algo-metric-value">75% / 25%</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='padding: 0 60px 16px;'>", unsafe_allow_html=True)
    ra, rb, rc, rd = st.columns(4)
    with ra:
        if st.button("View KNN Details"):
            st.switch_page("pages/2_KNN.py")
    with rb:
        if st.button("View Decision Tree"):
            st.switch_page("pages/3_Decision_Tree.py")
    with rc:
        if st.button("View SVM Details"):
            st.switch_page("pages/4_SVM.py")
    with rd:
        if st.button("Compare All Models"):
            st.switch_page("pages/5_Comparison.py")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='padding: 0 60px 60px;'>", unsafe_allow_html=True)
    st.info("**Key Finding:** KNN (K=5) achieved 95.83% accuracy with 97.44% recall — correctly identifying 97.44% of depressed students. High recall is critical in mental health screening to avoid missing at-risk students. Decision Tree achieved 85.50% accuracy with strong interpretability through its visual tree structure.")
    st.markdown("</div>", unsafe_allow_html=True)

with tab4:
    st.markdown("""
    <div class="section-wrap">
        <div class="section-label">Our Team</div>
        <div class="section-title">Group Members</div>
        <div class="section-sub">Tutorial Group 3 | Tutor: Dr Goh | BMCS2003 AI | 202605 Session</div>
        <div class="members-grid">
            <div class="member-card">
                <div class="member-header"><div class="member-avatar avatar-1">H</div><div class="member-name">Ho Jun Yon</div><div class="member-id">Student ID: 2612634</div></div>
                <div class="member-body">
                    <span class="member-algo">K-Nearest Neighbor (KNN)</span>
                    <div class="member-row"><span class="member-row-label">Target</span><span class="member-row-value">Depression</span></div>
                    <div class="member-row"><span class="member-row-label">Accuracy</span><span class="member-row-value">95.83%</span></div>
                    <div class="member-row"><span class="member-row-label">Best K</span><span class="member-row-value">K = 5</span></div>
                    <div class="member-row"><span class="member-row-label">Recall</span><span class="member-row-value">97.44%</span></div>
                    <div class="member-row"><span class="member-row-label">CV Mean</span><span class="member-row-value">86.67%</span></div>
                </div>
            </div>
            <div class="member-card">
                <div class="member-header"><div class="member-avatar avatar-2">I</div><div class="member-name">Irvin Tan Wei Shen</div><div class="member-id">Student ID: 2612638</div></div>
                <div class="member-body">
                    <span class="member-algo">Decision Tree (CART)</span>
                    <div class="member-row"><span class="member-row-label">Target</span><span class="member-row-value">Depression</span></div>
                    <div class="member-row"><span class="member-row-label">Accuracy</span><span class="member-row-value">85.50%</span></div>
                    <div class="member-row"><span class="member-row-label">Max Depth</span><span class="member-row-value">5 Levels</span></div>
                    <div class="member-row"><span class="member-row-label">Criterion</span><span class="member-row-value">Gini Impurity</span></div>
                    <div class="member-row"><span class="member-row-label">Root Split</span><span class="member-row-value">Marital Status</span></div>
                </div>
            </div>
            <div class="member-card">
                <div class="member-header"><div class="member-avatar avatar-3">C</div><div class="member-name">Chiang Jun Hang</div><div class="member-id">Student ID: 2612610</div></div>
                <div class="member-body">
                    <span class="member-algo">Support Vector Machine (SVM)</span>
                    <div class="member-row"><span class="member-row-label">Target</span><span class="member-row-value">Panic Attack</span></div>
                    <div class="member-row"><span class="member-row-label">Accuracy</span><span class="member-row-value">TBD</span></div>
                    <div class="member-row"><span class="member-row-label">Kernel</span><span class="member-row-value">RBF</span></div>
                    <div class="member-row"><span class="member-row-label">Scaling</span><span class="member-row-value">Standard Scaler</span></div>
                    <div class="member-row"><span class="member-row-label">Status</span><span class="member-row-value">In Progress</span></div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='padding: 0 60px 20px;'>", unsafe_allow_html=True)
    ta, tb, tc = st.columns(3)
    with ta:
        if st.button("Open KNN Page"):
            st.switch_page("pages/2_KNN.py")
    with tb:
        if st.button("Open Decision Tree Page"):
            st.switch_page("pages/3_Decision_Tree.py")
    with tc:
        if st.button("Open SVM Page"):
            st.switch_page("pages/4_SVM.py")
    st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("""
<div class="footer">
    <div>
        <div class="footer-brand">Mind<span>Check</span></div>
        <div class="footer-text">Student Mental Health Prediction System</div>
        <div class="footer-text" style="margin-top:3px;">Dataset: Kaggle — Shariful07 (2020) | IIUM Malaysia</div>
    </div>
    <div class="footer-right">
        BMCS2003 Artificial Intelligence<br>
        202605 Session | Tutorial Group 3<br>
        Tutor: Dr Goh | TARUMT
    </div>
</div>
""", unsafe_allow_html=True)
