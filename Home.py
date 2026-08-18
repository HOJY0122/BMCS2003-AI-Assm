import streamlit as st

st.set_page_config(
    page_title="MindCheck — Student Mental Health",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #1A1D2E !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background: #1A1D2E !important;
    padding: 0 !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #12152A !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebarNav"] { display: none !important; }

.stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.75) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    padding: 10px 16px !important;
    transition: all 0.2s !important;
    text-align: left !important;
}
.stButton > button:hover {
    background: rgba(61,82,255,0.2) !important;
    border-color: #3D52FF !important;
    color: white !important;
    transform: translateX(3px) !important;
}

.hero-section {
    min-height: 100vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 60px 40px;
    background: linear-gradient(135deg, #1A1D2E 0%, #2D3278 55%, #1A1D2E 100%);
    position: relative; overflow: hidden;
    text-align: center;
}
.hero-glow-1 {
    position: absolute; top: -100px; right: -100px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(61,82,255,0.22) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-glow-2 {
    position: absolute; bottom: -60px; left: 10%;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(99,179,237,0.1) 0%, transparent 70%);
    border-radius: 50%; pointer-events: none;
}
.hero-content { position: relative; z-index: 1; max-width: 760px; }
.hero-badge {
    display: inline-block;
    background: rgba(61,82,255,0.18);
    color: #93A3FF;
    font-size: 11px; font-weight: 700;
    padding: 7px 18px; border-radius: 20px;
    border: 1px solid rgba(61,82,255,0.35);
    margin-bottom: 28px;
    letter-spacing: 1.8px; text-transform: uppercase;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 56px; font-weight: 700;
    color: #FFFFFF; line-height: 1.12;
    letter-spacing: -2px; margin-bottom: 22px;
}
.hero-title .accent {
    background: linear-gradient(135deg, #6B86FF 0%, #93CFFF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    font-size: 17px; color: rgba(255,255,255,0.55);
    line-height: 1.75; margin-bottom: 48px;
    max-width: 540px; margin-left: auto; margin-right: auto;
}
.nav-cards-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    width: 100%; max-width: 780px;
    margin: 0 auto 44px;
}
.nav-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 22px 18px;
    text-align: left; cursor: default;
    transition: all 0.25s;
    position: relative; overflow: hidden;
}
.nav-card:hover {
    background: rgba(255,255,255,0.09);
    border-color: rgba(255,255,255,0.2);
    transform: translateY(-3px);
    box-shadow: 0 12px 32px rgba(0,0,0,0.3);
}
.nc-blue  { border-top: 3px solid #3D52FF; }
.nc-green { border-top: 3px solid #10B981; }
.nc-amber { border-top: 3px solid #F59E0B; }
.nc-violet{ border-top: 3px solid #8B5CF6; }
.nc-cyan  { border-top: 3px solid #06B6D4; }
.nc-rose  { border-top: 3px solid #EC4899; }
.nc-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 14px; font-weight: 600;
    color: white; margin-bottom: 5px; margin-top: 10px;
}
.nc-icon { font-size: 24px; }
.nc-desc { font-size: 11px; color: rgba(255,255,255,0.4); line-height: 1.5; }
.nc-arrow {
    position: absolute; top: 14px; right: 14px;
    color: rgba(255,255,255,0.2); font-size: 14px;
    transition: all 0.2s;
}
.nav-card:hover .nc-arrow { color: rgba(255,255,255,0.6); }

.stats-strip {
    display: flex; gap: 24px; justify-content: center;
    flex-wrap: wrap;
}
.stat-chip {
    text-align: center; padding: 14px 22px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; min-width: 100px;
}
.stat-n {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #6B86FF;
    line-height: 1;
}
.stat-l { font-size: 11px; color: rgba(255,255,255,0.35); margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:24px 20px 16px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:14px;">
        <div style="font-family:'Space Grotesk',sans-serif;font-size:21px;font-weight:700;color:white;">
            Mind<span style="color:#6B86FF;">Check</span>
        </div>
        <div style="font-size:11px;color:rgba(255,255,255,0.3);margin-top:4px;">
            BMCS2003 Artificial Intelligence
        </div>
    </div>
    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.25);
                letter-spacing:1.8px;text-transform:uppercase;padding:0 8px 8px;">
        MAIN
    </div>
    """, unsafe_allow_html=True)

    if st.button("Home", key="sb_home", width='stretch'):
        st.switch_page("Home.py")

    st.markdown("""
    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.25);
                letter-spacing:1.8px;text-transform:uppercase;
                padding:14px 8px 8px;">ANALYSIS</div>
    """, unsafe_allow_html=True)

    if st.button("Exploratory Data Analysis", key="sb_eda", width='stretch'):
        st.switch_page("pages/1_EDA.py")
    if st.button("Dataset Overview", key="sb_ds", width='stretch'):
        st.switch_page("pages/6_Dataset.py")

    st.markdown("""
    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.25);
                letter-spacing:1.8px;text-transform:uppercase;
                padding:14px 8px 8px;">MODELS</div>
    """, unsafe_allow_html=True)

    if st.button("KNN — Ho Jun Yon", key="sb_knn", width='stretch'):
        st.switch_page("pages/2_KNN.py")
    if st.button("Decision Tree — Irvin", key="sb_dt", width='stretch'):
        st.switch_page("pages/3_Decision_Tree.py")
    if st.button("SVM — Chiang Jun Hang", key="sb_svm", width='stretch'):
        st.switch_page("pages/4_SVM.py")

    st.markdown("""
    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.25);
                letter-spacing:1.8px;text-transform:uppercase;
                padding:14px 8px 8px;">COMPARE</div>
    """, unsafe_allow_html=True)

    if st.button("Compare All Models", key="sb_cmp", width='stretch'):
        st.switch_page("pages/5_Comparison.py")

    st.markdown("""
    <div style="position:fixed;bottom:0;left:0;width:250px;
                padding:14px 20px;border-top:1px solid rgba(255,255,255,0.06);
                font-size:11px;color:rgba(255,255,255,0.2);line-height:1.8;
                background:#12152A;">
        Tutorial Group 3 · Tutor: Dr Goh<br>
        202605 Session · TARUMT
    </div>
    """, unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-section">
    <div class="hero-glow-1"></div>
    <div class="hero-glow-2"></div>
    <div class="hero-content">

        <div class="hero-badge">Supervised Machine Learning · TARUMT · BMCS2003</div>

        <h1 class="hero-title">
            Student<br>
            <span class="accent">Mental Health</span><br>
            Prediction
        </h1>

        <p class="hero-desc">
            An AI-powered system that analyses student demographics
            and academic data to predict depression and panic attack
            risk — enabling early detection and timely support.
        </p>

        <div class="nav-cards-grid">
            <div class="nav-card nc-blue">
                <span class="nc-arrow">↗</span>
                <div class="nc-icon">📊</div>
                <div class="nc-title">EDA</div>
                <div class="nc-desc">Explore dataset visualizations and feature distributions</div>
            </div>
            <div class="nav-card nc-green">
                <span class="nc-arrow">↗</span>
                <div class="nc-icon">🔵</div>
                <div class="nc-title">KNN Model</div>
                <div class="nc-desc">K-Nearest Neighbor · 95.83% accuracy · K = 5</div>
            </div>
            <div class="nav-card nc-amber">
                <span class="nc-arrow">↗</span>
                <div class="nc-icon">🌳</div>
                <div class="nc-title">Decision Tree</div>
                <div class="nc-desc">CART · 85.50% accuracy · Max depth 5</div>
            </div>
            <div class="nav-card nc-violet">
                <span class="nc-arrow">↗</span>
                <div class="nc-icon">🔴</div>
                <div class="nc-title">SVM Model</div>
                <div class="nc-desc">Support Vector Machine · RBF Kernel</div>
            </div>
            <div class="nav-card nc-cyan">
                <span class="nc-arrow">↗</span>
                <div class="nc-icon">📈</div>
                <div class="nc-title">Compare Models</div>
                <div class="nc-desc">Side-by-side algorithm performance comparison</div>
            </div>
            <div class="nav-card nc-rose">
                <span class="nc-arrow">↗</span>
                <div class="nc-icon">📋</div>
                <div class="nc-title">Dataset</div>
                <div class="nc-desc">600 student records · IIUM Malaysia · Kaggle</div>
            </div>
        </div>

        <div class="stats-strip">
            <div class="stat-chip"><div class="stat-n">600</div><div class="stat-l">Student Records</div></div>
            <div class="stat-chip"><div class="stat-n">95.83%</div><div class="stat-l">Best Accuracy</div></div>
            <div class="stat-chip"><div class="stat-n">97.44%</div><div class="stat-l">Best Recall</div></div>
            <div class="stat-chip"><div class="stat-n">3</div><div class="stat-l">ML Algorithms</div></div>
            <div class="stat-chip"><div class="stat-n">11</div><div class="stat-l">Features</div></div>
        </div>

    </div>
</div>
""", unsafe_allow_html=True)

# ── Real Streamlit nav buttons below hero ─────────────────────
st.markdown("""
<div style="background:#12152A;padding:28px 40px;
            border-top:1px solid rgba(255,255,255,0.06);">
    <div style="font-size:12px;color:rgba(255,255,255,0.3);
                text-align:center;margin-bottom:16px;letter-spacing:0.5px;">
        Quick Navigation
    </div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1:
    if st.button("EDA", key="qn_eda", width='stretch'):
        st.switch_page("pages/1_EDA.py")
with c2:
    if st.button("KNN", key="qn_knn", width='stretch'):
        st.switch_page("pages/2_KNN.py")
with c3:
    if st.button("Decision Tree", key="qn_dt", width='stretch'):
        st.switch_page("pages/3_Decision_Tree.py")
with c4:
    if st.button("SVM", key="qn_svm", width='stretch'):
        st.switch_page("pages/4_SVM.py")
with c5:
    if st.button("Compare", key="qn_cmp", width='stretch'):
        st.switch_page("pages/5_Comparison.py")
with c6:
    if st.button("Dataset", key="qn_ds", width='stretch'):
        st.switch_page("pages/6_Dataset.py")
