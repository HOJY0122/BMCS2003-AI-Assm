import streamlit as st

st.set_page_config(
    page_title="MindCheck — Student Mental Health",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #F0F4FF !important;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] > .main {
    background: #F0F4FF !important;
    padding: 0 !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1A1D2E !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebarNav"] { display: none !important; }

/* Sidebar collapse button */
[data-testid="collapsedControl"] {
    color: #3D52FF !important;
    background: white !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 2px 0 8px rgba(61,82,255,0.15) !important;
}

/* ── Sidebar nav buttons ── */
.sidebar-logo {
    padding: 28px 24px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 16px;
}
.sidebar-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: white !important;
    letter-spacing: -0.5px;
}
.sidebar-brand span { color: #6B86FF !important; }
.sidebar-sub {
    font-size: 11px; color: rgba(255,255,255,0.4) !important;
    margin-top: 4px; letter-spacing: 0.5px;
}

.nav-section {
    font-size: 10px; font-weight: 700;
    color: rgba(255,255,255,0.3) !important;
    letter-spacing: 1.8px; text-transform: uppercase;
    padding: 0 24px 8px;
}

.nav-item {
    display: flex; align-items: center; gap: 12px;
    padding: 11px 24px; font-size: 14px; font-weight: 500;
    color: rgba(255,255,255,0.65) !important;
    border-radius: 0; cursor: pointer;
    transition: all 0.18s; text-decoration: none;
    border-left: 3px solid transparent;
}
.nav-item:hover {
    background: rgba(255,255,255,0.06) !important;
    color: white !important;
    border-left-color: #3D52FF;
}
.nav-item.active {
    background: rgba(61,82,255,0.15) !important;
    color: white !important;
    border-left-color: #3D52FF;
}
.nav-dot {
    width: 8px; height: 8px; border-radius: 50%;
    flex-shrink: 0;
}
.dot-blue { background: #3D52FF; }
.dot-green { background: #10B981; }
.dot-yellow { background: #F59E0B; }
.dot-purple { background: #8B5CF6; }
.dot-pink { background: #EC4899; }
.dot-cyan { background: #06B6D4; }

.sidebar-footer {
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 16px 24px;
    border-top: 1px solid rgba(255,255,255,0.06);
    font-size: 11px; color: rgba(255,255,255,0.25) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #3D52FF !important; color: white !important;
    border: none !important; border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important; font-size: 15px !important;
    padding: 14px 28px !important; transition: all 0.2s !important;
    width: 100% !important; cursor: pointer !important;
}
.stButton > button:hover {
    background: #2A3ECC !important; transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(61,82,255,0.35) !important;
}

/* ── Hero ── */
.hero-wrap {
    min-height: 100vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 60px 80px;
    background: linear-gradient(135deg, #1A1D2E 0%, #2D3278 55%, #1A1D2E 100%);
    position: relative; overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; top: -120px; right: -120px;
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(61,82,255,0.22) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-wrap::after {
    content: '';
    position: absolute; bottom: -80px; left: 10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-inner { position: relative; z-index: 1; text-align: center; max-width: 780px; }
.hero-badge {
    display: inline-block;
    background: rgba(61,82,255,0.2);
    color: #93A3FF; font-size: 11px; font-weight: 700;
    padding: 7px 18px; border-radius: 20px;
    border: 1px solid rgba(61,82,255,0.35);
    margin-bottom: 28px; letter-spacing: 1.8px;
    text-transform: uppercase;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 58px; font-weight: 700; color: #FFFFFF;
    line-height: 1.12; letter-spacing: -2px;
    margin-bottom: 24px;
}
.hero-title span {
    background: linear-gradient(135deg, #6B86FF 0%, #93CFFF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-desc {
    font-size: 18px; color: rgba(255,255,255,0.55);
    line-height: 1.75; margin-bottom: 48px; max-width: 580px; margin-left: auto; margin-right: auto;
}

/* ── Nav Cards Grid ── */
.nav-cards {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 16px; width: 100%; max-width: 800px; margin: 0 auto;
}
.nav-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 24px 20px;
    text-align: left; cursor: pointer;
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
    position: relative; overflow: hidden;
}
.nav-card:hover {
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,0.2);
    transform: translateY(-4px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.3);
}
.nav-card-icon {
    font-size: 28px; margin-bottom: 12px; display: block;
}
.nav-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px; font-weight: 600; color: white;
    margin-bottom: 6px;
}
.nav-card-desc { font-size: 12px; color: rgba(255,255,255,0.45); line-height: 1.5; }
.nav-card-arrow {
    position: absolute; top: 20px; right: 20px;
    color: rgba(255,255,255,0.25); font-size: 18px;
    transition: all 0.2s;
}
.nav-card:hover .nav-card-arrow { color: rgba(255,255,255,0.7); transform: translate(3px, -3px); }

.nc-blue { border-top: 3px solid #3D52FF; }
.nc-green { border-top: 3px solid #10B981; }
.nc-yellow { border-top: 3px solid #F59E0B; }
.nc-purple { border-top: 3px solid #8B5CF6; }
.nc-pink { border-top: 3px solid #EC4899; }
.nc-cyan { border-top: 3px solid #06B6D4; }

/* Stats row */
.stats-row {
    display: flex; gap: 32px; justify-content: center;
    margin-top: 48px; flex-wrap: wrap;
}
.stat-chip {
    text-align: center;
    padding: 14px 24px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
}
.stat-chip-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px; font-weight: 700; color: #6B86FF;
    line-height: 1;
}
.stat-chip-label { font-size: 11px; color: rgba(255,255,255,0.4); margin-top: 4px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="sidebar-brand">Mind<span>Check</span></div>
        <div class="sidebar-sub">BMCS2003 Artificial Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-section">Main</div>', unsafe_allow_html=True)
    if st.button("Home", key="nav_home", use_container_width=True):
        st.switch_page("Main.py")

    st.markdown('<div class="nav-section" style="margin-top:12px;">Analysis</div>', unsafe_allow_html=True)
    if st.button("Exploratory Data Analysis", key="nav_eda", use_container_width=True):
        st.switch_page("pages/1_EDA.py")
    if st.button("Dataset Overview", key="nav_dataset", use_container_width=True):
        st.switch_page("pages/6_Dataset.py")

    st.markdown('<div class="nav-section" style="margin-top:12px;">Models</div>', unsafe_allow_html=True)
    if st.button("KNN — Ho Jun Yon", key="nav_knn", use_container_width=True):
        st.switch_page("pages/2_KNN.py")
    if st.button("Decision Tree — Irvin", key="nav_dt", use_container_width=True):
        st.switch_page("pages/3_Decision_Tree.py")
    if st.button("SVM — Chiang Jun Hang", key="nav_svm", use_container_width=True):
        st.switch_page("pages/4_SVM.py")

    st.markdown('<div class="nav-section" style="margin-top:12px;">Compare</div>', unsafe_allow_html=True)
    if st.button("Compare All Models", key="nav_compare", use_container_width=True):
        st.switch_page("pages/5_Comparison.py")

    st.markdown("""
    <div class="sidebar-footer">
        Tutorial Group 3 · Tutor: Dr Goh<br>
        202605 Session · TARUMT
    </div>
    """, unsafe_allow_html=True)

# ── HERO / WELCOME ────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
  <div class="hero-inner">

    <div class="hero-badge">Supervised Machine Learning · TARUMT</div>

    <h1 class="hero-title">
      Student<br>
      <span>Mental Health</span><br>
      Prediction
    </h1>

    <p class="hero-desc">
      An AI-powered system that analyses student demographics and
      academic data to predict depression and panic attack risk —
      enabling early detection and timely support.
    </p>

    <div class="nav-cards">
      <div class="nav-card nc-blue">
        <span class="nav-card-arrow">↗</span>
        <span class="nav-card-icon">📊</span>
        <div class="nav-card-title">EDA</div>
        <div class="nav-card-desc">Explore dataset visualizations and distributions</div>
      </div>
      <div class="nav-card nc-green">
        <span class="nav-card-arrow">↗</span>
        <span class="nav-card-icon">🔵</span>
        <div class="nav-card-title">KNN Model</div>
        <div class="nav-card-desc">K-Nearest Neighbor · 95.83% accuracy</div>
      </div>
      <div class="nav-card nc-yellow">
        <span class="nav-card-arrow">↗</span>
        <span class="nav-card-icon">🌳</span>
        <div class="nav-card-title">Decision Tree</div>
        <div class="nav-card-desc">CART · 85.50% accuracy · Depth 5</div>
      </div>
      <div class="nav-card nc-purple">
        <span class="nav-card-arrow">↗</span>
        <span class="nav-card-icon">🔴</span>
        <div class="nav-card-title">SVM Model</div>
        <div class="nav-card-desc">Support Vector Machine · RBF Kernel</div>
      </div>
      <div class="nav-card nc-cyan">
        <span class="nav-card-arrow">↗</span>
        <span class="nav-card-icon">📈</span>
        <div class="nav-card-title">Compare Models</div>
        <div class="nav-card-desc">Side-by-side algorithm comparison</div>
      </div>
      <div class="nav-card nc-pink">
        <span class="nav-card-arrow">↗</span>
        <span class="nav-card-icon">📋</span>
        <div class="nav-card-title">Dataset</div>
        <div class="nav-card-desc">600 records · IIUM Malaysia · Kaggle</div>
      </div>
    </div>

    <div class="stats-row">
      <div class="stat-chip"><div class="stat-chip-num">600</div><div class="stat-chip-label">Student Records</div></div>
      <div class="stat-chip"><div class="stat-chip-num">95.83%</div><div class="stat-chip-label">Best Accuracy</div></div>
      <div class="stat-chip"><div class="stat-chip-num">97.44%</div><div class="stat-chip-label">Best Recall</div></div>
      <div class="stat-chip"><div class="stat-chip-num">3</div><div class="stat-chip-label">ML Algorithms</div></div>
      <div class="stat-chip"><div class="stat-chip-num">11</div><div class="stat-chip-label">Features</div></div>
    </div>

  </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation Buttons (Real Streamlit) ───────────────────────
st.markdown("<div style='background:#1A1D2E; padding: 0 80px 48px;'>", unsafe_allow_html=True)
b1, b2, b3, b4, b5, b6 = st.columns(6)
with b1:
    if st.button("Explore EDA"):
        st.switch_page("pages/1_EDA.py")
with b2:
    if st.button("KNN Prediction"):
        st.switch_page("pages/2_KNN.py")
with b3:
    if st.button("Decision Tree"):
        st.switch_page("pages/3_Decision_Tree.py")
with b4:
    if st.button("SVM Model"):
        st.switch_page("pages/4_SVM.py")
with b5:
    if st.button("Compare Models"):
        st.switch_page("pages/5_Comparison.py")
with b6:
    if st.button("View Dataset"):
        st.switch_page("pages/6_Dataset.py")
st.markdown("</div>", unsafe_allow_html=True)
