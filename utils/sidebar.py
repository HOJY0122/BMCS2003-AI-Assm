import streamlit as st

SIDEBAR_CSS = """
<style>
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

[data-testid="stSidebarNav"] { display: none !important; }

[data-testid="stSidebar"] {
    background: #0B0E1A !important;
    border-right: 1px solid #1A1F35 !important;
}

/* Remove default padding */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* All sidebar buttons — reset to clean nav style */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #8892A4 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px 8px 16px !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 1px !important;
    transition: all 0.15s ease !important;
    letter-spacing: 0.1px !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
    border: none !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    font-size: 13px !important;
    color: inherit !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #131929;
    border: 1px solid #1E2640;
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #6B7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: #5B7FFF !important;
}
</style>
"""


def sidebar(active="home"):
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────
        st.markdown("""
        <div style="padding: 28px 20px 20px;">
            <div style="font-size:22px; font-weight:800; color:#FFFFFF;
                        letter-spacing:-0.8px; line-height:1;">
                Mind<span style="color:#5B7FFF;">Check</span>
            </div>
            <div style="font-size:11px; color:#3D4455; margin-top:5px;
                        letter-spacing:0.3px;">
                BMCS2003 · Artificial Intelligence
            </div>
        </div>
        <div style="height:1px; background:#1A1F35; margin: 0 16px 16px;"></div>
        """, unsafe_allow_html=True)

        # ── MAIN ───────────────────────────────────────────────
        _section("MAIN")
        _nav_btn("🏠", "Home",          "home",    active, "Home.py")

        # ── ANALYSIS ───────────────────────────────────────────
        _section("ANALYSIS")
        _nav_btn("📊", "EDA",               "eda",     active, "pages/1_EDA.py")
        _nav_btn("🔬", "Feature Selection", "feature", active, "pages/9_Feature_Selection.py")
        _nav_btn("📋", "Dataset",           "dataset", active, "pages/6_Dataset.py")
        _nav_btn("📈", "Live Statistics",   "stats",   active, "pages/11_Live_Stats.py")

        # ── MODELS ─────────────────────────────────────────────
        _section("MODELS")
        _nav_btn("🔵", "KNN",           "knn",     active, "pages/2_KNN.py")
        _nav_btn("🌳", "Decision Tree", "dt",      active, "pages/3_Decision_Tree.py")
        _nav_btn("🔴", "SVM",           "svm",     active, "pages/4_SVM.py")

        # ── RESULTS ────────────────────────────────────────────
        _section("RESULTS")
        _nav_btn("📈", "Compare Models", "compare", active, "pages/5_Comparison.py")

        # ── INFO ───────────────────────────────────────────────
        _section("INFO")
        _nav_btn("ℹ️", "About",         "about",   active, "pages/7_About.py")
        _nav_btn("❓", "FAQ",           "faq",     active, "pages/8_FAQ.py")

        # ── Footer ─────────────────────────────────────────────
        st.markdown("""
        <div style="height:1px; background:#1A1F35; margin: 20px 16px 16px;"></div>
        <div style="padding: 0 20px 24px;">
            <div style="font-size:11px; color:#2E3447; line-height:1.9;">
                Tutorial Group 3<br>
                Tutor: Dr Goh · 202605<br>
                TARUMT
            </div>
        </div>
        """, unsafe_allow_html=True)


def _section(label):
    st.markdown(
        f"<div style='font-size:10px; font-weight:700; color:#2E3854; "
        f"letter-spacing:1.8px; text-transform:uppercase; "
        f"padding: 14px 20px 6px;'>{label}</div>",
        unsafe_allow_html=True
    )


def _nav_btn(icon, label, key, active, page):
    if active == key:
        # Active — highlighted, not clickable
        st.markdown(f"""
        <div style="
            display: flex; align-items: center; gap: 10px;
            background: rgba(91,127,255,0.12);
            border-left: 3px solid #5B7FFF;
            border-radius: 0 8px 8px 0;
            padding: 9px 12px 9px 17px;
            margin: 1px 8px 1px 0;
            font-size: 13px; font-weight: 600; color: #FFFFFF;
            cursor: default;">
            <span>{icon}</span>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button(f"{icon}  {label}", key=f"sb_{key}",
                     use_container_width=True):
            try:
                st.switch_page(page)
            except Exception:
                st.error(f"Page not found: {page}")