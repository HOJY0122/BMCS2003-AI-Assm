import streamlit as st

SIDEBAR_CSS = """
<style>
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }

[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar background ── */
[data-testid="stSidebar"] {
    background: #0B0E1A !important;
    border-right: 1px solid #1A1F35 !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── Nav buttons ── */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #8892A4 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 10px 7px 14px !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 1px !important;
    transition: all 0.15s ease !important;
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

/* ── Expander styling — tree sections ── */
section[data-testid="stSidebar"] .stExpander {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stExpander > details {
    border: none !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] .stExpander summary {
    font-size: 10px !important;
    font-weight: 700 !important;
    color: #4A5680 !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    padding: 10px 20px 6px !important;
    background: transparent !important;
    border: none !important;
    list-style: none !important;
}
section[data-testid="stSidebar"] .stExpander summary:hover {
    color: #7B88B0 !important;
}
section[data-testid="stSidebar"] .stExpander summary svg {
    color: #4A5680 !important;
    width: 12px !important;
    height: 12px !important;
}
section[data-testid="stSidebar"] details[open] summary {
    color: #6B7BA8 !important;
}
section[data-testid="stSidebar"] .stExpander > details > div {
    padding: 0 0 4px 0 !important;
    border: none !important;
}

/* ── Metric cards ── */
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

# Track which sections are active to auto-expand
_SECTION_MAP = {
    "home"   : "main",
    "eda"    : "analysis",
    "feature": "analysis",
    "dataset": "analysis",
    "stats"  : "analysis",
    "knn"    : "models",
    "dt"     : "models",
    "svm"    : "models",
    "compare": "results",
    "about"  : "info",
    "faq"    : "info",
    "batch"  : "models",
}


def sidebar(active="home"):
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────
        st.markdown("""
        <div style="padding: 28px 20px 18px;">
            <div style="font-size:22px; font-weight:800; color:#FFFFFF;
                        letter-spacing:-0.8px; line-height:1;">
                Mind<span style="color:#5B7FFF;">Check</span>
            </div>
            <div style="font-size:11px; color:#3D4455; margin-top:5px;
                        letter-spacing:0.3px;">
                BMCS2003 · Artificial Intelligence
            </div>
        </div>
        <div style="height:1px; background:#1A1F35; margin:0 16px 10px;"></div>
        """, unsafe_allow_html=True)

        active_section = _SECTION_MAP.get(active, "main")

        # ── MAIN (no expander — always visible) ───────────────
        _nav_btn("🏠", "Home", "home", active, "Home.py")

        st.markdown(
            "<div style='height:1px;background:#1A1F35;margin:6px 16px;'></div>",
            unsafe_allow_html=True
        )

        # ── ANALYSIS (collapseable) ────────────────────────────
        with st.expander("📂  ANALYSIS",
                         expanded=(active_section == "analysis")):
            _nav_btn("📊", "EDA",               "eda",     active, "pages/1_EDA.py")
            _nav_btn("🔬", "Feature Selection", "feature", active, "pages/9_Feature_Selection.py")
            _nav_btn("📋", "Dataset",           "dataset", active, "pages/6_Dataset.py")
            _nav_btn("📈", "Live Statistics",   "stats",   active, "pages/11_Live_Stats.py")

        # ── PREDICTORS (collapseable) ──────────────────────────
        with st.expander("🤖  PREDICTORS",
                         expanded=(active_section == "models")):
            _nav_btn("🔵", "KNN",           "knn", active, "pages/2_KNN.py")
            _nav_btn("🌳", "Decision Tree", "dt",  active, "pages/3_Decision_Tree.py")
            _nav_btn("🔴", "SVM",           "svm", active, "pages/4_SVM.py")

        # ── RESULTS (collapseable) ─────────────────────────────
        with st.expander("📊  RESULTS",
                         expanded=(active_section == "results")):
            _nav_btn("📈", "Compare Models", "compare",
                     active, "pages/5_Comparison.py")

        # ── INFO (collapseable) ────────────────────────────────
        with st.expander("ℹ️  INFO",
                         expanded=(active_section == "info")):
            _nav_btn("👥", "About", "about", active, "pages/7_About.py")
            _nav_btn("❓", "FAQ",   "faq",   active, "pages/8_FAQ.py")

        # ── Footer ────────────────────────────────────────────
        st.markdown("""
        <div style="height:1px; background:#1A1F35; margin:14px 16px 12px;"></div>
        <div style="padding:0 20px 24px;">
            <div style="font-size:11px; color:#2E3447; line-height:1.9;">
                Tutorial Group 3<br>
                Tutor: Dr Goh · 202605<br>
                TARUMT
            </div>
        </div>
        """, unsafe_allow_html=True)


def _nav_btn(icon, label, key, active, page):
    if active == key:
        st.markdown(f"""
        <div style="
            display:flex; align-items:center; gap:10px;
            background:rgba(91,127,255,0.12);
            border-left:3px solid #5B7FFF;
            border-radius:0 8px 8px 0;
            padding:8px 12px 8px 17px;
            margin:1px 8px 1px 0;
            font-size:13px; font-weight:600; color:#FFFFFF;
            cursor:default;">
            <span>{icon}</span><span>{label}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button(f"{icon}  {label}", key=f"sb_{key}",
                     use_container_width=True):
            try:
                st.switch_page(page)
            except Exception:
                st.error(f"Page not found: {page}")