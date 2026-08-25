import streamlit as st

SIDEBAR_CSS = """
<style>
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Fix collapsed sidebar gap ── */
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0px !important;
    min-width: 0px !important;
    overflow: hidden !important;
}
.main .block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1200px !important;
    transition: all 0.3s ease !important;
}

/* ── Sidebar base ── */
[data-testid="stSidebar"] {
    background: #1E293B !important;
    border-right: 1px solid #2D3F55 !important;
    min-width: 240px !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── All text inside sidebar white ── */
[data-testid="stSidebar"] * {
    color: #CBD5E1 !important;
}

/* ── Nav buttons ── */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #CBD5E1 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 9px 12px 9px 16px !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 2px !important;
    transition: background 0.15s ease, color 0.15s ease !important;
    letter-spacing: 0.1px !important;
    line-height: 1.4 !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.10) !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}
section[data-testid="stSidebar"] .stButton > button p {
    font-size: 14px !important;
    color: inherit !important;
    font-weight: 500 !important;
}

/* ── Expander (section groups) ── */
section[data-testid="stSidebar"] .stExpander {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    margin: 0 !important;
}
section[data-testid="stSidebar"] .stExpander > details {
    border: none !important;
    background: transparent !important;
}
section[data-testid="stSidebar"] .stExpander summary {
    font-size: 11px !important;
    font-weight: 700 !important;
    color: #64748B !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 20px 8px 20px !important;
    background: transparent !important;
    border: none !important;
    user-select: none !important;
}
section[data-testid="stSidebar"] .stExpander summary:hover {
    color: #94A3B8 !important;
}
section[data-testid="stSidebar"] .stExpander summary svg {
    color: #64748B !important;
    width: 14px !important;
    height: 14px !important;
}
section[data-testid="stSidebar"] details[open] summary {
    color: #94A3B8 !important;
}
section[data-testid="stSidebar"] .stExpander > details > div {
    padding: 0 0 6px 0 !important;
    border: none !important;
}

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #64748B !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-weight: 600 !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: #2563EB !important;
}

/* ── Cards ── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {
    background: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8 !important;
}

/* ── Expander in main ── */
.main .stExpander {
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}

/* ── Tab ── */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
}
</style>
"""

_SECTION_MAP = {
    "home"   : "main",
    "eda"    : "analysis",
    "feature": "analysis",
    "dataset": "analysis",
    "stats"  : "analysis",
    "split"  : "analysis",
    "knn"    : "models",
    "dt"     : "models",
    "svm"    : "models",
    "compare": "results",
    "about"  : "info",
    "faq"    : "info",
}


def sidebar(active="home"):
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────
        st.markdown("""
        <div style="padding:24px 20px 16px;">
            <div style="font-size:24px; font-weight:800; color:#FFFFFF;
                        letter-spacing:-0.5px; line-height:1.2;">
                Mind<span style="color:#60A5FA;">Check</span>
            </div>
            <div style="font-size:12px; color:#64748B; margin-top:4px;
                        font-weight:500; letter-spacing:0.2px;">
                BMCS2003 · Artificial Intelligence
            </div>
        </div>
        <div style="height:1px; background:#2D3F55; margin:0 16px 8px;"></div>
        """, unsafe_allow_html=True)

        active_section = _SECTION_MAP.get(active, "main")

        # ── HOME (always visible) ──────────────────────────────
        _nav_btn("🏠", "Home", "home", active, "Home.py")

        st.markdown(
            "<div style='height:1px;background:#2D3F55;margin:8px 16px;'></div>",
            unsafe_allow_html=True
        )

        # ── ANALYSIS ──────────────────────────────────────────
        with st.expander("📂  Analysis",
                         expanded=(active_section == "analysis")):
            _nav_btn("📊", "EDA",               "eda",     active, "pages/1_EDA.py")
            _nav_btn("🔬", "Feature Selection", "feature", active, "pages/9_Feature_Selection.py")
            _nav_btn("✂️", "Train/Test Split",  "split",   active, "pages/12_Train_Test_Split.py")
            _nav_btn("📋", "Dataset",           "dataset", active, "pages/6_Dataset.py")
            _nav_btn("📈", "Live Statistics",   "stats",   active, "pages/11_Live_Stats.py")

        # ── PREDICTORS ────────────────────────────────────────
        with st.expander("🤖  Predictors",
                         expanded=(active_section == "models")):
            _nav_btn("🔵", "KNN",           "knn", active, "pages/2_KNN.py")
            _nav_btn("🌳", "Decision Tree", "dt",  active, "pages/3_Decision_Tree.py")
            _nav_btn("🔴", "SVM",           "svm", active, "pages/4_SVM.py")

        # ── RESULTS ───────────────────────────────────────────
        with st.expander("📊  Results",
                         expanded=(active_section == "results")):
            _nav_btn("⚖️", "Compare Models", "compare",
                     active, "pages/5_Comparison.py")

        # ── INFO ──────────────────────────────────────────────
        with st.expander("ℹ️  Info",
                         expanded=(active_section == "info")):
            _nav_btn("👥", "About", "about", active, "pages/7_About.py")
            _nav_btn("❓", "FAQ",   "faq",   active, "pages/8_FAQ.py")

        # ── Footer ────────────────────────────────────────────
        st.markdown("""
        <div style="height:1px;background:#2D3F55;margin:12px 16px 12px;"></div>
        <div style="padding:0 20px 20px;">
            <div style="font-size:11px; color:#475569; line-height:2.0;
                        font-weight:500;">
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
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(96,165,250,0.18);
            border-left: 3px solid #60A5FA;
            border-radius: 0 8px 8px 0;
            padding: 9px 14px 9px 17px;
            margin: 2px 8px 2px 0;
            font-size: 14px;
            font-weight: 600;
            color: #FFFFFF !important;
            cursor: default;
            letter-spacing: 0.1px;
            line-height: 1.4;">
            <span style="font-size:16px;">{icon}</span>
            <span style="color:#FFFFFF;">{label}</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.button(f"{icon}  {label}", key=f"sb_{key}",
                     use_container_width=True):
            try:
                st.switch_page(page)
            except Exception:
                st.error(f"Page not found: {page}")