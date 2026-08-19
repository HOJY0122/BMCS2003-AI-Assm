import streamlit as st

SIDEBAR_CSS = """
<style>
/* Hide only what we need to hide - NOT the whole header */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Hide default streamlit page nav in sidebar */
[data-testid="stSidebarNav"] { display: none !important; }

/* Sidebar background */
[data-testid="stSidebar"] {
    background: #0F1117 !important;
}

/* All text inside sidebar white */
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.8) !important;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: rgba(255,255,255,0.65) !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 9px 14px !important;
    text-align: left !important;
    transition: all 0.15s !important;
    margin-bottom: 2px !important;
    width: 100% !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: white !important;
}

section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none !important;
    outline: none !important;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #1E2235;
    border: 1px solid #2A2D3E;
    border-radius: 10px;
    padding: 14px 16px;
}
div[data-testid="metric-container"] label {
    font-size: 11px !important;
    color: #8B92A5 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
div[data-testid="metric-container"] [data-testid="metric-value"] {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #4F7CFF !important;
}
</style>
"""


def sidebar(active="home"):
    st.markdown(SIDEBAR_CSS, unsafe_allow_html=True)

    with st.sidebar:
        # Brand
        st.markdown("""
        <div style="padding:20px 16px 14px; border-bottom:1px solid #1E2235; margin-bottom:8px;">
            <div style="font-size:20px; font-weight:800; color:white !important; letter-spacing:-0.5px;">
                Mind<span style="color:#4F7CFF;">Check</span>
            </div>
            <div style="font-size:11px; color:#4A5060 !important; margin-top:3px;">
                BMCS2003 · AI · TARUMT
            </div>
        </div>
        """, unsafe_allow_html=True)

        # MAIN
        _label("MAIN")
        _btn("🏠  Home",           "home",    active, "Home.py")

        # ANALYSIS
        _label("ANALYSIS")
        _btn("📊  EDA",            "eda",     active, "pages/1_EDA.py")
        _btn("📋  Dataset",        "dataset", active, "pages/6_Dataset.py")

        # MODELS
        _label("MODELS")
        _btn("🔵  KNN",            "knn",     active, "pages/2_KNN.py")
        _btn("🌳  Decision Tree",  "dt",      active, "pages/3_Decision_Tree.py")
        _btn("🔴  SVM",            "svm",     active, "pages/4_SVM.py")

        # RESULTS
        _label("RESULTS")
        _btn("📈  Compare Models", "compare", active, "pages/5_Comparison.py")

        # INFO
        _label("INFO")
        _btn("ℹ️  About",          "about",   active, "pages/7_About.py")
        _btn("❓  FAQ",            "faq",     active, "pages/8_FAQ.py")

        st.divider()
        st.caption("Group 3 · Tutor: Dr Goh\n202605 · TARUMT")


def _label(text):
    st.markdown(
        f"<p style='font-size:10px; font-weight:700; color:#555E75 !important; "
        f"letter-spacing:1.5px; text-transform:uppercase; "
        f"margin:14px 0 4px 4px; padding:0;'>{text}</p>",
        unsafe_allow_html=True
    )


def _btn(label, key, active, page):
    if active == key:
        # Active page — shown as highlighted div (not clickable)
        st.markdown(
            f"<div style='background:rgba(79,124,255,0.18); "
            f"border-left:3px solid #4F7CFF; border-radius:8px; "
            f"padding:9px 14px; margin-bottom:2px; "
            f"font-size:13px; font-weight:600; color:white !important;'>"
            f"{label}</div>",
            unsafe_allow_html=True
        )
    else:
        if st.button(label, key=f"sb_{key}", use_container_width=True):
            st.switch_page(page)
