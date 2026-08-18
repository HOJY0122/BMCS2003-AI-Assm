import streamlit as st

def render_sidebar(active_page=""):
    """Shared sidebar navigation for all pages"""
    with st.sidebar:
        st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background: #12152A !important;
            border-right: 1px solid rgba(255,255,255,0.06) !important;
        }
        [data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        section[data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,0.05) !important;
            color: rgba(255,255,255,0.7) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 8px !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            padding: 9px 14px !important;
            text-align: left !important;
            transition: all 0.18s !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(61,82,255,0.18) !important;
            border-color: #3D52FF !important;
            color: white !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Brand
        st.markdown("""
        <div style="padding:22px 18px 14px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:12px;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:700;color:white;">
                Mind<span style="color:#6B86FF;">Check</span>
            </div>
            <div style="font-size:11px;color:rgba(255,255,255,0.28);margin-top:3px;">
                BMCS2003 Artificial Intelligence
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Main
        _section("MAIN")
        if st.button("Home", key="sb_home", width='stretch'):
            st.switch_page("Home.py")

        # Analysis
        _section("ANALYSIS")
        if st.button("Exploratory Data Analysis", key="sb_eda", width='stretch'):
            st.switch_page("pages/1_EDA.py")
        if st.button("Dataset Overview", key="sb_ds", width='stretch'):
            st.switch_page("pages/6_Dataset.py")

        # Models
        _section("MODELS")
        if st.button("KNN — Ho Jun Yon", key="sb_knn", width='stretch'):
            st.switch_page("pages/2_KNN.py")
        if st.button("Decision Tree — Irvin", key="sb_dt", width='stretch'):
            st.switch_page("pages/3_Decision_Tree.py")
        if st.button("SVM — Chiang Jun Hang", key="sb_svm", width='stretch'):
            st.switch_page("pages/4_SVM.py")

        # Compare
        _section("COMPARE")
        if st.button("Compare All Models", key="sb_cmp", width='stretch'):
            st.switch_page("pages/5_Comparison.py")

        # Footer
        st.markdown("""
        <div style="margin-top:24px;padding:12px 0;
                    border-top:1px solid rgba(255,255,255,0.06);
                    font-size:11px;color:rgba(255,255,255,0.2);line-height:1.8;">
            Tutorial Group 3 · Tutor: Dr Goh<br>
            202605 Session · TARUMT
        </div>
        """, unsafe_allow_html=True)

def _section(label):
    st.markdown(f"""
    <div style="font-size:10px;font-weight:700;
                color:rgba(255,255,255,0.22);
                letter-spacing:1.8px;text-transform:uppercase;
                padding:12px 0 6px;">
        {label}
    </div>
    """, unsafe_allow_html=True)
