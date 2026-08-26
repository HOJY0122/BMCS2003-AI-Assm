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

# ── Load data ─────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_and_clean_dataset('dataset/Student_Mental_health.csv')

df = get_data()

@st.cache_data
def get_raw_missing_age():
    """Missing-value count is measured on the RAW file before cleaning,
    since load_and_clean_dataset() already fills Age NaNs with the
    median — so it can't be recovered from the cleaned df anymore."""
    raw = pd.read_csv('dataset/Student_Mental_health.csv')
    return int(raw['Age'].isna().sum())

# Metrics computed live from the loaded dataset, not hardcoded — these
# stay correct if the dataset file is ever swapped or grows.
_n           = len(df)
_dep_n       = int((df['Depression'] == 1).sum())
_anx_n       = int((df['Anxiety'] == 1).sum())
_pan_n       = int((df['Panic_Attack'] == 1).sum())
_missing_age = get_raw_missing_age()
_median_age  = int(df['Age'].median())

# ── Header ────────────────────────────────────────────────────
st.markdown("##### 📋 DATASET")
st.title("Student Mental Health Dataset")
st.write(
    f"Source: Kaggle — Shariful07 (2020) · {_n} student records · "
    "IIUM Malaysia · 11 original features"
)
st.divider()

# ── Overview metrics ──────────────────────────────────────────
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Total Records", f"{_n}")
m2.metric("Original Features", "11")
m3.metric("Depression (Yes)", f"{_dep_n}", delta=f"{_dep_n/_n*100:.0f}% of total", delta_color="off")
m4.metric("Anxiety (Yes)", f"{_anx_n}", delta=f"{_anx_n/_n*100:.0f}% of total", delta_color="off")
m5.metric("Panic Attack (Yes)", f"{_pan_n}", delta=f"{_pan_n/_n*100:.0f}% of total", delta_color="off")
m6.metric("Missing Values", f"{_missing_age}", delta="Age column", delta_color="off")

st.divider()

# ── Feature description table ────────────────────────────────
st.subheader("Feature Description")
feature_table = pd.DataFrame([
    ["Gender",         "Choose your gender",              "Student gender",                 "Male / Female → 1 / 0",       "Input"],
    ["Age",            "Age",                             "Student age",                     "17–24 (numeric)",             "Input"],
    ["Course",         "What is your course?",            "Field of study (grouped)",        "10 categories",               "Input"],
    ["Year_of_Study",  "Your current year of Study",       "Academic year",                   "Year 1–4",                    "Input"],
    ["CGPA",           "What is your CGPA?",               "GPA range",                       "0–1.99 to 3.50–4.00",         "Input"],
    ["Marital_Status", "Marital status",                   "Marital status",                  "Yes / No → 1 / 0",            "Input"],
    ["Depression",     "Do you have Depression?",          "Has depression?",                 "Yes / No → 1 / 0",            "Target (KNN & DT)"],
    ["Anxiety",        "Do you have Anxiety?",             "Has anxiety?",                    "Yes / No → 1 / 0",            "Input"],
    ["Panic_Attack",   "Do you have Panic attack?",        "Has panic attacks?",              "Yes / No → 1 / 0",            "Target (SVM)"],
    ["Seek_Treatment", "Did you seek any specialist...",   "Sought specialist help?",         "Yes / No → 1 / 0",            "Input"],
], columns=["Feature Name", "Original Column", "Description", "Values", "Role"])
feature_table.index = range(1, len(feature_table) + 1)

with st.expander("View All Features", expanded=True):
    st.dataframe(feature_table, width='stretch')

st.divider()

# ── Engineered features ──────────────────────────────────────
st.subheader("Engineered Features")
e1, e2, e3 = st.columns(3)
with e1:
    with st.container(border=True):
        st.markdown("**🧠 Mental Health Score**")
        st.write("**Formula:** Depression + Anxiety + Panic_Attack")
        st.write("**Range:** 0 to 3")
        st.caption("0 = Healthy · 1 = Mild · 2 = Moderate · 3 = Severe")
with e2:
    with st.container(border=True):
        st.markdown("**🎂 Age Group**")
        st.write("Groups student age into categories:")
        st.write("**Early** = Age 18–19")
        st.write("**Mid** = Age 20–21")
        st.write("**Senior** = Age 22+")
with e3:
    with st.container(border=True):
        st.markdown("**📊 CGPA Numeric**")
        st.write("Converts CGPA range to midpoint:")
        st.caption(
            "0–1.99 → 1.00 · 2.00–2.49 → 2.25 · 2.50–2.99 → 2.75 · "
            "3.00–3.49 → 3.25 · 3.50–4.00 → 3.75"
        )

st.divider()

# ── Preprocessing steps ───────────────────────────────────────
st.subheader("Preprocessing Steps Applied")
with st.container(border=True):
    st.markdown("**1. Column Renaming**")
    st.caption(
        "All 11 column names standardized to short machine-readable "
        'identifiers. Example: "Choose your gender" → "Gender", '
        '"What is your CGPA?" → "CGPA".'
    )
    st.divider()
    st.markdown("**2. Missing Value Handling**")
    st.caption(
        f"{_missing_age} missing values in the Age column were filled "
        f"with the median age value ({_median_age} years) to preserve "
        "dataset integrity without removing rows."
    )
    st.divider()
    st.markdown("**3. Inconsistency Correction**")
    st.caption(
        'Course name variations were grouped into standard categories '
        '(e.g., "KOE", "koe", "Koe" → "Engineering"). Year of study '
        'inconsistencies ("year 1" vs "Year 1") were standardized.'
    )
    st.divider()
    st.markdown("**4. Binary Encoding**")
    st.caption(
        "All Yes/No columns (Depression, Anxiety, Panic Attack, Seek "
        "Treatment, Marital Status) and Gender were encoded as binary "
        "integers: 1 = Yes/Male, 0 = No/Female."
    )
    st.divider()
    st.markdown("**5. Feature Engineering**")
    st.caption(
        "Three new derived features were created: Mental Health Score "
        "(sum of 3 conditions, range 0–3), Age Group (Early/Mid/Senior), "
        "and CGPA Numeric (midpoint value of CGPA range)."
    )

st.divider()

# ── Dataset preview ───────────────────────────────────────────
st.subheader("Cleaned Dataset Preview")
st.dataframe(df.head(20), width='stretch')

st.divider()

# ── Statistical summary ──────────────────────────────────────
st.subheader("Statistical Summary")
st.dataframe(df.describe().style.format("{:.2f}"), width='stretch')

st.divider()

# ── Navigate ──────────────────────────────────────────────────
n1, n2, _ = st.columns([1, 1, 4])
with n1:
    if st.button("Explore EDA Visualizations", width='stretch'):
        st.switch_page("pages/1_EDA.py")
with n2:
    if st.button("Back to Home", width='stretch'):
        st.switch_page("Home.py")