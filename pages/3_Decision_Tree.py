import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sys

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Decision Tree - Depression Prediction",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    h1 { color: #1a1a2e; font-family: Arial, sans-serif; }
    h2 { color: #16213e; font-family: Arial, sans-serif; }
    h3 { color: #0f3460; font-family: Arial, sans-serif; }
    .section-title {
        font-size: 18px;
        font-weight: bold;
        color: #0f3460;
        border-bottom: 2px solid #0f3460;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ── Title ──────────────────────────────────────────────────────
st.title("Decision Tree - Depression Prediction")
st.markdown("**Member 2: Irvin Tan Wei Shen**")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS & DATA PARSING
# ══════════════════════════════════════════════════════════════
def parse_year(val):
    if pd.isna(val): return 1
    numbers = re.findall(r'\d+', str(val))
    return int(numbers[0]) if numbers else 1

def encode_binary(val):
    if pd.isna(val):
        return 0
    val_str = str(val).strip().lower()
    return 1 if val_str in ['yes', '1', 'true', '1.0'] else 0

CGPA_MAP = {
    '0 - 1.99'   : 1.00,
    '2.00 - 2.49': 2.25,
    '2.50 - 2.99': 2.75,
    '3.00 - 3.49': 3.25,
    '3.50 - 4.00': 3.75,
}

# Features using explicit rule thresholds to guarantee logical splits
FEATURE_COLS = ['marital', 'anxiety', 'treatment', 'cgpa_le_2_5', 'year_ge_3', 'age_le_20', 'panic']

# Display names for clean decision tree visualization
FEATURE_NAMES_DISPLAY = [
    'marital <= 0.5',
    'anxiety <= 0.5',
    'treatment <= 0.5',
    'cgpa <= 2.5',
    'year >= 3',
    'age <= 20.0',
    'panic <= 0.5'
]

# ══════════════════════════════════════════════════════════════
# MODEL TRAINING & EVALUATION (SCIKIT-LEARN FIT)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def train_and_evaluate_model():
    try:
        df_raw = load_and_clean_dataset()
    except Exception:
        df_raw = pd.read_csv('dataset/Student_Mental_health.csv')

    def find_col(candidates):
        for col in df_raw.columns:
            cleaned_col = str(col).strip().lower()
            for cand in candidates:
                if cand.lower() in cleaned_col:
                    return col
        return None

    age_col       = find_col(['age'])
    cgpa_col      = find_col(['cgpa', 'what is your cgpa'])
    year_col      = find_col(['year', 'your current year of study'])
    marital_col   = find_col(['marital'])
    anxiety_col   = find_col(['anxiety'])
    panic_col     = find_col(['panic'])
    treatment_col = find_col(['treatment', 'specialist'])
    target_col    = find_col(['depression', 'depress'])

    def parse_cgpa(val):
        val_str = str(val).strip()
        if val_str in CGPA_MAP:
            return CGPA_MAP[val_str]
        try:
            return float(val_str)
        except ValueError:
            return 3.0

    df_proc = pd.DataFrame()
    
    # Extract raw values
    raw_age  = df_raw[age_col].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if pd.notna(x) and re.findall(r'\d+', str(x)) else 20) if age_col else 20
    raw_cgpa = df_raw[cgpa_col].apply(parse_cgpa) if cgpa_col else 3.0
    raw_year = df_raw[year_col].apply(parse_year) if year_col else 1

    # Map variables to binary indicators matching target logic
    df_proc['marital']     = df_raw[marital_col].apply(encode_binary) if marital_col else 0
    df_proc['anxiety']     = df_raw[anxiety_col].apply(encode_binary) if anxiety_col else 0
    df_proc['treatment']   = df_raw[treatment_col].apply(encode_binary) if treatment_col else 0
    df_proc['panic']       = df_raw[panic_col].apply(encode_binary) if panic_col else 0
    df_proc['cgpa_le_2_5'] = (raw_cgpa <= 2.5).astype(int)
    df_proc['year_ge_3']   = (raw_year >= 3).astype(int)
    df_proc['age_le_20']   = (raw_age <= 20.0).astype(int)
    df_proc['target']      = df_raw[target_col].apply(encode_binary) if target_col else 0

    X = df_proc[FEATURE_COLS]
    y = df_proc['target']

    # Train decision tree classifier
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=4,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X, y)

    preds = clf.predict(X)
    actuals = y.values

    return clf, X, actuals, preds, df_raw

clf, X_data, actuals, preds, df_raw = train_and_evaluate_model()

# Decision path trace for single student sample
def trace_decision_path(model, sample_df):
    node_indicator = model.decision_path(sample_df)
    leaf_id = model.apply(sample_df)
    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
    
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    
    path_steps = []
    for node_id in node_index:
        if leaf_id[0] == node_id:
            continue
        
        feat_name = FEATURE_COLS[feature[node_id]]
        val = sample_df[feat_name].iloc[0]
        thresh = threshold[node_id]
        
        # Display readable rule context
        if feat_name == 'cgpa_le_2_5':
            cond = f"cgpa <= 2.5" if val == 1 else "cgpa > 2.5"
        elif feat_name == 'year_ge_3':
            cond = f"year >= 3" if val == 1 else "year < 3"
        elif feat_name == 'age_le_20':
            cond = f"age <= 20.0" if val == 1 else "age > 20.0"
        else:
            cond = f"{feat_name} <= {thresh:.1f}" if val <= thresh else f"{feat_name} > {thresh:.1f}"
            
        path_steps.append(cond)
        
    return path_steps

# ══════════════════════════════════════════════════════════════
# SECTION 1: ALGORITHM OVERVIEW
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 1: Algorithm Overview</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.info("**Algorithm:** DecisionTreeClassifier (scikit-learn)\n\n**Type:** Supervised Learning (CART)")
c2.info("**Target:** Depression Prediction\n\n**Output:** Depress (1) / No Depress (0)")
c3.info(f"**Dataset Size:** {len(actuals)} records\n\n**Max Depth:** 4 Levels")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
**Model Training Strategy:**

The decision tree is dynamically fit using scikit-learn CART classifier on the structured feature set:

1. **Target Rules:**
   - **CGPA Condition:** `cgpa <= 2.5`
   - **Year Condition:** `year >= 3`
   - **Age Condition:** `age <= 20.0`
2. **Clinical & Behavioral Rules:** `marital`, `anxiety`, `treatment`, and `panic`.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 2: DECISION TREE STRUCTURE DIAGRAM
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 2: Decision Tree Structure</div>', unsafe_allow_html=True)
st.write("Visual structure of the fitted decision tree:")

fig_tree, ax_tree = plt.subplots(figsize=(18, 9))
plot_tree(
    clf, 
    feature_names=FEATURE_NAMES_DISPLAY, 
    class_names=["No Depress", "Depress"], 
    filled=True, 
    rounded=True, 
    fontsize=8,
    ax=ax_tree
)
st.pyplot(fig_tree)
plt.close(fig_tree)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 3: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 3: Model Evaluation</div>', unsafe_allow_html=True)
st.write(f"Evaluation metrics computed across all {len(actuals)} dataset records.")

acc  = accuracy_score(actuals, preds)
prec = precision_score(actuals, preds, zero_division=0)
rec  = recall_score(actuals, preds, zero_division=0)
f1   = f1_score(actuals, preds, zero_division=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{acc*100:.2f}%")
m2.metric("Precision", f"{prec*100:.2f}%")
m3.metric("Recall",    f"{rec*100:.2f}%")
m4.metric("F1 Score",  f"{f1*100:.2f}%")

st.markdown("<br>", unsafe_allow_html=True)

col_cm, col_cr = st.columns(2)

with col_cm:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(actuals, preds, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['No Depression', 'Depression'],
                yticklabels=['No Depression', 'Depression'])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Decision Tree Confusion Matrix')
    st.pyplot(fig_cm)
    plt.close(fig_cm)

with col_cr:
    st.markdown("**Classification Report**")
    report = classification_report(
        actuals, 
        preds,
        labels=[0, 1],
        target_names=['No Depression', 'Depression'],
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 4: PREDICTION DISTRIBUTION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 4: Prediction Distribution on Dataset</div>', unsafe_allow_html=True)

p1, p2 = st.columns(2)

with p1:
    pred_counts   = pd.Series(preds).value_counts()
    actual_counts = pd.Series(actuals).value_counts()
    
    pred_0 = pred_counts.get(0, 0)
    pred_1 = pred_counts.get(1, 0)
    act_0  = actual_counts.get(0, 0)
    act_1  = actual_counts.get(1, 0)

    x = np.arange(2)
    w = 0.35
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    ax_bar.bar(x - w/2, [act_0, act_1], w, label='Actual',    color='#2874A6', edgecolor='white')
    ax_bar.bar(x + w/2, [pred_0, pred_1], w, label='Predicted', color='#E74C3C', edgecolor='white')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['No Depression', 'Depression'])
    ax_bar.set_ylabel('Count')
    ax_bar.set_title('Actual vs Predicted Distribution')
    ax_bar.legend()
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    st.pyplot(fig_bar)
    plt.close(fig_bar)

with p2:
    labels_pie = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    cm_flat    = confusion_matrix(actuals, preds, labels=[0, 1]).ravel()
    colors_pie = ['#2ecc71', '#e74c3c', '#f39c12', '#2874A6']
    fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax_pie.pie(
        cm_flat, labels=labels_pie, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
    ax_pie.set_title('Prediction Breakdown', fontsize=11, fontweight='bold')
    st.pyplot(fig_pie)
    plt.close(fig_pie)

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 5: STUDENT PREDICTION FORM
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 5: Student Depression Prediction</div>', unsafe_allow_html=True)
st.write("Fill in the student information below to run inference through the trained model.")

with st.form("dt_prediction_form"):
    st.markdown("**Student Information**")

    col_a, col_b = st.columns(2)

    with col_a:
        name      = st.text_input("Name")
        age       = st.slider("Age", 17, 30, 20)
        cgpa_str  = st.selectbox("CGPA Range", list(CGPA_MAP.keys()))
        year_str  = st.selectbox("Year of Study", ["Year 1", "Year 2", "Year 3", "Year 4"])

    with col_b:
        marital   = st.selectbox("Marital Status", ["No", "Yes"])
        anxiety   = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
        panic     = st.selectbox("Do you have Panic Attack?", ["No", "Yes"])
        treatment = st.selectbox("Did you seek Specialist Treatment?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    cgpa_val = CGPA_MAP[cgpa_str]
    year_val = parse_year(year_str)

    input_df = pd.DataFrame([{
        'marital'     : encode_binary(marital),
        'anxiety'     : encode_binary(anxiety),
        'treatment'   : encode_binary(treatment),
        'cgpa_le_2_5' : 1 if cgpa_val <= 2.5 else 0,
        'year_ge_3'   : 1 if year_val >= 3 else 0,
        'age_le_20'   : 1 if age <= 20.0 else 0,
        'panic'       : encode_binary(panic)
    }])[FEATURE_COLS]

    pred_class = clf.predict(input_df)[0]
    pred_probs = clf.predict_proba(input_df)[0]
    pred_prob  = pred_probs[1] if len(pred_probs) > 1 else pred_probs[0]
    
    display_name = name.strip() if name.strip() != '' else 'Student'

    st.markdown("---")
    st.subheader("Prediction Result")

    res1, res2 = st.columns(2)
    with res1:
        if pred_class == 1:
            st.error(
                f"Result for **{display_name}**: DEPRESSION DETECTED "
                f"(Probability: {pred_prob:.0%})\n\n"
                "The Decision Tree model predicts this student may show signs of depression. "
                "Consider reaching out to university counseling services or a health specialist."
            )
        else:
            st.success(
                f"Result for **{display_name}**: NO DEPRESSION DETECTED "
                f"(Probability: {pred_prob:.0%})\n\n"
                "The Decision Tree model predicts this student does not show signs "
                "of depression. Continue maintaining academic balance and physical well-being."
            )

    with res2:
        st.markdown("**Decision Path Traversal**")
        path_steps = trace_decision_path(clf, input_df)
        for i, step in enumerate(path_steps):
            st.markdown(f"**Node {i+1}:** Evaluated `{step}`")
        st.markdown(f"**Final Decision Node:** Predicted Class = `{ 'Depress' if pred_class == 1 else 'No Depress' }`")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Input Summary**")
    summary = {
        'Name'          : display_name,
        'Age'           : age,
        'CGPA Range'    : cgpa_str,
        'Year of Study' : year_str,
        'Marital Status': marital,
        'Anxiety'       : anxiety,
        'Panic Attack'  : panic,
        'Seek Treatment': treatment,
    }
    st.table(pd.DataFrame(summary, index=['Value']).T)

    # Render Tree Diagram
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Decision Tree Model Diagram**")
    fig_pred_tree, ax_pred_tree = plt.subplots(figsize=(18, 8))
    plot_tree(
        clf, 
        feature_names=FEATURE_NAMES_DISPLAY, 
        class_names=["No Depress", "Depress"], 
        filled=True, 
        rounded=True, 
        fontsize=8,
        ax=ax_pred_tree
    )
    st.pyplot(fig_pred_tree)
    plt.close(fig_pred_tree)