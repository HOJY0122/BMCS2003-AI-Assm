import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import re
import os, sys

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

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
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════
def parse_year(val):
    if pd.isna(val): return 1
    numbers = re.findall(r'\d+', str(val).lower())
    return int(numbers[0]) if numbers else 1

def is_yes(val):
    return str(val).strip().lower() == 'yes'

CGPA_MAP = {
    '0 - 1.99'   : 1.00,
    '2.00 - 2.49': 2.25,
    '2.50 - 2.99': 2.75,
    '3.00 - 3.49': 3.25,
    '3.50 - 4.00': 3.75,
}

# ══════════════════════════════════════════════════════════════
# CUSTOM DECISION TREE LOGIC (Irvin's tree)
# ══════════════════════════════════════════════════════════════
def traverse_health_subtree(inputs, path_nodes, prefix='R'):
    path_nodes.append(f'Marital ? ({prefix})')
    if is_yes(inputs['marital']):
        path_nodes.append(f'Depress ({prefix}_Marital)')
        return 'Depress', path_nodes
    path_nodes.append(f'Anxiety ? ({prefix})')
    if is_yes(inputs['anxiety']):
        path_nodes.append(f'Depress ({prefix}_Anxiety)')
        return 'Depress', path_nodes
    path_nodes.append(f'Panic ? ({prefix})')
    if is_yes(inputs['panic']):
        path_nodes.append(f'Depress ({prefix}_Panic)')
        return 'Depress', path_nodes
    path_nodes.append(f'Treatment ? ({prefix})')
    if is_yes(inputs['treatment']):
        path_nodes.append(f'Depress ({prefix}_Treatment)')
        return 'Depress', path_nodes
    else:
        path_nodes.append(f'No Depress ({prefix}_Treatment)')
        return 'No Depress', path_nodes

def predict_custom_tree(inputs):
    path_nodes = []
    path_nodes.append('age >= 20 ?')
    if inputs['age'] >= 20:
        path_nodes.append('Year of Study >= 3 ? (R)')
        if inputs['year'] >= 3:
            path_nodes.append('Depress (R_Year)')
            return 'Depress', path_nodes
        else:
            path_nodes.append('CGPA <= 2.0 ? (R)')
            if inputs['cgpa'] <= 2.0:
                path_nodes.append('Depress (R_CGPA)')
                return 'Depress', path_nodes
            else:
                return traverse_health_subtree(inputs, path_nodes, prefix='R')
    else:
        path_nodes.append('CGPA <= 2.0 ? (L)')
        if inputs['cgpa'] <= 2.0:
            path_nodes.append('Depress (L_CGPA)')
            return 'Depress', path_nodes
        else:
            return traverse_health_subtree(inputs, path_nodes, prefix='L')

# ══════════════════════════════════════════════════════════════
# EVALUATE ON DATASET
# ══════════════════════════════════════════════════════════════
@st.cache_data
def evaluate_on_dataset():
    df_raw = pd.read_csv('dataset/Student_Mental_health.csv')
    preds, actuals = [], []
    for _, row in df_raw.iterrows():
        inp = {
            'age'      : int(row['Age']) if pd.notna(row['Age']) else 20,
            'cgpa'     : CGPA_MAP.get(str(row['What is your CGPA?']).strip(), 3.0),
            'year'     : parse_year(row['Your current year of Study']),
            'marital'  : row['Marital status'],
            'anxiety'  : row['Do you have Anxiety?'],
            'panic'    : row['Do you have Panic attack?'],
            'treatment': row['Did you seek any specialist for a treatment?'],
        }
        pred, _ = predict_custom_tree(inp)
        preds.append(1 if pred == 'Depress' else 0)
        actuals.append(1 if row['Do you have Depression?'] == 'Yes' else 0)
    return actuals, preds, df_raw

actuals, preds, df_raw = evaluate_on_dataset()

# ══════════════════════════════════════════════════════════════
# SECTION 1: ALGORITHM OVERVIEW
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 1: Algorithm Overview</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.info("**Algorithm:** Custom Decision Tree\n\n**Type:** Rule-based Classification")
c2.info("**Target:** Depression Prediction\n\n**Output:** Depress / No Depress")
c3.info(f"**Dataset Size:** {len(actuals)} records\n\n**Approach:** Manual Tree Traversal")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
**Decision Rules Applied (in order):**

The custom Decision Tree evaluates student features using the following rule hierarchy:

1. **Age >= 20** — Root split. Separates older and younger students.
2. **Year of Study >= 3** (if Age >= 20) — Senior students more likely depressed.
3. **CGPA <= 2.0** — Low CGPA students flagged as depressed.
4. **Marital Status** — Married students flagged as depressed.
5. **Anxiety** — Students with anxiety flagged as depressed.
6. **Panic Attack** — Students with panic attacks flagged as depressed.
7. **Seek Treatment** — Students seeking treatment flagged as depressed.
8. If none of the above apply — **No Depression** predicted.
""")

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 2: DECISION TREE DIAGRAM (no user input yet)
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 2: Decision Tree Structure</div>',
            unsafe_allow_html=True)
st.write("Full decision tree structure. Red highlighted path shows the active prediction route after input.")

def render_tree(active_path=[]):
    G = nx.DiGraph()

    pos = {
        "age >= 20 ?"              : (0, 7),
        "CGPA <= 2.0 ? (L)"        : (-4, 6),
        "Depress (L_CGPA)"         : (-2, 5),
        "Marital ? (L)"            : (-6, 5),
        "Depress (L_Marital)"      : (-4, 4),
        "Anxiety ? (L)"            : (-8, 4),
        "Depress (L_Anxiety)"      : (-6, 3),
        "Panic ? (L)"              : (-10, 3),
        "Depress (L_Panic)"        : (-8, 2),
        "Treatment ? (L)"          : (-12, 2),
        "Depress (L_Treatment)"    : (-10, 1),
        "No Depress (L_Treatment)" : (-14, 1),
        "Year of Study >= 3 ? (R)" : (4, 6),
        "Depress (R_Year)"         : (8, 5),
        "CGPA <= 2.0 ? (R)"        : (2, 5),
        "Depress (R_CGPA)"         : (4, 4),
        "Marital ? (R)"            : (0, 4),
        "Depress (R_Marital)"      : (2, 3),
        "Anxiety ? (R)"            : (-2, 3),
        "Depress (R_Anxiety)"      : (0, 2),
        "Panic ? (R)"              : (-4, 2),
        "Depress (R_Panic)"        : (-2, 1),
        "Treatment ? (R)"          : (-6, 1),
        "Depress (R_Treatment)"    : (-4, 0),
        "No Depress (R_Treatment)" : (-8, 0),
    }

    labels = {
        "age >= 20 ?"              : "age >= 20 ?",
        "CGPA <= 2.0 ? (L)"        : "CGPA <= 2.0 ?",
        "Depress (L_CGPA)"         : "Depress",
        "Marital ? (L)"            : "Marital ?",
        "Depress (L_Marital)"      : "Depress",
        "Anxiety ? (L)"            : "Anxiety ?",
        "Depress (L_Anxiety)"      : "Depress",
        "Panic ? (L)"              : "Panic ?",
        "Depress (L_Panic)"        : "Depress",
        "Treatment ? (L)"          : "Treatment ?",
        "Depress (L_Treatment)"    : "Depress",
        "No Depress (L_Treatment)" : "No Depress",
        "Year of Study >= 3 ? (R)" : "Year >= 3 ?",
        "Depress (R_Year)"         : "Depress",
        "CGPA <= 2.0 ? (R)"        : "CGPA <= 2.0 ?",
        "Depress (R_CGPA)"         : "Depress",
        "Marital ? (R)"            : "Marital ?",
        "Depress (R_Marital)"      : "Depress",
        "Anxiety ? (R)"            : "Anxiety ?",
        "Depress (R_Anxiety)"      : "Depress",
        "Panic ? (R)"              : "Panic ?",
        "Depress (R_Panic)"        : "Depress",
        "Treatment ? (R)"          : "Treatment ?",
        "Depress (R_Treatment)"    : "Depress",
        "No Depress (R_Treatment)" : "No Depress",
    }

    edges = [
        ("age >= 20 ?", "CGPA <= 2.0 ? (L)", "False"),
        ("age >= 20 ?", "Year of Study >= 3 ? (R)", "True"),
        ("CGPA <= 2.0 ? (L)", "Depress (L_CGPA)", "True"),
        ("CGPA <= 2.0 ? (L)", "Marital ? (L)", "False"),
        ("Marital ? (L)", "Depress (L_Marital)", "True"),
        ("Marital ? (L)", "Anxiety ? (L)", "False"),
        ("Anxiety ? (L)", "Depress (L_Anxiety)", "True"),
        ("Anxiety ? (L)", "Panic ? (L)", "False"),
        ("Panic ? (L)", "Depress (L_Panic)", "True"),
        ("Panic ? (L)", "Treatment ? (L)", "False"),
        ("Treatment ? (L)", "Depress (L_Treatment)", "True"),
        ("Treatment ? (L)", "No Depress (L_Treatment)", "False"),
        ("Year of Study >= 3 ? (R)", "Depress (R_Year)", "True"),
        ("Year of Study >= 3 ? (R)", "CGPA <= 2.0 ? (R)", "False"),
        ("CGPA <= 2.0 ? (R)", "Depress (R_CGPA)", "True"),
        ("CGPA <= 2.0 ? (R)", "Marital ? (R)", "False"),
        ("Marital ? (R)", "Depress (R_Marital)", "True"),
        ("Marital ? (R)", "Anxiety ? (R)", "False"),
        ("Anxiety ? (R)", "Depress (R_Anxiety)", "True"),
        ("Anxiety ? (R)", "Panic ? (R)", "False"),
        ("Panic ? (R)", "Depress (R_Panic)", "True"),
        ("Panic ? (R)", "Treatment ? (R)", "False"),
        ("Treatment ? (R)", "Depress (R_Treatment)", "True"),
        ("Treatment ? (R)", "No Depress (R_Treatment)", "False"),
    ]

    for u, v, w in edges:
        G.add_edge(u, v, label=w)

    # Node colors
    node_colors, border_colors, border_widths = [], [], []
    for node in G.nodes():
        if node in active_path:
            border_colors.append('#e74c3c')
            border_widths.append(3.0)
        else:
            border_colors.append('#2c3e50')
            border_widths.append(1.0)

        lbl = labels.get(node, node)
        if lbl == 'Depress':
            node_colors.append('#fadbd8')
        elif lbl == 'No Depress':
            node_colors.append('#d5f5e3')
        elif '?' in lbl:
            node_colors.append('#d6eaf8')
        else:
            node_colors.append('#ffffff')

    fig, ax = plt.subplots(figsize=(20, 11))
    nx.draw_networkx_nodes(G, pos, node_shape='s', node_size=4500,
                           node_color=node_colors, edgecolors=border_colors,
                           linewidths=border_widths, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7.5,
                            font_family='sans-serif', ax=ax)

    edge_colors, edge_widths = [], []
    for u, v in G.edges():
        if (u in active_path and v in active_path and
                active_path.index(v) == active_path.index(u) + 1):
            edge_colors.append('#e74c3c')
            edge_widths.append(2.5)
        else:
            edge_colors.append('#7f8c8d')
            edge_widths.append(1.0)

    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=15, ax=ax)
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=nx.get_edge_attributes(G, 'label'),
                                 font_size=7.5, font_color='#2874A6', ax=ax)

    # Legend
    legend_items = [
        mpatches.Patch(color='#d6eaf8', label='Decision Node'),
        mpatches.Patch(color='#fadbd8', label='Depress (Leaf)'),
        mpatches.Patch(color='#d5f5e3', label='No Depress (Leaf)'),
        mpatches.Patch(edgecolor='#e74c3c', facecolor='white',
                       label='Active Path', linewidth=2),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor='#cccccc')

    ax.set_title('Decision Tree Structure — Red border shows active prediction path',
                 fontsize=13, fontweight='bold', pad=12)
    ax.axis('off')
    plt.tight_layout()
    return fig

# Show tree with no active path initially
st.pyplot(render_tree([]))

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 3: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 3: Model Evaluation</div>',
            unsafe_allow_html=True)
st.write(f"Evaluation of the custom Decision Tree on all {len(actuals)} records in the dataset.")

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
    import seaborn as sns
    cm  = confusion_matrix(actuals, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Depression', 'Depression'],
                yticklabels=['No Depression', 'Depression'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Decision Tree Confusion Matrix')
    st.pyplot(fig)
    plt.close()

with col_cr:
    st.markdown("**Classification Report**")
    report    = classification_report(actuals, preds,
                                      target_names=['No Depression', 'Depression'],
                                      output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Evaluation Notes**")
    st.info(
        "The custom Decision Tree achieves high Recall, meaning it effectively "
        "identifies most students who are depressed. Precision is moderate, "
        "indicating some false positives. This is acceptable for mental health "
        "screening where missing a depressed student (false negative) is more "
        "costly than a false alarm."
    )

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 4: PREDICTION DISTRIBUTION
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 4: Prediction Distribution on Dataset</div>',
            unsafe_allow_html=True)

p1, p2 = st.columns(2)

with p1:
    pred_counts   = pd.Series(preds).value_counts().sort_index()
    actual_counts = pd.Series(actuals).value_counts().sort_index()
    x  = np.arange(2)
    w  = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w/2, actual_counts.values, w, label='Actual',    color='#2874A6', edgecolor='white')
    ax.bar(x + w/2, pred_counts.values,   w, label='Predicted', color='#E74C3C', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(['No Depression', 'Depression'])
    ax.set_ylabel('Count')
    ax.set_title('Actual vs Predicted Distribution')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)
    plt.close()

with p2:
    labels_pie = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    cm_flat    = confusion_matrix(actuals, preds).ravel()
    colors_pie = ['#2ecc71', '#e74c3c', '#f39c12', '#2874A6']
    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        cm_flat, labels=labels_pie, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=2)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight('bold')
    ax.set_title('Prediction Breakdown', fontsize=11, fontweight='bold')
    st.pyplot(fig)
    plt.close()

st.markdown("---")

# ══════════════════════════════════════════════════════════════
# SECTION 5: STUDENT PREDICTION FORM
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Step 5: Student Depression Prediction</div>',
            unsafe_allow_html=True)
st.write("Fill in the student information below to predict depression status using the Decision Tree.")

with st.form("dt_prediction_form"):
    st.markdown("**Student Information**")

    col_a, col_b = st.columns(2)

    with col_a:
        name      = st.text_input("Name")
        age       = st.slider("Age", 17, 30, 20)
        cgpa_str  = st.selectbox("CGPA Range", list(CGPA_MAP.keys()))
        year_str  = st.selectbox("Year of Study",
                                 ["Year 1", "Year 2", "Year 3", "Year 4"])

    with col_b:
        marital   = st.selectbox("Marital Status",                        ["No", "Yes"])
        anxiety   = st.selectbox("Do you have Anxiety?",                  ["No", "Yes"])
        panic     = st.selectbox("Do you have Panic Attack?",             ["No", "Yes"])
        treatment = st.selectbox("Did you seek Specialist Treatment?",    ["No", "Yes"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    parsed_inputs = {
        'age'      : age,
        'cgpa'     : CGPA_MAP[cgpa_str],
        'year'     : parse_year(year_str),
        'marital'  : marital,
        'anxiety'  : anxiety,
        'panic'    : panic,
        'treatment': treatment,
    }

    prediction, path_taken = predict_custom_tree(parsed_inputs)
    display_name = name.strip() if name.strip() != '' else 'Student'

    st.markdown("---")
    st.subheader("Prediction Result")

    res1, res2 = st.columns(2)
    with res1:
        if prediction == 'Depress':
            st.error(
                f"Result for {display_name}: DEPRESSION DETECTED\n\n"
                "The Decision Tree model predicts this student may have depression. "
                "Please consider seeking professional help or speaking with a counsellor."
            )
        else:
            st.success(
                f"Result for {display_name}: NO DEPRESSION\n\n"
                "The Decision Tree model predicts this student does not show signs "
                "of depression. Keep maintaining a healthy lifestyle and academic balance."
            )

    with res2:
        st.markdown("**Decision Path Taken**")
        for i, step in enumerate(path_taken):
            if 'Depress' in step and 'No' not in step:
                st.markdown(f"**Step {i+1}: {step}** → Depressed")
            elif 'No Depress' in step:
                st.markdown(f"**Step {i+1}: {step}** → Not Depressed")
            else:
                st.markdown(f"Step {i+1}: {step}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Decision Tree — Active Path Highlighted**")
    fig = render_tree(path_taken)
    st.pyplot(fig)
    plt.close()

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