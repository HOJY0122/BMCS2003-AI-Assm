import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import sidebar

st.set_page_config(page_title="Decision Tree Predictor", page_icon="🌳",
                   layout="wide", initial_sidebar_state="expanded")
sidebar("dt")

@st.cache_resource
def get_dt():
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())
    le_c = LabelEncoder(); le_y = LabelEncoder()
    df['Course_Enc'] = le_c.fit_transform(df['Course'])
    df['Year_Enc']   = le_y.fit_transform(df['Year_of_Study'])
    feat = ['Gender','Age','Course_Enc','Year_Enc','CGPA_Numeric',
            'Anxiety','Panic_Attack','Marital_Status']
    X = df[feat]; y = df['Depression']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    model = DecisionTreeClassifier(max_depth=5,criterion='gini',random_state=42)
    model.fit(Xtr,ytr); yp = model.predict(Xte)
    return {
        'model': model, 'feat': feat, 'le_c': le_c, 'le_y': le_y,
        'acc' : accuracy_score(yte,yp)*100,
        'prec': precision_score(yte,yp,zero_division=0)*100,
        'rec' : recall_score(yte,yp,zero_division=0)*100,
        'f1'  : f1_score(yte,yp,zero_division=0)*100,
        'fi'  : dict(zip(feat, model.feature_importances_)),
    }

M = get_dt()
CGPA_MAP = {'0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
            '3.00 - 3.49':3.25,'3.50 - 4.00':3.75}

if 'dt_result' not in st.session_state:
    st.session_state.dt_result = None

st.markdown("##### DECISION TREE PREDICTOR")
st.title("Depression Risk Predictor")
st.caption("Decision Tree (CART, Depth 5) · Member 2: Irvin Tan Wei Shen · Live trained on 600 records")
st.divider()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{M['acc']:.2f}%",  help="Live model accuracy on test set")
c2.metric("Precision", f"{M['prec']:.2f}%", help="Of predicted depression, how many correct")
c3.metric("Recall",    f"{M['rec']:.2f}%",  help="Of actual depression cases, how many caught")
c4.metric("F1 Score",  f"{M['f1']:.2f}%",   help="Harmonic mean of precision and recall")
st.divider()

st.subheader("Student Information")
st.caption("Fill in the details below and click Predict")

col1, col2, col3 = st.columns(3)
with col1:
    name    = st.text_input("Name", placeholder="e.g. Siti", key="dt_name")
    gender  = st.selectbox("Gender", ["Female","Male"], key="dt_gender")
    age     = st.slider("Age", 17, 30, 20, key="dt_age")
with col2:
    course  = st.selectbox("Course", [
                "Computer Science","Information Technology","Engineering",
                "Law","Psychology","Language","Islamic Studies",
                "Health Sciences","Business","Science & Math","Arts & Social","Others"],
                key="dt_course")
    year    = st.selectbox("Year of Study",
                ["Year 1","Year 2","Year 3","Year 4"], key="dt_year")
    cgpa    = st.selectbox("CGPA Range", list(CGPA_MAP.keys()), key="dt_cgpa")
with col3:
    marital = st.selectbox("Marital Status",              ["No","Yes"], key="dt_marital")
    anxiety = st.selectbox("Do you have Anxiety?",        ["No","Yes"], key="dt_anxiety")
    panic   = st.selectbox("Do you have Panic Attack?",   ["No","Yes"], key="dt_panic")
    st.write("")
    predict_btn = st.button("Predict Depression Risk",
                            use_container_width=True, key="dt_predict")

if predict_btn:
    g  = 1 if gender  == "Male" else 0
    ax = 1 if anxiety == "Yes"  else 0
    pa = 1 if panic   == "Yes"  else 0
    ma = 1 if marital == "Yes"  else 0
    ce = M['le_c'].transform([course])[0] if course in M['le_c'].classes_ else 0
    ye = M['le_y'].transform([year])[0]   if year   in M['le_y'].classes_ else 0
    cn = CGPA_MAP[cgpa]
    inp  = pd.DataFrame([[g,age,ce,ye,cn,ax,pa,ma]], columns=M['feat'])
    pred = M['model'].predict(inp)[0]
    prob = M['model'].predict_proba(inp)[0]
    st.session_state.dt_result = {
        'pred': int(pred), 'prob': prob.tolist(),
        'name': name.strip() or "Student",
        'gender': gender, 'age': age, 'course': course,
        'year': year, 'cgpa': cgpa, 'marital': marital,
        'anxiety': anxiety, 'panic': panic,
    }

if st.session_state.dt_result:
    R = st.session_state.dt_result
    pred = R['pred']; prob = R['prob']; name_lbl = R['name']
    st.divider()

    if pred == 1:
        st.error(f"## ⚠️ {name_lbl} — Depression Risk Detected")
        st.write("The Decision Tree model predicts a **high risk of depression**. "
                 "Please consider speaking with a counsellor or mental health professional.")
    else:
        st.success(f"## ✅ {name_lbl} — No Depression Detected")
        st.write("The Decision Tree model predicts **low depression risk**. "
                 "Keep maintaining a healthy academic and social lifestyle.")

    st.write("")
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("**Prediction Confidence**")
        fig, ax2 = plt.subplots(figsize=(4, 0.8))
        ax2.barh([""], [prob[0]*100], color="#10B981", height=0.5)
        ax2.barh([""], [prob[1]*100], left=[prob[0]*100], color="#EF4444", height=0.5)
        ax2.set_xlim(0,100); ax2.axis('off')
        for x, val, lbl in [
            (prob[0]*50, prob[0], "No Risk"),
            (prob[0]*100+prob[1]*50, prob[1], "At Risk"),
        ]:
            if val > 0.12:
                ax2.text(x, 0, f"{lbl}\n{val*100:.0f}%", ha='center', va='center',
                         fontsize=8, color='white', fontweight='bold')
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True); plt.close()
        pa_col, pb_col = st.columns(2)
        pa_col.metric("No Depression", f"{prob[0]*100:.1f}%")
        pb_col.metric("Depression",    f"{prob[1]*100:.1f}%")

    with r2:
        st.markdown("**Input Summary**")
        st.dataframe(pd.DataFrame({
            "Field": ["Name","Gender","Age","Course","Year","CGPA",
                      "Marital","Anxiety","Panic Attack"],
            "Value": [R['name'], R['gender'], R['age'], R['course'],
                      R['year'], R['cgpa'], R['marital'], R['anxiety'], R['panic']]
        }).set_index("Field"), use_container_width=True)

    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write("**Algorithm:** Decision Tree (CART)")
            st.write("**Max Depth:** 5")
            st.write("**Criterion:** Gini Impurity")
            st.write("**Split:** 70% / 30%")
            st.write(f"**Live Accuracy:** {M['acc']:.2f}%")

    # Feature importance
    st.write("")
    st.markdown("**Live Feature Importance**")
    fi_df = pd.DataFrame({'Feature': list(M['fi'].keys()),
                          'Importance': list(M['fi'].values())
                         }).sort_values('Importance', ascending=True)
    fig2, ax3 = plt.subplots(figsize=(7, 3))
    colors = ['#10B981' if v >= fi_df['Importance'].mean() else '#6B7280'
              for v in fi_df['Importance']]
    ax3.barh(fi_df['Feature'], fi_df['Importance'], color=colors, edgecolor='none')
    ax3.set_xlabel("Importance Score")
    ax3.set_title("Feature Importance (Live)", fontweight='bold')
    ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True); plt.close()

    st.write("")
    if st.button("Clear Result", key="dt_clear"):
        st.session_state.dt_result = None
        st.rerun()

# ── Always show tree diagram ───────────────────────────────────
st.divider()
st.subheader("Decision Tree Structure")
st.caption("Visual structure of the trained Decision Tree (Depth 5). "
           "Orange = tends to Depression, Blue = tends to No Depression.")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

@st.cache_resource
def get_tree_fig(_model, feat_names):
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(
        _model,
        feature_names=feat_names,
        class_names=['No Depression', 'Depression'],
        filled=True, rounded=True,
        fontsize=7, ax=ax,
        impurity=True, proportion=False
    )
    ax.set_title("Decision Tree Structure — Live Trained (Depth 5)",
                 fontsize=13, fontweight='bold', pad=16)
    plt.tight_layout()
    return fig

feat_display = ['Gender','Age','Course','Year','CGPA',
                'Anxiety','Panic Attack','Marital Status']
tree_fig = get_tree_fig(M['model'], feat_display)
st.pyplot(tree_fig, use_container_width=True)
plt.close('all')

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")