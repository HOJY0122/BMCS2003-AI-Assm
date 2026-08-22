import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import sidebar

st.set_page_config(page_title="KNN Predictor", page_icon="🔵",
                   layout="wide", initial_sidebar_state="expanded")
sidebar("knn")

# ── Train model (live) ─────────────────────────────────────────
@st.cache_resource
def get_knn():
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())
    le_c = LabelEncoder(); le_y = LabelEncoder()
    df['Course_Enc'] = le_c.fit_transform(df['Course'])
    df['Year_Enc']   = le_y.fit_transform(df['Year_of_Study'])

    feat = ['Gender','Age','Course_Enc','Year_Enc','CGPA_Numeric','Anxiety','Panic_Attack']
    X = df[feat]; y = df['Depression']
    sc = MinMaxScaler(); Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(Xs,y,test_size=0.2,random_state=42,stratify=y)

    best_k, best_a = 5, 0
    for k in range(1,21):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(Xtr,ytr)
        a = accuracy_score(yte,m.predict(Xte))
        if a > best_a: best_a, best_k = a, k

    model = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)

    return {
        'model': model, 'scaler': sc,
        'le_c': le_c, 'le_y': le_y, 'feat': feat,
        'best_k': best_k,
        'acc' : accuracy_score(yte,yp)*100,
        'prec': precision_score(yte,yp,zero_division=0)*100,
        'rec' : recall_score(yte,yp,zero_division=0)*100,
        'f1'  : f1_score(yte,yp,zero_division=0)*100,
    }

M = get_knn()

# ── Header ─────────────────────────────────────────────────────
st.markdown("##### KNN PREDICTOR")
st.title("Depression Risk Predictor")
st.caption(f"K-Nearest Neighbor (K={M['best_k']}) · Member 1: Ho Jun Yon · Live trained on {600} records")
st.divider()

# ── Live metrics strip ─────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{M['acc']:.2f}%",  help="Live model accuracy on test set")
c2.metric("Precision", f"{M['prec']:.2f}%", help="Of predicted depression, how many correct")
c3.metric("Recall",    f"{M['rec']:.2f}%",  help="Of actual depression cases, how many caught")
c4.metric("F1 Score",  f"{M['f1']:.2f}%",   help="Harmonic mean of precision and recall")
st.divider()

# ── Input form ─────────────────────────────────────────────────
st.subheader("Student Information")
st.caption("Fill in the details below and click Predict")

CGPA_MAP = {'0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
            '3.00 - 3.49':3.25,'3.50 - 4.00':3.75}

with st.form("knn_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        name    = st.text_input("Name", placeholder="e.g. Ahmad")
        gender  = st.selectbox("Gender", ["Female","Male"])
        age     = st.slider("Age", 17, 30, 20)
    with col2:
        course  = st.selectbox("Course", [
                    "Computer Science","Information Technology","Engineering",
                    "Law","Psychology","Language","Islamic Studies",
                    "Health Sciences","Business","Science & Math","Arts & Social","Others"])
        year    = st.selectbox("Year of Study", ["Year 1","Year 2","Year 3","Year 4"])
        cgpa    = st.selectbox("CGPA Range", list(CGPA_MAP.keys()))
    with col3:
        anxiety = st.selectbox("Do you have Anxiety?",       ["No","Yes"])
        panic   = st.selectbox("Do you have Panic Attack?",  ["No","Yes"])
        st.write("")
        st.write("")
        predict_btn = st.form_submit_button("Predict Depression Risk",
                                             use_container_width=True)

# ── Prediction result ──────────────────────────────────────────
if predict_btn:
    g  = 1 if gender  == "Male" else 0
    ax = 1 if anxiety == "Yes"  else 0
    pa = 1 if panic   == "Yes"  else 0
    le_c = M['le_c']; le_y = M['le_y']
    ce = le_c.transform([course])[0] if course in le_c.classes_ else 0
    ye = le_y.transform([year])[0]   if year   in le_y.classes_ else 0
    cn = CGPA_MAP[cgpa]

    inp   = pd.DataFrame([[g,age,ce,ye,cn,ax,pa]], columns=M['feat'])
    inp_s = M['scaler'].transform(inp)
    pred  = M['model'].predict(inp_s)[0]
    prob  = M['model'].predict_proba(inp_s)[0]
    name_lbl = name.strip() or "Student"

    st.divider()

    # Result banner
    if pred == 1:
        st.error(f"## ⚠️ {name_lbl} — Depression Risk Detected")
        st.write("The model predicts a **high risk of depression**. "
                 "Please consider speaking with a counsellor or mental health professional.")
    else:
        st.success(f"## ✅ {name_lbl} — No Depression Detected")
        st.write("The model predicts **low depression risk**. "
                 "Keep maintaining a healthy academic and social lifestyle.")

    st.write("")

    # Confidence + summary
    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown("**Prediction Confidence**")
        fig, ax2 = plt.subplots(figsize=(4, 0.7))
        ax2.barh([""], [prob[0]*100], color="#10B981", height=0.5)
        ax2.barh([""], [prob[1]*100], left=[prob[0]*100],
                 color="#EF4444", height=0.5)
        ax2.set_xlim(0,100); ax2.axis('off')
        for x, val, lbl in [
            (prob[0]*50, prob[0], "No Risk"),
            (prob[0]*100+prob[1]*50, prob[1], "At Risk")
        ]:
            if val > 0.12:
                ax2.text(x, 0, f"{lbl}\n{val*100:.0f}%",
                         ha='center', va='center',
                         fontsize=8, color='white', fontweight='bold')
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True); plt.close()

        pa_col, pb_col = st.columns(2)
        pa_col.metric("No Depression", f"{prob[0]*100:.1f}%")
        pb_col.metric("Depression",    f"{prob[1]*100:.1f}%")

    with r2:
        st.markdown("**Input Summary**")
        st.dataframe(pd.DataFrame({
            "Field": ["Name","Gender","Age","Course","Year","CGPA","Anxiety","Panic Attack"],
            "Value": [name_lbl, gender, age, course, year, cgpa, anxiety, panic]
        }).set_index("Field"), use_container_width=True)

    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write(f"**Algorithm:** K-Nearest Neighbor")
            st.write(f"**Best K:** {M['best_k']}")
            st.write(f"**Scaling:** MinMax Scaler")
            st.write(f"**Distance:** Euclidean")
            st.write(f"**Split:** 80% train / 20% test")
            st.write(f"**Live Accuracy:** {M['acc']:.2f}%")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")