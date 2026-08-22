import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar

st.set_page_config(page_title="SVM Predictor", page_icon="🔴",
                   layout="wide", initial_sidebar_state="expanded")
sidebar("svm")

# ── Train model (live) ─────────────────────────────────────────
@st.cache_resource
def get_svm():
    df_raw = pd.read_csv('dataset/Student_Mental_health.csv')
    df_raw.columns = df_raw.columns.str.strip()
    df_raw['Age'] = df_raw['Age'].fillna(df_raw['Age'].median())
    df_raw['Your current year of Study'] = df_raw['Your current year of Study'].str.strip().str.lower()
    df_raw['What is your CGPA?'] = df_raw['What is your CGPA?'].str.strip()

    def cat_course(c):
        c = str(c).lower()
        return 'STEM/IT' if any(x in c for x in [
            'technology','it','computer','cs','system','software','se','bit','bcs','cts'
        ]) else 'Other'

    df_raw['Course_Category'] = df_raw['What is your course?'].apply(cat_course)
    df_raw = df_raw.drop(columns=['Timestamp','What is your course?'], errors='ignore')

    X = df_raw.drop(columns=['Do you have Depression?'])
    y = (df_raw['Do you have Depression?'] == 'Yes').astype(int)
    col_order = list(X.columns)

    pipe = Pipeline([
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scl', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42))
    ])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)

    return {
        'pipe': pipe, 'col_order': col_order,
        'X_all': X, 'y_all': y,
        'cat_course': cat_course,
        'acc' : accuracy_score(yte,yp)*100,
        'prec': precision_score(yte,yp,zero_division=0)*100,
        'rec' : recall_score(yte,yp,zero_division=0)*100,
        'f1'  : f1_score(yte,yp,zero_division=0)*100,
    }

M = get_svm()

# ── Header ─────────────────────────────────────────────────────
st.markdown("##### SVM PREDICTOR")
st.title("Depression Risk Predictor")
st.caption("Support Vector Machine (RBF Kernel) · Member 3: Chiang Jun Hang · Live trained on 600 records")
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

with st.form("svm_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        name    = st.text_input("Name", placeholder="e.g. Wei Ming")
        gender  = st.selectbox("Gender", ["Male","Female"])
        age     = st.slider("Age", 15, 40, 20)
    with col2:
        course  = st.selectbox("Course Field", [
                    "Information Technology (IT)",
                    "Computer Science (CS)",
                    "Information System (IS)",
                    "Software Engineering (SE)",
                    "Other"])
        year    = st.selectbox("Year of Study", ["Year 1","Year 2","Year 3","Year 4"])
        cgpa    = st.selectbox("CGPA Range", [
                    "0 - 1.99","2.00 - 2.49","2.50 - 2.99",
                    "3.00 - 3.49","3.50 - 4.00"])
    with col3:
        marital = st.selectbox("Marital Status",                      ["No","Yes"])
        anxiety = st.selectbox("Do you have Anxiety?",                ["No","Yes"])
        panic   = st.selectbox("Do you have Panic Attack?",           ["No","Yes"])
        treat   = st.selectbox("Sought Specialist Treatment?",        ["No","Yes"])
        predict_btn = st.form_submit_button("Predict Depression Risk",
                                             use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────
if predict_btn:
    year_num = year.split()[-1]
    inp = pd.DataFrame([{
        'Choose your gender'                           : gender,
        'Age'                                          : age,
        'Your current year of Study'                   : f'year {year_num}',
        'What is your CGPA?'                           : cgpa,
        'Marital status'                               : marital,
        'Do you have Anxiety?'                         : anxiety,
        'Do you have Panic attack?'                    : panic,
        'Did you seek any specialist for a treatment?' : treat,
        'Course_Category'                              : M['cat_course'](course),
    }])[M['col_order']]

    pred = M['pipe'].predict(inp)[0]
    prob = M['pipe'].predict_proba(inp)[0]
    name_lbl = name.strip() or "Student"

    st.divider()

    if pred == 1:
        st.error(f"## ⚠️ {name_lbl} — Depression Risk Detected")
        st.write("The SVM model predicts a **high risk of depression**. "
                 "Please consider speaking with a counsellor or mental health professional.")
    else:
        st.success(f"## ✅ {name_lbl} — No Depression Detected")
        st.write("The SVM model predicts **low depression risk**. "
                 "Keep maintaining a healthy academic and social lifestyle.")

    st.write("")
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
            "Field": ["Name","Gender","Age","Course","Year","CGPA",
                      "Marital","Anxiety","Panic","Treatment"],
            "Value": [name_lbl, gender, age, course, year, cgpa,
                      marital, anxiety, panic, treat]
        }).set_index("Field"), use_container_width=True)

    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write("**Algorithm:** Support Vector Machine")
            st.write("**Kernel:** RBF")
            st.write("**Class Weight:** Balanced")
            st.write("**Scaling:** Standard Scaler")
            st.write("**Split:** 75% train / 25% test")
            st.write(f"**Live Accuracy:** {M['acc']:.2f}%")

    # SVM PCA boundary visualization
    st.write("")
    st.markdown("**SVM Decision Boundary (PCA Projection)**")
    st.caption("Your input plotted against the dataset in 2D PCA space")
    try:
        prep = M['pipe'].named_steps['enc']
        X_bg_proc   = prep.transform(M['X_all'])
        X_user_proc = prep.transform(inp)
        sc2d = StandardScaler()
        X_bg_sc  = sc2d.fit_transform(X_bg_proc)
        X_us_sc  = sc2d.transform(X_user_proc)
        pca = PCA(n_components=2, random_state=42)
        X_bg_2d  = pca.fit_transform(X_bg_sc)
        X_us_2d  = pca.transform(X_us_sc)
        svm2d = LinearSVC(C=1.0, class_weight='balanced',
                           random_state=42, max_iter=5000)
        svm2d.fit(X_bg_2d, M['y_all'])

        fig3, ax3 = plt.subplots(figsize=(8,4))
        y_all = M['y_all'].values
        ax3.scatter(X_bg_2d[y_all==1,0], X_bg_2d[y_all==1,1],
                    color='#EF4444', s=30, alpha=0.5, label='Depression', zorder=3)
        ax3.scatter(X_bg_2d[y_all==0,0], X_bg_2d[y_all==0,1],
                    color='#3B82F6', s=30, alpha=0.5, label='No Depression', zorder=3)
        ax3.scatter(X_us_2d[0,0], X_us_2d[0,1],
                    color='black', s=300, marker='*',
                    label=f'Input: {name_lbl}', zorder=6)
        xmin=X_bg_2d[:,0].min()-1; xmax=X_bg_2d[:,0].max()+1
        xmin=min(xmin,X_us_2d[0,0]-1); xmax=max(xmax,X_us_2d[0,0]+1)
        w=svm2d.coef_[0]; b=svm2d.intercept_[0]
        xpts=np.linspace(xmin,xmax,200)
        if w[1]!=0:
            ypts=-(w[0]*xpts+b)/w[1]
            ax3.plot(xpts,ypts,'--',color='#1E3A5F',lw=2,label='Decision Boundary',zorder=5)
        ax3.set_xlim(xmin,xmax)
        ax3.set_ylim(X_bg_2d[:,1].min()-1, X_bg_2d[:,1].max()+1)
        ax3.set_xlabel('PC1 — Mental Health Risk Factors', fontsize=9)
        ax3.set_ylabel('PC2 — Academic & Demographic Profile', fontsize=9)
        ax3.set_title('Live SVM Decision Boundary', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9); ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True); plt.close()
    except Exception as e:
        st.warning(f"Visualization unavailable: {e}")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")