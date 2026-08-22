import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

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

    feat = ['Gender','Age','Course_Enc','Year_Enc',
            'CGPA_Numeric','Anxiety','Panic_Attack']
    X = df[feat]; y = df['Depression']
    sc = MinMaxScaler(); Xs = sc.fit_transform(X)
    Xtr,Xte,ytr,yte = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y)

    # Find best K
    k_scores = []
    for k in range(1, 21):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(Xtr, ytr)
        k_scores.append(accuracy_score(yte, m.predict(Xte)))

    best_k  = k_scores.index(max(k_scores)) + 1
    model   = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)

    # Cross validation
    cv_scores = cross_val_score(
        KNeighborsClassifier(n_neighbors=best_k, metric='euclidean'),
        Xs, y, cv=5, scoring='accuracy')

    return {
        'model': model, 'scaler': sc,
        'le_c': le_c, 'le_y': le_y, 'feat': feat,
        'best_k': best_k, 'k_scores': k_scores,
        'cv_scores': cv_scores,
        'Xtr': Xtr, 'Xte': Xte, 'ytr': ytr, 'yte': yte,
        'acc' : accuracy_score(yte,yp)*100,
        'prec': precision_score(yte,yp,zero_division=0)*100,
        'rec' : recall_score(yte,yp,zero_division=0)*100,
        'f1'  : f1_score(yte,yp,zero_division=0)*100,
        'cm'  : confusion_matrix(yte,yp),
    }

M = get_knn()

CGPA_MAP = {'0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
            '3.00 - 3.49':3.25,'3.50 - 4.00':3.75}
COURSES   = ["Computer Science","Information Technology","Engineering",
             "Law","Psychology","Language","Islamic Studies",
             "Health Sciences","Business","Science & Math","Arts & Social","Others"]

# ── Session state ──────────────────────────────────────────────
if 'knn_result' not in st.session_state:
    st.session_state.knn_result = None

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("##### KNN PREDICTOR")
st.title("Depression Risk Predictor")
st.caption(f"K-Nearest Neighbor (K={M['best_k']}) · "
           f"Member 1: Ho Jun Yon · Live trained on 600 records")
st.divider()

# ── Live metrics ───────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{M['acc']:.2f}%",
          help="Live model accuracy on test set (120 records)")
c2.metric("Precision", f"{M['prec']:.2f}%",
          help="Of predicted depression, how many are correct")
c3.metric("Recall",    f"{M['rec']:.2f}%",
          help="Of actual depression cases, how many were caught")
c4.metric("F1 Score",  f"{M['f1']:.2f}%",
          help="Harmonic mean of precision and recall")
st.divider()

# ══════════════════════════════════════════════════════════════
# PREDICTOR FORM
# ══════════════════════════════════════════════════════════════
st.subheader("Student Information")
st.caption("Fill in the details below and click Predict")

col1, col2, col3 = st.columns(3)
with col1:
    name    = st.text_input("Name", placeholder="e.g. Ahmad", key="knn_name")
    gender  = st.selectbox("Gender", ["Female","Male"], key="knn_gender")
    age     = st.slider("Age", 17, 30, 20, key="knn_age")
with col2:
    course  = st.selectbox("Course", COURSES, key="knn_course")
    year    = st.selectbox("Year of Study",
                ["Year 1","Year 2","Year 3","Year 4"], key="knn_year")
    cgpa    = st.selectbox("CGPA Range", list(CGPA_MAP.keys()), key="knn_cgpa")
with col3:
    anxiety = st.selectbox("Do you have Anxiety?",
                ["No","Yes"], key="knn_anxiety")
    panic   = st.selectbox("Do you have Panic Attack?",
                ["No","Yes"], key="knn_panic")
    st.write(""); st.write(""); st.write("")
    predict_btn = st.button("🔍  Predict Depression Risk",
                            use_container_width=True, key="knn_predict",
                            type="primary")

# ── Run prediction → save to session state ─────────────────────
if predict_btn:
    g  = 1 if gender  == "Male" else 0
    ax = 1 if anxiety == "Yes"  else 0
    pa = 1 if panic   == "Yes"  else 0
    ce = M['le_c'].transform([course])[0] if course in M['le_c'].classes_ else 0
    ye = M['le_y'].transform([year])[0]   if year   in M['le_y'].classes_ else 0
    cn = CGPA_MAP[cgpa]

    inp   = pd.DataFrame([[g, age, ce, ye, cn, ax, pa]], columns=M['feat'])
    inp_s = M['scaler'].transform(inp)
    pred  = int(M['model'].predict(inp_s)[0])
    prob  = M['model'].predict_proba(inp_s)[0].tolist()

    st.session_state.knn_result = {
        'pred': pred, 'prob': prob,
        'name': name.strip() or "Student",
        'gender': gender, 'age': age, 'course': course,
        'year': year, 'cgpa': cgpa,
        'anxiety': anxiety, 'panic': panic,
    }

# ══════════════════════════════════════════════════════════════
# RESULT (persists via session state)
# ══════════════════════════════════════════════════════════════
if st.session_state.knn_result:
    R        = st.session_state.knn_result
    pred     = R['pred']
    prob     = R['prob']
    name_lbl = R['name']

    st.divider()

    # ── Banner ─────────────────────────────────────────────────
    if pred == 1:
        st.error(f"## ⚠️  {name_lbl} — Depression Risk Detected")
        st.write("The KNN model predicts a **high risk of depression**. "
                 "Consider speaking with a counsellor or mental health professional.")
    else:
        st.success(f"## ✅  {name_lbl} — No Depression Detected")
        st.write("The KNN model predicts **low depression risk**. "
                 "Keep maintaining a healthy academic and social lifestyle.")

    st.write("")

    # ── 3 columns: Confidence | Table | Model Info ─────────────
    r1, r2, r3 = st.columns([1.2, 1.2, 1])

    with r1:
        st.markdown("**Prediction Confidence**")
        fig, ax2 = plt.subplots(figsize=(4, 0.8))
        ax2.barh([""], [prob[0]*100], color="#10B981", height=0.5)
        ax2.barh([""], [prob[1]*100], left=[prob[0]*100],
                 color="#EF4444", height=0.5)
        ax2.set_xlim(0,100); ax2.axis('off')
        for xp, val, lbl in [
            (prob[0]*50,             prob[0], "No Risk"),
            (prob[0]*100+prob[1]*50, prob[1], "At Risk"),
        ]:
            if val > 0.12:
                ax2.text(xp, 0, f"{lbl}\n{val*100:.0f}%",
                         ha='center', va='center',
                         fontsize=8, color='white', fontweight='bold')
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.write("")
        pa_c, pb_c = st.columns(2)
        pa_c.metric("No Depression", f"{prob[0]*100:.1f}%")
        pb_c.metric("Depression",    f"{prob[1]*100:.1f}%")

    with r2:
        st.markdown("**Input Summary**")
        table_data = {
            "Field": ["Name","Gender","Age","Course",
                      "Year","CGPA","Anxiety","Panic Attack"],
            "Value": [R['name'], R['gender'], str(R['age']),
                      R['course'], R['year'], R['cgpa'],
                      R['anxiety'], R['panic']]
        }
        st.table(pd.DataFrame(table_data).set_index("Field"))

    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write(f"**Algorithm:** K-Nearest Neighbor")
            st.write(f"**Best K:** {M['best_k']}")
            st.write(f"**Scaling:** MinMax Scaler")
            st.write(f"**Distance:** Euclidean")
            st.write(f"**Train:** 480 (80%)")
            st.write(f"**Test:** 120 (20%)")
            st.write(f"**CV Mean:** {M['cv_scores'].mean()*100:.2f}%")
            st.write(f"**Live Acc:** {M['acc']:.2f}%")

    st.write("")
    if st.button("Clear Result", key="knn_clear"):
        st.session_state.knn_result = None
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# K VALUE CHART (always shown)
# ══════════════════════════════════════════════════════════════
st.subheader("K-Value Optimization")
st.caption("How accuracy changes across K=1 to K=20. Best K selected automatically.")

fig_k, ax_k = plt.subplots(figsize=(10, 4))
k_range = list(range(1, 21))
ax_k.plot(k_range, [s*100 for s in M['k_scores']],
          'r-o', label='Test Accuracy', markersize=5, linewidth=2)
ax_k.axvline(x=M['best_k'], color='green', linestyle='--', linewidth=2,
             label=f'Best K = {M["best_k"]}')
ax_k.set_xlabel('K Value'); ax_k.set_ylabel('Accuracy (%)')
ax_k.set_title('KNN: Test Accuracy for Different K Values', fontweight='bold')
ax_k.set_xticks(k_range)
ax_k.legend(); ax_k.grid(True, alpha=0.3)
ax_k.spines['top'].set_visible(False); ax_k.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig_k, use_container_width=True)
plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# 5-FOLD CROSS VALIDATION (always shown)
# ══════════════════════════════════════════════════════════════
st.subheader("5-Fold Cross Validation")
cv = M['cv_scores']
cv1, cv2, cv3 = st.columns(3)
cv1.metric("CV Mean Accuracy", f"{cv.mean()*100:.2f}%")
cv2.metric("CV Std Dev",       f"{cv.std()*100:.2f}%")
cv3.metric("CV Max Score",     f"{cv.max()*100:.2f}%")

st.write("")
fig_cv, ax_cv = plt.subplots(figsize=(8, 3))
folds  = [f"Fold {i+1}" for i in range(5)]
colors = ['#3B82F6','#10B981','#EF4444','#F59E0B','#8B5CF6']
bars   = ax_cv.bar(folds, cv*100, color=colors, edgecolor='none', alpha=0.9)
ax_cv.axhline(y=cv.mean()*100, color='red', linestyle='--', linewidth=1.5,
              label=f'Mean = {cv.mean()*100:.2f}%')
ax_cv.set_ylabel('Accuracy (%)'); ax_cv.set_ylim(0, 110)
ax_cv.set_title('5-Fold Cross Validation Scores', fontweight='bold')
ax_cv.legend(); ax_cv.grid(axis='y', alpha=0.3)
ax_cv.spines['top'].set_visible(False); ax_cv.spines['right'].set_visible(False)
for bar, val in zip(bars, cv*100):
    ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
               f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
plt.tight_layout()
st.pyplot(fig_cv, use_container_width=True)
plt.close()

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")