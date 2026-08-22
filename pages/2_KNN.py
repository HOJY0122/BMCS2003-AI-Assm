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

    k_train, k_test = [], []
    for k in range(1, 21):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(Xtr, ytr)
        k_train.append(accuracy_score(ytr, m.predict(Xtr)))
        k_test.append(accuracy_score(yte, m.predict(Xte)))

    best_k  = k_test.index(max(k_test)) + 1
    model   = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    cv = cross_val_score(
        KNeighborsClassifier(n_neighbors=best_k, metric='euclidean'),
        Xs, y, cv=5, scoring='accuracy')

    # Feature importance via correlation
    corr = df[feat + ['Depression']].corr()['Depression'].drop('Depression').abs()
    corr.index = ['Gender','Age','Course','Year','CGPA','Anxiety','Panic Attack']

    return {
        'model': model, 'scaler': sc,
        'le_c': le_c, 'le_y': le_y, 'feat': feat,
        'best_k': best_k,
        'k_train': k_train, 'k_test': k_test,
        'cv': cv,
        'corr': corr,
        'acc' : accuracy_score(yte,yp)*100,
        'prec': precision_score(yte,yp,zero_division=0)*100,
        'rec' : recall_score(yte,yp,zero_division=0)*100,
        'f1'  : f1_score(yte,yp,zero_division=0)*100,
        'cm'  : confusion_matrix(yte,yp),
    }

M = get_knn()

CGPA_MAP = {'0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
            '3.00 - 3.49':3.25,'3.50 - 4.00':3.75}
COURSES = ["Computer Science","Information Technology","Engineering",
           "Law","Psychology","Language","Islamic Studies",
           "Health Sciences","Business","Science & Math","Arts & Social","Others"]

if 'knn_result' not in st.session_state:
    st.session_state.knn_result = None

# ── Header ─────────────────────────────────────────────────────
st.markdown("##### 🔵 KNN PREDICTOR")
st.title("Depression Risk Predictor")
st.caption(f"K-Nearest Neighbor · K={M['best_k']} · "
           f"Member 1: Ho Jun Yon · Live trained on 600 records")
st.divider()

# ── Live metrics ───────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{M['acc']:.2f}%")
c2.metric("Precision", f"{M['prec']:.2f}%")
c3.metric("Recall",    f"{M['rec']:.2f}%")
c4.metric("F1 Score",  f"{M['f1']:.2f}%")
st.divider()

# ══════════════════════════════════════════════════════════════
# 1. USER INPUT
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
                            use_container_width=True,
                            key="knn_predict", type="primary")

if predict_btn:
    g  = 1 if gender  == "Male" else 0
    ax = 1 if anxiety == "Yes"  else 0
    pa = 1 if panic   == "Yes"  else 0
    ce = M['le_c'].transform([course])[0] if course in M['le_c'].classes_ else 0
    ye = M['le_y'].transform([year])[0]   if year   in M['le_y'].classes_ else 0
    cn = CGPA_MAP[cgpa]
    inp   = pd.DataFrame([[g,age,ce,ye,cn,ax,pa]], columns=M['feat'])
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
# 2. TEST RESULT
# ══════════════════════════════════════════════════════════════
if st.session_state.knn_result:
    R = st.session_state.knn_result
    pred = R['pred']; prob = R['prob']; name_lbl = R['name']

    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"### ⚠️  {name_lbl} — Depression Risk Detected\n\n"
                 "The KNN model predicts a **high risk of depression**. "
                 "Please consider seeking professional support.")
    else:
        st.success(f"### ✅  {name_lbl} — No Depression Detected\n\n"
                   "The KNN model predicts **low depression risk**. "
                   "Keep maintaining a healthy lifestyle!")

    st.write("")
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
        st.pyplot(fig, use_container_width=True); plt.close()
        st.write("")
        pa_c, pb_c = st.columns(2)
        pa_c.metric("No Depression", f"{prob[0]*100:.1f}%")
        pb_c.metric("Depression",    f"{prob[1]*100:.1f}%")

    with r2:
        st.markdown("**Input Summary**")
        st.table(pd.DataFrame({
            "Field": ["Name","Gender","Age","Course",
                      "Year","CGPA","Anxiety","Panic Attack"],
            "Value": [R['name'], R['gender'], str(R['age']),
                      R['course'], R['year'], R['cgpa'],
                      R['anxiety'], R['panic']]
        }).set_index("Field"))

    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write(f"**Algorithm:** KNN")
            st.write(f"**Best K:** {M['best_k']}")
            st.write(f"**Scaling:** MinMax")
            st.write(f"**Distance:** Euclidean")
            st.write(f"**Split:** 80 / 20")
            st.write(f"**CV Mean:** {M['cv'].mean()*100:.2f}%")
            st.write(f"**Accuracy:** {M['acc']:.2f}%")

    st.write("")
    if st.button("Clear Result", key="knn_clear"):
        st.session_state.knn_result = None
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# 3. LIVE FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
st.subheader("Live Feature Importance")
st.caption("Pearson correlation of each feature with Depression. "
           "Computed live from dataset — no hardcode.")

fi = M['corr'].sort_values(ascending=True)
fi_c1, fi_c2 = st.columns([2, 1])
with fi_c1:
    fig_fi, ax_fi = plt.subplots(figsize=(7, 3.5))
    colors = ['#3B82F6' if v >= fi.mean() else '#9CA3AF' for v in fi.values]
    bars = ax_fi.barh(fi.index, fi.values, color=colors,
                      edgecolor='none', height=0.6)
    ax_fi.axvline(fi.mean(), color='red', linestyle='--',
                  linewidth=1.2, alpha=0.7, label=f'Mean = {fi.mean():.3f}')
    for bar, val in zip(bars, fi.values):
        ax_fi.text(val+0.005, bar.get_y()+bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax_fi.set_xlabel('Absolute Correlation with Depression')
    ax_fi.set_title('Feature Importance — Pearson Correlation (Live)',
                    fontweight='bold')
    ax_fi.legend(fontsize=9)
    ax_fi.spines['top'].set_visible(False)
    ax_fi.spines['right'].set_visible(False)
    ax_fi.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_fi, use_container_width=True); plt.close()

with fi_c2:
    with st.container(border=True):
        st.markdown("**Feature Ranking**")
        for feat_n, val in fi.sort_values(ascending=False).items():
            icon = "🔵" if val >= fi.mean() else "⚪"
            st.write(f"{icon} **{feat_n}** — {val:.3f}")
        st.caption("🔵 Above average importance")

st.divider()

# ══════════════════════════════════════════════════════════════
# 4. LEARN MORE (K-OPT + CV + CONFUSION MATRIX)
# ══════════════════════════════════════════════════════════════
with st.expander("📚  Learn More — K-Value Optimization, Cross Validation & Confusion Matrix"):

    st.markdown("### K-Value Optimization")
    st.caption("How test accuracy changes across K=1 to K=20. Best K selected automatically.")
    fig_k, ax_k = plt.subplots(figsize=(10, 4))
    k_range = list(range(1, 21))
    ax_k.plot(k_range, [s*100 for s in M['k_train']],
              'b-o', label='Train Accuracy', markersize=4, linewidth=1.5)
    ax_k.plot(k_range, [s*100 for s in M['k_test']],
              'r-o', label='Test Accuracy', markersize=4, linewidth=1.5)
    ax_k.axvline(x=M['best_k'], color='green', linestyle='--',
                 linewidth=2, label=f'Best K = {M["best_k"]}')
    ax_k.set_xlabel('K Value'); ax_k.set_ylabel('Accuracy (%)')
    ax_k.set_title('KNN: Train vs Test Accuracy for Different K Values', fontweight='bold')
    ax_k.set_xticks(k_range)
    ax_k.legend(); ax_k.grid(True, alpha=0.3)
    ax_k.spines['top'].set_visible(False); ax_k.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_k, use_container_width=True); plt.close()

    st.write("")
    st.markdown("### 5-Fold Cross Validation")
    cv = M['cv']
    cv1, cv2, cv3 = st.columns(3)
    cv1.metric("CV Mean", f"{cv.mean()*100:.2f}%")
    cv2.metric("CV Std Dev", f"{cv.std()*100:.2f}%")
    cv3.metric("CV Max", f"{cv.max()*100:.2f}%")
    st.write("")
    fig_cv, ax_cv = plt.subplots(figsize=(8, 3))
    folds  = [f"Fold {i+1}" for i in range(5)]
    colors = ['#3B82F6','#10B981','#EF4444','#F59E0B','#8B5CF6']
    bars   = ax_cv.bar(folds, cv*100, color=colors, edgecolor='none', alpha=0.9)
    ax_cv.axhline(y=cv.mean()*100, color='red', linestyle='--',
                  linewidth=1.5, label=f'Mean = {cv.mean()*100:.2f}%')
    ax_cv.set_ylabel('Accuracy (%)'); ax_cv.set_ylim(0,110)
    ax_cv.set_title('5-Fold Cross Validation Scores', fontweight='bold')
    ax_cv.legend(); ax_cv.grid(axis='y', alpha=0.3)
    ax_cv.spines['top'].set_visible(False); ax_cv.spines['right'].set_visible(False)
    for bar, val in zip(bars, cv*100):
        ax_cv.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                   f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_cv, use_container_width=True); plt.close()

    st.write("")
    st.markdown("### Confusion Matrix")
    import seaborn as sns
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(M['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                xticklabels=['No Depression','Depression'],
                yticklabels=['No Depression','Depression'],
                linewidths=0.5, annot_kws={'size':12,'weight':'bold'})
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    ax_cm.set_title('KNN Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True); plt.close()
    tn,fp,fn,tp = M['cm'].ravel()
    a,b,c,d = st.columns(4)
    a.metric("TN", str(tn)); b.metric("FP", str(fp))
    c.metric("FN", str(fn)); d.metric("TP", str(tp))

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")