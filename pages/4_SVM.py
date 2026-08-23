import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar

st.set_page_config(page_title="SVM Predictor", page_icon="🔴",
                   layout="wide", initial_sidebar_state="expanded")
sidebar("svm")

@st.cache_resource
def get_svm():
    df = pd.read_csv('dataset/Student_Mental_health.csv')
    df.columns = df.columns.str.strip()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Your current year of Study'] = df['Your current year of Study'].str.strip().str.lower()
    df['What is your CGPA?'] = df['What is your CGPA?'].str.strip()

    def cat(c):
        c = str(c).lower()
        return 'STEM/IT' if any(x in c for x in [
            'technology','it','computer','cs','system',
            'software','se','bit','bcs','cts']) else 'Other'

    df['Course_Category'] = df['What is your course?'].apply(cat)
    df = df.drop(columns=['Timestamp','What is your course?'], errors='ignore')
    X = df.drop(columns=['Do you have Depression?'])
    y = (df['Do you have Depression?'] == 'Yes').astype(int)
    col_order = list(X.columns)

    pipe = Pipeline([
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scl', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42))
    ])
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
    pipe.fit(Xtr,ytr); yp = pipe.predict(Xte)

    # Permutation importance
    perm = permutation_importance(pipe, Xte, yte,
                                  n_repeats=10, random_state=42,
                                  scoring='accuracy')
    fi_labels = ['Gender','Age','Year of Study','CGPA','Marital Status',
                 'Anxiety','Panic Attack','Seek Treatment','Course Category']
    fi_df = pd.DataFrame({
        'Feature': fi_labels[:len(perm.importances_mean)],
        'Importance': perm.importances_mean,
        'Std': perm.importances_std
    })

    return {
        'pipe': pipe, 'col_order': col_order, 'cat': cat,
        'X_all': X, 'y_all': y,
        'fi_df': fi_df,
        'acc' : accuracy_score(yte,yp)*100,
        'prec': precision_score(yte,yp,zero_division=0)*100,
        'rec' : recall_score(yte,yp,zero_division=0)*100,
        'f1'  : f1_score(yte,yp,zero_division=0)*100,
        'cm'  : confusion_matrix(yte,yp),
    }

with st.spinner("Training SVM model..."):
    M = get_svm()

if 'svm_result' not in st.session_state:
    st.session_state.svm_result = None

# ── Header ─────────────────────────────────────────────────────
st.markdown("##### 🔴 SVM PREDICTOR")
st.title("Depression Risk Predictor")
st.caption("Support Vector Machine (RBF Kernel) · Member 3: Chiang Jun Hang · Live trained on 600 records")
st.divider()

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
    name    = st.text_input("Name", placeholder="e.g. Wei Ming", key="svm_name")
    gender  = st.selectbox("Gender", ["Male","Female"], key="svm_gender")
    age     = st.slider("Age", 15, 40, 20, key="svm_age")
with col2:
    course  = st.selectbox("Course Field", [
                "Information Technology (IT)","Computer Science (CS)",
                "Information System (IS)","Software Engineering (SE)","Other"],
                key="svm_course")
    year    = st.selectbox("Year of Study",
                ["Year 1","Year 2","Year 3","Year 4"], key="svm_year")
    cgpa    = st.selectbox("CGPA Range", [
                "0 - 1.99","2.00 - 2.49","2.50 - 2.99",
                "3.00 - 3.49","3.50 - 4.00"], key="svm_cgpa")
with col3:
    marital = st.selectbox("Marital Status",               ["No","Yes"], key="svm_marital")
    anxiety = st.selectbox("Do you have Anxiety?",         ["No","Yes"], key="svm_anxiety")
    panic   = st.selectbox("Do you have Panic Attack?",    ["No","Yes"], key="svm_panic")
    treat   = st.selectbox("Sought Specialist Treatment?", ["No","Yes"], key="svm_treat")
    predict_btn = st.button("🔍  Predict Depression Risk",
                            use_container_width=True,
                            key="svm_predict", type="primary")

if predict_btn:
    # ── Input Validation ───────────────────────────────────────
    _errors = []
    if name.strip() and name.strip().isdigit():
        _errors.append("❌ Name cannot be numbers only.")
    if age < 15 or age > 40:
        _errors.append(f"❌ Age {age} is outside valid range (15–40).")

    # ── Business Rules ─────────────────────────────────────────
    _high_concern  = (anxiety == "Yes" and panic == "Yes")
    _treat_flag    = (treat == "Yes")
    _academic_risk = (year in ["Year 3","Year 4"] and cgpa == "0 - 1.99")

    if _errors:
        for e in _errors:
            st.error(e)
        st.warning("⚠️ Please fix the errors above before predicting.")
    else:
        try:
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
                'Course_Category'                              : M['cat'](course),
            }])[M['col_order']]
            pred = int(M['pipe'].predict(inp)[0])
            prob = M['pipe'].predict_proba(inp)[0].tolist()
            st.session_state.svm_result = {
                'pred': pred, 'prob': prob,
                'inp': inp,
                'name': name.strip() or "Student",
                'gender': gender, 'age': age, 'course': course,
                'year': year, 'cgpa': cgpa, 'marital': marital,
                'anxiety': anxiety, 'panic': panic, 'treat': treat,
                'high_concern': _high_concern,
                'treat_flag': _treat_flag,
                'academic_risk': _academic_risk,
            }
        except Exception as ex:
            st.error(f"❌ Prediction failed: {ex}")
            st.caption("Please check your inputs and try again.")

# ══════════════════════════════════════════════════════════════
# 2. TEST RESULT
# ══════════════════════════════════════════════════════════════
if st.session_state.svm_result:
    R = st.session_state.svm_result
    pred = R['pred']; prob = R['prob']; name_lbl = R['name']

    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"### ⚠️  {name_lbl} — Depression Risk Detected\n\n"
                 "The SVM model predicts a **high risk of depression**. "
                 "Please consider seeking professional support.")
    else:
        st.success(f"### ✅  {name_lbl} — No Depression Detected\n\n"
                   "The SVM model predicts **low depression risk**. "
                   "Keep maintaining a healthy lifestyle!")

    # ── Business Rule Alerts ──────────────────────────────────
    if R.get('high_concern'):
        st.warning("⚠️ **High Concern:** Student has both Anxiety AND Panic Attack. "
                   "Immediate counselling referral is strongly recommended.")
    if R.get('treat_flag'):
        st.info("ℹ️ **Treatment Noted:** This student has already sought specialist help. "
                "Consider follow-up support and progress monitoring.")
    if R.get('academic_risk'):
        st.warning("⚠️ **Academic Risk:** Senior year with very low CGPA (0–1.99). "
                   "Combined academic and mental health support is advised.")

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
            "Field": ["Name","Gender","Age","Course","Year","CGPA",
                      "Marital","Anxiety","Panic","Treatment"],
            "Value": [R['name'], R['gender'], str(R['age']),
                      R['course'], R['year'], R['cgpa'],
                      R['marital'], R['anxiety'], R['panic'], R['treat']]
        }).set_index("Field"))

    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write("**Algorithm:** SVM")
            st.write("**Kernel:** RBF")
            st.write("**Class Weight:** Balanced")
            st.write("**Scaling:** Standard")
            st.write("**Split:** 75 / 25")
            st.write(f"**Accuracy:** {M['acc']:.2f}%")

    # SVM PCA boundary
    st.write("")
    st.markdown("**SVM Decision Boundary (PCA Projection)**")
    try:
        inp_stored = R['inp']
        prep = M['pipe'].named_steps['enc']
        X_bg_proc   = prep.transform(M['X_all'])
        X_user_proc = prep.transform(inp_stored)
        sc2d = StandardScaler()
        X_bg_sc = sc2d.fit_transform(X_bg_proc)
        X_us_sc = sc2d.transform(X_user_proc)
        pca = PCA(n_components=2, random_state=42)
        X_bg_2d = pca.fit_transform(X_bg_sc)
        X_us_2d = pca.transform(X_us_sc)
        svm2d = LinearSVC(C=1.0, class_weight='balanced',
                           random_state=42, max_iter=5000)
        svm2d.fit(X_bg_2d, M['y_all'])

        fig3, ax3 = plt.subplots(figsize=(8,4))
        y_all = M['y_all'].values
        ax3.scatter(X_bg_2d[y_all==1,0], X_bg_2d[y_all==1,1],
                    color='#EF4444', s=30, alpha=0.5, label='Depression', zorder=3)
        ax3.scatter(X_bg_2d[y_all==0,0], X_bg_2d[y_all==0,1],
                    color='#3B82F6', s=30, alpha=0.5, label='No Depression', zorder=3)
        ax3.scatter(X_us_2d[0,0], X_us_2d[0,1], color='black',
                    s=300, marker='*', label=f'Input: {name_lbl}', zorder=6)
        xmin=X_bg_2d[:,0].min()-1; xmax=X_bg_2d[:,0].max()+1
        xmin=min(xmin,X_us_2d[0,0]-1); xmax=max(xmax,X_us_2d[0,0]+1)
        w=svm2d.coef_[0]; b=svm2d.intercept_[0]
        xpts=np.linspace(xmin,xmax,200)
        if w[1]!=0:
            ypts=-(w[0]*xpts+b)/w[1]
            ax3.plot(xpts,ypts,'--',color='#1E3A5F',lw=2,
                     label='Decision Boundary',zorder=5)
        ax3.set_xlim(xmin,xmax)
        ax3.set_ylim(X_bg_2d[:,1].min()-1, X_bg_2d[:,1].max()+1)
        ax3.set_xlabel('PC1 — Mental Health Risk Factors', fontsize=9)
        ax3.set_ylabel('PC2 — Academic & Demographic', fontsize=9)
        ax3.set_title('Live SVM Decision Boundary (PCA 2D Projection)',
                      fontsize=11, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True); plt.close()
    except Exception as e:
        st.warning(f"Visualization unavailable: {e}")


    # ══════════════════════════════════════════════════════════
    # PREDICTION EXPLANATION — WHY did SVM decide this?
    # ══════════════════════════════════════════════════════════
    st.write("")
    st.markdown("---")
    st.markdown("### 🧠 Why did SVM predict this?")
    st.caption("Feature importance and confidence breakdown for this prediction.")

    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown("**Permutation Feature Importance**")
        st.caption("Which features matter most to the SVM model overall.")
        _fi_svm = M['fi_df'].sort_values('Importance', ascending=True)
        _mean_sv = _fi_svm['Importance'].mean()
        fig_sv, ax_sv = plt.subplots(figsize=(5, 3.5))
        _sv_colors = ['#EF4444' if v >= _mean_sv else '#9CA3AF'
                      for v in _fi_svm['Importance']]
        ax_sv.barh(_fi_svm['Feature'], _fi_svm['Importance'],
                   color=_sv_colors, edgecolor='none', height=0.6)
        ax_sv.axvline(_mean_sv, color='red', ls='--', lw=1.2,
                      alpha=0.7, label=f'Mean = {_mean_sv:.4f}')
        ax_sv.set_xlabel('Mean Accuracy Decrease')
        ax_sv.set_title('SVM Feature Importance (Live)', fontweight='bold')
        ax_sv.legend(fontsize=8)
        ax_sv.spines['top'].set_visible(False)
        ax_sv.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_sv, use_container_width=True); plt.close()

    with ex2:
        st.markdown("**Risk Factor Analysis for This Student**")
        st.caption("How this student's profile compares to key risk indicators.")

        _risk_factors = []
        _protect_factors = []

        if R['anxiety'] == 'Yes':
            _risk_factors.append(("Anxiety", "Strong predictor of depression"))
        if R['panic'] == 'Yes':
            _risk_factors.append(("Panic Attack", "Strongest predictor (r=0.341)"))
        if R['marital'] == 'Yes':
            _risk_factors.append(("Marital Status", "Decision tree root feature"))
        if R['treat'] == 'Yes':
            _risk_factors.append(("Sought Treatment", "Indicates awareness of condition"))
        if CGPA_MAP.get(R['cgpa'], 3.25) < 2.5:
            _risk_factors.append(("Low CGPA", "Academic difficulty may increase risk"))

        if CGPA_MAP.get(R['cgpa'], 3.25) >= 3.0:
            _protect_factors.append(("High CGPA", "Academic success is a protective factor"))
        if R['anxiety'] == 'No' and R['panic'] == 'No':
            _protect_factors.append(("No Anxiety/Panic", "Absence of co-morbid conditions"))

        if _risk_factors:
            st.markdown("**⚠️ Risk Factors Present:**")
            for factor, reason in _risk_factors:
                st.write(f"• **{factor}** — {reason}")
        if _protect_factors:
            st.write("")
            st.markdown("**✅ Protective Factors:**")
            for factor, reason in _protect_factors:
                st.write(f"• **{factor}** — {reason}")

        st.write("")
        _total_risk = len(_risk_factors)
        _risk_pct   = prob[1] * 100
        with st.container(border=True):
            st.markdown(f"**Risk Summary**")
            st.write(f"Risk Factors Detected: **{_total_risk}**")
            st.write(f"SVM Depression Probability: **{_risk_pct:.1f}%**")
            if _risk_pct > 70:
                st.error("High confidence — multiple risk factors confirmed")
            elif _risk_pct > 40:
                st.warning("Moderate confidence — some risk factors present")
            else:
                st.success("Low confidence of depression — few risk factors")

    st.write("")
    if st.button("Clear Result", key="svm_clear"):
        st.session_state.svm_result = None
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# 3. LIVE FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
st.subheader("Live Feature Importance")
st.caption("Permutation importance — accuracy drop when each feature is shuffled. "
           "Computed live from trained model — no hardcode.")

fi_df = M['fi_df'].sort_values('Importance', ascending=True)
mean_v = fi_df['Importance'].mean()

fi_c1, fi_c2 = st.columns([2, 1])
with fi_c1:
    fig_fi, ax_fi = plt.subplots(figsize=(7, 3.5))
    colors = ['#EF4444' if v >= mean_v else '#9CA3AF'
              for v in fi_df['Importance']]
    ax_fi.barh(fi_df['Feature'], fi_df['Importance'],
               xerr=fi_df['Std'], color=colors, edgecolor='none',
               height=0.6, error_kw={'elinewidth':1.5,'ecolor':'#374151'})
    ax_fi.axvline(mean_v, color='red', linestyle='--',
                  linewidth=1.2, alpha=0.7, label=f'Mean = {mean_v:.4f}')
    for _, row in fi_df.iterrows():
        ax_fi.text(max(row['Importance'],0)+0.001,
                   fi_df.index.get_loc(_)+0 if False else
                   list(fi_df.index).index(_)*0 ,
                   '', va='center')
    ax_fi.set_xlabel('Mean Accuracy Decrease (Permutation)')
    ax_fi.set_title('SVM Feature Importance — Permutation Method (Live)',
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
        for _, row in fi_df.sort_values('Importance', ascending=False).iterrows():
            icon = "🔴" if row['Importance'] >= mean_v else "⚪"
            st.write(f"{icon} **{row['Feature']}** — {row['Importance']:.4f}")
        st.caption("🔴 Above average importance\n\nError bars = std across 10 repeats")

st.divider()

# ══════════════════════════════════════════════════════════════
# 4. LEARN MORE — Confusion Matrix
# ══════════════════════════════════════════════════════════════
with st.expander("📚  Learn More — Confusion Matrix & How SVM Works"):
    st.markdown("### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(M['cm'], annot=True, fmt='d', cmap='Reds', ax=ax_cm,
                xticklabels=['No Depression','Depression'],
                yticklabels=['No Depression','Depression'],
                linewidths=0.5, annot_kws={'size':12,'weight':'bold'})
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    ax_cm.set_title('SVM Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True); plt.close()
    tn,fp,fn,tp = M['cm'].ravel()
    a,b,c,d = st.columns(4)
    a.metric("TN", str(tn)); b.metric("FP", str(fp))
    c.metric("FN", str(fn)); d.metric("TP", str(tp))

    st.write("")
    st.markdown("### How SVM Works")
    st.write(
        "Support Vector Machine finds the **optimal hyperplane** that maximally "
        "separates two classes. The **RBF (Radial Basis Function) kernel** maps "
        "features into a higher-dimensional space to handle non-linearly separable data. "
        "**Balanced class weights** compensate for the class imbalance in the dataset. "
        "The PCA visualization above projects the high-dimensional SVM boundary "
        "into 2D for visualization."
    )


# ══════════════════════════════════════════════════════════════
# BATCH PREDICTION — SVM
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("Batch Prediction — 🔴 SVM")
st.caption("Upload a CSV with multiple students. SVM predicts each one instantly.")

_svm_sample = pd.DataFrame({
    'Name':           ['Ahmad','Siti','Wei Ming'],
    'Gender':         ['Male','Female','Male'],
    'Age':            [20,21,19],
    'Course':         ['Computer Science','Engineering','Information Technology'],
    'Year_of_Study':  ['Year 2','Year 3','Year 1'],
    'CGPA':           ['3.00 - 3.49','2.50 - 2.99','3.50 - 4.00'],
    'Marital_Status': ['No','No','No'],
    'Anxiety':        ['Yes','No','No'],
    'Panic_Attack':   ['No','Yes','No'],
    'Seek_Treatment': ['No','No','No'],
})

with st.expander("📋  View / Download CSV Template"):
    st.dataframe(_svm_sample, use_container_width=True, hide_index=True)
    st.download_button("⬇️  Download SVM Template",
        data=_svm_sample.to_csv(index=False).encode(),
        file_name="svm_batch_template.csv", mime="text/csv",
        use_container_width=True, key="svm_tmpl_dl")

_svm_file = st.file_uploader("Upload CSV for SVM Batch",
                               type=["csv"], key="svm_batch_file")
if _svm_file:
    try:
        _sdf = pd.read_csv(_svm_file)
        _sdf.columns = _sdf.columns.str.strip()
        st.success(f"✅ {len(_sdf)} students loaded")
        _sresults = []
        for _si, _srow in _sdf.iterrows():
            try:
                _sname = str(_srow.get('Name', f'Student {_si+1}')).strip()
                _sgender  = str(_srow.get('Gender','Male'))
                _smarital = str(_srow.get('Marital_Status','No'))
                _sanxiety = str(_srow.get('Anxiety','No'))
                _spanic   = str(_srow.get('Panic_Attack','No'))
                _streat   = str(_srow.get('Seek_Treatment','No'))
                try: _sage = int(float(_srow.get('Age',20)))
                except: _sage = 20
                _scourse = str(_srow.get('Course','Others'))
                _syear   = str(_srow.get('Year_of_Study','Year 1'))
                _scgpa   = str(_srow.get('CGPA','3.00 - 3.49')).strip()
                _syear_n = ''.join(filter(str.isdigit, _syear)) or '1'
                _sinp = pd.DataFrame([{
                    'Choose your gender'                           : _sgender,
                    'Age'                                          : _sage,
                    'Your current year of Study'                   : f'year {_syear_n}',
                    'What is your CGPA?'                           : _scgpa,
                    'Marital status'                               : _smarital,
                    'Do you have Anxiety?'                         : _sanxiety,
                    'Do you have Panic attack?'                    : _spanic,
                    'Did you seek any specialist for a treatment?' : _streat,
                    'Course_Category'                              : M['cat'](_scourse),
                }])[M['svm_col_order']]
                _spred = int(M['svm'].predict(_sinp)[0])
                _sprob = M['svm'].predict_proba(_sinp)[0][1]
                _sresults.append({
                    'Name': _sname, 'Gender': _sgender, 'Age': _sage,
                    'Course': _scourse, 'Year': _syear, 'CGPA': _scgpa,
                    'Marital': _smarital, 'Anxiety': _sanxiety,
                    'Panic': _spanic, 'Treatment': _streat,
                    'Result':     '⚠️ Depression' if _spred==1 else '✅ No Depression',
                    'Confidence': f"{_sprob*100:.1f}%",
                    'Risk':       'HIGH' if _spred==1 else 'LOW',
                    '_pred': _spred, '_prob': _sprob,
                })
            except Exception as _se:
                _sresults.append({'Name': str(_srow.get('Name','')), 'Error': str(_se)})

        _sres = pd.DataFrame(_sresults)
        _stotal = len(_sres)
        _sdep   = int(_sres['_pred'].sum()) if '_pred' in _sres else 0
        _snodep = _stotal - _sdep

        _sm1,_sm2,_sm3 = st.columns(3)
        _sm1.metric("Total Students", str(_stotal))
        _sm2.metric("⚠️ At Risk",     str(_sdep),
                    delta=f"{_sdep/_stotal*100:.0f}%", delta_color="inverse")
        _sm3.metric("✅ No Risk",      str(_snodep),
                    delta=f"{_snodep/_stotal*100:.0f}%")

        _sc1,_sc2 = st.columns(2)
        with _sc1:
            _sfig,_sax2 = plt.subplots(figsize=(4,3))
            _sax2.pie([_sdep,_snodep],
                labels=[f'Depression ({_sdep})',f'No Depression ({_snodep})'],
                colors=['#EF4444','#10B981'], autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2},
                textprops={'fontsize':10,'fontweight':'bold'})
            _sax2.set_title('SVM Batch Results', fontweight='bold')
            plt.tight_layout(); st.pyplot(_sfig, use_container_width=True); plt.close()
        with _sc2:
            _sfig2,_sax3 = plt.subplots(figsize=(4,3))
            _sax3.hist(_sres['_prob']*100, bins=10, color='#EF4444', edgecolor='white', alpha=0.85)
            _sax3.axvline(50, color='navy', ls='--', lw=1.5, label='Threshold 50%')
            _sax3.set_xlabel('Depression Probability (%)'); _sax3.set_ylabel('Count')
            _sax3.set_title('Confidence Distribution', fontweight='bold'); _sax3.legend(fontsize=9)
            _sax3.spines['top'].set_visible(False); _sax3.spines['right'].set_visible(False)
            plt.tight_layout(); st.pyplot(_sfig2, use_container_width=True); plt.close()

        _sdisp = _sres[['Name','Gender','Age','Course','Year','CGPA',
                         'Anxiety','Panic','Result','Confidence','Risk']]
        def _sstyle(val):
            if '⚠️' in str(val) or val=='HIGH': return 'background-color:#FEE2E2;color:#991B1B;font-weight:bold'
            if '✅' in str(val) or val=='LOW':  return 'background-color:#DCFCE7;color:#166534;font-weight:bold'
            return ''
        try:    _sstyled = _sdisp.style.map(_sstyle, subset=['Result','Risk'])
        except: _sstyled = _sdisp.style.applymap(_sstyle, subset=['Result','Risk'])
        st.dataframe(_sstyled, use_container_width=True, hide_index=True)

        st.write("")
        st.markdown("**Individual Student Cards**")
        for _, _sr in _sres.iterrows():
            if '_pred' not in _sr: continue
            _sicon = "⚠️" if _sr['_pred']==1 else "✅"
            with st.expander(f"{_sicon} {_sr['Name']} — {_sr['Result']} ({_sr['Confidence']})"):
                if _sr['_pred']==1: st.error(f"**Depression Risk** | SVM Confidence: {_sr['Confidence']}")
                else:               st.success(f"**No Depression** | SVM Confidence: {_sr['Confidence']}")

        _sdl1,_sdl2 = st.columns(2)
        with _sdl1:
            st.download_button("⬇️  Download All Results",
                data=_sdisp.to_csv(index=False).encode(),
                file_name="svm_batch_results.csv", mime="text/csv",
                use_container_width=True, key="svm_dl_all")
        with _sdl2:
            _shigh = _sres[_sres['_pred']==1][['Name','Gender','Age','Course',
                'Year','CGPA','Anxiety','Panic','Result','Confidence']]
            if len(_shigh):
                st.download_button(f"⬇️  At-Risk Only ({len(_shigh)})",
                    data=_shigh.to_csv(index=False).encode(),
                    file_name="svm_at_risk.csv", mime="text/csv",
                    use_container_width=True, key="svm_dl_risk")
    except Exception as _serr:
        st.error(f"Error processing file: {str(_serr)}")
        st.caption("Please check your CSV matches the template format.")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")