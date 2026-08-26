import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family':'sans-serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#E2E8F0','axes.labelcolor':'#1E293B','xtick.color':'#64748B','ytick.color':'#64748B','text.color':'#1E293B','grid.color':'#F1F5F9','grid.linewidth':0.8})
import seaborn as sns
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar, came_from
from utils.models import load_all_models, MAX_BATCH_ROWS, MAX_BATCH_MB
from utils.pdf_report import generate_pdf

st.set_page_config(page_title="KNN Predictor — MindCheck",
                   page_icon="🔵", layout="wide",
                   initial_sidebar_state="expanded")
sidebar("knn")

if came_from('compare'):
    if st.button("← Back to Compare Models", key="knn_back_to_compare"):
        st.switch_page("pages/5_Comparison.py")

M = load_all_models()

# ── Restore result carried from Comparison page ──────────────
if 'knn_carry' in st.session_state and st.session_state['knn_carry'] is not None:
    st.session_state['knn_result'] = st.session_state.pop('knn_carry')

if 'knn_result' not in st.session_state:
    st.session_state['knn_result'] = None

# ── Prefill from Comparison page ──────────────────────────────
_pre = st.session_state.pop('knn_prefill', None)
if _pre:
    st.info(f"📋 Data from Comparison page: **{_pre.get('name','Student')}** — form pre-filled below.")

# ── Header ─────────────────────────────────────────────────────
st.markdown("##### 🔵 KNN PREDICTOR")
st.title("Depression Risk Predictor")
st.caption(f"K-Nearest Neighbor · K={M['best_k']} · "
           f"Member: Ho Jun Yon · Live trained on {M['n_records']} records")
st.divider()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{M['knn_m']['acc']:.2f}%")
c2.metric("Precision", f"{M['knn_m']['prec']:.2f}%")
c3.metric("Recall",    f"{M['knn_m']['rec']:.2f}%")
c4.metric("F1 Score",  f"{M['knn_m']['f1']:.2f}%")
st.divider()

# ── Input form ─────────────────────────────────────────────────
st.subheader("Student Information")
col1, col2, col3 = st.columns(3)
with col1:
    _pn = _pre.get('name','')  if _pre else ''
    name   = st.text_input("Name", value=_pn, placeholder="e.g. Ahmad", key="knn_name")
    _gi = ["Female","Male"].index(_pre['gender']) if _pre and _pre.get('gender') in ["Female","Male"] else 0
    gender = st.selectbox("Gender", ["Female","Male"], index=_gi, key="knn_gender")
    _ai = int(_pre.get('age',20)) if _pre else 20
    age    = st.slider("Age", 17, 30, _ai, key="knn_age")
with col2:
    course = st.selectbox("Course", M['courses'], key="knn_course")
    _yi = ["Year 1","Year 2","Year 3","Year 4"].index(_pre['year']) if _pre and _pre.get('year') in ["Year 1","Year 2","Year 3","Year 4"] else 0
    year   = st.selectbox("Year of Study", ["Year 1","Year 2","Year 3","Year 4"], index=_yi, key="knn_year")
    cgpa   = st.selectbox("CGPA Range", list(M['cgpa_map'].keys()), key="knn_cgpa")
with col3:
    _axi = ["No","Yes"].index(_pre['anxiety']) if _pre and _pre.get('anxiety') in ["No","Yes"] else 0
    anxiety = st.selectbox("Do you have Anxiety?",      ["No","Yes"], index=_axi, key="knn_anxiety")
    _pai = ["No","Yes"].index(_pre['panic']) if _pre and _pre.get('panic') in ["No","Yes"] else 0
    panic   = st.selectbox("Do you have Panic Attack?", ["No","Yes"], index=_pai, key="knn_panic")
    st.write(""); st.write(""); st.write("")
    predict_btn = st.button("🔍  Predict Depression Risk",
                            width='stretch', type="primary", key="knn_predict")

if predict_btn:
    _errors = []
    if name.strip() and name.strip().isdigit():
        _errors.append("❌ Name cannot be numbers only.")
    if age < 17 or age > 35:
        _errors.append(f"❌ Age {age} is outside valid range (17–35).")
    _high_concern  = (anxiety == "Yes" and panic == "Yes")
    _academic_risk = (year in ["Year 3","Year 4"] and cgpa == "0 - 1.99")
    if _errors:
        for e in _errors: st.error(e)
        st.warning("⚠️ Please fix the errors above before predicting.")
    else:
        try:
            g  = 1 if gender  == "Male" else 0
            ax = 1 if anxiety == "Yes"  else 0
            pa = 1 if panic   == "Yes"  else 0
            ce = M['le_c'].transform([course])[0] if course in M['le_c'].classes_ else 0
            ye = M['le_y'].transform([year])[0]   if year   in M['le_y'].classes_ else 0
            cn = M['cgpa_map'].get(cgpa, 3.25)
            inp   = pd.DataFrame([[g,age,ce,ye,cn,ax,pa]], columns=M['knn_feat'])
            inp_s = M['sc_knn'].transform(inp)
            pred  = int(M['knn'].predict(inp_s)[0])
            prob  = M['knn'].predict_proba(inp_s)[0].tolist()
            st.session_state['knn_result'] = {
                'pred': pred, 'prob': prob,
                'name': name.strip() or "Student",
                'gender': gender, 'age': age, 'course': course,
                'year': year, 'cgpa': cgpa,
                'anxiety': anxiety, 'panic': panic,
                'high_concern': _high_concern,
                'academic_risk': _academic_risk,
                'inp_s': inp_s,
            }
        except Exception as ex:
            st.error(f"❌ Prediction failed: {ex}")

# ── Result ─────────────────────────────────────────────────────
if st.session_state['knn_result']:
    R    = st.session_state['knn_result']
    pred = R['pred']; prob = R['prob']; name_lbl = R['name']
    st.divider()
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"### ⚠️  {name_lbl} — Depression Risk Detected\n\n"
                 "Please consider speaking with a counsellor or mental health professional.")
    else:
        st.success(f"### ✅  {name_lbl} — No Depression Detected\n\n"
                   "Keep maintaining a healthy academic and social lifestyle!")

    if R.get('high_concern'):
        st.warning("⚠️ **High Concern:** Student has both Anxiety AND Panic Attack. Immediate counselling referral recommended.")
    if R.get('academic_risk'):
        st.warning("⚠️ **Academic Risk:** Senior year student with very low CGPA (0–1.99). Combined support advised.")

    st.write("")
    r1, r2, r3 = st.columns([1.2, 1.2, 1])
    with r1:
        st.markdown("**Prediction Confidence**")
        fig, ax2 = plt.subplots(figsize=(4, 0.8))
        ax2.barh([""], [prob[0]*100], color="#10B981", height=0.5)
        ax2.barh([""], [prob[1]*100], left=[prob[0]*100], color="#EF4444", height=0.5)
        ax2.set_xlim(0,100); ax2.axis('off')
        for xp, val, lbl in [(prob[0]*50,prob[0],"No Risk"),(prob[0]*100+prob[1]*50,prob[1],"At Risk")]:
            if val > 0.12:
                ax2.text(xp,0,f"{lbl}\n{val*100:.0f}%",ha='center',va='center',fontsize=8,color='white',fontweight='bold')
        plt.tight_layout(pad=0); st.pyplot(fig,width='stretch'); plt.close()
        st.write("")
        pa_c,pb_c = st.columns(2)
        pa_c.metric("No Depression", f"{prob[0]*100:.1f}%")
        pb_c.metric("Depression",    f"{prob[1]*100:.1f}%")
    with r2:
        st.markdown("**Input Summary**")
        st.table(pd.DataFrame({"Field":["Name","Gender","Age","Course","Year","CGPA","Anxiety","Panic Attack"],
                               "Value":[R['name'],R['gender'],str(R['age']),R['course'],R['year'],R['cgpa'],R['anxiety'],R['panic']]
                              }).set_index("Field"))
    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write(f"**Algorithm:** KNN")
            st.write(f"**Best K:** {M['best_k']}")
            st.write(f"**Scaling:** MinMax")
            st.write(f"**Distance:** Euclidean")
            st.write(f"**Split:** 80 / 20")
            st.write(f"**CV Mean:** {M['cv_scores'].mean()*100:.2f}%")
            st.write(f"**Accuracy:** {M['knn_m']['acc']:.2f}%")

    # KNN Neighbours — PCA projection (same visual language as the SVM
    # decision-boundary chart, so the two predictor pages feel consistent)
    st.write(""); st.markdown("**KNN Neighbours (PCA Projection)**")
    try:
        # Recompute the scaled input from R's fields (not a stored inp_s
        # array) — this is the same pattern the "K Nearest Neighbours"
        # panel below already uses, and it's what makes this work whether
        # R came from a fresh prediction on this page OR a Quick Navigate
        # carry from the Comparison page (which doesn't carry inp_s).
        _pg = 1 if R['gender']=='Male' else 0
        _pax = 1 if R['anxiety']=='Yes' else 0
        _ppa = 1 if R['panic']=='Yes' else 0
        _pce = M['le_c'].transform([R['course']])[0] if R['course'] in M['le_c'].classes_ else 0
        _pye = M['le_y'].transform([R['year']])[0]   if R['year']   in M['le_y'].classes_ else 0
        _pcn = M['cgpa_map'].get(R['cgpa'], 3.25)
        _p_inp = pd.DataFrame([[_pg, R['age'], _pce, _pye, _pcn, _pax, _ppa]], columns=M['knn_feat'])
        _p_inp_s = M['sc_knn'].transform(_p_inp)

        pca = PCA(n_components=2, random_state=42)
        X_bg_2d = pca.fit_transform(M['knn_Xtr'])
        X_us_2d = pca.transform(_p_inp_s)

        _dists, _idxs = M['knn'].kneighbors(_p_inp_s)
        y_all = np.array(M['knn_ytr'])
        nbr_pts    = X_bg_2d[_idxs[0]]
        nbr_labels = y_all[_idxs[0]]

        fig4, ax4 = plt.subplots(figsize=(8, 4.5))
        # All training students, faint background
        ax4.scatter(X_bg_2d[y_all==1,0], X_bg_2d[y_all==1,1],
                    color='#EF4444', s=25, alpha=0.30, label='Depression', zorder=2)
        ax4.scatter(X_bg_2d[y_all==0,0], X_bg_2d[y_all==0,1],
                    color='#3B82F6', s=25, alpha=0.30, label='No Depression', zorder=2)

        # Dashed lines from the student to each of the K neighbours actually
        # used by the model (real neighbours from M['knn'].kneighbors(), not
        # just "nearest in this 2D picture" — the picture is a projection of
        # the real decision, not a re-derived one)
        for pt in nbr_pts:
            ax4.plot([X_us_2d[0,0], pt[0]], [X_us_2d[0,1], pt[1]],
                     color='#9CA3AF', lw=1, linestyle='--', zorder=3)

        ax4.scatter(nbr_pts[:,0], nbr_pts[:,1],
                    color=['#EF4444' if l==1 else '#3B82F6' for l in nbr_labels],
                    s=150, edgecolor='black', linewidth=1.5, zorder=4,
                    label=f'{M["best_k"]} Nearest Neighbours')

        ax4.scatter(X_us_2d[0,0], X_us_2d[0,1], color='black', s=300,
                    marker='*', label=f'Input: {name_lbl}', zorder=6)

        ax4.set_xlabel('PC1', fontsize=9)
        ax4.set_ylabel('PC2', fontsize=9)
        ax4.set_title(f'Live KNN Neighbours — PCA 2D Projection (K={M["best_k"]})',
                      fontsize=11, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4, width='stretch')
        plt.close()
        st.caption(
            "Full 7-feature space compressed to 2D via PCA for visualization — the "
            "highlighted neighbours are the model's real K nearest neighbours, not "
            "re-derived from this 2D picture."
        )
    except Exception as _ke2:
        st.warning(f"Visualization unavailable: {_ke2}")

    # Explanation
    st.write(""); st.markdown("---")
    st.markdown("### 🧠 Why did KNN predict this?")
    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown("**Feature Contribution**")
        _input_vals = {'Gender':1 if R['gender']=='Male' else 0,'Age':R['age'],
                       'Course':0,'Year':0,'CGPA':M['cgpa_map'].get(R['cgpa'],3.25),
                       'Anxiety':1 if R['anxiety']=='Yes' else 0,'Panic Attack':1 if R['panic']=='Yes' else 0}
        _contrib = {f: v * M['knn_corr'].get(f,0) if v>0.5 else -abs(M['knn_corr'].get(f,0))*0.3
                    for f,v in _input_vals.items()}
        _cs = pd.Series(_contrib).sort_values()
        fig_ex,ax_ex = plt.subplots(figsize=(5,3.5))
        colors_ex = ['#EF4444' if v>0 else '#10B981' for v in _cs.values]
        ax_ex.barh(list(_cs.index),_cs.values,color=colors_ex,edgecolor='none',height=0.6)
        ax_ex.axvline(0,color='black',lw=1)
        ax_ex.set_xlabel('Contribution (→ Depression  ← No Depression)')
        ax_ex.set_title('Feature Contribution',fontweight='bold')
        ax_ex.spines['top'].set_visible(False); ax_ex.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_ex,width='stretch'); plt.close()
    with ex2:
        st.markdown("**K Nearest Neighbours**")
        try:
            g=1 if R['gender']=='Male' else 0; ax=1 if R['anxiety']=='Yes' else 0
            pa=1 if R['panic']=='Yes' else 0
            ce=M['le_c'].transform([R['course']])[0] if R['course'] in M['le_c'].classes_ else 0
            ye=M['le_y'].transform([R['year']])[0]   if R['year']   in M['le_y'].classes_ else 0
            cn=M['cgpa_map'].get(R['cgpa'],3.25)
            inp_r=pd.DataFrame([[g,R['age'],ce,ye,cn,ax,pa]],columns=M['knn_feat'])
            inp_s2=M['sc_knn'].transform(inp_r)
            dists,idxs=M['knn'].kneighbors(inp_s2)
            nbr_preds=[M['knn']._y[i] for i in idxs[0]]
            dep_n=sum(nbr_preds); ndep_n=len(nbr_preds)-dep_n
            with st.container(border=True):
                st.write(f"**Among {M['best_k']} nearest neighbours:**")
                st.write(f"🔴 Depressed: **{dep_n}** students")
                st.write(f"🟢 Not Depressed: **{ndep_n}** students")
                st.write("")
                for i,(dist,lbl) in enumerate(zip(dists[0],nbr_preds),1):
                    icon="🔴" if lbl==1 else "🟢"
                    st.write(f"Neighbour {i}: {icon} {'Depressed' if lbl==1 else 'Not Depressed'} (dist: {dist:.4f})")
                st.write("")
                st.caption(f"Majority vote: {dep_n}/{M['best_k']} → **{'Depression' if dep_n>ndep_n else 'No Depression'}**")
        except Exception as _ne: st.info(f"Neighbour analysis: {_ne}")

    st.write("")
    _alerts_knn=[]; _notes_knn=[]
    if R.get('high_concern'): _alerts_knn.append("High Concern: Both Anxiety and Panic Attack present.")
    if R.get('academic_risk'): _alerts_knn.append("Academic Risk: Senior year with CGPA 0-1.99.")
    if R['anxiety']=='Yes': _notes_knn.append("Anxiety = Yes — 2nd strongest predictor (r=0.257)")
    if R['panic']=='Yes':   _notes_knn.append("Panic Attack = Yes — strongest predictor (r=0.341)")
    try:
        # Build feature contribution for PDF
        _fi_for_pdf = {
            'Gender':       (1 if R['gender']=='Male' else 0) * M['knn_corr'].get('Gender',0),
            'Age':          -abs(M['knn_corr'].get('Age',0))*0.3,
            'CGPA':         -abs(M['knn_corr'].get('CGPA',0))*0.3,
            'Anxiety':      (1 if R['anxiety']=='Yes' else -0.3) * M['knn_corr'].get('Anxiety',0),
            'Panic Attack': (1 if R['panic']=='Yes'   else -0.3) * M['knn_corr'].get('Panic Attack',0),
        }
        _pdf=generate_pdf(
            model_name   = f"KNN (K={M['best_k']})",
            student_name = R['name'],
            result       = R['pred'],
            prob         = R['prob'],
            input_data   = {
                "Gender": R['gender'], "Age": str(R['age']),
                "Course": R['course'], "Year": R['year'],
                "CGPA":   R['cgpa'],   "Anxiety": R['anxiety'],
                "Panic Attack": R['panic'],
            },
            metrics            = M['knn_m'],
            business_alerts    = _alerts_knn,
            explanation_notes  = _notes_knn,
            feature_importance = _fi_for_pdf,
            cm_array           = M['knn_m']['cm'],
            model_color        = '#2563EB',
        )
        st.download_button("📥  Download PDF Report",data=_pdf,
                           file_name=f"mindcheck_knn_{R['name'].replace(' ','_')}.pdf",
                           mime="application/pdf",width='stretch',key="knn_pdf")
    except Exception as _pe: st.caption(f"PDF unavailable: {_pe}")

    st.write("")
    if st.button("Clear Result", key="knn_clear"):
        st.session_state['knn_result'] = None; st.rerun()

st.divider()

# ── Live Feature Importance ────────────────────────────────────
st.subheader("Live Feature Importance")
st.caption("Pearson correlation — computed live from dataset")
fi = M['knn_corr'].sort_values(ascending=True)
fi_c1,fi_c2 = st.columns([2,1])
with fi_c1:
    fig_fi,ax_fi=plt.subplots(figsize=(7,3.5))
    colors_fi=['#3B82F6' if v>=fi.mean() else '#64748B' for v in fi.values]
    bars_fi=ax_fi.barh(list(fi.index),fi.values,color=colors_fi,edgecolor='none',height=0.6)
    ax_fi.axvline(fi.mean(),color='red',ls='--',lw=1.2,alpha=0.7,label=f'Mean={fi.mean():.3f}')
    for bar,val in zip(bars_fi,fi.values):
        ax_fi.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=9,fontweight='bold')
    ax_fi.set_xlabel('Absolute Correlation with Depression')
    ax_fi.set_title('Feature Importance — Pearson Correlation (Live)',fontweight='bold')
    ax_fi.legend(fontsize=9); ax_fi.spines['top'].set_visible(False); ax_fi.spines['right'].set_visible(False)
    ax_fi.grid(axis='x',alpha=0.3); plt.tight_layout(); st.pyplot(fig_fi,width='stretch'); plt.close()
with fi_c2:
    with st.container(border=True):
        st.markdown("**Feature Ranking**")
        for feat_n,val in fi.sort_values(ascending=False).items():
            icon="🔵" if val>=fi.mean() else "⚪"
            st.write(f"{icon} **{feat_n}** — {val:.3f}")
        st.caption("🔵 Above average importance")

st.divider()

# ── Learn More ─────────────────────────────────────────────────
with st.expander("📚  Learn More — K-Value Optimization, Cross Validation & Confusion Matrix"):
    st.markdown("### K-Value Optimization")
    fig_k,ax_k=plt.subplots(figsize=(10,4))
    k_range=list(range(1,21))
    ax_k.plot(k_range,[s*100 for s in M['k_scores']],'r-o',label='Test Accuracy',markersize=4,lw=2)
    ax_k.axvline(x=M['best_k'],color='green',ls='--',lw=2,label=f'Best K={M["best_k"]}')
    ax_k.set_xlabel('K Value'); ax_k.set_ylabel('Accuracy (%)')
    ax_k.set_title('KNN: Test Accuracy for Different K Values',fontweight='bold')
    ax_k.set_xticks(k_range); ax_k.legend(); ax_k.grid(True,alpha=0.3)
    ax_k.spines['top'].set_visible(False); ax_k.spines['right'].set_visible(False)
    plt.tight_layout(); st.pyplot(fig_k,width='stretch'); plt.close()

    st.markdown("### 5-Fold Cross Validation")
    cv=M['cv_scores']
    cv1,cv2,cv3=st.columns(3)
    cv1.metric("CV Mean",f"{cv.mean()*100:.2f}%")
    cv2.metric("CV Std",f"{cv.std()*100:.2f}%")
    cv3.metric("CV Max",f"{cv.max()*100:.2f}%")
    fig_cv,ax_cv=plt.subplots(figsize=(8,3))
    colors_cv=['#3B82F6','#10B981','#EF4444','#F59E0B','#8B5CF6']
    bars_cv=ax_cv.bar([f"Fold {i+1}" for i in range(5)],cv*100,color=colors_cv,edgecolor='none',alpha=0.9)
    ax_cv.axhline(y=cv.mean()*100,color='red',ls='--',lw=1.5,label=f'Mean={cv.mean()*100:.2f}%')
    ax_cv.set_ylabel('Accuracy (%)'); ax_cv.set_ylim(0,110)
    ax_cv.set_title('5-Fold Cross Validation Scores',fontweight='bold')
    ax_cv.legend(); ax_cv.grid(axis='y',alpha=0.3)
    ax_cv.spines['top'].set_visible(False); ax_cv.spines['right'].set_visible(False)
    for bar,val in zip(bars_cv,cv*100):
        ax_cv.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.8,f'{val:.1f}%',ha='center',fontsize=9,fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_cv,width='stretch'); plt.close()

    st.markdown("### Confusion Matrix")
    fig_cm,ax_cm=plt.subplots(figsize=(5,4))
    sns.heatmap(M['knn_m']['cm'],annot=True,fmt='d',cmap='Blues',ax=ax_cm,
                xticklabels=['No Depression','Depression'],
                yticklabels=['No Depression','Depression'],
                linewidths=0.5,annot_kws={'size':12,'weight':'bold'})
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    ax_cm.set_title('KNN Confusion Matrix',fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_cm,width='stretch'); plt.close()
    tn,fp,fn,tp=M['knn_m']['cm'].ravel()
    a,b,c,d=st.columns(4)
    a.metric("TN",str(tn)); b.metric("FP",str(fp))
    c.metric("FN",str(fn)); d.metric("TP",str(tp))

# ── Batch Prediction ───────────────────────────────────────────
st.divider()
st.subheader("Batch Prediction — 🔵 KNN")
st.caption("Upload a CSV to predict multiple students at once using KNN.")
st.caption(f"⚠️ Limits: max **{MAX_BATCH_ROWS} records** and **{MAX_BATCH_MB} MB** per upload.")

_knn_sample=pd.DataFrame({'Name':['Ahmad','Siti','Wei Ming'],'Gender':['Male','Female','Male'],
    'Age':[20,21,19],'Course':['Computer Science','Engineering','Information Technology'],
    'Year_of_Study':['Year 2','Year 3','Year 1'],'CGPA':['3.00 - 3.49','2.50 - 2.99','3.50 - 4.00'],
    'Anxiety':['Yes','No','No'],'Panic_Attack':['No','Yes','No']})
with st.expander("📋  View / Download CSV Template"):
    st.dataframe(_knn_sample,width='stretch',hide_index=True)
    st.download_button("⬇️  Download KNN Template",
        data=_knn_sample.to_csv(index=False).encode(),
        file_name="knn_batch_template.csv",mime="text/csv",
        width='stretch',key="knn_tmpl_dl")

_knn_file=st.file_uploader(f"Upload CSV — drag & drop or click to browse (max {MAX_BATCH_ROWS} records, {MAX_BATCH_MB} MB)",
                            type=["csv"],key="knn_batch_file")
if _knn_file:
    _knn_size_mb = _knn_file.size / (1024*1024)
    if _knn_size_mb > MAX_BATCH_MB:
        st.error(f"❌ File is {_knn_size_mb:.2f} MB — exceeds the {MAX_BATCH_MB} MB limit. Please upload a smaller file.")
    else:
        try:
            _kdf=pd.read_csv(_knn_file); _kdf.columns=_kdf.columns.str.strip()
            if len(_kdf) > MAX_BATCH_ROWS:
                st.warning(f"⚠️ File has {len(_kdf)} records — only the first {MAX_BATCH_ROWS} will be processed.")
                _kdf = _kdf.head(MAX_BATCH_ROWS)
            st.success(f"✅ {len(_kdf)} students loaded")
            _kresults=[]
            for _ki,_krow in _kdf.iterrows():
                try:
                    _kname=str(_krow.get('Name',f'Student {int(_ki)+1}')).strip()
                    _kg=1 if str(_krow.get('Gender','')).lower()=='male' else 0
                    _kax=1 if str(_krow.get('Anxiety','')).lower()=='yes' else 0
                    _kpa=1 if str(_krow.get('Panic_Attack','')).lower()=='yes' else 0
                    try: _kage=int(float(_krow.get('Age',20)))
                    except: _kage=20
                    _kcourse=str(_krow.get('Course','Others'))
                    _kyear=str(_krow.get('Year_of_Study','Year 1'))
                    _kcgpa=str(_krow.get('CGPA','3.00 - 3.49')).strip()
                    _kce=M['le_c'].transform([_kcourse])[0] if _kcourse in M['le_c'].classes_ else 0
                    _kye=M['le_y'].transform([_kyear])[0]   if _kyear   in M['le_y'].classes_ else 0
                    _kcn=M['cgpa_map'].get(_kcgpa,3.25)
                    _kinp=pd.DataFrame([[_kg,_kage,_kce,_kye,_kcn,_kax,_kpa]],columns=M['knn_feat'])
                    _kinp_s=M['sc_knn'].transform(_kinp)
                    _kpred=int(M['knn'].predict(_kinp_s)[0])
                    _kprob=M['knn'].predict_proba(_kinp_s)[0][1]
                    _kresults.append({'Name':_kname,'Gender':_krow.get('Gender',''),
                        'Age':_kage,'Course':_kcourse,'Year':_kyear,'CGPA':_kcgpa,
                        'Anxiety':_krow.get('Anxiety',''),'Panic Attack':_krow.get('Panic_Attack',''),
                        'Result':'⚠️ Depression' if _kpred==1 else '✅ No Depression',
                        'Confidence':f"{_kprob*100:.1f}%",'Risk':'HIGH' if _kpred==1 else 'LOW',
                        '_pred':_kpred,'_prob':_kprob})
                except Exception as _ke: _kresults.append({'Name':str(_krow.get('Name','')),'Error':str(_ke)})
            _kres=pd.DataFrame(_kresults)
            _ktotal=len(_kres); _kdep=int(_kres['_pred'].sum()) if '_pred' in _kres else 0
            _bm1,_bm2,_bm3=st.columns(3)
            _bm1.metric("Total Students",str(_ktotal))
            _bm2.metric("⚠️ At Risk",str(_kdep),delta=f"{_kdep/_ktotal*100:.0f}%",delta_color="inverse")
            _bm3.metric("✅ No Risk",str(_ktotal-_kdep),delta=f"{(_ktotal-_kdep)/_ktotal*100:.0f}%")
            _bc1,_bc2=st.columns(2)
            with _bc1:
                fig_b,ax_b=plt.subplots(figsize=(4,3))
                ax_b.pie([_kdep,_ktotal-_kdep],labels=[f'Depression ({_kdep})',f'No Depression ({_ktotal-_kdep})'],
                    colors=['#EF4444','#10B981'],autopct='%1.1f%%',startangle=90,
                    wedgeprops={'edgecolor':'white','linewidth':2},textprops={'fontsize':10,'fontweight':'bold'})
                ax_b.set_title('KNN Batch Results',fontweight='bold')
                plt.tight_layout(); st.pyplot(fig_b,width='stretch'); plt.close()
            with _bc2:
                if '_prob' in _kres:
                    fig_p,ax_p=plt.subplots(figsize=(4,3))
                    ax_p.hist(_kres['_prob']*100,bins=min(10,_ktotal),color='#3B82F6',edgecolor='white',alpha=0.85)
                    ax_p.axvline(50,color='red',ls='--',lw=1.5,label='Threshold 50%')
                    ax_p.set_xlabel('Depression Probability (%)'); ax_p.set_ylabel('Count')
                    ax_p.set_title('Confidence Distribution',fontweight='bold')
                    ax_p.legend(fontsize=9); ax_p.spines['top'].set_visible(False); ax_p.spines['right'].set_visible(False)
                    plt.tight_layout(); st.pyplot(fig_p,width='stretch'); plt.close()
            _kdisp=_kres[['Name','Gender','Age','Course','Year','CGPA','Anxiety','Panic Attack','Result','Confidence','Risk']]
            def _kst(val):
                if '⚠️' in str(val) or val=='HIGH': return 'background-color:#FEE2E2;color:#991B1B;font-weight:bold'
                if '✅' in str(val) or val=='LOW':  return 'background-color:#DCFCE7;color:#166534;font-weight:bold'
                return ''
            _kstyled=_kdisp.style.map(_kst,subset=['Result','Risk'])
            st.dataframe(_kstyled,width='stretch',hide_index=True)
            st.write(""); st.markdown("**Individual Student Cards**")
            for _,_kr in _kres.iterrows():
                if '_pred' not in _kr: continue
                with st.expander(f"{'⚠️' if _kr['_pred']==1 else '✅'} {_kr['Name']} — {_kr['Result']} ({_kr['Confidence']})"):
                    if _kr['_pred']==1: st.error(f"**Depression Risk** | Confidence: {_kr['Confidence']}")
                    else:               st.success(f"**No Depression** | Confidence: {_kr['Confidence']}")
            _kdl1,_kdl2=st.columns(2)
            with _kdl1:
                st.download_button("⬇️  Download All Results",data=_kdisp.to_csv(index=False).encode(),
                    file_name="knn_batch_results.csv",mime="text/csv",width='stretch',key="knn_dl_all")
            with _kdl2:
                _khigh=_kres[_kres['_pred']==1][list(_kdisp.columns)] if '_pred' in _kres else pd.DataFrame()
                if len(_khigh):
                    st.download_button(f"⬇️  At-Risk Only ({len(_khigh)})",
                        data=_khigh.to_csv(index=False).encode(),
                        file_name="knn_at_risk.csv",mime="text/csv",width='stretch',key="knn_dl_risk")
        except Exception as _kerr:
            st.error(f"Error: {str(_kerr)}")
            st.caption("Please check your CSV matches the template format.")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")