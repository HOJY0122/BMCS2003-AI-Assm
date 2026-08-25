import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar
from utils.models import load_all_models
from utils.pdf_report import generate_pdf

st.set_page_config(page_title="SVM Predictor — MindCheck",
                   page_icon="🔴", layout="wide",
                   initial_sidebar_state="expanded")
sidebar("svm")

M = load_all_models()

CGPA_MAP = M['cgpa_map']

if 'svm_result' not in st.session_state:
    st.session_state.svm_result = None

_pre = st.session_state.pop('svm_prefill', None)
if _pre:
    st.info(f"📋 Data from Comparison page: **{_pre.get('name','Student')}** — form pre-filled below.")

st.markdown("##### 🔴 SVM PREDICTOR")
st.title("Depression Risk Predictor")
st.caption(f"Support Vector Machine (RBF Kernel) · Member: Chiang Jun Hang · Live trained on {M['n_records']} records")
st.divider()

c1,c2,c3,c4=st.columns(4)
c1.metric("Accuracy",  f"{M['svm_m']['acc']:.2f}%")
c2.metric("Precision", f"{M['svm_m']['prec']:.2f}%")
c3.metric("Recall",    f"{M['svm_m']['rec']:.2f}%")
c4.metric("F1 Score",  f"{M['svm_m']['f1']:.2f}%")
st.divider()

st.subheader("Student Information")
col1,col2,col3=st.columns(3)
with col1:
    _pn=_pre.get('name','') if _pre else ''
    name   = st.text_input("Name",value=_pn,placeholder="e.g. Wei Ming",key="svm_name")
    _gi=["Male","Female"].index(_pre['gender']) if _pre and _pre.get('gender') in ["Male","Female"] else 0
    gender = st.selectbox("Gender",["Male","Female"],index=_gi,key="svm_gender")
    _ai=int(_pre.get('age',20)) if _pre else 20
    age    = st.slider("Age",15,40,_ai,key="svm_age")
with col2:
    course = st.selectbox("Course Field",["Information Technology (IT)","Computer Science (CS)",
        "Information System (IS)","Software Engineering (SE)","Other"],key="svm_course")
    _yi=["Year 1","Year 2","Year 3","Year 4"].index(_pre['year']) if _pre and _pre.get('year') in ["Year 1","Year 2","Year 3","Year 4"] else 0
    year   = st.selectbox("Year of Study",["Year 1","Year 2","Year 3","Year 4"],index=_yi,key="svm_year")
    cgpa   = st.selectbox("CGPA Range",list(CGPA_MAP.keys()),key="svm_cgpa")
with col3:
    _mai=["No","Yes"].index(_pre['marital']) if _pre and _pre.get('marital') in ["No","Yes"] else 0
    marital = st.selectbox("Marital Status",               ["No","Yes"],index=_mai,key="svm_marital")
    _axi=["No","Yes"].index(_pre['anxiety']) if _pre and _pre.get('anxiety') in ["No","Yes"] else 0
    anxiety = st.selectbox("Do you have Anxiety?",         ["No","Yes"],index=_axi,key="svm_anxiety")
    _pai=["No","Yes"].index(_pre['panic']) if _pre and _pre.get('panic') in ["No","Yes"] else 0
    panic   = st.selectbox("Do you have Panic Attack?",    ["No","Yes"],index=_pai,key="svm_panic")
    treat   = st.selectbox("Sought Specialist Treatment?", ["No","Yes"],key="svm_treat")
    predict_btn=st.button("🔍  Predict Depression Risk",use_container_width=True,type="primary",key="svm_predict")

if predict_btn:
    _errors=[]
    if name.strip() and name.strip().isdigit(): _errors.append("❌ Name cannot be numbers only.")
    if age<15 or age>40: _errors.append(f"❌ Age {age} outside valid range (15–40).")
    _high_concern=(anxiety=="Yes" and panic=="Yes")
    _treat_flag=(treat=="Yes")
    _academic_risk=(year in ["Year 3","Year 4"] and cgpa=="0 - 1.99")
    if _errors:
        for e in _errors: st.error(e)
        st.warning("⚠️ Please fix errors above.")
    else:
        try:
            yr_num=year.split()[-1]
            inp=pd.DataFrame([{
                'Choose your gender'                           : gender,
                'Age'                                          : age,
                'Your current year of Study'                   : f'year {yr_num}',
                'What is your CGPA?'                           : cgpa,
                'Marital status'                               : marital,
                'Do you have Anxiety?'                         : anxiety,
                'Do you have Panic attack?'                    : panic,
                'Did you seek any specialist for a treatment?' : treat,
                'Course_Category'                              : M['cat_course'](course),
            }])[M['svm_col_order']]
            pred=int(M['svm'].predict(inp)[0])
            prob=M['svm'].predict_proba(inp)[0].tolist()
            st.session_state.svm_result={
                'pred':pred,'prob':prob,'inp':inp,'name':name.strip() or "Student",
                'gender':gender,'age':age,'course':course,'year':year,'cgpa':cgpa,
                'marital':marital,'anxiety':anxiety,'panic':panic,'treat':treat,
                'high_concern':_high_concern,'treat_flag':_treat_flag,'academic_risk':_academic_risk}
        except Exception as ex: st.error(f"❌ Prediction failed: {ex}")

if st.session_state.svm_result:
    R=st.session_state.svm_result; pred=R['pred']; prob=R['prob']; name_lbl=R['name']
    st.divider(); st.subheader("Prediction Result")
    if pred==1:
        st.error(f"### ⚠️  {name_lbl} — Depression Risk Detected\n\nPlease consider seeking professional support.")
    else:
        st.success(f"### ✅  {name_lbl} — No Depression Detected\n\nKeep maintaining a healthy lifestyle!")
    if R.get('high_concern'):  st.warning("⚠️ **High Concern:** Both Anxiety AND Panic Attack present.")
    if R.get('treat_flag'):    st.info("ℹ️ **Treatment Noted:** Student has already sought specialist help.")
    if R.get('academic_risk'): st.warning("⚠️ **Academic Risk:** Senior year with CGPA 0–1.99.")

    st.write(""); r1,r2,r3=st.columns([1.2,1.2,1])
    with r1:
        st.markdown("**Prediction Confidence**")
        fig,ax2=plt.subplots(figsize=(4,0.8))
        ax2.barh([""], [prob[0]*100],color="#10B981",height=0.5)
        ax2.barh([""], [prob[1]*100],left=[prob[0]*100],color="#EF4444",height=0.5)
        ax2.set_xlim(0,100); ax2.axis('off')
        for xp,val,lbl in [(prob[0]*50,prob[0],"No Risk"),(prob[0]*100+prob[1]*50,prob[1],"At Risk")]:
            if val>0.12: ax2.text(xp,0,f"{lbl}\n{val*100:.0f}%",ha='center',va='center',fontsize=8,color='white',fontweight='bold')
        plt.tight_layout(pad=0); st.pyplot(fig,use_container_width=True); plt.close()
        st.write(""); pa_c,pb_c=st.columns(2)
        pa_c.metric("No Depression",f"{prob[0]*100:.1f}%")
        pb_c.metric("Depression",   f"{prob[1]*100:.1f}%")
    with r2:
        st.markdown("**Input Summary**")
        st.table(pd.DataFrame({"Field":["Name","Gender","Age","Course","Year","CGPA","Marital","Anxiety","Panic","Treatment"],
            "Value":[R['name'],R['gender'],str(R['age']),R['course'],R['year'],R['cgpa'],R['marital'],R['anxiety'],R['panic'],R['treat']]
        }).set_index("Field"))
    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write("**Algorithm:** SVM"); st.write("**Kernel:** RBF")
            st.write("**Class Weight:** Balanced"); st.write("**Scaling:** Standard")
            st.write("**Split:** 75 / 25"); st.write(f"**Accuracy:** {M['svm_m']['acc']:.2f}%")

    # PCA boundary
    st.write(""); st.markdown("**SVM Decision Boundary (PCA Projection)**")
    try:
        prep=M['svm'].named_steps['enc']
        X_bg_proc=prep.transform(M['svm_X_all'])
        X_user_proc=prep.transform(R['inp'])
        sc2d=StandardScaler()
        X_bg_sc=sc2d.fit_transform(X_bg_proc); X_us_sc=sc2d.transform(X_user_proc)
        pca=PCA(n_components=2,random_state=42)
        X_bg_2d=pca.fit_transform(X_bg_sc); X_us_2d=pca.transform(X_us_sc)
        svm2d=LinearSVC(C=1.0,class_weight='balanced',random_state=42,max_iter=5000)
        svm2d.fit(X_bg_2d,M['svm_y_all'])
        fig3,ax3=plt.subplots(figsize=(8,4))
        y_all=M['svm_y_all'].values
        ax3.scatter(X_bg_2d[y_all==1,0],X_bg_2d[y_all==1,1],color='#EF4444',s=30,alpha=0.5,label='Depression',zorder=3)
        ax3.scatter(X_bg_2d[y_all==0,0],X_bg_2d[y_all==0,1],color='#3B82F6',s=30,alpha=0.5,label='No Depression',zorder=3)
        ax3.scatter(X_us_2d[0,0],X_us_2d[0,1],color='black',s=300,marker='*',label=f'Input: {name_lbl}',zorder=6)
        xmin=X_bg_2d[:,0].min()-1; xmax=X_bg_2d[:,0].max()+1
        xmin=min(xmin,X_us_2d[0,0]-1); xmax=max(xmax,X_us_2d[0,0]+1)
        w=svm2d.coef_[0]; b=svm2d.intercept_[0]; xpts=np.linspace(xmin,xmax,200)
        if w[1]!=0:
            ypts=-(w[0]*xpts+b)/w[1]
            ax3.plot(xpts,ypts,'--',color='#1E3A5F',lw=2,label='Decision Boundary',zorder=5)
        ax3.set_xlim(xmin,xmax); ax3.set_ylim(X_bg_2d[:,1].min()-1,X_bg_2d[:,1].max()+1)
        ax3.set_xlabel('PC1 — Mental Health Risk Factors',fontsize=9)
        ax3.set_ylabel('PC2 — Academic & Demographic',fontsize=9)
        ax3.set_title('Live SVM Decision Boundary (PCA 2D)',fontsize=11,fontweight='bold')
        ax3.legend(fontsize=9); ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig3,use_container_width=True); plt.close()
    except Exception as _e: st.warning(f"Visualization unavailable: {_e}")

    # Explanation
    st.write(""); st.markdown("---"); st.markdown("### 🧠 Why did SVM predict this?")
    ex1,ex2=st.columns(2)
    with ex1:
        st.markdown("**Permutation Feature Importance**")
        _fi_svm=M['svm_fi_df'].sort_values('Importance',ascending=True)
        _mean_sv=_fi_svm['Importance'].mean()
        fig_sv,ax_sv=plt.subplots(figsize=(5,3.5))
        _sv_colors=['#EF4444' if v>=_mean_sv else '#9CA3AF' for v in _fi_svm['Importance']]
        ax_sv.barh(_fi_svm['Feature'],_fi_svm['Importance'],xerr=_fi_svm['Std'],
                   color=_sv_colors,edgecolor='none',height=0.6,error_kw={'elinewidth':1.5,'ecolor':'#374151'})
        ax_sv.axvline(_mean_sv,color='red',ls='--',lw=1.2,alpha=0.7,label=f'Mean={_mean_sv:.4f}')
        ax_sv.set_xlabel('Mean Accuracy Decrease')
        ax_sv.set_title('SVM Feature Importance — Permutation (Live)',fontweight='bold')
        ax_sv.legend(fontsize=8); ax_sv.spines['top'].set_visible(False); ax_sv.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_sv,use_container_width=True); plt.close()
    with ex2:
        st.markdown("**Risk Factor Analysis**")
        _risk=[]; _prot=[]
        if R['anxiety']=='Yes': _risk.append(("Anxiety","Strong predictor of depression"))
        if R['panic']=='Yes':   _risk.append(("Panic Attack","Strongest predictor (r=0.341)"))
        if R['marital']=='Yes': _risk.append(("Marital Status","Decision tree root feature"))
        if R['treat']=='Yes':   _risk.append(("Sought Treatment","Indicates awareness of condition"))
        if CGPA_MAP.get(R['cgpa'],3.25)<2.5: _risk.append(("Low CGPA","Academic difficulty increases risk"))
        if CGPA_MAP.get(R['cgpa'],3.25)>=3.0: _prot.append(("High CGPA","Academic success is protective"))
        if R['anxiety']=='No' and R['panic']=='No': _prot.append(("No Anxiety/Panic","Absence of co-morbid conditions"))
        if _risk:
            st.markdown("**⚠️ Risk Factors:**")
            for f,r in _risk: st.write(f"• **{f}** — {r}")
        if _prot:
            st.markdown("**✅ Protective Factors:**")
            for f,r in _prot: st.write(f"• **{f}** — {r}")
        with st.container(border=True):
            st.write(f"Risk Factors: **{len(_risk)}**")
            st.write(f"Depression Probability: **{prob[1]*100:.1f}%**")
            if prob[1]>0.7: st.error("High confidence — multiple risk factors")
            elif prob[1]>0.4: st.warning("Moderate risk — some factors present")
            else: st.success("Low risk — few risk factors")

    # PDF
    st.write("")
    _alerts_sv=[]; _notes_sv=[]
    if R.get('high_concern'): _alerts_sv.append("High Concern: Both Anxiety and Panic Attack present.")
    if R.get('treat_flag'):   _alerts_sv.append("Treatment Noted: Student has already sought specialist help.")
    if R.get('academic_risk'): _alerts_sv.append("Academic Risk: Senior year with CGPA 0-1.99.")
    if R['anxiety']=='Yes': _notes_sv.append("Anxiety = Yes — strong SVM predictor")
    if R['panic']=='Yes':   _notes_sv.append("Panic Attack = Yes — highest correlation (r=0.341)")
    try:
        _pdf=generate_pdf(model_name="SVM (RBF Kernel, Balanced)",student_name=R['name'],
            result=R['pred'],prob=R['prob'],
            input_data={"Gender":R['gender'],"Age":str(R['age']),"Course":R['course'],
                        "Year":R['year'],"CGPA":R['cgpa'],"Marital":R['marital'],
                        "Anxiety":R['anxiety'],"Panic":R['panic'],"Treatment":R['treat']},
            metrics=M['svm_m'],business_alerts=_alerts_sv,explanation_notes=_notes_sv)
        st.download_button("📥  Download PDF Report",data=_pdf,
            file_name=f"mindcheck_svm_{R['name'].replace(' ','_')}.pdf",
            mime="application/pdf",use_container_width=True,key="svm_pdf")
    except Exception as _pe: st.caption(f"PDF unavailable: {_pe}")

    st.write("")
    if st.button("Clear Result",key="svm_clear"):
        st.session_state.svm_result=None; st.rerun()

st.divider()

# Feature Importance
st.subheader("Live Feature Importance")
st.caption("Permutation importance — computed live from trained SVM model")
_fi_sv=M['svm_fi_df'].sort_values('Importance',ascending=True)
_mean_sv2=_fi_sv['Importance'].mean()
fi_c1,fi_c2=st.columns([2,1])
with fi_c1:
    fig_fi,ax_fi=plt.subplots(figsize=(7,3.5))
    colors_fi=['#EF4444' if v>=_mean_sv2 else '#9CA3AF' for v in _fi_sv['Importance']]
    ax_fi.barh(_fi_sv['Feature'],_fi_sv['Importance'],xerr=_fi_sv['Std'],
               color=colors_fi,edgecolor='none',height=0.6,error_kw={'elinewidth':1.5,'ecolor':'#374151'})
    ax_fi.axvline(_mean_sv2,color='red',ls='--',lw=1.2,alpha=0.7,label=f'Mean={_mean_sv2:.4f}')
    ax_fi.set_xlabel('Mean Accuracy Decrease (Permutation)')
    ax_fi.set_title('SVM Feature Importance — Permutation Method (Live)',fontweight='bold')
    ax_fi.legend(fontsize=9); ax_fi.spines['top'].set_visible(False); ax_fi.spines['right'].set_visible(False)
    ax_fi.grid(axis='x',alpha=0.3); plt.tight_layout(); st.pyplot(fig_fi,use_container_width=True); plt.close()
with fi_c2:
    with st.container(border=True):
        st.markdown("**Feature Ranking**")
        for _,row in _fi_sv.sort_values('Importance',ascending=False).iterrows():
            icon="🔴" if row['Importance']>=_mean_sv2 else "⚪"
            st.write(f"{icon} **{row['Feature']}** — {row['Importance']:.4f}")
        st.caption("🔴 Above average importance")

st.divider()

with st.expander("📚  Learn More — Confusion Matrix & How SVM Works"):
    fig_cm,ax_cm=plt.subplots(figsize=(5,4))
    sns.heatmap(M['svm_m']['cm'],annot=True,fmt='d',cmap='Reds',ax=ax_cm,
                xticklabels=['No Depression','Depression'],yticklabels=['No Depression','Depression'],
                linewidths=0.5,annot_kws={'size':12,'weight':'bold'})
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    ax_cm.set_title('SVM Confusion Matrix',fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_cm,use_container_width=True); plt.close()
    tn,fp,fn,tp=M['svm_m']['cm'].ravel()
    a,b,c,d=st.columns(4)
    a.metric("TN",str(tn)); b.metric("FP",str(fp)); c.metric("FN",str(fn)); d.metric("TP",str(tp))
    st.write("")
    st.write("**How SVM Works:** SVM finds the optimal hyperplane that maximally separates two classes. "
             "The **RBF kernel** handles non-linearly separable data. **Balanced class weights** "
             "compensate for class imbalance. PCA above projects the boundary into 2D for visualization.")

st.divider()

# Batch
st.subheader("Batch Prediction — 🔴 SVM")
_svm_sample=pd.DataFrame({'Name':['Ahmad','Siti','Wei Ming'],'Gender':['Male','Female','Male'],
    'Age':[20,21,19],'Course':['Computer Science','Engineering','Information Technology'],
    'Year_of_Study':['Year 2','Year 3','Year 1'],'CGPA':['3.00 - 3.49','2.50 - 2.99','3.50 - 4.00'],
    'Marital_Status':['No','No','No'],'Anxiety':['Yes','No','No'],
    'Panic_Attack':['No','Yes','No'],'Seek_Treatment':['No','No','No']})
with st.expander("📋  View / Download CSV Template"):
    st.dataframe(_svm_sample,use_container_width=True,hide_index=True)
    st.download_button("⬇️  Download SVM Template",data=_svm_sample.to_csv(index=False).encode(),
        file_name="svm_batch_template.csv",mime="text/csv",use_container_width=True,key="svm_tmpl_dl")

_svm_file=st.file_uploader("Upload CSV — drag & drop or click to browse",type=["csv"],key="svm_batch_file")
if _svm_file:
    try:
        _sdf=pd.read_csv(_svm_file); _sdf.columns=_sdf.columns.str.strip()
        st.success(f"✅ {len(_sdf)} students loaded")
        _sresults=[]
        for _si,_srow in _sdf.iterrows():
            try:
                _sname=str(_srow.get('Name',f'Student {_si+1}')).strip()
                _sgender=str(_srow.get('Gender','Male'))
                _smarital=str(_srow.get('Marital_Status','No'))
                _sanxiety=str(_srow.get('Anxiety','No'))
                _spanic=str(_srow.get('Panic_Attack','No'))
                _streat=str(_srow.get('Seek_Treatment','No'))
                try: _sage=int(float(_srow.get('Age',20)))
                except: _sage=20
                _scourse=str(_srow.get('Course','Others'))
                _syear=str(_srow.get('Year_of_Study','Year 1'))
                _scgpa=str(_srow.get('CGPA','3.00 - 3.49')).strip()
                _syear_n=''.join(filter(str.isdigit,_syear)) or '1'
                _sinp=pd.DataFrame([{
                    'Choose your gender':_sgender,'Age':_sage,
                    'Your current year of Study':f'year {_syear_n}',
                    'What is your CGPA?':_scgpa,'Marital status':_smarital,
                    'Do you have Anxiety?':_sanxiety,'Do you have Panic attack?':_spanic,
                    'Did you seek any specialist for a treatment?':_streat,
                    'Course_Category':M['cat_course'](_scourse)
                }])[M['svm_col_order']]
                _spred=int(M['svm'].predict(_sinp)[0])
                _sprob=M['svm'].predict_proba(_sinp)[0][1]
                _sresults.append({'Name':_sname,'Gender':_sgender,'Age':_sage,'Course':_scourse,
                    'Year':_syear,'CGPA':_scgpa,'Anxiety':_sanxiety,'Panic':_spanic,
                    'Result':'⚠️ Depression' if _spred==1 else '✅ No Depression',
                    'Confidence':f"{_sprob*100:.1f}%",'Risk':'HIGH' if _spred==1 else 'LOW',
                    '_pred':_spred,'_prob':_sprob})
            except Exception as _se: _sresults.append({'Name':str(_srow.get('Name','')),'Error':str(_se)})
        _sres=pd.DataFrame(_sresults); _stotal=len(_sres)
        _sdep=int(_sres['_pred'].sum()) if '_pred' in _sres else 0
        _sm1,_sm2,_sm3=st.columns(3)
        _sm1.metric("Total Students",str(_stotal))
        _sm2.metric("⚠️ At Risk",str(_sdep),delta=f"{_sdep/_stotal*100:.0f}%",delta_color="inverse")
        _sm3.metric("✅ No Risk",str(_stotal-_sdep),delta=f"{(_stotal-_sdep)/_stotal*100:.0f}%")
        _sdisp=_sres[['Name','Gender','Age','Course','Year','CGPA','Anxiety','Panic','Result','Confidence','Risk']]
        def _sst(val):
            if '⚠️' in str(val) or val=='HIGH': return 'background-color:#FEE2E2;color:#991B1B;font-weight:bold'
            if '✅' in str(val) or val=='LOW':  return 'background-color:#DCFCE7;color:#166534;font-weight:bold'
            return ''
        try:    _sstyled=_sdisp.style.map(_sst,subset=['Result','Risk'])
        except: _sstyled=_sdisp.style.applymap(_sst,subset=['Result','Risk'])
        st.dataframe(_sstyled,use_container_width=True,hide_index=True)
        _sdl1,_sdl2=st.columns(2)
        with _sdl1:
            st.download_button("⬇️  Download All Results",data=_sdisp.to_csv(index=False).encode(),
                file_name="svm_batch_results.csv",mime="text/csv",use_container_width=True,key="svm_dl_all")
        with _sdl2:
            _shigh=_sres[_sres['_pred']==1][list(_sdisp.columns)] if '_pred' in _sres else pd.DataFrame()
            if len(_shigh):
                st.download_button(f"⬇️  At-Risk Only ({len(_shigh)})",
                    data=_shigh.to_csv(index=False).encode(),
                    file_name="svm_at_risk.csv",mime="text/csv",use_container_width=True,key="svm_dl_risk")
    except Exception as _serr: st.error(f"Error: {str(_serr)}")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")