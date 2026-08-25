import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from sklearn.tree import _tree as _sk_tree
import sys, os, warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sidebar import sidebar
from utils.models import load_all_models, MAX_BATCH_ROWS, MAX_BATCH_MB
from utils.pdf_report import generate_pdf


def _draw_path_tree(model, feature_names, path_node_ids):
    """Draw the full decision tree with the route a specific prediction
    took highlighted in red, so the exact path is visible at a glance —
    not just a text list of the steps."""
    tree_ = model.tree_
    path_set = set(path_node_ids)

    # In-order layout (left subtree, self, right subtree) gives a clean,
    # non-overlapping x-position for every node in a binary tree.
    positions = {}
    _x = [0]
    def _assign(node_id, depth):
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        is_leaf = left == _sk_tree.TREE_LEAF
        if not is_leaf:
            _assign(left, depth + 1)
        positions[node_id] = (_x[0], -depth)
        _x[0] += 1
        if not is_leaf:
            _assign(right, depth + 1)
    _assign(0, 0)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    fig, ax = plt.subplots(figsize=(max(14, len(xs) * 0.55), 8))

    # edges
    for node_id in range(tree_.node_count):
        left = tree_.children_left[node_id]
        right = tree_.children_right[node_id]
        if left == _sk_tree.TREE_LEAF:
            continue
        for child in (left, right):
            x1, y1 = positions[node_id]
            x2, y2 = positions[child]
            on_path = node_id in path_set and child in path_set
            ax.plot([x1, x2], [y1, y2],
                    color='#DC2626' if on_path else '#CBD5E1',
                    linewidth=3 if on_path else 1, zorder=2)

    # nodes
    for node_id in range(tree_.node_count):
        x, y = positions[node_id]
        left = tree_.children_left[node_id]
        is_leaf = left == _sk_tree.TREE_LEAF
        val = tree_.value[node_id][0]
        majority_dep = val[1] > val[0]
        base_color = '#FCA5A5' if majority_dep else '#93C5FD'
        on_path = node_id in path_set
        if is_leaf:
            label = f"No Dep: {int(val[0])}\nDep: {int(val[1])}"
        else:
            fname = feature_names[tree_.feature[node_id]]
            label = f"{fname}\n≤ {tree_.threshold[node_id]:.2f}"
        ax.text(x, y, label, ha='center', va='center', fontsize=7,
                fontweight='bold' if on_path else 'normal',
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=base_color,
                          edgecolor='#DC2626' if on_path else 'white',
                          linewidth=2.5 if on_path else 1),
                zorder=3)

    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.axis('off')
    ax.set_title("Decision Path Highlighted — Red = This Student's Route",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig

st.set_page_config(page_title="Decision Tree Predictor — MindCheck",
                   page_icon="🌳", layout="wide",
                   initial_sidebar_state="expanded")
sidebar("dt")

M = load_all_models()

if 'dt_carry' in st.session_state and st.session_state['dt_carry'] is not None:
    st.session_state['dt_result'] = st.session_state.pop('dt_carry')

if 'dt_result' not in st.session_state:
    st.session_state.dt_result = None

_pre = st.session_state.pop('dt_prefill', None)
if _pre:
    st.info(f"📋 Data from Comparison page: **{_pre.get('name','Student')}** — form pre-filled below.")

st.markdown("##### 🌳 DECISION TREE PREDICTOR")
st.title("Depression Risk Predictor")
st.caption(f"Decision Tree (CART, Depth 5) · Member: Irvin Tan Wei Shen · Live trained on {M['n_records']} records")
st.divider()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Accuracy",  f"{M['dt_m']['acc']:.2f}%")
c2.metric("Precision", f"{M['dt_m']['prec']:.2f}%")
c3.metric("Recall",    f"{M['dt_m']['rec']:.2f}%")
c4.metric("F1 Score",  f"{M['dt_m']['f1']:.2f}%")
st.divider()

st.subheader("Student Information")
col1,col2,col3 = st.columns(3)
with col1:
    _pn=_pre.get('name','') if _pre else ''
    name   = st.text_input("Name",value=_pn,placeholder="e.g. Siti",key="dt_name")
    _gi=["Female","Male"].index(_pre['gender']) if _pre and _pre.get('gender') in ["Female","Male"] else 0
    gender = st.selectbox("Gender",["Female","Male"],index=_gi,key="dt_gender")
    _ai=int(_pre.get('age',20)) if _pre else 20
    age    = st.slider("Age",17,30,_ai,key="dt_age")
with col2:
    course = st.selectbox("Course",M['courses'],key="dt_course")
    _yi=["Year 1","Year 2","Year 3","Year 4"].index(_pre['year']) if _pre and _pre.get('year') in ["Year 1","Year 2","Year 3","Year 4"] else 0
    year   = st.selectbox("Year of Study",["Year 1","Year 2","Year 3","Year 4"],index=_yi,key="dt_year")
    cgpa   = st.selectbox("CGPA Range",list(M['cgpa_map'].keys()),key="dt_cgpa")
with col3:
    _mai=["No","Yes"].index(_pre['marital']) if _pre and _pre.get('marital') in ["No","Yes"] else 0
    marital = st.selectbox("Marital Status",["No","Yes"],index=_mai,key="dt_marital")
    _axi=["No","Yes"].index(_pre['anxiety']) if _pre and _pre.get('anxiety') in ["No","Yes"] else 0
    anxiety = st.selectbox("Do you have Anxiety?",["No","Yes"],index=_axi,key="dt_anxiety")
    _pai=["No","Yes"].index(_pre['panic']) if _pre and _pre.get('panic') in ["No","Yes"] else 0
    panic   = st.selectbox("Do you have Panic Attack?",["No","Yes"],index=_pai,key="dt_panic")
    predict_btn = st.button("🔍  Predict Depression Risk",use_container_width=True,type="primary",key="dt_predict")

if predict_btn:
    _errors=[]
    if name.strip() and name.strip().isdigit(): _errors.append("❌ Name cannot be numbers only.")
    if age<17 or age>35: _errors.append(f"❌ Age {age} outside valid range (17–35).")
    _high_concern=(anxiety=="Yes" and panic=="Yes")
    _married_risk=(marital=="Yes" and age<21)
    _academic_risk=(year in ["Year 3","Year 4"] and cgpa=="0 - 1.99")
    if _errors:
        for e in _errors: st.error(e)
        st.warning("⚠️ Please fix errors above.")
    else:
        try:
            g=1 if gender=="Male" else 0; ax=1 if anxiety=="Yes" else 0
            pa=1 if panic=="Yes" else 0;  ma=1 if marital=="Yes" else 0
            ce=M['le_c'].transform([course])[0] if course in M['le_c'].classes_ else 0
            ye=M['le_y'].transform([year])[0]   if year   in M['le_y'].classes_ else 0
            cn=M['cgpa_map'].get(cgpa,3.25)
            inp=pd.DataFrame([[g,age,ce,ye,cn,ax,pa,ma]],columns=M['dt_feat'])
            pred=int(M['dt'].predict(inp)[0])
            prob=M['dt'].predict_proba(inp)[0].tolist()
            st.session_state.dt_result={
                'pred':pred,'prob':prob,'name':name.strip() or "Student",
                'gender':gender,'age':age,'course':course,'year':year,'cgpa':cgpa,
                'marital':marital,'anxiety':anxiety,'panic':panic,
                'high_concern':_high_concern,'married_risk':_married_risk,'academic_risk':_academic_risk}
        except Exception as ex: st.error(f"❌ Prediction failed: {ex}")

if st.session_state.dt_result:
    R=st.session_state.dt_result; pred=R['pred']; prob=R['prob']; name_lbl=R['name']
    st.divider(); st.subheader("Prediction Result")
    if pred==1:
        st.error(f"### ⚠️  {name_lbl} — Depression Risk Detected\n\nPlease consider seeking professional support.")
    else:
        st.success(f"### ✅  {name_lbl} — No Depression Detected\n\nKeep maintaining a healthy lifestyle!")
    if R.get('high_concern'):  st.warning("⚠️ **High Concern:** Both Anxiety AND Panic Attack present.")
    if R.get('married_risk'):  st.warning("⚠️ **Married Student:** Under 21 and married — additional support advised.")
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
        st.table(pd.DataFrame({"Field":["Name","Gender","Age","Course","Year","CGPA","Marital","Anxiety","Panic Attack"],
            "Value":[R['name'],R['gender'],str(R['age']),R['course'],R['year'],R['cgpa'],R['marital'],R['anxiety'],R['panic']]
        }).set_index("Field"))
    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write("**Algorithm:** Decision Tree"); st.write("**Max Depth:** 5")
            st.write("**Criterion:** Gini"); st.write("**Split:** 70 / 30")
            st.write(f"**Accuracy:** {M['dt_m']['acc']:.2f}%")

    # DT Explanation — decision path
    st.write(""); st.markdown("---"); st.markdown("### 🧠 Why did the Decision Tree predict this?")
    _path_node_ids = None
    ex1,ex2=st.columns(2)
    with ex1:
        st.markdown("**Decision Path Traced**")
        try:
            _tree=M['dt'].tree_
            _g=1 if R['gender']=='Male' else 0; _ax=1 if R['anxiety']=='Yes' else 0
            _pa=1 if R['panic']=='Yes' else 0;  _ma=1 if R['marital']=='Yes' else 0
            _ce=M['le_c'].transform([R['course']])[0] if R['course'] in M['le_c'].classes_ else 0
            _ye=M['le_y'].transform([R['year']])[0]   if R['year']   in M['le_y'].classes_ else 0
            _cn=M['cgpa_map'].get(R['cgpa'],3.25)
            _inp=[[_g,R['age'],_ce,_ye,_cn,_ax,_pa,_ma]]
            _node=0; _path=[]; _path_ids=[0]
            while _tree.children_left[_node]!=_sk_tree.TREE_LEAF:
                _fi=_tree.feature[_node]; _th=_tree.threshold[_node]
                _val=_inp[0][_fi]; _go_left=_val<=_th
                _path.append({'Feature':M['dt_fi_labels'][_fi],
                              'Condition':f"{'≤' if _go_left else '>'} {_th:.2f}",
                              'Value':f"{_val:.2f}",'Decision':"→ Left" if _go_left else "→ Right"})
                _node=_tree.children_left[_node] if _go_left else _tree.children_right[_node]
                _path_ids.append(int(_node))
            _leaf=_tree.value[_node][0]
            for i,step in enumerate(_path,1):
                with st.container(border=True):
                    st.write(f"**Node {i}:** {step['Feature']} {step['Condition']}")
                    st.caption(f"Value: {step['Value']} {step['Decision']}")
            st.success(f"**Leaf:** {int(_leaf[0])} No Dep / {int(_leaf[1])} Dep → **{'Depression' if _leaf[1]>_leaf[0] else 'No Depression'}**")
            _path_node_ids = _path_ids
        except Exception as _dte: st.info(f"Path unavailable: {_dte}")
    with ex2:
        st.markdown("**Feature Importance**")
        _fi_df=pd.DataFrame({'Feature':M['dt_fi_labels'],'Importance':M['dt_fi_vals']}).sort_values('Importance',ascending=True)
        fig_ex2,ax_ex2=plt.subplots(figsize=(5,3.5))
        _mean_fi=_fi_df['Importance'].mean()
        _ex_colors=['#EF4444' if v>=_mean_fi else '#9CA3AF' for v in _fi_df['Importance']]
        ax_ex2.barh(_fi_df['Feature'],_fi_df['Importance'],color=_ex_colors,edgecolor='none',height=0.6)
        ax_ex2.axvline(_mean_fi,color='red',ls='--',lw=1.2,alpha=0.7,label=f'Mean={_mean_fi:.3f}')
        ax_ex2.set_xlabel('Gini Importance'); ax_ex2.set_title('Feature Importance (Live)',fontweight='bold')
        ax_ex2.legend(fontsize=8); ax_ex2.spines['top'].set_visible(False); ax_ex2.spines['right'].set_visible(False)
        plt.tight_layout(); st.pyplot(fig_ex2,use_container_width=True); plt.close()

    if _path_node_ids:
        st.write("")
        st.markdown("**🌳 Full Decision Path Visualized**")
        st.caption("Red route = exactly how this student's data traveled through the tree")
        fig_path = _draw_path_tree(M['dt'], M['dt_fi_labels'], _path_node_ids)
        st.pyplot(fig_path, use_container_width=True)
        plt.close(fig_path)

    # PDF
    st.write("")
    _alerts_dt=[]; _notes_dt=[]
    if R.get('high_concern'): _alerts_dt.append("High Concern: Both Anxiety and Panic Attack present.")
    if R.get('married_risk'): _alerts_dt.append("Married Student Alert: Under 21 and married.")
    if R.get('academic_risk'): _alerts_dt.append("Academic Risk: Senior year with CGPA 0-1.99.")
    if R['marital']=='Yes': _notes_dt.append("Marital Status = Yes — DT root feature, highest predictor")
    if R['anxiety']=='Yes': _notes_dt.append("Anxiety = Yes — early DT split feature")
    try:
        _pdf=generate_pdf(model_name="Decision Tree (CART, Depth 5)",student_name=R['name'],
            result=R['pred'],prob=R['prob'],
            input_data={"Gender":R['gender'],"Age":str(R['age']),"Course":R['course'],
                        "Year":R['year'],"CGPA":R['cgpa'],"Marital":R['marital'],
                        "Anxiety":R['anxiety'],"Panic":R['panic']},
            metrics=M['dt_m'],business_alerts=_alerts_dt,explanation_notes=_notes_dt)
        st.download_button("📥  Download PDF Report",data=_pdf,
            file_name=f"mindcheck_dt_{R['name'].replace(' ','_')}.pdf",
            mime="application/pdf",use_container_width=True,key="dt_pdf")
    except Exception as _pe: st.caption(f"PDF unavailable: {_pe}")

    st.write("")
    if st.button("Clear Result",key="dt_clear"):
        st.session_state.dt_result=None; st.rerun()

st.divider()

# Feature Importance
st.subheader("Live Feature Importance")
st.caption("Gini impurity reduction — computed live from trained model")
_fi_df2=pd.DataFrame({'Feature':M['dt_fi_labels'],'Importance':M['dt_fi_vals']}).sort_values('Importance',ascending=True)
fi_c1,fi_c2=st.columns([2,1])
with fi_c1:
    fig_fi,ax_fi=plt.subplots(figsize=(7,3.5))
    _mean_v=_fi_df2['Importance'].mean()
    colors_fi=['#10B981' if v>=_mean_v else '#9CA3AF' for v in _fi_df2['Importance']]
    bars_fi=ax_fi.barh(_fi_df2['Feature'],_fi_df2['Importance'],color=colors_fi,edgecolor='none',height=0.6)
    ax_fi.axvline(_mean_v,color='red',ls='--',lw=1.2,alpha=0.7,label=f'Mean={_mean_v:.3f}')
    for bar,val in zip(bars_fi,_fi_df2['Importance']):
        ax_fi.text(val+0.005,bar.get_y()+bar.get_height()/2,f'{val:.3f}',va='center',fontsize=9,fontweight='bold')
    ax_fi.set_xlabel('Feature Importance (Gini Reduction)')
    ax_fi.set_title('Decision Tree Feature Importance (Live)',fontweight='bold')
    ax_fi.legend(fontsize=9); ax_fi.spines['top'].set_visible(False); ax_fi.spines['right'].set_visible(False)
    ax_fi.grid(axis='x',alpha=0.3); plt.tight_layout(); st.pyplot(fig_fi,use_container_width=True); plt.close()
with fi_c2:
    with st.container(border=True):
        st.markdown("**Feature Ranking**")
        for _,row in _fi_df2.sort_values('Importance',ascending=False).iterrows():
            icon="🟢" if row['Importance']>=_mean_v else "⚪"
            st.write(f"{icon} **{row['Feature']}** — {row['Importance']:.3f}")
        st.caption("🟢 Above average importance")

st.divider()

# Decision Tree Diagram
st.subheader("Decision Tree Structure")
st.caption("Visual tree trained live — Orange=Depression, Blue=No Depression")
@st.cache_resource
def get_tree_fig(_model):
    fig,ax=plt.subplots(figsize=(22,9))
    plot_tree(_model,feature_names=['Gender','Age','Course','Year','CGPA','Anxiety','Panic Attack','Marital Status'],
              class_names=['No Depression','Depression'],filled=True,rounded=True,fontsize=7,ax=ax)
    ax.set_title("Decision Tree — Live Trained (Depth 5)",fontsize=13,fontweight='bold',pad=16)
    plt.tight_layout(); return fig
st.pyplot(get_tree_fig(M['dt']),use_container_width=True); plt.close('all')

st.divider()

# Learn More
with st.expander("📚  Learn More — Confusion Matrix & How Decision Tree Works"):
    fig_cm,ax_cm=plt.subplots(figsize=(5,4))
    sns.heatmap(M['dt_m']['cm'],annot=True,fmt='d',cmap='Greens',ax=ax_cm,
                xticklabels=['No Depression','Depression'],yticklabels=['No Depression','Depression'],
                linewidths=0.5,annot_kws={'size':12,'weight':'bold'})
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Decision Tree Confusion Matrix',fontweight='bold')
    plt.tight_layout(); st.pyplot(fig_cm,use_container_width=True); plt.close()
    tn,fp,fn,tp=M['dt_m']['cm'].ravel()
    a,b,c,d=st.columns(4)
    a.metric("TN",str(tn)); b.metric("FP",str(fp)); c.metric("FN",str(fn)); d.metric("TP",str(tp))

st.divider()

# Batch
st.subheader("Batch Prediction — 🌳 Decision Tree")
st.caption(f"⚠️ Limits: max **{MAX_BATCH_ROWS} records** and **{MAX_BATCH_MB} MB** per upload.")
_dt_sample=pd.DataFrame({'Name':['Ahmad','Siti','Wei Ming'],'Gender':['Male','Female','Male'],
    'Age':[20,21,19],'Course':['Computer Science','Engineering','Information Technology'],
    'Year_of_Study':['Year 2','Year 3','Year 1'],'CGPA':['3.00 - 3.49','2.50 - 2.99','3.50 - 4.00'],
    'Marital_Status':['No','No','No'],'Anxiety':['Yes','No','No'],'Panic_Attack':['No','Yes','No']})
with st.expander("📋  View / Download CSV Template"):
    st.dataframe(_dt_sample,use_container_width=True,hide_index=True)
    st.download_button("⬇️  Download DT Template",data=_dt_sample.to_csv(index=False).encode(),
        file_name="dt_batch_template.csv",mime="text/csv",use_container_width=True,key="dt_tmpl_dl")

_dt_file=st.file_uploader(f"Upload CSV — drag & drop or click to browse (max {MAX_BATCH_ROWS} records, {MAX_BATCH_MB} MB)",type=["csv"],key="dt_batch_file")
if _dt_file:
    _dt_size_mb = _dt_file.size / (1024*1024)
    if _dt_size_mb > MAX_BATCH_MB:
        st.error(f"❌ File is {_dt_size_mb:.2f} MB — exceeds the {MAX_BATCH_MB} MB limit. Please upload a smaller file.")
    else:
        try:
            _ddf=pd.read_csv(_dt_file); _ddf.columns=_ddf.columns.str.strip()
            if len(_ddf) > MAX_BATCH_ROWS:
                st.warning(f"⚠️ File has {len(_ddf)} records — only the first {MAX_BATCH_ROWS} will be processed.")
                _ddf = _ddf.head(MAX_BATCH_ROWS)
            st.success(f"✅ {len(_ddf)} students loaded")
            _dresults=[]
            for _di,_drow in _ddf.iterrows():
                try:
                    _dname=str(_drow.get('Name',f'Student {_di+1}')).strip()
                    _dg=1 if str(_drow.get('Gender','')).lower()=='male' else 0
                    _dax=1 if str(_drow.get('Anxiety','')).lower()=='yes' else 0
                    _dpa=1 if str(_drow.get('Panic_Attack','')).lower()=='yes' else 0
                    _dma=1 if str(_drow.get('Marital_Status','')).lower()=='yes' else 0
                    try: _dage=int(float(_drow.get('Age',20)))
                    except: _dage=20
                    _dcourse=str(_drow.get('Course','Others')); _dyear=str(_drow.get('Year_of_Study','Year 1'))
                    _dcgpa=str(_drow.get('CGPA','3.00 - 3.49')).strip()
                    _dce=M['le_c'].transform([_dcourse])[0] if _dcourse in M['le_c'].classes_ else 0
                    _dye=M['le_y'].transform([_dyear])[0]   if _dyear   in M['le_y'].classes_ else 0
                    _dcn=M['cgpa_map'].get(_dcgpa,3.25)
                    _dinp=pd.DataFrame([[_dg,_dage,_dce,_dye,_dcn,_dax,_dpa,_dma]],columns=M['dt_feat'])
                    _dpred=int(M['dt'].predict(_dinp)[0])
                    _dprob=M['dt'].predict_proba(_dinp)[0][1]
                    _dresults.append({'Name':_dname,'Gender':_drow.get('Gender',''),'Age':_dage,
                        'Course':_dcourse,'Year':_dyear,'CGPA':_dcgpa,'Anxiety':_drow.get('Anxiety',''),
                        'Panic':_drow.get('Panic_Attack',''),'Result':'⚠️ Depression' if _dpred==1 else '✅ No Depression',
                        'Confidence':f"{_dprob*100:.1f}%",'Risk':'HIGH' if _dpred==1 else 'LOW',
                        '_pred':_dpred,'_prob':_dprob})
                except Exception as _de: _dresults.append({'Name':str(_drow.get('Name','')),'Error':str(_de)})
            _dres=pd.DataFrame(_dresults); _dtotal=len(_dres)
            _ddep=int(_dres['_pred'].sum()) if '_pred' in _dres else 0
            _dm1,_dm2,_dm3=st.columns(3)
            _dm1.metric("Total Students",str(_dtotal))
            _dm2.metric("⚠️ At Risk",str(_ddep),delta=f"{_ddep/_dtotal*100:.0f}%",delta_color="inverse")
            _dm3.metric("✅ No Risk",str(_dtotal-_ddep),delta=f"{(_dtotal-_ddep)/_dtotal*100:.0f}%")
            _ddisp=_dres[['Name','Gender','Age','Course','Year','CGPA','Anxiety','Panic','Result','Confidence','Risk']]
            def _dst(val):
                if '⚠️' in str(val) or val=='HIGH': return 'background-color:#FEE2E2;color:#991B1B;font-weight:bold'
                if '✅' in str(val) or val=='LOW':  return 'background-color:#DCFCE7;color:#166534;font-weight:bold'
                return ''
            try:    _dstyled=_ddisp.style.map(_dst,subset=['Result','Risk'])
            except: _dstyled=_ddisp.style.applymap(_dst,subset=['Result','Risk'])
            st.dataframe(_dstyled,use_container_width=True,hide_index=True)
            _ddl1,_ddl2=st.columns(2)
            with _ddl1:
                st.download_button("⬇️  Download All Results",data=_ddisp.to_csv(index=False).encode(),
                    file_name="dt_batch_results.csv",mime="text/csv",use_container_width=True,key="dt_dl_all")
            with _ddl2:
                _dhigh=_dres[_dres['_pred']==1][list(_ddisp.columns)] if '_pred' in _dres else pd.DataFrame()
                if len(_dhigh):
                    st.download_button(f"⬇️  At-Risk Only ({len(_dhigh)})",
                        data=_dhigh.to_csv(index=False).encode(),
                        file_name="dt_at_risk.csv",mime="text/csv",use_container_width=True,key="dt_dl_risk")
        except Exception as _derr: st.error(f"Error: {str(_derr)}")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")