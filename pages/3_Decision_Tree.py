import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

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
    fi_labels = ['Gender','Age','Course','Year','CGPA',
                 'Anxiety','Panic Attack','Marital Status']
    return {
        'model': model, 'feat': feat,
        'le_c': le_c, 'le_y': le_y,
        'fi_labels': fi_labels,
        'fi_vals': model.feature_importances_,
        'acc' : accuracy_score(yte,yp)*100,
        'prec': precision_score(yte,yp,zero_division=0)*100,
        'rec' : recall_score(yte,yp,zero_division=0)*100,
        'f1'  : f1_score(yte,yp,zero_division=0)*100,
        'cm'  : confusion_matrix(yte,yp),
    }

M = get_dt()
CGPA_MAP = {'0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
            '3.00 - 3.49':3.25,'3.50 - 4.00':3.75}
COURSES = ["Computer Science","Information Technology","Engineering",
           "Law","Psychology","Language","Islamic Studies",
           "Health Sciences","Business","Science & Math","Arts & Social","Others"]

if 'dt_result' not in st.session_state:
    st.session_state.dt_result = None

# ── Header ─────────────────────────────────────────────────────
st.markdown("##### 🌳 DECISION TREE PREDICTOR")
st.title("Depression Risk Predictor")
st.caption("Decision Tree (CART, Depth 5) · Member 2: Irvin Tan Wei Shen · Live trained on 600 records")
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
    name    = st.text_input("Name", placeholder="e.g. Siti", key="dt_name")
    gender  = st.selectbox("Gender", ["Female","Male"], key="dt_gender")
    age     = st.slider("Age", 17, 30, 20, key="dt_age")
with col2:
    course  = st.selectbox("Course", COURSES, key="dt_course")
    year    = st.selectbox("Year of Study",
                ["Year 1","Year 2","Year 3","Year 4"], key="dt_year")
    cgpa    = st.selectbox("CGPA Range", list(CGPA_MAP.keys()), key="dt_cgpa")
with col3:
    marital = st.selectbox("Marital Status",            ["No","Yes"], key="dt_marital")
    anxiety = st.selectbox("Do you have Anxiety?",      ["No","Yes"], key="dt_anxiety")
    panic   = st.selectbox("Do you have Panic Attack?", ["No","Yes"], key="dt_panic")
    st.write("")
    predict_btn = st.button("🔍  Predict Depression Risk",
                            use_container_width=True,
                            key="dt_predict", type="primary")

if predict_btn:
    # ── Input Validation ───────────────────────────────────────
    _errors = []
    if name.strip() and name.strip().isdigit():
        _errors.append("❌ Name cannot be numbers only.")
    if age < 17 or age > 35:
        _errors.append(f"❌ Age {age} is outside valid range (17–35).")
    if cgpa not in CGPA_MAP:
        _errors.append("❌ Invalid CGPA range selected.")

    # ── Business Rules ─────────────────────────────────────────
    _high_concern  = (anxiety == "Yes" and panic == "Yes")
    _married_risk  = (marital == "Yes" and age < 21)
    _academic_risk = (year in ["Year 3","Year 4"] and cgpa == "0 - 1.99")

    if _errors:
        for e in _errors:
            st.error(e)
        st.warning("⚠️ Please fix the errors above before predicting.")
    else:
        try:
            g  = 1 if gender  == "Male" else 0
            ax = 1 if anxiety == "Yes"  else 0
            pa = 1 if panic   == "Yes"  else 0
            ma = 1 if marital == "Yes"  else 0
            ce = M['le_c'].transform([course])[0] if course in M['le_c'].classes_ else 0
            ye = M['le_y'].transform([year])[0]   if year   in M['le_y'].classes_ else 0
            cn = CGPA_MAP.get(cgpa, 3.25)
            inp  = pd.DataFrame([[g,age,ce,ye,cn,ax,pa,ma]], columns=M['feat'])
            pred = int(M['model'].predict(inp)[0])
            prob = M['model'].predict_proba(inp)[0].tolist()
            st.session_state.dt_result = {
                'pred': pred, 'prob': prob,
                'name': name.strip() or "Student",
                'gender': gender, 'age': age, 'course': course,
                'year': year, 'cgpa': cgpa,
                'marital': marital, 'anxiety': anxiety, 'panic': panic,
                'high_concern': _high_concern,
                'married_risk': _married_risk,
                'academic_risk': _academic_risk,
            }
        except Exception as ex:
            st.error(f"❌ Prediction failed: {ex}")
            st.caption("Please check your inputs and try again.")

# ══════════════════════════════════════════════════════════════
# 2. TEST RESULT
# ══════════════════════════════════════════════════════════════
if st.session_state.dt_result:
    R = st.session_state.dt_result
    pred = R['pred']; prob = R['prob']; name_lbl = R['name']

    st.divider()
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"### ⚠️  {name_lbl} — Depression Risk Detected\n\n"
                 "The Decision Tree model predicts a **high risk of depression**. "
                 "Please consider seeking professional support.")
    else:
        st.success(f"### ✅  {name_lbl} — No Depression Detected\n\n"
                   "The Decision Tree model predicts **low depression risk**. "
                   "Keep maintaining a healthy lifestyle!")

    # ── Business Rule Alerts ──────────────────────────────────
    if R.get('high_concern'):
        st.warning("⚠️ **High Concern:** Student has both Anxiety AND Panic Attack. "
                   "Immediate counselling referral is strongly recommended.")
    if R.get('married_risk'):
        st.warning("⚠️ **Married Student Alert:** Student is married and under 21. "
                   "Additional personal support may be needed.")
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
            "Field": ["Name","Gender","Age","Course","Year",
                      "CGPA","Marital","Anxiety","Panic Attack"],
            "Value": [R['name'], R['gender'], str(R['age']),
                      R['course'], R['year'], R['cgpa'],
                      R['marital'], R['anxiety'], R['panic']]
        }).set_index("Field"))

    with r3:
        st.markdown("**Model Info**")
        with st.container(border=True):
            st.write("**Algorithm:** Decision Tree")
            st.write("**Max Depth:** 5")
            st.write("**Criterion:** Gini")
            st.write("**Split:** 70 / 30")
            st.write(f"**Accuracy:** {M['acc']:.2f}%")


    # ══════════════════════════════════════════════════════════
    # PREDICTION EXPLANATION — WHY did DT decide this?
    # ══════════════════════════════════════════════════════════
    st.write("")
    st.markdown("---")
    st.markdown("### 🧠 Why did the Decision Tree predict this?")
    st.caption("Decision path traced through the live trained tree.")

    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown("**Decision Path Traced**")
        st.caption("Which nodes the tree evaluated for this student.")
        try:
            from sklearn.tree import _tree as _sk_tree
            _tree   = M['model'].tree_
            _f_names = ['Gender','Age','Course','Year','CGPA',
                        'Anxiety','Panic Attack','Marital Status']
            _g  = 1 if R['gender']  == 'Male' else 0
            _ax = 1 if R['anxiety'] == 'Yes'  else 0
            _pa = 1 if R['panic']   == 'Yes'  else 0
            _ma = 1 if R['marital'] == 'Yes'  else 0
            _ce = M['le_c'].transform([R['course']])[0] if R['course'] in M['le_c'].classes_ else 0
            _ye = M['le_y'].transform([R['year']])[0]   if R['year']   in M['le_y'].classes_ else 0
            _cn = CGPA_MAP.get(R['cgpa'], 3.25)
            _inp = [[_g, R['age'], _ce, _ye, _cn, _ax, _pa, _ma]]

            _node = 0
            _path = []
            while _tree.children_left[_node] != _sk_tree.TREE_LEAF:
                _feat_i  = _tree.feature[_node]
                _thresh  = _tree.threshold[_node]
                _val     = _inp[0][_feat_i]
                _go_left = _val <= _thresh
                _direction = "≤" if _go_left else ">"
                _path.append({
                    'Feature'  : _f_names[_feat_i],
                    'Condition': f"{_direction} {_thresh:.2f}",
                    'Value'    : f"{_val:.2f}",
                    'Decision' : "→ Left" if _go_left else "→ Right",
                })
                _node = _tree.children_left[_node] if _go_left else _tree.children_right[_node]

            _leaf_vals = _tree.value[_node][0]
            _leaf_lbl  = "Depression" if _leaf_vals[1] > _leaf_vals[0] else "No Depression"

            for i, step in enumerate(_path, 1):
                with st.container(border=True):
                    st.write(f"**Node {i}:** {step['Feature']} {step['Condition']}")
                    st.caption(f"Student value: {step['Value']} {step['Decision']}")

            st.success(f"**Leaf Node:** {int(_leaf_vals[0])} No Depression / "
                       f"{int(_leaf_vals[1])} Depression → **{_leaf_lbl}**")
        except Exception as _dtex:
            st.info(f"Path tracing unavailable: {_dtex}")

    with ex2:
        st.markdown("**Feature Importance in This Decision**")
        st.caption("Which features the tree relied on most (from trained model).")
        _fi_df2 = pd.DataFrame({
            'Feature'   : M['fi_labels'],
            'Importance': M['fi_vals']
        }).sort_values('Importance', ascending=True)
        fig_ex2, ax_ex2 = plt.subplots(figsize=(5, 3.5))
        _mean_fi = _fi_df2['Importance'].mean()
        _ex_colors = ['#EF4444' if v >= _mean_fi else '#9CA3AF'
                      for v in _fi_df2['Importance']]
        ax_ex2.barh(_fi_df2['Feature'], _fi_df2['Importance'],
                    color=_ex_colors, edgecolor='none', height=0.6)
        ax_ex2.axvline(_mean_fi, color='red', ls='--', lw=1.2,
                       alpha=0.7, label=f'Mean = {_mean_fi:.3f}')
        ax_ex2.set_xlabel('Gini Importance Score')
        ax_ex2.set_title('What the Tree Relies On (Live)', fontweight='bold')
        ax_ex2.legend(fontsize=8)
        ax_ex2.spines['top'].set_visible(False)
        ax_ex2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_ex2, use_container_width=True); plt.close()

        st.markdown("**Key factors for this student:**")
        if R['marital'] == 'Yes':
            st.write("• **Married** — Root split feature; married = higher risk branch")
        if R['anxiety'] == 'Yes':
            st.write("• **Anxiety = Yes** — Tree splits on this early in the path")
        if R['panic'] == 'Yes':
            st.write("• **Panic Attack = Yes** — Strong branch predictor")
        if CGPA_MAP.get(R['cgpa'], 3.25) >= 3.0:
            st.write("• **High CGPA** — Suggests academic stability, lower risk")

    st.write("")
    if st.button("Clear Result", key="dt_clear"):
        st.session_state.dt_result = None
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# 3. LIVE FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
st.subheader("Live Feature Importance")
st.caption("Gini impurity reduction per feature. "
           "Computed live from the trained model — no hardcode.")

fi_df = pd.DataFrame({
    'Feature': M['fi_labels'],
    'Importance': M['fi_vals']
}).sort_values('Importance', ascending=True)

fi_c1, fi_c2 = st.columns([2, 1])
with fi_c1:
    fig_fi, ax_fi = plt.subplots(figsize=(7, 3.5))
    mean_v = fi_df['Importance'].mean()
    colors = ['#10B981' if v >= mean_v else '#9CA3AF'
              for v in fi_df['Importance']]
    bars = ax_fi.barh(fi_df['Feature'], fi_df['Importance'],
                      color=colors, edgecolor='none', height=0.6)
    ax_fi.axvline(mean_v, color='red', linestyle='--',
                  linewidth=1.2, alpha=0.7, label=f'Mean = {mean_v:.3f}')
    for bar, val in zip(bars, fi_df['Importance']):
        ax_fi.text(val+0.005, bar.get_y()+bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    ax_fi.set_xlabel('Feature Importance (Gini Reduction)')
    ax_fi.set_title('Decision Tree Feature Importance (Live)', fontweight='bold')
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
            icon = "🟢" if row['Importance'] >= mean_v else "⚪"
            st.write(f"{icon} **{row['Feature']}** — {row['Importance']:.3f}")
        st.caption("🟢 Above average importance")

st.divider()

# ══════════════════════════════════════════════════════════════
# 4. DECISION TREE DIAGRAM
# ══════════════════════════════════════════════════════════════
st.subheader("Decision Tree Structure")
st.caption("Visual tree trained live on the dataset. "
           "Orange = Depression, Blue = No Depression.")

@st.cache_resource
def get_tree_fig(_model):
    fig, ax = plt.subplots(figsize=(22, 9))
    plot_tree(_model,
              feature_names=['Gender','Age','Course','Year','CGPA',
                             'Anxiety','Panic Attack','Marital Status'],
              class_names=['No Depression','Depression'],
              filled=True, rounded=True, fontsize=7, ax=ax)
    ax.set_title("Decision Tree — Live Trained (Depth 5)",
                 fontsize=13, fontweight='bold', pad=16)
    plt.tight_layout()
    return fig

st.pyplot(get_tree_fig(M['model']), use_container_width=True)
plt.close('all')

st.divider()

# ══════════════════════════════════════════════════════════════
# 5. LEARN MORE — Confusion Matrix
# ══════════════════════════════════════════════════════════════
with st.expander("📚  Learn More — Confusion Matrix & Classification Report"):
    st.markdown("### Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(M['cm'], annot=True, fmt='d', cmap='Greens', ax=ax_cm,
                xticklabels=['No Depression','Depression'],
                yticklabels=['No Depression','Depression'],
                linewidths=0.5, annot_kws={'size':12,'weight':'bold'})
    ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Decision Tree Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig_cm, use_container_width=True); plt.close()

    tn,fp,fn,tp = M['cm'].ravel()
    a,b,c,d = st.columns(4)
    a.metric("TN", str(tn)); b.metric("FP", str(fp))
    c.metric("FN", str(fn)); d.metric("TP", str(tp))

    st.write("")
    st.markdown("### How Decision Tree Works")
    st.write(
        "CART (Classification and Regression Trees) builds a binary tree by "
        "recursively splitting data on the feature that maximizes **Gini impurity reduction**. "
        "The tree stops splitting when it reaches **max depth = 5**. "
        "The root split feature (**Marital Status**) was automatically selected "
        "as the most discriminative feature in the dataset."
    )


# ══════════════════════════════════════════════════════════════
# BATCH PREDICTION — DECISION TREE
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("Batch Prediction — 🌳 Decision Tree")
st.caption("Upload a CSV with multiple students. Decision Tree predicts each one instantly.")

_dt_sample = pd.DataFrame({
    'Name':          ['Ahmad','Siti','Wei Ming'],
    'Gender':        ['Male','Female','Male'],
    'Age':           [20,21,19],
    'Course':        ['Computer Science','Engineering','Information Technology'],
    'Year_of_Study': ['Year 2','Year 3','Year 1'],
    'CGPA':          ['3.00 - 3.49','2.50 - 2.99','3.50 - 4.00'],
    'Marital_Status':['No','No','No'],
    'Anxiety':       ['Yes','No','No'],
    'Panic_Attack':  ['No','Yes','No'],
})

with st.expander("📋  View / Download CSV Template"):
    st.dataframe(_dt_sample, use_container_width=True, hide_index=True)
    st.download_button("⬇️  Download DT Template",
        data=_dt_sample.to_csv(index=False).encode(),
        file_name="dt_batch_template.csv", mime="text/csv",
        use_container_width=True, key="dt_tmpl_dl")

_dt_file = st.file_uploader("Upload CSV for Decision Tree Batch",
                              type=["csv"], key="dt_batch_file")
if _dt_file:
    try:
        _ddf = pd.read_csv(_dt_file)
        _ddf.columns = _ddf.columns.str.strip()
        st.success(f"✅ {len(_ddf)} students loaded")
        _dresults = []
        for _di, _drow in _ddf.iterrows():
            try:
                _dname = str(_drow.get('Name', f'Student {_di+1}')).strip()
                _dg  = 1 if str(_drow.get('Gender','')).lower()=='male' else 0
                _dax = 1 if str(_drow.get('Anxiety','')).lower()=='yes' else 0
                _dpa = 1 if str(_drow.get('Panic_Attack','')).lower()=='yes' else 0
                _dma = 1 if str(_drow.get('Marital_Status','')).lower()=='yes' else 0
                try: _dage = int(float(_drow.get('Age',20)))
                except: _dage = 20
                _dcourse = str(_drow.get('Course','Others'))
                _dyear   = str(_drow.get('Year_of_Study','Year 1'))
                _dcgpa   = str(_drow.get('CGPA','3.00 - 3.49')).strip()
                _dce = M['le_c'].transform([_dcourse])[0] if _dcourse in M['le_c'].classes_ else 0
                _dye = M['le_y'].transform([_dyear])[0]   if _dyear   in M['le_y'].classes_ else 0
                _dcn = CGPA_MAP.get(_dcgpa, 3.25)
                _dinp = pd.DataFrame([[_dg,_dage,_dce,_dye,_dcn,_dax,_dpa,_dma]],columns=M['feat'])
                _dpred = int(M['model'].predict(_dinp)[0])
                _dprob = M['model'].predict_proba(_dinp)[0][1]
                _dresults.append({
                    'Name': _dname, 'Gender': _drow.get('Gender',''),
                    'Age': _dage, 'Course': _dcourse, 'Year': _dyear,
                    'CGPA': _dcgpa, 'Marital': _drow.get('Marital_Status',''),
                    'Anxiety': _drow.get('Anxiety',''), 'Panic': _drow.get('Panic_Attack',''),
                    'Result':     '⚠️ Depression' if _dpred==1 else '✅ No Depression',
                    'Confidence': f"{_dprob*100:.1f}%",
                    'Risk':       'HIGH' if _dpred==1 else 'LOW',
                    '_pred': _dpred, '_prob': _dprob,
                })
            except Exception as _de:
                _dresults.append({'Name': str(_drow.get('Name','')), 'Error': str(_de)})

        _dres = pd.DataFrame(_dresults)
        _dtotal = len(_dres)
        _ddep   = int(_dres['_pred'].sum()) if '_pred' in _dres else 0
        _dnodep = _dtotal - _ddep

        _dm1,_dm2,_dm3 = st.columns(3)
        _dm1.metric("Total Students", str(_dtotal))
        _dm2.metric("⚠️ At Risk",     str(_ddep),
                    delta=f"{_ddep/_dtotal*100:.0f}%", delta_color="inverse")
        _dm3.metric("✅ No Risk",      str(_dnodep),
                    delta=f"{_dnodep/_dtotal*100:.0f}%")

        _dc1,_dc2 = st.columns(2)
        with _dc1:
            _dfig,_dax2 = plt.subplots(figsize=(4,3))
            _dax2.pie([_ddep,_dnodep],
                labels=[f'Depression ({_ddep})',f'No Depression ({_dnodep})'],
                colors=['#EF4444','#10B981'], autopct='%1.1f%%', startangle=90,
                wedgeprops={'edgecolor':'white','linewidth':2},
                textprops={'fontsize':10,'fontweight':'bold'})
            _dax2.set_title('Decision Tree Batch Results', fontweight='bold')
            plt.tight_layout(); st.pyplot(_dfig, use_container_width=True); plt.close()
        with _dc2:
            _dfig2,_dax3 = plt.subplots(figsize=(4,3))
            _dax3.hist(_dres['_prob']*100, bins=10, color='#10B981', edgecolor='white', alpha=0.85)
            _dax3.axvline(50, color='red', ls='--', lw=1.5, label='Threshold 50%')
            _dax3.set_xlabel('Depression Probability (%)'); _dax3.set_ylabel('Count')
            _dax3.set_title('Confidence Distribution', fontweight='bold'); _dax3.legend(fontsize=9)
            _dax3.spines['top'].set_visible(False); _dax3.spines['right'].set_visible(False)
            plt.tight_layout(); st.pyplot(_dfig2, use_container_width=True); plt.close()

        _ddisp = _dres[['Name','Gender','Age','Course','Year','CGPA',
                         'Anxiety','Panic','Result','Confidence','Risk']]
        def _dstyle(val):
            if '⚠️' in str(val) or val=='HIGH': return 'background-color:#FEE2E2;color:#991B1B;font-weight:bold'
            if '✅' in str(val) or val=='LOW':  return 'background-color:#DCFCE7;color:#166534;font-weight:bold'
            return ''
        try:    _dstyled = _ddisp.style.map(_dstyle, subset=['Result','Risk'])
        except: _dstyled = _ddisp.style.applymap(_dstyle, subset=['Result','Risk'])
        st.dataframe(_dstyled, use_container_width=True, hide_index=True)

        st.write("")
        st.markdown("**Individual Student Cards**")
        for _, _dr in _dres.iterrows():
            if '_pred' not in _dr: continue
            _dicon = "⚠️" if _dr['_pred']==1 else "✅"
            with st.expander(f"{_dicon} {_dr['Name']} — {_dr['Result']} ({_dr['Confidence']})"):
                if _dr['_pred']==1: st.error(f"**Depression Risk** | DT Confidence: {_dr['Confidence']}")
                else:               st.success(f"**No Depression** | DT Confidence: {_dr['Confidence']}")

        _ddl1,_ddl2 = st.columns(2)
        with _ddl1:
            st.download_button("⬇️  Download All Results",
                data=_ddisp.to_csv(index=False).encode(),
                file_name="dt_batch_results.csv", mime="text/csv",
                use_container_width=True, key="dt_dl_all")
        with _ddl2:
            _dhigh = _dres[_dres['_pred']==1][['Name','Gender','Age','Course',
                'Year','CGPA','Anxiety','Panic','Result','Confidence']]
            if len(_dhigh):
                st.download_button(f"⬇️  At-Risk Only ({len(_dhigh)})",
                    data=_dhigh.to_csv(index=False).encode(),
                    file_name="dt_at_risk.csv", mime="text/csv",
                    use_container_width=True, key="dt_dl_risk")
    except Exception as _derr:
        st.error(f"Error processing file: {str(_derr)}")
        st.caption("Please check your CSV matches the template format.")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 · Group 3 · Dr Goh · TARUMT")