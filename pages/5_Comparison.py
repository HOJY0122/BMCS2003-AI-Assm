import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.family':'sans-serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white','axes.edgecolor':'#E2E8F0','axes.labelcolor':'#1E293B','xtick.color':'#64748B','ytick.color':'#64748B','text.color':'#1E293B','grid.color':'#F1F5F9','grid.linewidth':0.8})
import seaborn as sns
import warnings, sys, os
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import sidebar
from utils.models import load_all_models as _load_shared


def _hex_to_rgba(hex_color, alpha):
    """Convert a '#RRGGBB' hex color to an 'rgba(r,g,b,a)' string, so a
    radar-chart fill can be made transparent WITHOUT also dimming the
    trace's line/markers (which happens if you set trace-level `opacity`
    instead — that was the cause of the radar chart's hover/interaction
    being unreliable, since the whole trace including its hoverable line
    became faint and hard to target)."""
    hex_color = hex_color.lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'

st.set_page_config(
    page_title="Compare Models — MindCheck",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("compare")

# ══════════════════════════════════════════════════════════════
# LOAD DATA + TRAIN ALL 3 MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all_models():
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())

    le_course = LabelEncoder()
    le_year   = LabelEncoder()
    df['Course_Enc']        = le_course.fit_transform(df['Course'])
    df['Year_of_Study_Enc'] = le_year.fit_transform(df['Year_of_Study'])

    # ── KNN — Target: Depression ─────────────────────────────
    knn_feat = ['Gender','Age','Course_Enc','Year_of_Study_Enc',
                'CGPA_Numeric','Anxiety','Panic_Attack']
    X_k = df[knn_feat]; y_k = df['Depression']
    sc_knn = MinMaxScaler()
    X_ks   = sc_knn.fit_transform(X_k)
    Xtr_k, Xte_k, ytr_k, yte_k = train_test_split(
        X_ks, y_k, test_size=0.2, random_state=42, stratify=y_k)
    best_k, best_a = 5, 0
    for k in range(1, 21):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(Xtr_k, ytr_k)
        a = accuracy_score(yte_k, m.predict(Xte_k))
        if a > best_a: best_a, best_k = a, k
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn.fit(Xtr_k, ytr_k)
    knn_p = knn.predict(Xte_k)

    # ── DT — Target: Depression ──────────────────────────────
    dt_feat = ['Gender','Age','Course_Enc','Year_of_Study_Enc',
               'CGPA_Numeric','Anxiety','Panic_Attack','Marital_Status']
    X_d = df[dt_feat]; y_d = df['Depression']
    Xtr_d, Xte_d, ytr_d, yte_d = train_test_split(
        X_d, y_d, test_size=0.3, random_state=42, stratify=y_d)
    dt = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
    dt.fit(Xtr_d, ytr_d)
    dt_p = dt.predict(Xte_d)

    # ── SVM — Target: Depression (same encoded features as KNN/DT) ──
    from sklearn.pipeline import Pipeline
    svm_feat = ['Gender','Age','Course_Enc','Year_of_Study_Enc',
                'CGPA_Numeric','Anxiety','Panic_Attack','Marital_Status']
    X_s = df[svm_feat]; y_s = df['Depression']
    svm_col_order = svm_feat
    svm_pipe = Pipeline([
        ('scl', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42))
    ])
    Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
        X_s, y_s, test_size=0.25, random_state=42, stratify=y_s)
    svm_pipe.fit(Xtr_s, ytr_s)
    svm_p = svm_pipe.predict(Xte_s)
    svm   = svm_pipe   # alias
    sc_svm = None      # pipeline handles scaling internally

    def _m(yt, yp):
        return {
            'acc' : accuracy_score(yt, yp)*100,
            'prec': precision_score(yt, yp, zero_division=0)*100,
            'rec' : recall_score(yt, yp, zero_division=0)*100,
            'f1'  : f1_score(yt, yp, zero_division=0)*100,
            'cm'  : confusion_matrix(yt, yp),
        }

    return {
        'knn': knn, 'sc_knn': sc_knn, 'knn_feat': knn_feat,
        'svm_col_order': list(X_s.columns),
        'best_k': best_k, 'knn_m': _m(yte_k, knn_p),
        'dt' : dt,  'dt_feat': dt_feat, 'dt_m': _m(yte_d, dt_p),
        'svm': svm, 'sc_svm': sc_svm, 'svm_feat': svm_feat,
        'svm_m': _m(yte_s, svm_p),
        'le_course': le_course, 'le_year': le_year,
        'df': df,
        'knn_test': (Xte_k, yte_k),
        'dt_test' : (Xte_d, yte_d),
        'svm_test': (Xte_s, yte_s),
    }

with st.spinner("Training all 3 models..."):
    M = load_all_models()

# Session state
for k, v in [('log',[]),('n',0),('knn_c',0),('dt_c',0),('svm_c',0),
                ('as_result', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Page Header ────────────────────────────────────────────────
st.markdown("##### RESULTS")
st.title("Live Model Comparison")
st.write(
    "Enter student information and instantly compare predictions from "
    "**KNN**, **Decision Tree** and **SVM** simultaneously. "
    "Track accuracy across multiple test runs."
)
st.divider()

# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# SECTION 1 — SCORE CARDS (collapseable)
# ══════════════════════════════════════════════════════════════
with st.expander("📊  Trained Model Performance", expanded=True):
    st.caption("Live computed metrics — same numbers shown on each predictor page")
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.markdown("### 🔵 KNN")
            st.caption(f"Ho Jun Yon · K={M['best_k']} · Target: Depression")
            st.divider()
            st.metric("Accuracy",  f"{M['knn_m']['acc']:.2f}%")
            ma, mb = st.columns(2)
            ma.metric("Precision", f"{M['knn_m']['prec']:.2f}%")
            mb.metric("Recall",    f"{M['knn_m']['rec']:.2f}%")
            mc, md = st.columns(2)
            mc.metric("F1 Score",  f"{M['knn_m']['f1']:.2f}%")
            md.metric("Split",     "80 / 20")

    with c2:
        with st.container(border=True):
            st.markdown("### 🌳 Decision Tree")
            st.caption("Irvin Tan · Depth 5 · Target: Depression")
            st.divider()
            st.metric("Accuracy",  f"{M['dt_m']['acc']:.2f}%")
            ma, mb = st.columns(2)
            ma.metric("Precision", f"{M['dt_m']['prec']:.2f}%")
            mb.metric("Recall",    f"{M['dt_m']['rec']:.2f}%")
            mc, md = st.columns(2)
            mc.metric("F1 Score",  f"{M['dt_m']['f1']:.2f}%")
            md.metric("Split",     "70 / 30")

    with c3:
        with st.container(border=True):
            st.markdown("### 🔴 SVM")
            st.caption("Chiang Jun Hang · RBF · Target: Depression")
            st.divider()
            st.metric("Accuracy",  f"{M['svm_m']['acc']:.2f}%")
            ma, mb = st.columns(2)
            ma.metric("Precision", f"{M['svm_m']['prec']:.2f}%")
            mb.metric("Recall",    f"{M['svm_m']['rec']:.2f}%")
            mc, md = st.columns(2)
            mc.metric("F1 Score",  f"{M['svm_m']['f1']:.2f}%")
            md.metric("Split",     "75 / 25")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2 — LIVE TEST FORM
# ══════════════════════════════════════════════════════════════
st.subheader("Run Live Test — All 3 Models")
st.write("Fill in student details. All 3 models predict simultaneously.")

CGPA_MAP = {
    '0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
    '3.00 - 3.49':3.25,'3.50 - 4.00':3.75
}

with st.form("live_form", clear_on_submit=False):
    fa, fb, fc = st.columns(3)

    with fa:
        t_name    = st.text_input("Name (optional)")
        t_gender  = st.selectbox("Gender", ["Female","Male"])
        t_age     = st.slider("Age", 17, 30, 20)
        t_course  = st.selectbox("Course", [
            "Computer Science","Information Technology","Engineering",
            "Law","Psychology","Language","Islamic Studies",
            "Health Sciences","Business","Science & Math",
            "Arts & Social","Others"])

    with fb:
        t_year    = st.selectbox("Year of Study",
                                 ["Year 1","Year 2","Year 3","Year 4"])
        t_cgpa    = st.selectbox("CGPA Range", list(CGPA_MAP.keys()))
        t_marital = st.selectbox("Marital Status", ["No","Yes"])

    with fc:
        t_anxiety = st.selectbox("Has Anxiety?",      ["No","Yes"])
        t_panic   = st.selectbox("Has Panic Attack?", ["No","Yes"])
        st.markdown("**Actual answers (for tracking)**")
        t_actual_dep   = st.selectbox("Actual Depression?",
                                      ["Unknown","No","Yes"])

    run = st.form_submit_button("Run All 3 Models Now",
                                width='stretch')

def _show_result(col, icon, title, member, target, pred, prob,
                 act, label_pos, label_neg):
    """Render one model's result card. Pure render — correctness is
    computed separately (see _score()) so this can be safely called on
    every rerun for persistent display, without double-counting the
    scoreboard tally."""
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.caption(f"{member} · Target: {target}")
            st.divider()
            if pred == 1:
                st.error(f"**{label_pos} DETECTED**")
            else:
                st.success(f"**NO {label_pos}**")

            pa_col, pb_col = st.columns(2)
            pa_col.metric(label_neg, f"{prob[0]*100:.1f}%")
            pb_col.metric(label_pos, f"{prob[1]*100:.1f}%")

            fig, ax = plt.subplots(figsize=(3.5, 0.6))
            ax.barh([""], [prob[0]*100], color="#10B981",
                    height=0.5, label=label_neg)
            ax.barh([""], [prob[1]*100], left=[prob[0]*100],
                    color="#EF4444", height=0.5, label=label_pos)
            ax.set_xlim(0, 100)
            ax.axis('off')
            if prob[0] > 0.15:
                ax.text(prob[0]*50, 0, f"{prob[0]*100:.0f}%",
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            if prob[1] > 0.15:
                ax.text(prob[0]*100 + prob[1]*50, 0,
                        f"{prob[1]*100:.0f}%",
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            plt.tight_layout(pad=0)
            st.pyplot(fig, width='stretch')
            plt.close()

            if act is not None:
                if pred == act:
                    st.success("✅ Correct")
                else:
                    st.error("❌ Wrong")


def _score(pred, act):
    return None if act is None else (pred == act)


if run:
    le_c = M['le_course']; le_y = M['le_year']
    g  = 1 if t_gender  == "Male" else 0
    ax = 1 if t_anxiety == "Yes"  else 0
    pa = 1 if t_panic   == "Yes"  else 0
    ma = 1 if t_marital == "Yes"  else 0
    cg = CGPA_MAP[t_cgpa]
    ce = le_c.transform([t_course])[0] if t_course in le_c.classes_ else 0
    ye = le_y.transform([t_year])[0]   if t_year   in le_y.classes_ else 0

    # ── KNN ──────────────────────────────────────────────────
    knn_in = pd.DataFrame(
        [[g, t_age, ce, ye, cg, ax, pa]],
        columns=M['knn_feat'])
    knn_in_s  = M['sc_knn'].transform(knn_in)
    knn_pred  = M['knn'].predict(knn_in_s)[0]
    knn_prob  = M['knn'].predict_proba(knn_in_s)[0]

    # ── DT ───────────────────────────────────────────────────
    dt_in = pd.DataFrame(
        [[g, t_age, ce, ye, cg, ax, pa, ma]],
        columns=M['dt_feat'])
    dt_pred  = M['dt'].predict(dt_in)[0]
    dt_prob  = M['dt'].predict_proba(dt_in)[0]

    # ── SVM ── same encoded feature set as KNN/DT ──────────────
    svm_in = pd.DataFrame([[g, t_age, ce, ye, cg, ax, pa, ma]],
                          columns=M['svm_col_order'])
    svm_pred = M['svm'].predict(svm_in)[0]
    svm_prob = M['svm'].predict_proba(svm_in)[0]

    # ── Actual labels ─────────────────────────────────────────
    act_dep   = None if t_actual_dep   == "Unknown" else (1 if t_actual_dep   == "Yes" else 0)

    name = t_name.strip() or "Student"
    dep_agree = (knn_pred == dt_pred)

    # Correctness computed ONCE here (not inside the render function) so
    # the scoreboard tally below never double-counts on a later rerun.
    knn_c = _score(knn_pred, act_dep)
    dt_c  = _score(dt_pred,  act_dep)
    svm_c = _score(svm_pred, act_dep)

    # ── Persist everything needed to redisplay + navigate on every
    # future rerun — NOT just the one rerun where the form was submitted.
    # (A button nested inside `if run:` would only ever fire once: the
    # very next click makes `run` False again, hiding the button before
    # its own click can be processed — that was the original bug here.)
    st.session_state['live_result'] = {
        'name': name, 'gender': t_gender, 'age': t_age,
        'course': t_course, 'year': t_year, 'cgpa': t_cgpa,
        'marital': t_marital, 'anxiety': t_anxiety,
        'panic': t_panic,
        'knn_pred': int(knn_pred), 'knn_prob': knn_prob.tolist(),
        'dt_pred':  int(dt_pred),  'dt_prob':  dt_prob.tolist(),
        'svm_pred': int(svm_pred), 'svm_prob': svm_prob.tolist(),
        'svm_in': svm_in,
        'act_dep': act_dep,
        'dep_agree': dep_agree,
    }

    # ── Update session state / log — only on the actual submit ────
    st.session_state['n'] += 1
    if knn_c is not None: st.session_state['knn_c'] += int(knn_c)
    if dt_c  is not None: st.session_state['dt_c']  += int(dt_c)
    if svm_c is not None: st.session_state['svm_c'] += int(svm_c)

    st.session_state['log'].insert(0, {
        "Test #"    : st.session_state['n'],
        "Name"      : name,
        "Gender"    : t_gender,
        "Age"       : t_age,
        "CGPA"      : t_cgpa,
        "Anxiety"   : t_anxiety,
        "Panic"     : t_panic,
        "KNN (Dep)" : "Dep" if knn_pred==1 else "No",
        "DT (Dep)"  : "Dep" if dt_pred==1  else "No",
        "SVM (Dep)" : "Dep" if svm_pred==1 else "No",
        "KNN%"      : f"{knn_prob[1]*100:.0f}%",
        "DT%"       : f"{dt_prob[1]*100:.0f}%",
        "SVM%"      : f"{svm_prob[1]*100:.0f}%",
        "Agree KNN-DT-SVM": "✅" if (knn_pred==dt_pred==svm_pred) else "❌",
        "Actual Dep"  : t_actual_dep,
    })

# ══════════════════════════════════════════════════════════════
# Persistent display — reruns EVERY time (button clicks included),
# reading from session_state instead of local variables that only
# existed during the `if run:` rerun. This is what actually fixes the
# "Open KNN/DT/SVM" buttons: they now exist on every rerun after a test
# has been run, not just the one immediately after clicking "Run All 3
# Models Now".
# ══════════════════════════════════════════════════════════════
if st.session_state.get('live_result'):
    LR = st.session_state['live_result']

    st.markdown("---")
    st.subheader(f"Prediction Results — {LR['name']}")

    rc1, rc2, rc3 = st.columns(3)
    _show_result(rc1,"🔵","KNN","Ho Jun Yon","Depression",
                 LR['knn_pred'], LR['knn_prob'], LR['act_dep'], "Depression","No Dep")
    _show_result(rc2,"🌳","Decision Tree","Irvin","Depression",
                 LR['dt_pred'], LR['dt_prob'], LR['act_dep'], "Depression","No Dep")
    _show_result(rc3,"🔴","SVM","Chiang Jun Hang","Depression",
                 LR['svm_pred'], LR['svm_prob'], LR['act_dep'], "Depression","No Dep")

    dep_label = "Depression" if LR['knn_pred'] == 1 else "No Depression"
    svm_label = "Depression" if LR['svm_pred'] == 1 else "No Depression"

    st.markdown("")
    a1, a2 = st.columns(2)
    with a1:
        if LR['dep_agree']:
            st.info(f"KNN & DT **AGREE** on Depression: **{dep_label}**")
        else:
            st.warning(
                f"KNN & DT **DISAGREE** — "
                f"KNN: **{'Depression' if LR['knn_pred']==1 else 'No Depression'}** | "
                f"DT: **{'Depression' if LR['dt_pred']==1 else 'No Depression'}**"
            )
    with a2:
        st.info(f"SVM predicts Depression: **{svm_label}**")

    # ── Navigate to predictor with result pre-loaded ──────────
    st.write("")
    st.markdown("**Open in Individual Predictor — result pre-loaded, no need to predict again**")
    _live_shared = {
        'from_comparison': True,
        'name': LR['name'], 'gender': LR['gender'], 'age': LR['age'],
        'course': LR['course'], 'year': LR['year'], 'cgpa': LR['cgpa'],
        'marital': LR['marital'], 'anxiety': LR['anxiety'],
        'panic': LR['panic'],
    }
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("🔵 Open KNN →", key="live_nav_knn", width='stretch'):
            st.session_state['knn_prefill'] = _live_shared
            st.session_state['knn_carry']  = {
                'pred': LR['knn_pred'], 'prob': LR['knn_prob'],
                'name': LR['name'], 'gender': LR['gender'], 'age': LR['age'],
                'course': LR['course'], 'year': LR['year'], 'cgpa': LR['cgpa'],
                'anxiety': LR['anxiety'], 'panic': LR['panic'],
                'high_concern': (LR['anxiety']=="Yes" and LR['panic']=="Yes"),
                'academic_risk': (LR['year'] in ["Year 3","Year 4"] and LR['cgpa']=="0 - 1.99"),
            }
            st.switch_page("pages/2_KNN.py")
    with nav2:
        if st.button("🌳 Open DT →", key="live_nav_dt", width='stretch'):
            st.session_state['dt_prefill'] = _live_shared
            st.session_state['dt_carry']  = {
                'pred': LR['dt_pred'], 'prob': LR['dt_prob'],
                'name': LR['name'], 'gender': LR['gender'], 'age': LR['age'],
                'course': LR['course'], 'year': LR['year'], 'cgpa': LR['cgpa'],
                'marital': LR['marital'], 'anxiety': LR['anxiety'], 'panic': LR['panic'],
                'high_concern': (LR['anxiety']=="Yes" and LR['panic']=="Yes"),
                'married_risk': (LR['marital']=="Yes" and LR['age']<21),
                'academic_risk': (LR['year'] in ["Year 3","Year 4"] and LR['cgpa']=="0 - 1.99"),
            }
            st.switch_page("pages/3_Decision_Tree.py")
    with nav3:
        if st.button("🔴 Open SVM →", key="live_nav_svm", width='stretch'):
            st.session_state['svm_prefill'] = _live_shared
            st.session_state['svm_carry']  = {
                'pred': LR['svm_pred'], 'prob': LR['svm_prob'],
                'inp': LR['svm_in'],
                'name': LR['name'], 'gender': LR['gender'], 'age': LR['age'],
                'course': LR['course'], 'year': LR['year'], 'cgpa': LR['cgpa'],
                'marital': LR['marital'], 'anxiety': LR['anxiety'],
                'panic': LR['panic'],
                'high_concern': (LR['anxiety']=="Yes" and LR['panic']=="Yes"),
                'academic_risk': (LR['year'] in ["Year 3","Year 4"] and LR['cgpa']=="0 - 1.99"),
            }
            st.switch_page("pages/4_SVM.py")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3 — LIVE SCOREBOARD
# ══════════════════════════════════════════════════════════════
st.subheader("Live Test Scoreboard")

n = st.session_state['n']
if n == 0:
    st.info("No tests run yet. Fill in the form above and click Run All 3 Models.")
else:
    sb = st.columns(5)
    sb[0].metric("Tests Run",    str(n))
    sb[1].metric("KNN Correct",  str(st.session_state['knn_c']),
                 help="Depression — only counted when Actual provided")
    sb[2].metric("DT Correct",   str(st.session_state['dt_c']),
                 help="Depression — only counted when Actual provided")
    sb[3].metric("SVM Correct",  str(st.session_state['svm_c']),
                 help="Depression — only counted when Actual provided")
    log_df = pd.DataFrame(st.session_state['log'])
    agree_n = (log_df["Agree KNN-DT-SVM"] == "✅").sum()
    sb[4].metric("All 3 Agree", f"{agree_n}/{n}")

    # Live accuracy chart (if actuals provided)
    actuals_dep = sum(1 for e in st.session_state['log'] if e['Actual Dep'] != 'Unknown')

    if actuals_dep > 0:
        st.write("")
        st.markdown("**Live Accuracy on Your Test Cases**")
        fig, ax = plt.subplots(figsize=(8, 2.2))
        models, accs, colors = [], [], []
        models += ['KNN (Dep)', 'DT (Dep)', 'SVM (Dep)']
        accs   += [st.session_state['knn_c']/actuals_dep*100,
                   st.session_state['dt_c'] /actuals_dep*100,
                   st.session_state['svm_c']/actuals_dep*100]
        colors += ['#5B7FFF','#10B981','#EF4444']
        bars = ax.barh(models, accs, color=colors, height=0.45, edgecolor='none')
        for bar, val in zip(bars, accs):
            ax.text(val+1, bar.get_y()+bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10, fontweight='bold')
        ax.set_xlim(0, 118)
        ax.axvline(50, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_xlabel("Your live test accuracy (%)")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close()

    st.markdown("**Test History**")
    st.dataframe(log_df, width='stretch', hide_index=True)

    if st.button("Clear All Tests", key="clear"):
        for k in ['log','n','knn_c','dt_c','svm_c']:
            st.session_state[k] = [] if k=='log' else 0
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — INTERACTIVE METRIC CHARTS (Plotly)
# ══════════════════════════════════════════════════════════════
st.subheader("Interactive Metric Charts")
st.caption("Hover over bars for details · Click legend to show/hide models · "
           "Double-click to isolate · Drag to zoom · Click metric tabs to drill down")

metric_names = ["Accuracy","Precision","Recall","F1 Score"]
knn_v = [M['knn_m']['acc'], M['knn_m']['prec'], M['knn_m']['rec'], M['knn_m']['f1']]
dt_v  = [M['dt_m']['acc'],  M['dt_m']['prec'],  M['dt_m']['rec'],  M['dt_m']['f1']]
svm_v = [M['svm_m']['acc'], M['svm_m']['prec'], M['svm_m']['rec'], M['svm_m']['f1']]

# ── Tabs for each metric drill-down ───────────────────────────
tab_all, tab_acc, tab_prec, tab_rec, tab_f1 = st.tabs([
    "📊 All Metrics", "🎯 Accuracy", "🔬 Precision", "📡 Recall", "⚖️ F1 Score"
])

with tab_all:
    ic1, ic2 = st.columns(2)

    with ic1:
        # Interactive grouped bar chart
        fig_bar = go.Figure()
        for name, vals, color in [
            ('🔵 KNN',           knn_v, '#3B82F6'),
            ('🌳 Decision Tree', dt_v,  '#10B981'),
            ('🔴 SVM',           svm_v, '#EF4444'),
        ]:
            fig_bar.add_trace(go.Bar(
                name=name, x=metric_names, y=vals,
                marker_color=color, opacity=0.9,
                hovertemplate='<b>%{x}</b><br>Score: <b>%{y:.2f}%</b><extra>' + name + '</extra>',
                text=[f'{v:.1f}%' for v in vals],
                textposition='outside',
                textfont=dict(size=10, color='white'),
            ))
        fig_bar.update_layout(
            barmode='group',
            title=dict(text='All 3 Models — Click legend to toggle',
                      font=dict(size=14)),
            yaxis=dict(range=[0,115], title='Score (%)',
                      gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(title='Metric'),
            legend=dict(orientation='h', yanchor='bottom',
                       y=1.02, xanchor='right', x=1),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            hovermode='x unified',
            height=380,
        )
        fig_bar.update_xaxes(showgrid=False)
        st.plotly_chart(fig_bar, width='stretch')

    with ic2:
        # Interactive radar chart
        fig_radar = go.Figure()
        for name, vals, color in [
            ('🔵 KNN',           knn_v, '#3B82F6'),
            ('🌳 Decision Tree', dt_v,  '#10B981'),
            ('🔴 SVM',           svm_v, '#EF4444'),
        ]:
            fig_radar.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=metric_names + [metric_names[0]],
                fill='toself', fillcolor=_hex_to_rgba(color, 0.25),
                mode='lines+markers',
                marker=dict(size=7, color=color),
                name=name,
                line=dict(color=color, width=2),
                hoveron='points+fills',
                hovertemplate='<b>%{theta}</b><br>%{r:.2f}%<extra>' + name + '</extra>',
            ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,110],
                               gridcolor='rgba(255,255,255,0.15)',
                               tickfont=dict(size=9, color='grey')),
                angularaxis=dict(gridcolor='rgba(255,255,255,0.15)'),
                bgcolor='rgba(0,0,0,0)',
            ),
            title=dict(text='Radar Chart — Hover to compare',
                      font=dict(size=14)),
            legend=dict(orientation='h', yanchor='bottom',
                       y=-0.15, xanchor='center', x=0.5),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=380,
        )
        st.plotly_chart(fig_radar, width='stretch')

# ── Metric drill-down tabs ─────────────────────────────────────
for tab, metric, idx in [
    (tab_acc,  'Accuracy',  0),
    (tab_prec, 'Precision', 1),
    (tab_rec,  'Recall',    2),
    (tab_f1,   'F1 Score',  3),
]:
    with tab:
        vals_3 = {'KNN': knn_v[idx], 'Decision Tree': dt_v[idx], 'SVM': svm_v[idx]}
        best   = max(vals_3, key=vals_3.get)
        colors_3 = {'KNN':'#3B82F6','Decision Tree':'#10B981','SVM':'#EF4444'}

        dd1, dd2 = st.columns([2, 1])
        with dd1:
            fig_dd = go.Figure(go.Bar(
                x=list(vals_3.keys()),
                y=list(vals_3.values()),
                marker_color=[colors_3[k] for k in vals_3],
                text=[f'{v:.2f}%' for v in vals_3.values()],
                textposition='outside',
                textfont=dict(size=13, color='white', family='bold'),
                hovertemplate='<b>%{x}</b><br>' + metric + ': <b>%{y:.2f}%</b><extra></extra>',
                width=0.5,
            ))
            fig_dd.update_layout(
                title=dict(text=f'{metric} — Click bar for details',
                          font=dict(size=13)),
                yaxis=dict(range=[0,115], title=f'{metric} (%)',
                          gridcolor='rgba(255,255,255,0.1)'),
                xaxis=dict(title='Model'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300,
            )
            fig_dd.update_xaxes(showgrid=False)
            st.plotly_chart(fig_dd, width='stretch')

        with dd2:
            with st.container(border=True):
                st.markdown(f"**{metric} Analysis**")
                st.write(f"🏆 Best: **{best}** ({vals_3[best]:.2f}%)")
                st.write("")
                for model, val in sorted(vals_3.items(),
                                          key=lambda x: -x[1]):
                    diff = val - list(vals_3.values())[0]
                    icon = "🥇" if model==best else "🥈" if val==sorted(vals_3.values())[-2] else "🥉"
                    st.write(f"{icon} {model}: **{val:.2f}%**")
                st.write("")
                # Metric explanation
                explanations = {
                    'Accuracy':  'Overall correct predictions out of all predictions.',
                    'Precision': 'Of all Depression predictions, how many were correct.',
                    'Recall':    'Of all actual Depression cases, how many were caught. Most critical for mental health!',
                    'F1 Score':  'Balance between Precision and Recall.',
                }
                st.caption(explanations[metric])

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — INTERACTIVE CONFUSION MATRICES (Plotly)
# ══════════════════════════════════════════════════════════════
st.subheader("Interactive Confusion Matrices")
st.caption("Hover over each cell for details · Click model tabs to switch")

cm_tab1, cm_tab2, cm_tab3 = st.tabs(["🔵 KNN", "🌳 Decision Tree", "🔴 SVM"])

for cm_tab, title, cm_data, color, split, member in [
    (cm_tab1, "KNN",           M['knn_m']['cm'], "#3B82F6", "80/20", "Ho Jun Yon"),
    (cm_tab2, "Decision Tree", M['dt_m']['cm'],  "#10B981", "70/30", "Irvin Tan"),
    (cm_tab3, "SVM",           M['svm_m']['cm'], "#EF4444", "75/25", "Chiang Jun Hang"),
]:
    with cm_tab:
        tn,fp,fn,tp = cm_data.ravel()
        acc  = (tn+tp)/(tn+fp+fn+tp)*100
        prec = tp/(tp+fp)*100 if (tp+fp)>0 else 0
        rec  = tp/(tp+fn)*100 if (tp+fn)>0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0

        cmt1, cmt2 = st.columns([1.5, 1])
        with cmt1:
            # Interactive heatmap
            z   = [[tn, fp], [fn, tp]]
            txt = [
                [f'TN = {tn}<br>Correctly identified<br>as No Depression',
                 f'FP = {fp}<br>Incorrectly predicted<br>as Depression'],
                [f'FN = {fn}<br>Missed Depression<br>cases (most costly!)',
                 f'TP = {tp}<br>Correctly identified<br>Depression cases'],
            ]
            fig_cm = go.Figure(go.Heatmap(
                z=z, x=['Predicted: No', 'Predicted: Yes'],
                y=['Actual: No', 'Actual: Yes'],
                text=[[str(tn), str(fp)],[str(fn), str(tp)]],
                texttemplate='<b>%{text}</b>',
                textfont=dict(size=20, color='white'),
                hovertext=txt,
                hovertemplate='%{hovertext}<extra></extra>',
                colorscale=[[0,'rgba(30,30,50,1)'],[1,color]],
                showscale=False,
            ))
            fig_cm.update_layout(
                title=dict(text=f'{title} Confusion Matrix — {member} · Split {split}',
                          font=dict(size=13)),
                xaxis=dict(title='Predicted', side='bottom'),
                yaxis=dict(title='Actual', autorange='reversed'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                height=320,
            )
            st.plotly_chart(fig_cm, width='stretch')

        with cmt2:
            with st.container(border=True):
                st.markdown(f"**{title} Results**")
                st.metric("Accuracy",  f"{acc:.2f}%")
                st.metric("Precision", f"{prec:.2f}%")
                st.metric("Recall",    f"{rec:.2f}%")
                st.metric("F1 Score",  f"{f1:.2f}%")
            st.write("")
            with st.container(border=True):
                st.markdown("**Cell Breakdown**")
                st.write(f"✅ **TN = {tn}** — Correctly no depression")
                st.write(f"⚠️ **FP = {fp}** — False alarm")
                st.write(f"❌ **FN = {fn}** — Missed depression")
                st.write(f"✅ **TP = {tp}** — Caught depression")
                st.write("")
                if fn > 0:
                    st.error(f"Missed {fn} depressed student(s)! "
                             f"False Negative Rate: {fn/(fn+tp)*100:.1f}%")

# ══════════════════════════════════════════════════════════════
# MODEL AUTO-SELECTOR
# ══════════════════════════════════════════════════════════════
st.divider()
st.subheader("Predictor Auto Selector")
st.write(
    "Not sure which model to use? Fill in the student profile below and the system "
    "will **automatically recommend the best model** based on the student's characteristics "
    "and each model's strengths."
)

with st.container(border=True):
    st.markdown("**Student Profile**")
    as1, as2, as3 = st.columns(3)
    with as1:
        as_name    = st.text_input("Name", placeholder="e.g. Ahmad", key="as_name")
        as_gender  = st.selectbox("Gender", ["Female","Male"], key="as_gender")
        as_age     = st.slider("Age", 17, 30, 20, key="as_age")
    with as2:
        as_course  = st.selectbox("Course", [
            "Computer Science","Information Technology","Engineering",
            "Law","Psychology","Language","Islamic Studies",
            "Health Sciences","Business","Science & Math","Arts & Social","Others"],
            key="as_course")
        as_year    = st.selectbox("Year of Study",
            ["Year 1","Year 2","Year 3","Year 4"], key="as_year")
        as_cgpa    = st.selectbox("CGPA Range", [
            "0 - 1.99","2.00 - 2.49","2.50 - 2.99",
            "3.00 - 3.49","3.50 - 4.00"], key="as_cgpa")
    with as3:
        as_marital = st.selectbox("Marital Status",           ["No","Yes"], key="as_marital")
        as_anxiety = st.selectbox("Do you have Anxiety?",     ["No","Yes"], key="as_anxiety")
        as_panic   = st.selectbox("Do you have Panic Attack?",["No","Yes"], key="as_panic")

    as_btn = st.button("🔍  Find Best Model for This Student",
                       width='stretch', type="primary", key="as_btn")

if as_btn:
    as_name_lbl = as_name.strip() or "Student"
    CGPA_NUM = {'0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
                '3.00 - 3.49':3.25,'3.50 - 4.00':3.75}

    # ── Run all 3 models ──────────────────────────────────────
    _g  = 1 if as_gender  == "Male" else 0
    _ax = 1 if as_anxiety == "Yes"  else 0
    _pa = 1 if as_panic   == "Yes"  else 0
    _ma = 1 if as_marital == "Yes"  else 0
    _cn = CGPA_NUM.get(as_cgpa, 3.25)
    # Year and course encoding for SVM (same as KNN/DT)
    _ce = M['le_course'].transform([as_course])[0] if as_course in M['le_course'].classes_ else 0
    _ye = M['le_year'].transform([as_year])[0] if as_year   in M['le_year'].classes_ else 0

    # KNN
    _knn_inp   = pd.DataFrame([[_g,as_age,_ce,_ye,_cn,_ax,_pa]], columns=M['knn_feat'])
    _knn_inp_s = M['sc_knn'].transform(_knn_inp)
    _knn_pred  = int(M['knn'].predict(_knn_inp_s)[0])
    _knn_prob  = M['knn'].predict_proba(_knn_inp_s)[0]

    # DT
    _dt_inp  = pd.DataFrame([[_g,as_age,_ce,_ye,_cn,_ax,_pa,_ma]], columns=M['dt_feat'])
    _dt_pred = int(M['dt'].predict(_dt_inp)[0])
    _dt_prob = M['dt'].predict_proba(_dt_inp)[0]

    # SVM — same encoded feature set as KNN/DT
    _svm_inp2 = pd.DataFrame([[_g,as_age,_ce,_ye,_cn,_ax,_pa,_ma]],
                             columns=M['svm_col_order'])
    _svm_pred = int(M['svm'].predict(_svm_inp2)[0])
    _svm_prob = M['svm'].predict_proba(_svm_inp2)[0]

    # ── Scoring logic — find best model ───────────────────────
    # Score each model based on profile characteristics
    _scores = {'KNN': 0, 'Decision Tree': 0, 'SVM': 0}
    _reasons = {'KNN': [], 'Decision Tree': [], 'SVM': []}

    # Rule 1: All 3 models agree → highest confidence model wins
    _all_agree = (_knn_pred == _dt_pred == _svm_pred)
    if _all_agree:
        _scores['KNN']          += 3
        _scores['Decision Tree'] += 2
        _scores['SVM']           += 2
        _reasons['KNN'].append(f"All 3 models agree — KNN selected as highest accuracy model ({M['knn_m']['acc']:.2f}%)")

    # Rule 2: Marital status = Yes → DT is better (root feature)
    if as_marital == "Yes":
        _scores['Decision Tree'] += 3
        _reasons['Decision Tree'].append("Marital Status = Yes → DT's root feature, gives best insight")

    # Rule 3: Both anxiety + panic → KNN better (captures combined patterns)
    if as_anxiety == "Yes" and as_panic == "Yes":
        _scores['KNN'] += 2
        _scores['SVM'] += 2
        _reasons['KNN'].append("Both Anxiety + Panic present → KNN excels at multi-symptom patterns")
        _reasons['SVM'].append("Complex symptom combination → SVM handles non-linear boundaries well")

    # Rule 4: High CGPA + no symptoms → KNN better (similar student profiles)
    if _cn >= 3.0 and as_anxiety == "No" and as_panic == "No":
        _scores['KNN'] += 2
        _reasons['KNN'].append("High CGPA + no symptoms → KNN finds similar healthy student profiles")

    # Rule 5: Low CGPA academic risk → DT better (explicit rules)
    if _cn < 2.5:
        _scores['Decision Tree'] += 2
        _reasons['Decision Tree'].append("Low CGPA → DT's explicit CGPA threshold rules apply well")

    # Rule 6: Model confidence tiebreaker
    _knn_conf  = max(_knn_prob)
    _dt_conf   = max(_dt_prob)
    _svm_conf  = max(_svm_prob)
    _confs = {'KNN': _knn_conf, 'Decision Tree': _dt_conf, 'SVM': _svm_conf}
    _most_conf = max(_confs, key=_confs.get)
    _scores[_most_conf] += 1
    _reasons[_most_conf].append(f"Highest prediction confidence: {_confs[_most_conf]*100:.1f}%")

    # Determine winner
    _best = max(_scores, key=_scores.get)
    _best_pred = {'KNN': _knn_pred, 'Decision Tree': _dt_pred, 'SVM': _svm_pred}[_best]
    _best_prob = {'KNN': _knn_prob, 'Decision Tree': _dt_prob, 'SVM': _svm_prob}[_best]

    # ── Display Result ─────────────────────────────────────────
    st.divider()
    st.subheader(f"Auto-Selector Result for {as_name_lbl}")

    # Winner announcement
    _icons = {'KNN':'🔵','Decision Tree':'🌳','SVM':'🔴'}
    if _best_pred == 1:
        st.error(f"### {_icons[_best]}  Recommended: **{_best}** → Depression Risk Detected")
    else:
        st.success(f"### {_icons[_best]}  Recommended: **{_best}** → No Depression Detected")

    st.write("")

    # Score cards
    sc1, sc2, sc3 = st.columns(3)
    for col, model, icon, pred, prob, score in [
        (sc1, 'KNN',           '🔵', _knn_pred, _knn_prob, _scores['KNN']),
        (sc2, 'Decision Tree', '🌳', _dt_pred,  _dt_prob,  _scores['Decision Tree']),
        (sc3, 'SVM',           '🔴', _svm_pred, _svm_prob, _scores['SVM']),
    ]:
        with col:
            is_best = (model == _best)
            with st.container(border=True):
                if is_best:
                    st.markdown(f"### {icon} {model} ⭐ RECOMMENDED")
                else:
                    st.markdown(f"### {icon} {model}")
                st.metric("Auto-Selector Score", f"{score} pts")
                st.metric("Prediction",
                          "Depression" if pred==1 else "No Depression")
                st.metric("Confidence", f"{max(prob)*100:.1f}%")
                if _reasons[model]:
                    st.write("")
                    st.caption("**Why selected/scored:**")
                    for r in _reasons[model]:
                        st.caption(f"• {r}")

    st.write("")

    # Comparison summary
    _agree_models = sum([_knn_pred==_best_pred,
                         _dt_pred ==_best_pred,
                         _svm_pred==_best_pred])
    with st.container(border=True):
        st.markdown("**Summary**")
        st.write(f"**Recommended Model:** {_icons[_best]} {_best} "
                 f"(Score: {_scores[_best]} pts)")
        st.write(f"**Prediction:** "
                 f"{'⚠️ Depression Risk' if _best_pred==1 else '✅ No Depression'}")
        st.write(f"**Confidence:** {max(_best_prob)*100:.1f}%")
        st.write(f"**Model Agreement:** {_agree_models}/3 models predict the same result")
        if _agree_models == 3:
            st.success("All 3 models agree — high reliability prediction!")
        elif _agree_models == 2:
            st.info("2/3 models agree — moderate reliability.")
        else:
            st.warning("Models disagree — treat prediction with caution. "
                       "Consider running individual model pages for more details.")

        # ── Save results to session state so nav buttons can use them ──
        st.session_state['as_result'] = {
            'name'      : as_name_lbl,
            'gender'    : as_gender, 'age'    : as_age,
            'course'    : as_course, 'year'   : as_year, 'cgpa': as_cgpa,
            'marital'   : as_marital,'anxiety': as_anxiety,
            'panic'     : as_panic,
            'knn_pred'  : int(_knn_pred), 'knn_prob': _knn_prob.tolist(),
            'dt_pred'   : int(_dt_pred),  'dt_prob' : _dt_prob.tolist(),
            'svm_pred'  : int(_svm_pred), 'svm_prob': _svm_prob.tolist(),
            'svm_inp'   : _svm_inp2,
            'best'      : _best,
        }

        st.info("✅ Result saved — use the **Quick Navigate** banner below to open any predictor page with this result pre-loaded.")


# AUTO-SELECTOR RESULT — COMPACT QUICK NAVIGATE BANNER
# ══════════════════════════════════════════════════════════════
if st.session_state.get('as_result'):
    _ar        = st.session_state['as_result']
    _best_lbl  = _ar['best']
    _icons     = {'KNN':'🔵','Decision Tree':'🌳','SVM':'🔴'}
    _preds     = {'KNN':(_ar['knn_pred'],_ar['knn_prob']),
                  'Decision Tree':(_ar['dt_pred'],_ar['dt_prob']),
                  'SVM':(_ar['svm_pred'],_ar['svm_prob'])}
    _pages_map = {'KNN':'pages/2_KNN.py',
                  'Decision Tree':'pages/3_Decision_Tree.py',
                  'SVM':'pages/4_SVM.py'}
    _ckeys     = {'KNN':'knn_carry','Decision Tree':'dt_carry','SVM':'svm_carry'}
    _cdata     = {
        'KNN': {
            'pred':_ar['knn_pred'],'prob':_ar['knn_prob'],
            'name':_ar['name'],'gender':_ar['gender'],'age':_ar['age'],
            'course':_ar['course'],'year':_ar['year'],'cgpa':_ar['cgpa'],
            'anxiety':_ar['anxiety'],'panic':_ar['panic'],
            'high_concern':(_ar['anxiety']=="Yes" and _ar['panic']=="Yes"),
            'academic_risk':(_ar['year'] in ["Year 3","Year 4"] and _ar['cgpa']=="0 - 1.99"),
        },
        'Decision Tree': {
            'pred':_ar['dt_pred'],'prob':_ar['dt_prob'],
            'name':_ar['name'],'gender':_ar['gender'],'age':_ar['age'],
            'course':_ar['course'],'year':_ar['year'],'cgpa':_ar['cgpa'],
            'marital':_ar['marital'],'anxiety':_ar['anxiety'],'panic':_ar['panic'],
            'high_concern':(_ar['anxiety']=="Yes" and _ar['panic']=="Yes"),
            'married_risk':(_ar['marital']=="Yes" and _ar['age']<21),
            'academic_risk':(_ar['year'] in ["Year 3","Year 4"] and _ar['cgpa']=="0 - 1.99"),
        },
        'SVM': {
            'pred':_ar['svm_pred'],'prob':_ar['svm_prob'],
            'inp':_ar.get('svm_inp'),
            'name':_ar['name'],'gender':_ar['gender'],'age':_ar['age'],
            'course':_ar['course'],'year':_ar['year'],'cgpa':_ar['cgpa'],
            'marital':_ar['marital'],'anxiety':_ar['anxiety'],
            'panic':_ar['panic'],
            'high_concern':(_ar['anxiety']=="Yes" and _ar['panic']=="Yes"),
            'academic_risk':(_ar['year'] in ["Year 3","Year 4"] and _ar['cgpa']=="0 - 1.99"),
        },
    }

    st.divider()
    with st.container(border=True):
        _hcol, _xcol = st.columns([5, 1])
        with _hcol:
            st.markdown(f"#### ⚡ Quick Navigate — **{_ar['name']}**")
            st.caption("Result already computed — no re-prediction needed. Click **Open →** to view full details.")
        with _xcol:
            st.write("")
            if st.button("✖ Clear", key="ar_clear", width='stretch'):
                st.session_state['as_result'] = None
                st.rerun()

        st.write("")
        _qcols = st.columns(3)
        for _qcol, _model in zip(_qcols, ['KNN','Decision Tree','SVM']):
            _pred, _prob = _preds[_model]
            _is_best     = (_model == _best_lbl)
            _star        = " ⭐ BEST" if _is_best else ""
            _lbl         = "⚠️ Depression" if _pred==1 else "✅ No Depression"
            _conf        = f"{_prob[1]*100:.1f}%" if _pred==1 else f"{_prob[0]*100:.1f}%"
            with _qcol:
                with st.container(border=True):
                    st.markdown(f"**{_icons[_model]} {_model}**{_star}")
                    if _pred==1: st.error(f"{_lbl} · {_conf} confidence")
                    else:        st.success(f"{_lbl} · {_conf} confidence")
                    if st.button(f"Open {_model.split()[0]} →",
                                  key=f"qn_{_model.replace(' ','_')}",
                                  width='stretch',
                                  type="primary" if _is_best else "secondary"):
                        st.session_state[_ckeys[_model]] = _cdata[_model]
                        st.switch_page(_pages_map[_model])
st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")