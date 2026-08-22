import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, sys, os
warnings.filterwarnings('ignore')

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

    # ── SVM — Target: Panic Attack ───────────────────────────
    svm_feat = ['Gender','Age','Course_Enc','Year_of_Study_Enc',
                'CGPA_Numeric','Anxiety','Marital_Status','Seek_Treatment']
    X_s = df[svm_feat]; y_s = df['Panic_Attack']
    sc_svm = StandardScaler()
    X_ss   = sc_svm.fit_transform(X_s)
    Xtr_s, Xte_s, ytr_s, yte_s = train_test_split(
        X_ss, y_s, test_size=0.25, random_state=42, stratify=y_s)
    svm = SVC(kernel='rbf', probability=True, random_state=42)
    svm.fit(Xtr_s, ytr_s)
    svm_p = svm.predict(Xte_s)

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
for k, v in [('log',[]),('n',0),('knn_c',0),('dt_c',0),('svm_c',0)]:
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
# SECTION 1 — SCORE CARDS WITH NAVIGATION
# ══════════════════════════════════════════════════════════════
c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("### 🔵 KNN")
        st.caption(f"Ho Jun Yon · K={M['best_k']} · Target: Depression")
        st.divider()
        # Accuracy big
        st.metric("Accuracy",  f"{M['knn_m']['acc']:.2f}%")
        ma, mb = st.columns(2)
        ma.metric("Precision", f"{M['knn_m']['prec']:.2f}%")
        mb.metric("Recall",    f"{M['knn_m']['rec']:.2f}%")
        mc, md = st.columns(2)
        mc.metric("F1 Score",  f"{M['knn_m']['f1']:.2f}%")
        md.metric("Split",     "80 / 20")
        st.divider()
        if st.button("Open KNN Page →", key="go_knn", use_container_width=True):
            st.switch_page("pages/2_KNN.py")

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
        st.divider()
        if st.button("Open Decision Tree Page →", key="go_dt", use_container_width=True):
            st.switch_page("pages/3_Decision_Tree.py")

with c3:
    with st.container(border=True):
        st.markdown("### 🔴 SVM")
        st.caption("Chiang Jun Hang · RBF · Target: Panic Attack")
        st.divider()
        st.metric("Accuracy",  f"{M['svm_m']['acc']:.2f}%")
        ma, mb = st.columns(2)
        ma.metric("Precision", f"{M['svm_m']['prec']:.2f}%")
        mb.metric("Recall",    f"{M['svm_m']['rec']:.2f}%")
        mc, md = st.columns(2)
        mc.metric("F1 Score",  f"{M['svm_m']['f1']:.2f}%")
        md.metric("Split",     "75 / 25")
        st.divider()
        if st.button("Open SVM Page →", key="go_svm", use_container_width=True):
            st.switch_page("pages/4_SVM.py")

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
        t_seek    = st.selectbox("Seek Treatment?", ["No","Yes"])

    with fc:
        t_anxiety = st.selectbox("Has Anxiety?",      ["No","Yes"])
        t_panic   = st.selectbox("Has Panic Attack?", ["No","Yes"])
        st.markdown("**Actual answers (for tracking)**")
        t_actual_dep   = st.selectbox("Actual Depression?",
                                      ["Unknown","No","Yes"])
        t_actual_panic = st.selectbox("Actual Panic Attack?",
                                      ["Unknown","No","Yes"])

    run = st.form_submit_button("Run All 3 Models Now",
                                use_container_width=True)

if run:
    le_c = M['le_course']; le_y = M['le_year']
    g  = 1 if t_gender  == "Male" else 0
    ax = 1 if t_anxiety == "Yes"  else 0
    pa = 1 if t_panic   == "Yes"  else 0
    ma = 1 if t_marital == "Yes"  else 0
    sk = 1 if t_seek    == "Yes"  else 0
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

    # ── SVM ──────────────────────────────────────────────────
    svm_in = pd.DataFrame(
        [[g, t_age, ce, ye, cg, ax, ma, sk]],
        columns=M['svm_feat'])
    svm_in_s = M['sc_svm'].transform(svm_in)
    svm_pred  = M['svm'].predict(svm_in_s)[0]
    svm_prob  = M['svm'].predict_proba(svm_in_s)[0]

    # ── Actual labels ─────────────────────────────────────────
    act_dep   = None if t_actual_dep   == "Unknown" else (1 if t_actual_dep   == "Yes" else 0)
    act_panic = None if t_actual_panic == "Unknown" else (1 if t_actual_panic == "Yes" else 0)

    name = t_name.strip() or "Student"

    # ── Display results ───────────────────────────────────────
    st.markdown("---")
    st.subheader(f"Prediction Results — {name}")

    rc1, rc2, rc3 = st.columns(3)

    def show_result(col, icon, title, member, target, pred, prob,
                    act, label_pos, label_neg):
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

                # Stacked bar
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
                st.pyplot(fig, use_container_width=True)
                plt.close()

                if act is not None:
                    correct = (pred == act)
                    st.success("✅ Correct") if correct else st.error("❌ Wrong")
                    return correct
                return None

    knn_c = show_result(rc1,"🔵","KNN","Ho Jun Yon","Depression",
                        knn_pred,knn_prob,act_dep,"Depression","No Dep")
    dt_c  = show_result(rc2,"🌳","Decision Tree","Irvin","Depression",
                        dt_pred,dt_prob,act_dep,"Depression","No Dep")
    svm_c = show_result(rc3,"🔴","SVM","Chiang Jun Hang","Panic Attack",
                        svm_pred,svm_prob,act_panic,"Panic Attack","No Panic")

    # Agreement line
    dep_agree  = (knn_pred == dt_pred)
    dep_label  = "Depression" if knn_pred == 1 else "No Depression"
    svm_label  = "Panic Attack" if svm_pred == 1 else "No Panic Attack"

    st.markdown("")
    a1, a2 = st.columns(2)
    with a1:
        if dep_agree:
            st.info(f"KNN & DT **AGREE** on Depression: **{dep_label}**")
        else:
            st.warning(
                f"KNN & DT **DISAGREE** — "
                f"KNN: **{'Depression' if knn_pred==1 else 'No Depression'}** | "
                f"DT: **{'Depression' if dt_pred==1 else 'No Depression'}**"
            )
    with a2:
        st.info(f"SVM predicts Panic Attack: **{svm_label}**")

    # ── Update session state ──────────────────────────────────
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
        "SVM (Panic)": "Panic" if svm_pred==1 else "No",
        "KNN%"      : f"{knn_prob[1]*100:.0f}%",
        "DT%"       : f"{dt_prob[1]*100:.0f}%",
        "SVM%"      : f"{svm_prob[1]*100:.0f}%",
        "Agree KNN-DT": "✅" if dep_agree else "❌",
        "Actual Dep"  : t_actual_dep,
        "Actual Panic": t_actual_panic,
    })

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
                 help="Panic Attack — only counted when Actual provided")
    log_df = pd.DataFrame(st.session_state['log'])
    agree_n = (log_df["Agree KNN-DT"] == "✅").sum()
    sb[4].metric("KNN-DT Agree", f"{agree_n}/{n}")

    # Live accuracy chart (if actuals provided)
    actuals_dep   = sum(1 for e in st.session_state['log'] if e['Actual Dep']   != 'Unknown')
    actuals_panic = sum(1 for e in st.session_state['log'] if e['Actual Panic'] != 'Unknown')

    if actuals_dep > 0 or actuals_panic > 0:
        st.write("")
        st.markdown("**Live Accuracy on Your Test Cases**")
        fig, ax = plt.subplots(figsize=(8, 2.2))
        models, accs, colors = [], [], []
        if actuals_dep > 0:
            models += ['KNN (Dep)', 'DT (Dep)']
            accs   += [st.session_state['knn_c']/actuals_dep*100,
                       st.session_state['dt_c'] /actuals_dep*100]
            colors += ['#5B7FFF','#10B981']
        if actuals_panic > 0:
            models += ['SVM (Panic)']
            accs   += [st.session_state['svm_c']/actuals_panic*100]
            colors += ['#EF4444']
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
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("**Test History**")
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    if st.button("Clear All Tests", key="clear"):
        for k in ['log','n','knn_c','dt_c','svm_c']:
            st.session_state[k] = [] if k=='log' else 0
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4 — TRAINED METRIC CHARTS
# ══════════════════════════════════════════════════════════════
st.subheader("Trained Metric Charts")

metric_names = ["Accuracy","Precision","Recall","F1 Score"]
knn_v = [M['knn_m']['acc'], M['knn_m']['prec'], M['knn_m']['rec'], M['knn_m']['f1']]
dt_v  = [M['dt_m']['acc'],  M['dt_m']['prec'],  M['dt_m']['rec'],  M['dt_m']['f1']]
svm_v = [M['svm_m']['acc'], M['svm_m']['prec'], M['svm_m']['rec'], M['svm_m']['f1']]

ch1, ch2 = st.columns(2)

with ch1:
    x = np.arange(len(metric_names)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x-w,   knn_v, w, label='KNN',           color='#5B7FFF', alpha=0.9)
    b2 = ax.bar(x,     dt_v,  w, label='Decision Tree', color='#10B981', alpha=0.9)
    b3 = ax.bar(x+w,   svm_v, w, label='SVM',           color='#EF4444', alpha=0.9)
    for bars, vals in [(b1,knn_v),(b2,dt_v),(b3,svm_v)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                    f'{val:.1f}', ha='center', fontsize=7.5, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylim(0, 118); ax.set_ylabel("Score (%)")
    ax.set_title("All 3 Models — Metric Comparison", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

with ch2:
    N = len(metric_names)
    angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
    fig2, ax2 = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    for vals, color, label in [
        (knn_v,'#5B7FFF','KNN'),
        (dt_v, '#10B981','Decision Tree'),
        (svm_v,'#EF4444','SVM'),
    ]:
        r = vals + [vals[0]]
        ax2.plot(angles, r, 'o-', lw=2, color=color, label=label)
        ax2.fill(angles, r, alpha=0.08, color=color)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metric_names, fontsize=10)
    ax2.set_ylim(0, 110); ax2.set_yticks([20,40,60,80,100])
    ax2.tick_params(axis='y', labelsize=7)
    ax2.set_title("Radar Chart — All 3 Models", fontsize=12,
                  fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.35,1.15), fontsize=9)
    st.pyplot(fig2, use_container_width=True); plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5 — CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════
st.subheader("Confusion Matrices")
cm1, cm2, cm3 = st.columns(3)

for col, title, cm, color, detail in [
    (cm1,"🔵 KNN",          M['knn_m']['cm'],'Blues',  "Test set (80/20)"),
    (cm2,"🌳 Decision Tree", M['dt_m']['cm'], 'Greens', "Test set (70/30)"),
    (cm3,"🔴 SVM",           M['svm_m']['cm'],'Reds',   "Test set (75/25)"),
]:
    with col:
        st.markdown(f"**{title}** · {detail}")
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                    xticklabels=['No','Yes'], yticklabels=['No','Yes'],
                    annot_kws={'size':12,'weight':'bold'}, linewidths=0.5)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual',    fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()
        tn,fp,fn,tp = cm.ravel()
        a,b,c,d = st.columns(4)
        a.metric("TN",str(tn)); b.metric("FP",str(fp))
        c.metric("FN",str(fn)); d.metric("TP",str(tp))

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")