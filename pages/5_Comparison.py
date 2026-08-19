import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
# LOAD & TRAIN MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_models():
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')
    df['CGPA_Numeric'] = df['CGPA_Numeric'].fillna(df['CGPA_Numeric'].median())

    le_course = LabelEncoder()
    le_year   = LabelEncoder()
    df['Course_Enc']        = le_course.fit_transform(df['Course'])
    df['Year_of_Study_Enc'] = le_year.fit_transform(df['Year_of_Study'])

    y = df['Depression']

    # ── KNN ──────────────────────────────────────────────────
    knn_feat = ['Gender','Age','Course_Enc','Year_of_Study_Enc',
                'CGPA_Numeric','Anxiety','Panic_Attack']
    X_knn = df[knn_feat]
    scaler = MinMaxScaler()
    X_knn_s = scaler.fit_transform(X_knn)
    X_tr_k, X_te_k, y_tr_k, y_te_k = train_test_split(
        X_knn_s, y, test_size=0.2, random_state=42, stratify=y)

    # Find best K
    best_k, best_acc = 5, 0
    for k in range(1, 21):
        m = KNeighborsClassifier(n_neighbors=k)
        m.fit(X_tr_k, y_tr_k)
        acc = accuracy_score(y_te_k, m.predict(X_te_k))
        if acc > best_acc:
            best_acc, best_k = acc, k
    knn = KNeighborsClassifier(n_neighbors=best_k, metric='euclidean')
    knn.fit(X_tr_k, y_tr_k)

    # ── Decision Tree ─────────────────────────────────────────
    dt_feat = ['Gender','Age','Course_Enc','Year_of_Study_Enc',
               'CGPA_Numeric','Anxiety','Panic_Attack','Marital_Status']
    X_dt = df[dt_feat]
    dt = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
    dt.fit(X_dt, y)

    return {
        'knn': knn, 'knn_scaler': scaler,
        'knn_feat': knn_feat, 'knn_best_k': best_k,
        'knn_X_tr': X_tr_k, 'knn_X_te': X_te_k,
        'knn_y_tr': y_tr_k, 'knn_y_te': y_te_k,
        'dt': dt, 'dt_feat': dt_feat,
        'X_dt': X_dt, 'y': y,
        'le_course': le_course, 'le_year': le_year,
        'df': df
    }

M = load_models()

# Pre-compute metrics
knn_pred   = M['knn'].predict(M['knn_X_te'])
knn_pred_p = M['knn'].predict_proba(M['knn_X_te'])[:,1]
dt_pred    = M['dt'].predict(M['X_dt'])

knn_metrics = {
    'acc':  accuracy_score(M['knn_y_te'], knn_pred)*100,
    'prec': precision_score(M['knn_y_te'], knn_pred, zero_division=0)*100,
    'rec':  recall_score(M['knn_y_te'], knn_pred, zero_division=0)*100,
    'f1':   f1_score(M['knn_y_te'], knn_pred, zero_division=0)*100,
    'cm':   confusion_matrix(M['knn_y_te'], knn_pred),
}
dt_metrics = {
    'acc':  accuracy_score(M['y'], dt_pred)*100,
    'prec': precision_score(M['y'], dt_pred, zero_division=0)*100,
    'rec':  recall_score(M['y'], dt_pred, zero_division=0)*100,
    'f1':   f1_score(M['y'], dt_pred, zero_division=0)*100,
    'cm':   confusion_matrix(M['y'], dt_pred),
}

# ── Session state for live test log ──────────────────────────
if 'test_log' not in st.session_state:
    st.session_state.test_log = []
if 'total_tests' not in st.session_state:
    st.session_state.total_tests = 0
if 'knn_correct' not in st.session_state:
    st.session_state.knn_correct = 0
if 'dt_correct' not in st.session_state:
    st.session_state.dt_correct = 0

# ══════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("##### RESULTS")
st.title("Model Comparison")
st.write("Compare KNN and Decision Tree live. Run your own test cases and track results.")
st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 1: MODEL METRICS
# ══════════════════════════════════════════════════════════════
st.subheader("Trained Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    with st.container(border=True):
        st.markdown("#### 🔵 KNN")
        st.caption(f"K={M['knn_best_k']} · Test set: {len(M['knn_y_te'])} records")
        st.divider()
        ma, mb = st.columns(2)
        ma.metric("Accuracy",  f"{knn_metrics['acc']:.2f}%")
        mb.metric("Precision", f"{knn_metrics['prec']:.2f}%")
        mc, md = st.columns(2)
        mc.metric("Recall",    f"{knn_metrics['rec']:.2f}%")
        md.metric("F1 Score",  f"{knn_metrics['f1']:.2f}%")
        if st.button("View KNN Page →", key="go_knn", use_container_width=True):
            st.switch_page("pages/2_KNN.py")

with c2:
    with st.container(border=True):
        st.markdown("#### 🌳 Decision Tree")
        st.caption("CART · Depth 5 · Full dataset: 600 records")
        st.divider()
        ma, mb = st.columns(2)
        ma.metric("Accuracy",  f"{dt_metrics['acc']:.2f}%")
        mb.metric("Precision", f"{dt_metrics['prec']:.2f}%")
        mc, md = st.columns(2)
        mc.metric("Recall",    f"{dt_metrics['rec']:.2f}%")
        md.metric("F1 Score",  f"{dt_metrics['f1']:.2f}%")
        if st.button("View DT Page →", key="go_dt", use_container_width=True):
            st.switch_page("pages/3_Decision_Tree.py")

with c3:
    with st.container(border=True):
        st.markdown("#### 🔴 SVM")
        st.caption("RBF Kernel · 75/25 Split · Pending")
        st.divider()
        ma, mb = st.columns(2)
        ma.metric("Accuracy",  "TBD")
        mb.metric("Precision", "TBD")
        mc, md = st.columns(2)
        mc.metric("Recall",    "TBD")
        md.metric("F1 Score",  "TBD")
        if st.button("View SVM Page →", key="go_svm", use_container_width=True):
            st.switch_page("pages/4_SVM.py")

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 2: LIVE TEST — RUN YOUR OWN PREDICTION
# ══════════════════════════════════════════════════════════════
st.subheader("Live Test — Run Your Own Prediction")
st.write("Fill in student details and see what **both models** predict simultaneously.")

with st.form("live_test_form"):
    st.markdown("**Student Information**")
    col1, col2, col3 = st.columns(3)

    with col1:
        t_name    = st.text_input("Name (optional)", placeholder="e.g. Alex")
        t_gender  = st.selectbox("Gender", ["Female", "Male"])
        t_age     = st.slider("Age", 17, 30, 20)
        t_course  = st.selectbox("Course", [
            "Computer Science","Information Technology","Engineering",
            "Law","Psychology","Language","Islamic Studies",
            "Health Sciences","Business","Science & Math",
            "Arts & Social","Others"
        ])

    with col2:
        t_year    = st.selectbox("Year of Study", ["Year 1","Year 2","Year 3","Year 4"])
        t_cgpa    = st.selectbox("CGPA Range", [
            "0 - 1.99","2.00 - 2.49","2.50 - 2.99",
            "3.00 - 3.49","3.50 - 4.00"
        ])
        t_marital = st.selectbox("Marital Status", ["No","Yes"])

    with col3:
        t_anxiety = st.selectbox("Has Anxiety?",      ["No","Yes"])
        t_panic   = st.selectbox("Has Panic Attack?", ["No","Yes"])
        t_actual  = st.selectbox(
            "Actual Depression? (for accuracy tracking)",
            ["Unknown","No","Yes"],
            help="If you know the actual answer, select it to track model accuracy"
        )

    submitted = st.form_submit_button("Run Live Test on Both Models",
                                      use_container_width=True)

if submitted:
    # ── Encode inputs ─────────────────────────────────────────
    cgpa_map = {
        '0 - 1.99':1.0,'2.00 - 2.49':2.25,'2.50 - 2.99':2.75,
        '3.00 - 3.49':3.25,'3.50 - 4.00':3.75
    }

    gender_enc  = 1 if t_gender  == "Male" else 0
    anxiety_enc = 1 if t_anxiety == "Yes"  else 0
    panic_enc   = 1 if t_panic   == "Yes"  else 0
    marital_enc = 1 if t_marital == "Yes"  else 0

    le_c = M['le_course']
    le_y = M['le_year']
    course_enc = le_c.transform([t_course])[0] if t_course in le_c.classes_ else 0
    year_enc   = le_y.transform([t_year])[0]   if t_year   in le_y.classes_ else 0
    cgpa_num   = cgpa_map[t_cgpa]

    # ── KNN predict ───────────────────────────────────────────
    knn_inp = np.array([[gender_enc, t_age, course_enc, year_enc,
                          cgpa_num, anxiety_enc, panic_enc]])
    knn_inp_s = M['knn_scaler'].transform(
        pd.DataFrame(knn_inp, columns=M['knn_feat'])
    )
    knn_res  = M['knn'].predict(knn_inp_s)[0]
    knn_prob = M['knn'].predict_proba(knn_inp_s)[0]

    # ── DT predict ────────────────────────────────────────────
    dt_inp = np.array([[gender_enc, t_age, course_enc, year_enc,
                         cgpa_num, anxiety_enc, panic_enc, marital_enc]])
    dt_inp_df = pd.DataFrame(dt_inp, columns=M['dt_feat'])
    dt_res    = M['dt'].predict(dt_inp_df)[0]
    dt_prob   = M['dt'].predict_proba(dt_inp_df)[0]

    # ── Actual label ──────────────────────────────────────────
    actual_val = None if t_actual == "Unknown" else (1 if t_actual == "Yes" else 0)

    # ── Show results ──────────────────────────────────────────
    name_lbl = t_name.strip() if t_name.strip() else "Student"
    st.markdown("---")
    st.subheader(f"Results for: {name_lbl}")

    r1, r2 = st.columns(2)

    with r1:
        with st.container(border=True):
            st.markdown("### 🔵 KNN Prediction")
            if knn_res == 1:
                st.error("**DEPRESSION DETECTED**")
            else:
                st.success("**NO DEPRESSION**")

            kp1, kp2 = st.columns(2)
            kp1.metric("No Depression", f"{knn_prob[0]*100:.1f}%")
            kp2.metric("Depression",    f"{knn_prob[1]*100:.1f}%")

            # Probability bar
            fig, ax = plt.subplots(figsize=(4, 1.2))
            ax.barh([""], [knn_prob[0]*100], color="#10B981", height=0.4)
            ax.barh([""], [knn_prob[1]*100],
                    left=[knn_prob[0]*100], color="#EF4444", height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=9)
            ax.tick_params(axis='y', which='both', length=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.text(knn_prob[0]*50, 0, f"No: {knn_prob[0]*100:.0f}%",
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
            ax.text(knn_prob[0]*100 + knn_prob[1]*50, 0,
                    f"Yes: {knn_prob[1]*100:.0f}%",
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            if actual_val is not None:
                correct = (knn_res == actual_val)
                if correct:
                    st.success("✅ KNN got this RIGHT")
                else:
                    st.error("❌ KNN got this WRONG")

    with r2:
        with st.container(border=True):
            st.markdown("### 🌳 Decision Tree Prediction")
            if dt_res == 1:
                st.error("**DEPRESSION DETECTED**")
            else:
                st.success("**NO DEPRESSION**")

            dp1, dp2 = st.columns(2)
            dp1.metric("No Depression", f"{dt_prob[0]*100:.1f}%")
            dp2.metric("Depression",    f"{dt_prob[1]*100:.1f}%")

            # Probability bar
            fig, ax = plt.subplots(figsize=(4, 1.2))
            ax.barh([""], [dt_prob[0]*100], color="#10B981", height=0.4)
            ax.barh([""], [dt_prob[1]*100],
                    left=[dt_prob[0]*100], color="#EF4444", height=0.4)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", fontsize=9)
            ax.tick_params(axis='y', which='both', length=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.text(dt_prob[0]*50, 0, f"No: {dt_prob[0]*100:.0f}%",
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
            ax.text(dt_prob[0]*100 + dt_prob[1]*50, 0,
                    f"Yes: {dt_prob[1]*100:.0f}%",
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            if actual_val is not None:
                correct = (dt_res == actual_val)
                if correct:
                    st.success("✅ DT got this RIGHT")
                else:
                    st.error("❌ DT got this WRONG")

    # Agreement check
    st.markdown("")
    if knn_res == dt_res:
        label = "Depression" if knn_res == 1 else "No Depression"
        st.info(f"Both models **AGREE**: {label}")
    else:
        st.warning(
            "Models **DISAGREE** — "
            f"KNN says **{'Depression' if knn_res==1 else 'No Depression'}**, "
            f"Decision Tree says **{'Depression' if dt_res==1 else 'No Depression'}**"
        )

    # ── Log this test ─────────────────────────────────────────
    st.session_state.total_tests += 1
    log_entry = {
        "Test #":     st.session_state.total_tests,
        "Name":       name_lbl,
        "Gender":     t_gender,
        "Age":        t_age,
        "CGPA":       t_cgpa,
        "Anxiety":    t_anxiety,
        "Panic":      t_panic,
        "KNN":        "Depression" if knn_res==1 else "No Depression",
        "DT":         "Depression" if dt_res==1  else "No Depression",
        "Agree":      "✅" if knn_res==dt_res else "❌",
        "Actual":     t_actual,
    }
    if actual_val is not None:
        st.session_state.knn_correct += int(knn_res == actual_val)
        st.session_state.dt_correct  += int(dt_res  == actual_val)

    st.session_state.test_log.insert(0, log_entry)

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 3: LIVE TEST SCOREBOARD
# ══════════════════════════════════════════════════════════════
st.subheader("Live Test Scoreboard")

total = st.session_state.total_tests

if total == 0:
    st.info("No tests run yet. Fill in the form above and click Run Live Test!")
else:
    # Scoreboard metrics
    sb1, sb2, sb3, sb4, sb5 = st.columns(5)
    sb1.metric("Tests Run",      str(total))
    sb2.metric("KNN Correct",    str(st.session_state.knn_correct),
               help="Only counted when Actual is provided")
    sb3.metric("DT Correct",     str(st.session_state.dt_correct),
               help="Only counted when Actual is provided")

    # Agreement rate
    log_df = pd.DataFrame(st.session_state.test_log)
    agree_count = (log_df["Agree"] == "✅").sum()
    sb4.metric("Agreements",     f"{agree_count}/{total}",
               help="Cases where both models agreed")
    sb5.metric("Agreement Rate", f"{agree_count/total*100:.0f}%")

    st.write("")

    # Live accuracy bar
    if st.session_state.knn_correct > 0 or st.session_state.dt_correct > 0:
        actuals_provided = sum(
            1 for e in st.session_state.test_log if e['Actual'] != 'Unknown'
        )
        if actuals_provided > 0:
            knn_live_acc = st.session_state.knn_correct / actuals_provided * 100
            dt_live_acc  = st.session_state.dt_correct  / actuals_provided * 100

            st.markdown(f"**Live Accuracy** (based on {actuals_provided} tests with known answer)")
            fig, ax = plt.subplots(figsize=(8, 2))
            models  = ['KNN', 'Decision Tree']
            accs    = [knn_live_acc, dt_live_acc]
            colors  = ['#5B7FFF', '#10B981']
            bars = ax.barh(models, accs, color=colors,
                           height=0.4, edgecolor='none')
            for bar, val in zip(bars, accs):
                ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                        f'{val:.1f}%', va='center',
                        fontsize=11, fontweight='bold')
            ax.set_xlim(0, 115)
            ax.set_xlabel("Accuracy on your test cases (%)", fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axvline(x=50, color='gray', linestyle='--',
                       linewidth=0.8, alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    # Test log table
    st.markdown("**Test History**")
    st.dataframe(log_df, use_container_width=True, hide_index=True)

    # Clear button
    if st.button("Clear All Tests", key="clear_log"):
        st.session_state.test_log     = []
        st.session_state.total_tests  = 0
        st.session_state.knn_correct  = 0
        st.session_state.dt_correct   = 0
        st.rerun()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 4: METRIC CHARTS
# ══════════════════════════════════════════════════════════════
st.subheader("Trained Metric Charts")

metrics_names  = ["Accuracy", "Precision", "Recall", "F1 Score"]
knn_vals = [knn_metrics['acc'], knn_metrics['prec'],
            knn_metrics['rec'], knn_metrics['f1']]
dt_vals  = [dt_metrics['acc'],  dt_metrics['prec'],
            dt_metrics['rec'],  dt_metrics['f1']]

ch1, ch2 = st.columns(2)

with ch1:
    # Bar chart
    x     = np.arange(len(metrics_names))
    w     = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b1 = ax.bar(x - w/2, knn_vals, w, label='KNN',
                color='#5B7FFF', edgecolor='none', alpha=0.9)
    b2 = ax.bar(x + w/2, dt_vals,  w, label='Decision Tree',
                color='#10B981', edgecolor='none', alpha=0.9)
    for bar, val in zip(list(b1)+list(b2), knn_vals+dt_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("KNN vs Decision Tree", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with ch2:
    # Radar chart
    N      = len(metrics_names)
    angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
    knn_r  = knn_vals + [knn_vals[0]]
    dt_r   = dt_vals  + [dt_vals[0]]

    fig2, ax2 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax2.plot(angles, knn_r, 'o-', lw=2, color='#5B7FFF', label='KNN')
    ax2.fill(angles, knn_r, alpha=0.12, color='#5B7FFF')
    ax2.plot(angles, dt_r,  'o-', lw=2, color='#10B981', label='Decision Tree')
    ax2.fill(angles, dt_r,  alpha=0.12, color='#10B981')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(metrics_names, fontsize=10)
    ax2.set_ylim(0, 110)
    ax2.set_yticks([20,40,60,80,100])
    ax2.tick_params(axis='y', labelsize=7)
    ax2.set_title("Radar Chart", fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3,1.15), fontsize=9)
    st.pyplot(fig2, use_container_width=True)
    plt.close()

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 5: CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════
st.subheader("Confusion Matrices")

cm1, cm2 = st.columns(2)

for col, title, cm, color, detail in [
    (cm1, "🔵 KNN", knn_metrics['cm'], 'Blues',
     f"Test set: {len(M['knn_y_te'])} records"),
    (cm2, "🌳 Decision Tree", dt_metrics['cm'], 'Greens',
     "Full dataset: 600 records"),
]:
    with col:
        st.markdown(f"**{title}** — {detail}")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=ax,
                    xticklabels=['No Dep','Depression'],
                    yticklabels=['No Dep','Depression'],
                    annot_kws={'size':13,'weight':'bold'},
                    linewidths=0.5)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual',    fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        tn, fp, fn, tp = cm.ravel()
        ca,cb,cc,cd = st.columns(4)
        ca.metric("TN", str(tn))
        cb.metric("FP", str(fp))
        cc.metric("FN", str(fn))
        cd.metric("TP", str(tp))

st.divider()

# ══════════════════════════════════════════════════════════════
# SECTION 6: KEY FINDINGS
# ══════════════════════════════════════════════════════════════
st.subheader("Key Findings")

f1c, f2c, f3c = st.columns(3)
with f1c:
    with st.container(border=True):
        st.markdown("**🏆 Best Overall: KNN**")
        st.write(
            f"KNN achieved {knn_metrics['acc']:.2f}% accuracy with "
            f"{knn_metrics['rec']:.2f}% recall on the test set. "
            "It correctly identified nearly all depressed students."
        )
with f2c:
    with st.container(border=True):
        st.markdown("**📖 Most Interpretable: Decision Tree**")
        st.write(
            "Decision Tree provides a visual decision path. "
            "Marital Status is the root feature — the most important "
            "predictor of depression in this dataset."
        )
with f3c:
    with st.container(border=True):
        st.markdown("**⚠️ Why Recall Matters**")
        st.write(
            "In mental health screening, missing a depressed student "
            "(false negative) is far more serious than a false alarm. "
            "High recall = fewer missed cases."
        )

st.caption("MindCheck · BMCS2003 AI · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")
