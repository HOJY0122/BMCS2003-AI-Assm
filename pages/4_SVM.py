import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, sys, os
warnings.filterwarnings('ignore')

from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import sidebar

st.set_page_config(
    page_title="SVM — MindCheck",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)
sidebar("svm")

# ══════════════════════════════════════════════════════════════
# LOAD & TRAIN SVM (Chiang Jun Hang's approach)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def train_svm():
    df_raw = pd.read_csv('dataset/Student_Mental_health.csv')
    df_raw.columns = df_raw.columns.str.strip()
    df_raw['Age'] = df_raw['Age'].fillna(df_raw['Age'].median())
    df_raw['Your current year of Study'] = (
        df_raw['Your current year of Study'].str.strip().str.lower()
    )
    df_raw['What is your CGPA?'] = df_raw['What is your CGPA?'].str.strip()

    def categorize_course(c_str):
        c = str(c_str).lower()
        if any(x in c for x in ['technology','it','computer','cs',
                                  'system','is','software','se',
                                  'bit','bcs','cts']):
            return 'STEM/IT'
        return 'Other'

    df_raw['Course_Category'] = df_raw['What is your course?'].apply(categorize_course)
    df_raw = df_raw.drop(columns=['Timestamp','What is your course?'], errors='ignore')

    # Features & target (Depression — same as original)
    X = df_raw.drop(columns=['Do you have Depression?'])
    y = (df_raw['Do you have Depression?'] == 'Yes').astype(int)

    # Build pipeline (OrdinalEncoder + StandardScaler + SVC)
    pipe = Pipeline([
        ('enc', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ('scl', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42))
    ])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    metrics = {
        'acc' : accuracy_score(y_te, y_pred)*100,
        'prec': precision_score(y_te, y_pred, zero_division=0)*100,
        'rec' : recall_score(y_te, y_pred, zero_division=0)*100,
        'f1'  : f1_score(y_te, y_pred, zero_division=0)*100,
        'cm'  : confusion_matrix(y_te, y_pred),
        'report': classification_report(y_te, y_pred,
                    target_names=['No Depression','Depression'],
                    output_dict=True),
    }

    return pipe, X, y, X_te, y_te, y_pred, metrics, df_raw, categorize_course

with st.spinner("Training SVM model..."):
    pipe, X_all, y_all, X_te, y_te, y_pred, metrics, df_raw, categorize_course = train_svm()

prep_stage = pipe.named_steps['enc']

# ── Page Header ────────────────────────────────────────────────
st.markdown("##### SVM — SUPPORT VECTOR MACHINE")
st.title("SVM - Depression Prediction")
st.write("**Member 3: Chiang Jun Hang** | Algorithm: SVC (RBF Kernel) | Target: Depression")
st.divider()

# ══════════════════════════════════════════════════════════════
# STEP 1: ALGORITHM OVERVIEW
# ══════════════════════════════════════════════════════════════
st.subheader("Step 1: Algorithm Overview")

c1, c2, c3 = st.columns(3)
c1.info("**Algorithm:** Support Vector Machine (SVC)\n\n**Kernel:** RBF (Radial Basis Function)")
c2.info("**Preprocessing:** OrdinalEncoder + StandardScaler\n\n**Class Weight:** Balanced")
c3.info(f"**Train Size:** {len(X_all) - len(X_te)} (75%)\n\n**Test Size:** {len(X_te)} (25%)")

st.write("")
with st.expander("What is SVM?"):
    st.write(
        "Support Vector Machine (SVM) finds the **optimal hyperplane** that "
        "maximally separates two classes in high-dimensional feature space. "
        "The distance from the hyperplane to the nearest data points of each "
        "class (called support vectors) is maximized — this is the margin. "
        "The **RBF kernel** maps features into a higher-dimensional space to "
        "handle non-linearly separable data. The PCA visualization below shows "
        "the decision boundary projected into 2D using Principal Component Analysis."
    )

st.divider()

# ══════════════════════════════════════════════════════════════
# STEP 2: MODEL EVALUATION
# ══════════════════════════════════════════════════════════════
st.subheader("Step 2: Model Evaluation")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{metrics['acc']:.2f}%")
m2.metric("Precision", f"{metrics['prec']:.2f}%")
m3.metric("Recall",    f"{metrics['rec']:.2f}%")
m4.metric("F1 Score",  f"{metrics['f1']:.2f}%")

st.write("")
col_cm, col_cr = st.columns(2)

with col_cm:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(metrics['cm'], annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=['No Depression','Depression'],
                yticklabels=['No Depression','Depression'],
                linewidths=0.5, annot_kws={'size':12,'weight':'bold'})
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual',    fontsize=10)
    ax.set_title('SVM Confusion Matrix', fontsize=11, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    tn, fp, fn, tp = metrics['cm'].ravel()
    a, b, c, d = st.columns(4)
    a.metric("TN", str(tn)); b.metric("FP", str(fp))
    c.metric("FN", str(fn)); d.metric("TP", str(tp))

with col_cr:
    st.markdown("**Classification Report**")
    report_df = pd.DataFrame(metrics['report']).transpose()
    st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)

    st.write("")
    st.info(
        "The SVM with RBF kernel and balanced class weights effectively handles "
        "the class imbalance in the dataset. High recall means fewer depressed "
        "students are missed — which is critical for mental health screening."
    )

st.divider()

# ══════════════════════════════════════════════════════════════
# STEP 3: LIVE PREDICTION + SVM BOUNDARY
# ══════════════════════════════════════════════════════════════
st.subheader("Step 3: Student Depression Prediction")
st.write("Fill in the student details below to predict depression status using SVM.")

with st.form("svm_form"):
    st.markdown("**Student Information**")
    col_a, col_b = st.columns(2)

    with col_a:
        t_name    = st.text_input("Name (optional)")
        t_age     = st.slider("Age", 15, 40, 20)
        t_gender  = st.selectbox("Gender", ["Male", "Female"])
        t_year    = st.selectbox("Year of Study",
                                 ["Year 1","Year 2","Year 3","Year 4"])
        t_cgpa    = st.selectbox("CGPA", [
                       "0 - 1.99","2.00 - 2.49","2.50 - 2.99",
                       "3.00 - 3.49","3.50 - 4.00"])

    with col_b:
        t_course  = st.selectbox("Course Field", [
                       "Information Technology (IT)",
                       "Computer Science (CS)",
                       "Information System (IS)",
                       "Software Engineering (SE)",
                       "Other"])
        t_marital = st.selectbox("Marital Status",                     ["No","Yes"])
        t_anxiety = st.selectbox("Do you have Anxiety?",               ["No","Yes"])
        t_panic   = st.selectbox("Do you have Panic Attack?",          ["No","Yes"])
        t_treat   = st.selectbox("Did you seek Specialist Treatment?", ["No","Yes"])

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    name = t_name.strip() or "Student"
    year_num = t_year.split()[-1]

    input_data = pd.DataFrame([{
        'Age'                                        : t_age,
        'Choose your gender'                         : t_gender,
        'Your current year of Study'                 : f'year {year_num}',
        'What is your CGPA?'                         : t_cgpa,
        'Marital status'                             : t_marital,
        'Do you have Anxiety?'                       : t_anxiety,
        'Do you have Panic attack?'                  : t_panic,
        'Did you seek any specialist for a treatment?': t_treat,
        'Course_Category'                            : categorize_course(t_course),
    }])

    try:
        prediction = pipe.predict(input_data)[0]
        prob       = pipe.predict_proba(input_data)[0]

        st.markdown("---")
        st.subheader(f"Prediction Result — {name}")

        res1, res2 = st.columns(2)

        with res1:
            if prediction == 1:
                st.error(f"**Result for {name}: DEPRESSION DETECTED**\n\n"
                         "The SVM model predicts this student may have depression. "
                         "Please consider seeking professional support.")
            else:
                st.success(f"**Result for {name}: NO DEPRESSION**\n\n"
                           "The SVM model predicts this student does not show "
                           "signs of depression. Keep maintaining a healthy lifestyle!")

            pa, pb = st.columns(2)
            pa.metric("No Depression", f"{prob[0]*100:.1f}%")
            pb.metric("Depression",    f"{prob[1]*100:.1f}%")

            # Prob bar
            fig, ax = plt.subplots(figsize=(4, 1))
            ax.barh([""], [prob[0]*100], color="#10B981", height=0.5)
            ax.barh([""], [prob[1]*100], left=[prob[0]*100],
                    color="#EF4444", height=0.5)
            ax.set_xlim(0, 100); ax.axis('off')
            if prob[0] > 0.15:
                ax.text(prob[0]*50, 0, f"No: {prob[0]*100:.0f}%",
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
            if prob[1] > 0.15:
                ax.text(prob[0]*100+prob[1]*50, 0,
                        f"Yes: {prob[1]*100:.0f}%",
                        ha='center', va='center', fontsize=9,
                        color='white', fontweight='bold')
            plt.tight_layout(pad=0)
            st.pyplot(fig, use_container_width=True); plt.close()

        with res2:
            # Input summary
            st.markdown("**Input Summary**")
            summary = {
                'Name': name, 'Age': t_age, 'Gender': t_gender,
                'Year': t_year, 'CGPA': t_cgpa, 'Course': t_course,
                'Marital': t_marital, 'Anxiety': t_anxiety,
                'Panic': t_panic, 'Treatment': t_treat,
            }
            st.table(pd.DataFrame(summary, index=['Value']).T)

        # ── SVM Decision Boundary (Chiang's PCA visualization) ──
        st.write("")
        st.markdown("**Live SVM Decision Boundary (PCA Projection)**")
        st.caption("Your input plotted against the dataset in 2D PCA space with the SVM hyperplane.")

        try:
            X_bg_proc   = prep_stage.transform(X_all)
            X_user_proc = prep_stage.transform(input_data)

            scaler_2d   = StandardScaler()
            X_bg_scaled = scaler_2d.fit_transform(X_bg_proc)
            X_user_scaled = scaler_2d.transform(X_user_proc)

            pca = PCA(n_components=2, random_state=42)
            X_bg_2d   = pca.fit_transform(X_bg_scaled)
            X_user_2d = pca.transform(X_user_scaled)

            svm_2d = LinearSVC(C=1.0, class_weight='balanced',
                                random_state=42, max_iter=5000)
            svm_2d.fit(X_bg_2d, y_all)

            fig2, ax2 = plt.subplots(figsize=(8, 5), facecolor='white')
            ax2.set_facecolor('white')

            ax2.scatter(X_bg_2d[y_all==1, 0], X_bg_2d[y_all==1, 1],
                        color='#EF4444', s=40, alpha=0.55,
                        label='Depression Data', zorder=3)
            ax2.scatter(X_bg_2d[y_all==0, 0], X_bg_2d[y_all==0, 1],
                        color='#3B82F6', s=40, alpha=0.55,
                        label='No Depression Data', zorder=3)
            ax2.scatter(X_user_2d[0,0], X_user_2d[0,1],
                        color='black', s=280, marker='*',
                        label=f'Your Input: {name}', zorder=6)

            x_min = X_bg_2d[:,0].min()-1; x_max = X_bg_2d[:,0].max()+1
            y_min = X_bg_2d[:,1].min()-1; y_max = X_bg_2d[:,1].max()+1
            x_min = min(x_min, X_user_2d[0,0]-1)
            x_max = max(x_max, X_user_2d[0,0]+1)

            w = svm_2d.coef_[0]; b = svm_2d.intercept_[0]
            x_pts = np.linspace(x_min, x_max, 200)
            if w[1] != 0:
                y_opt = -(w[0]*x_pts + b) / w[1]
                ax2.plot(x_pts, y_opt, color='#1E3A5F', lw=2.5,
                         label='Optimal Hyperplane', zorder=5)

            ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_xlabel('Mental Health Risk Factors (PC1)',
                           fontsize=10, fontweight='bold')
            ax2.set_ylabel('Academic & Demographic Profile (PC2)',
                           fontsize=10, fontweight='bold')
            ax2.set_title('Live Student Classification & SVM Decision Hyperplane',
                          fontsize=12, fontweight='bold', pad=15)
            ax2.legend(bbox_to_anchor=(1.02,1), loc='upper left',
                       borderaxespad=0, frameon=True)
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        except Exception as viz_err:
            st.warning(f"Visualization unavailable: {viz_err}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("MindCheck · BMCS2003 AI · 202605 Session · Tutorial Group 3 · Tutor: Dr Goh · TARUMT")