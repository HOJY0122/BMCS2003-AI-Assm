import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, os, sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import load_and_clean_dataset
from utils.sidebar import render_sidebar

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="KNN - Depression Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar("knn")

st.title("KNN - Depression Prediction")
st.markdown("**Member 1: Ho Jun Yon**")
st.markdown("---")

# ══════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS DATA
# ══════════════════════════════════════════════════════════════
@st.cache_data
def prepare_knn_data():
    df = load_and_clean_dataset('dataset/Student_Mental_health.csv')

    df_knn = df.drop(columns=['CGPA', 'Age_Group', 'Seek_Treatment',
                               'Marital_Status', 'Mental_Health_Score'])

    # Fix NaN in CGPA_Numeric
    df_knn['CGPA_Numeric'] = df_knn['CGPA_Numeric'].fillna(df_knn['CGPA_Numeric'].median())

    le_course = LabelEncoder()
    le_year   = LabelEncoder()
    df_knn['Course_Enc']        = le_course.fit_transform(df_knn['Course'])
    df_knn['Year_of_Study_Enc'] = le_year.fit_transform(df_knn['Year_of_Study'])

    feature_cols = ['Gender', 'Age', 'Course_Enc', 'Year_of_Study_Enc',
                    'CGPA_Numeric', 'Anxiety', 'Panic_Attack']

    X = df_knn[feature_cols]
    y = df_knn['Depression']

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler, feature_cols, le_course, le_year, df_knn


X_train, X_test, y_train, y_test, scaler, feature_cols, le_course, le_year, df_knn = prepare_knn_data()

# ══════════════════════════════════════════════════════════════
# 2. FIND BEST K
# ══════════════════════════════════════════════════════════════
st.subheader("Step 1: K-Value Optimization")

@st.cache_data
def find_best_k(X_tr, y_tr, X_te, y_te):
    k_range   = range(1, 21)
    train_acc = []
    test_acc  = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_tr, y_tr)
        train_acc.append(accuracy_score(y_tr, knn.predict(X_tr)))
        test_acc.append(accuracy_score(y_te, knn.predict(X_te)))
    best_k = list(k_range)[test_acc.index(max(test_acc))]
    return list(k_range), train_acc, test_acc, best_k

k_range, train_acc, test_acc, best_k = find_best_k(X_train, y_train, X_test, y_test)

col1, col2 = st.columns([2, 1])
with col1:
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(k_range, train_acc, 'b-o', label='Train Accuracy', markersize=5)
    ax.plot(k_range, test_acc,  'r-o', label='Test Accuracy',  markersize=5)
    ax.axvline(x=best_k, color='green', linestyle='--', label=f'Best K = {best_k}')
    ax.set_xlabel('K Value')
    ax.set_ylabel('Accuracy')
    ax.set_title('KNN: Train vs Test Accuracy for Different K Values')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

with col2:
    st.metric("Best K Value",       f"K = {best_k}")
    st.metric("Best Test Accuracy", f"{max(test_acc)*100:.2f}%")
    st.info(f"Best K = {best_k} was selected based on highest test accuracy.")

# ══════════════════════════════════════════════════════════════
# 3. TRAIN MODEL
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Step 2: Train KNN Model")

@st.cache_resource
def train_knn(X_tr, y_tr, k):
    model = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    model.fit(X_tr, y_tr)
    return model

knn_model = train_knn(X_train, y_train, best_k)
y_pred    = knn_model.predict(X_test)

os.makedirs('models', exist_ok=True)
with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump({'model': knn_model, 'scaler': scaler,
                 'features': feature_cols, 'best_k': best_k}, f)

st.success(f"KNN Model trained successfully with K = {best_k}")

c1, c2, c3 = st.columns(3)
c1.info(f"**Algorithm:** K-Nearest Neighbor\n\n**K Value:** {best_k}")
c2.info(f"**Scaling:** MinMax Scaler\n\n**Distance:** Euclidean")
c3.info(f"**Train Size:** {len(X_train)} (80%)\n\n**Test Size:** {len(X_test)} (20%)")

# ══════════════════════════════════════════════════════════════
# 4. EVALUATION
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Step 3: Model Evaluation")

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy",  f"{acc*100:.2f}%")
m2.metric("Precision", f"{prec*100:.2f}%")
m3.metric("Recall",    f"{rec*100:.2f}%")
m4.metric("F1 Score",  f"{f1*100:.2f}%")

col_cm, col_cr = st.columns(2)

with col_cm:
    st.markdown("**Confusion Matrix**")
    cm  = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Depression', 'Depression'],
                yticklabels=['No Depression', 'Depression'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('KNN Confusion Matrix')
    st.pyplot(fig)
    plt.close()

with col_cr:
    st.markdown("**Classification Report**")
    report    = classification_report(y_test, y_pred,
                                      target_names=['No Depression', 'Depression'],
                                      output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"), width='stretch')

# ══════════════════════════════════════════════════════════════
# 5. CROSS VALIDATION
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Step 4: Cross Validation (5-Fold)")

@st.cache_data
def run_cv(X_tr, y_tr, k):
    model  = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    scores = cross_val_score(model, X_tr, y_tr, cv=5, scoring='accuracy')
    return scores

cv_scores = run_cv(X_train, y_train, best_k)

cv1, cv2, cv3 = st.columns(3)
cv1.metric("CV Mean Accuracy", f"{cv_scores.mean()*100:.2f}%")
cv2.metric("CV Std Dev",       f"{cv_scores.std()*100:.2f}%")
cv3.metric("CV Max Score",     f"{cv_scores.max()*100:.2f}%")

fig, ax = plt.subplots(figsize=(8, 3))
folds   = [f"Fold {i+1}" for i in range(5)]
colors  = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
bars    = ax.bar(folds, cv_scores * 100, color=colors)
ax.axhline(y=cv_scores.mean() * 100, color='red', linestyle='--',
           label=f'Mean = {cv_scores.mean()*100:.2f}%')
ax.set_ylabel('Accuracy (%)')
ax.set_title('5-Fold Cross Validation Scores')
ax.set_ylim(0, 110)
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{score*100:.1f}%', ha='center', fontsize=9)
ax.legend()
st.pyplot(fig)
plt.close()

# ══════════════════════════════════════════════════════════════
# 6. PREDICTION FORM
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Step 5: Student Depression Prediction")
st.write("Please fill in the student information below to predict depression status.")

with st.form("knn_prediction_form"):
    st.markdown("**Student Information**")

    col_a, col_b = st.columns(2)

    with col_a:
        name    = st.text_input("Name")
        gender  = st.selectbox("Gender", ["Female", "Male"])
        age     = st.slider("Age", 18, 24, 20)
        course  = st.selectbox("Course", [
                      "Computer Science", "Information Technology",
                      "Engineering", "Law", "Psychology", "Language",
                      "Islamic Studies", "Health Sciences",
                      "Business", "Science & Math", "Arts & Social", "Others"
                  ])

    with col_b:
        year         = st.selectbox("Year of Study", ["Year 1", "Year 2", "Year 3", "Year 4"])
        cgpa         = st.selectbox("CGPA Range", [
                           "0 - 1.99", "2.00 - 2.49", "2.50 - 2.99",
                           "3.00 - 3.49", "3.50 - 4.00"
                       ])
        anxiety      = st.selectbox("Do you have Anxiety?",      ["No", "Yes"])
        panic_attack = st.selectbox("Do you have Panic Attack?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict", width='stretch')

if submitted:
    # ── Encode inputs ──────────────────────────────────────────
    gender_enc  = 1 if gender == "Male" else 0
    anxiety_enc = 1 if anxiety == "Yes" else 0
    panic_enc   = 1 if panic_attack == "Yes" else 0

    course_list = list(le_course.classes_)
    course_enc  = le_course.transform([course])[0] if course in course_list else 0

    year_list   = list(le_year.classes_)
    year_enc    = le_year.transform([year])[0] if year in year_list else 0

    cgpa_map    = {
        '0 - 1.99'   : 1.00, '2.00 - 2.49': 2.25,
        '2.50 - 2.99': 2.75, '3.00 - 3.49': 3.25,
        '3.50 - 4.00': 3.75
    }
    cgpa_num = cgpa_map[cgpa]

    input_data   = np.array([[gender_enc, age, course_enc, year_enc,
                               cgpa_num, anxiety_enc, panic_enc]])
    input_scaled = scaler.transform(input_data)
    prediction   = knn_model.predict(input_scaled)[0]
    probability  = knn_model.predict_proba(input_scaled)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    display_name = name if name.strip() != "" else "Student"

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if prediction == 1:
            st.error(
                f"Result for {display_name}: DEPRESSION DETECTED\n\n"
                "The model predicts this student may have depression. "
                "Please consider seeking professional help or speaking to a counsellor."
            )
        else:
            st.success(
                f"Result for {display_name}: NO DEPRESSION\n\n"
                "The model predicts this student does not show signs of depression. "
                "Keep maintaining a healthy lifestyle and academic balance."
            )

    with res_col2:
        fig, ax = plt.subplots(figsize=(4, 3))
        labels  = ['No Depression', 'Depression']
        colors  = ['#2ecc71', '#e74c3c']
        ax.barh(labels, probability * 100, color=colors)
        ax.set_xlabel('Probability (%)')
        ax.set_title('Prediction Confidence')
        ax.set_xlim(0, 100)
        for i, v in enumerate(probability * 100):
            ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    st.markdown("**Input Summary**")
    summary = {
        'Name'         : display_name,
        'Gender'       : gender,
        'Age'          : age,
        'Course'       : course,
        'Year of Study': year,
        'CGPA'         : cgpa,
        'Anxiety'      : anxiety,
        'Panic Attack' : panic_attack,
    }
    st.table(pd.DataFrame(summary, index=['Value']).T)

# ══════════════════════════════════════════════════════════════
# 7. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("Feature Correlation with Depression")

corr_vals = df_knn[feature_cols + ['Depression']].corr()['Depression'].drop('Depression')
fig, ax   = plt.subplots(figsize=(8, 4))
colors    = ['#e74c3c' if v > 0 else '#3498db' for v in corr_vals.values]
bars      = ax.barh(corr_vals.index, corr_vals.values, color=colors)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Correlation with Depression')
ax.set_title('Feature Correlation with Target (Depression)')
for bar, val in zip(bars, corr_vals.values):
    ax.text(
        val + 0.005 if val >= 0 else val - 0.005,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.3f}', va='center',
        ha='left' if val >= 0 else 'right', fontsize=9
    )
st.pyplot(fig)
plt.close()

st.info(
    "Red bars indicate a positive correlation with Depression.\n"
    "Blue bars indicate a negative correlation with Depression."
)
