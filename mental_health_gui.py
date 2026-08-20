import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# --- 1. Set File Paths Safely ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
RESULTS_PATH = os.path.join(BASE_DIR, "results.json")
DATA_PATH = os.path.join(BASE_DIR, "Student Mental health.csv")

# --- 2. Load Model Safely ---
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(
        f"Failed to load model.joblib: {e}. Please run 'python train_model.py' first."
    )
    st.stop()

# --- 3. Sidebar Performance Metrics ---
try:
    with open(RESULTS_PATH, "r") as f:
        metrics = json.load(f)

    st.sidebar.title("📊 Model Performance")
    st.sidebar.metric(
        label="Model Accuracy", value=f"{metrics['accuracy'] * 100:.1f}%"
    )
    st.sidebar.metric(label="F1-Score", value=f"{metrics['f1_score']:.2f}")
    st.sidebar.metric(
        label="Precision", value=f"{metrics['precision'] * 100:.1f}%"
    )
except FileNotFoundError:
    st.sidebar.warning("Run train_model.py first to generate results.json.")

# --- 4. Web Interface Header ---
st.title("🧠 Student Mental Health Predictor")
st.write("Provide the student profile details below to predict the risk status.")

# --- 5. Interactive Input Form ---
age = st.number_input("Age", min_value=15, max_value=40, value=20, step=1)
gender = st.selectbox("Choose your gender", ["Male", "Female"])
year_of_study = st.selectbox(
    "Your current year of Study",
    ["Year 1", "Year 2", "Year 3", "Year 4"],
)
cgpa = st.selectbox(
    "What is your CGPA?",
    ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"],
)
course = st.selectbox(
    "What is your course field?",
    ["Information Technology (IT)", "Computer Science (CS)", "Information System (IS)", "Software Engineering (SE)","Other"]
)
marital = st.selectbox("Marital status", ["No", "Yes"])
anxiety = st.selectbox("Do you have Anxiety?", ["No", "Yes"])
panic = st.selectbox("Do you have Panic attack?", ["No", "Yes"])
treatment = st.selectbox(
    "Did you seek any specialist for a treatment?", ["No", "Yes"]
)

def categorize_course(c_str):
    c = str(c_str).lower()
    # If the user selected any of the 4 IT/Tech fields, map to STEM/IT
    if any(x in c for x in ["technology", "it", "computer", "cs", "system", "is", "software", "se", "bit", "bcs", "cts"]):
        return "STEM/IT"
    else:
        return "Other"

def plot_live_svm_boundary(input_df, prep_stage):
    df_raw = pd.read_csv(DATA_PATH)
    df_raw.columns = df_raw.columns.str.strip()

    df_raw["Your current year of Study"] = (
        df_raw["Your current year of Study"].str.strip().str.lower()
    )
    df_raw["What is your CGPA?"] = df_raw["What is your CGPA?"].str.strip()
    df_raw["Age"] = df_raw["Age"].fillna(df_raw["Age"].median())

    # Feature Engineering for background data
    df_raw["Course_Category"] = df_raw["What is your course?"].apply(
        categorize_course
    )
    df_raw = df_raw.drop(
        columns=["Timestamp", "What is your course?"], errors="ignore"
    )

    X_background = df_raw.drop(columns=["Do you have Depression?"])
    y_background = (df_raw["Do you have Depression?"] == "Yes").astype(int)

    # Transform through pipeline preprocessing
    X_bg_proc = prep_stage.transform(X_background)
    X_user_proc = prep_stage.transform(input_df)

    # PCA down to 2 dimensions for visual plotting
    pca = PCA(n_components=2, random_state=42)
    X_bg_2d = pca.fit_transform(X_bg_proc)
    X_user_2d = pca.transform(X_user_proc)

    # 2D Linear SVM specifically for rendering the hyperplane
    svm_2d = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    svm_2d.fit(X_bg_2d, y_background)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="white")
    ax.set_facecolor("white")

    # 1. Scatter background data with transparency
    ax.scatter(
        X_bg_2d[y_background == 1, 0],
        X_bg_2d[y_background == 1, 1],
        color="#FF0000",
        s=45,
        alpha=0.6,
        label="Depression Data",
        zorder=3,
    )
    ax.scatter(
        X_bg_2d[y_background == 0, 0],
        X_bg_2d[y_background == 0, 1],
        color="#0000FF",
        s=45,
        alpha=0.6,
        label="No Depression Data",
        zorder=3,
    )

    # 2. User's Input Point (Black Star)
    ax.scatter(
        X_user_2d[0, 0],
        X_user_2d[0, 1],
        color="black",
        s=250,
        marker="*",
        label="Your Input Student",
        zorder=6,
    )

    # 3. Dynamic Axis Range calculation
    x_min, x_max = X_bg_2d[:, 0].min() - 1.0, X_bg_2d[:, 0].max() + 1.0
    y_min, y_max = X_bg_2d[:, 1].min() - 1.0, X_bg_2d[:, 1].max() + 1.0

    # Ensure user input point stays inside view bounds
    x_min = min(x_min, X_user_2d[0, 0] - 1.0)
    x_max = max(x_max, X_user_2d[0, 0] + 1.0)

    # 4. Draw Optimal Hyperplane Line across visible X-range
    w = svm_2d.coef_[0]
    b = svm_2d.intercept_[0]
    x_points = np.linspace(x_min, x_max, 200)

    if w[1] != 0:
        y_opt = -(w[0] * x_points + b) / w[1]
        ax.plot(
            x_points,
            y_opt,
            color="#1E3A5F",
            linewidth=2.8,
            label="Optimal Hyperplane",
            zorder=5,
        )

    # Set calculated bounds cleanly
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Arrow Spines styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.8)
    ax.spines["bottom"].set_linewidth(1.8)

    ax.set_xlabel(
        "Overall Mental Health Risk Factors (PC1)",
        fontsize=10,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Academic & Demographic Profile (PC2)", fontsize=10, fontweight="bold"
    )
    ax.set_title(
        "Live Student Classification & Decision Hyperplane",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )

    # Position Legend completely outside on the right side
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
        frameon=True,
    )
    fig.tight_layout()

    return fig

# --- 7. Prediction Action Button ---
if st.button("Predict Mental Health Risk"):
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Choose your gender": gender,
                "Your current year of Study": f"year {year_of_study.split()[-1]}",
                "What is your CGPA?": cgpa,
                "Marital status": marital,
                "Do you have Anxiety?": anxiety,
                "Do you have Panic attack?": panic,
                "Did you seek any specialist for a treatment?": treatment,
                "Course_Category": categorize_course(course),
            }
        ]
    )

    try:
        prediction = model.predict(input_data)

        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(
                "⚠️ High risk of depression predicted. Consider speaking with someone or seeking support."
            )
        else:
            st.success("✅ Low risk of depression predicted.")

        st.subheader("📈 Live SVM Decision Boundary Position")
        fig = plot_live_svm_boundary(
            input_data, model.named_steps[list(model.named_steps.keys())[0]]
        )
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
