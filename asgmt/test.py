import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Mental Health Predictor", layout="centered")

st.title("Student Mental Health Prediction System")
st.write("Predict likelihood of depression using a Decision Tree model.")

# 1. Load Data
@st.cache_data
def load_and_prep_data():
    file_path = 'Student Mental health.csv'
    df = pd.read_csv(file_path)

    # Clean string columns
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

    df['Your current year of Study'] = df['Your current year of Study'].str.lower()
    df['What is your course?'] = df['What is your course?'].str.lower()

    # Handle missing age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df['Age'] = df['Age'].fillna(df['Age'].median())

    return df

try:
    df = load_and_prep_data()

    target_col = 'Do you have Depression?'
    feature_cols = [
        'Choose your gender', 
        'Age', 
        'What is your course?', 
        'Your current year of Study', 
        'What is your CGPA?', 
        'Marital status', 
        'Do you have Anxiety?', 
        'Do you have Panic attack?'
    ]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)

    # Train model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 2. Interactive Input UI
    st.header("Enter Student Information")
    
    user_inputs = {}

    # Validation: Restrict Age to 18–24
    user_inputs['Age'] = st.number_input(
        "Age (18 to 24)", 
        min_value=18, 
        max_value=24, 
        value=20, 
        step=1
    )

    for col in feature_cols:
        if col == 'Age':
            continue
        
        options = list(label_encoders[col].classes_)
        selected = st.selectbox(f"Select {col}", options)
        user_inputs[col] = label_encoders[col].transform([selected])[0]

    # 3. Prediction Button
    if st.button("Predict Mental Health Status", type="primary"):
        input_df = pd.DataFrame([user_inputs])
        prediction_encoded = model.predict(input_df)[0]
        prediction = target_le.inverse_transform([prediction_encoded])[0]

        if prediction.lower() == 'yes':
            st.error(f"Prediction Result: **{prediction.upper()}** (High indication of Depression)")
        else:
            st.success(f"Prediction Result: **{prediction.upper()}** (Low indication of Depression)")

except Exception as e:
    st.error(f"Error loading dataset or model: {e}")