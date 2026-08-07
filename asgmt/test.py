import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Load Dataset
file_path = 'Student Mental health.csv'
df = pd.read_csv(file_path)

# 2. Data Preprocessing & Cleaning
# Strip leading/trailing whitespaces from column names and string columns
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype(str).str.strip()

# Standardize year of study and course names
df['Your current year of Study'] = df['Your current year of Study'].str.lower()
df['What is your course?'] = df['What is your course?'].str.lower()

# Impute missing values in 'Age' with median
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].fillna(df['Age'].median())

# Target variable selection
target_col = 'Do you have Depression?'

# Select feature columns
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

# Encode Categorical Features using LabelEncoder
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode Target Variable
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y)

# 3. Model Training
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Training Complete! Accuracy on Test Set: {acc * 100:.2f}%\n")


# 4. Interactive User Input Prompt with Validation
def get_user_prediction():
    print("=" * 50)
    print("   STUDENT MENTAL HEALTH PREDICTION SYSTEM   ")
    print("=" * 50)

    user_input = {}
    
    for col in feature_cols:
        # Age Validation (Must be an integer/float between 18 and 24)
        if col == 'Age':
            while True:
                user_val = input(f"\nEnter Age (18 - 24): ").strip()
                try:
                    val = float(user_val)
                    if 18 <= val <= 24:
                        user_input[col] = val
                        break
                    else:
                        print("Error: Invalid range. Age must be between 18 and 24.")
                except ValueError:
                    print("Error: Invalid input. Please enter a numerical value for age.")
        
        # Categorical Options Validation
        elif col in label_encoders:
            le = label_encoders[col]
            valid_options = list(le.classes_)
            
            print(f"\nSelect option for [{col}]:")
            for idx, option in enumerate(valid_options, 1):
                print(f"  {idx}. {option}")
            
            while True:
                choice = input(f"Enter choice (1-{len(valid_options)}) or type exact option name: ").strip()
                
                # Validation by option number index
                if choice.isdigit():
                    num = int(choice)
                    if 1 <= num <= len(valid_options):
                        val = valid_options[num - 1]
                        user_input[col] = le.transform([val])[0]
                        break
                    else:
                        print(f"Error: Choice out of bounds. Please enter a number between 1 and {len(valid_options)}.")
                
                # Validation by text option name (case-insensitive)
                else:
                    matched = [opt for opt in valid_options if opt.lower() == choice.lower()]
                    if matched:
                        user_input[col] = le.transform([matched[0]])[0]
                        break
                    else:
                        print("Error: Option not recognized. Please choose a valid option from the list.")

    # Convert user input into Pandas DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Predict using Decision Tree Model
    prediction_encoded = model.predict(input_df)[0]
    prediction = target_le.inverse_transform([prediction_encoded])[0]
    
    # Output Result
    print("\n" + "=" * 50)
    print(f"PREDICTION RESULT ({target_col}): {prediction.upper()}")
    print("=" * 50)


if __name__ == '__main__':
    get_user_prediction()