import pandas as pd
import numpy as np

def load_and_clean_dataset(filepath='dataset/Student_Mental_health.csv'):
    """
    Shared preprocessing for Student Mental Health Dataset
    Used by ALL members (KNN, Decision Tree, SVM)
    """
    df = pd.read_csv(filepath)

    # ── 1. Rename columns ──────────────────────────────────────
    df.rename(columns={
        'Choose your gender'                            : 'Gender',
        'What is your course?'                          : 'Course',
        'Your current year of Study'                    : 'Year_of_Study',
        'What is your CGPA?'                            : 'CGPA',
        'Marital status'                                : 'Marital_Status',
        'Do you have Depression?'                       : 'Depression',
        'Do you have Anxiety?'                          : 'Anxiety',
        'Do you have Panic attack?'                     : 'Panic_Attack',
        'Did you seek any specialist for a treatment?'  : 'Seek_Treatment'
    }, inplace=True)

    # ── 2. Drop Timestamp ──────────────────────────────────────
    df.drop(columns=['Timestamp'], inplace=True)

    # ── 3. Fix missing Age ─────────────────────────────────────
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Age'] = df['Age'].astype(int)

    # ── 4. Standardize Year_of_Study ──────────────────────────
    df['Year_of_Study'] = df['Year_of_Study'].str.strip().str.lower().map({
        'year 1': 'Year 1',
        'year 2': 'Year 2',
        'year 3': 'Year 3',
        'year 4': 'Year 4',
    })

    # ── 5. Group Course names ──────────────────────────────────
    course_map = {
        'BCS':'Computer Science','BIT':'Information Technology',
        'IT':'Information Technology','CTS':'Information Technology',
        'Engineering':'Engineering','Engine':'Engineering',
        'engin':'Engineering','ENM':'Engineering',
        'KOE':'Engineering','koe':'Engineering','Koe':'Engineering',
        'Laws':'Law','Law':'Law',
        'Psychology':'Psychology','psychology':'Psychology',
        'BENL':'Language','Benl':'Language','TAASL':'Language',
        'ALA':'Language','DIPLOMA TESL':'Language',
        'Islamic education':'Islamic Studies',
        'Islamic Education':'Islamic Studies',
        'Pendidikan islam':'Islamic Studies',
        'Pendidikan Islam':'Islamic Studies',
        'Usuluddin':'Islamic Studies','Fiqh fatwa':'Islamic Studies',
        'Fiqh':'Islamic Studies','Irkhs':'Islamic Studies',
        'KIRKHS':'Islamic Studies','Kirkhs':'Islamic Studies',
        'KENMS':'Islamic Studies','Kop':'Islamic Studies',
        'Malcom':'Islamic Studies',
        'Biomedical science':'Health Sciences',
        'BioMedical science':'Health Sciences',
        'Biotechnology':'Health Sciences',
        'Marine science':'Health Sciences',
        'Radiography':'Health Sciences',
        'Diploma Nursing':'Health Sciences',
        'Nursing':'Health Sciences','MHSC':'Health Sciences',
        'Mathemathics':'Science & Math',
        'Econs':'Business','Banking Studies':'Business',
        'Business Administration':'Business',
        'Accounting':'Business','Human Resources':'Business',
        'Human Sciences':'Business',
        'Communication':'Arts & Social',
    }
    df['Course'] = df['Course'].map(course_map).fillna('Others')

    # ── 6. Binary encode Yes/No ────────────────────────────────
    for col in ['Depression','Anxiety','Panic_Attack','Seek_Treatment','Marital_Status']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # ── 7. Feature Engineering ─────────────────────────────────
    # Mental Health Score (sum of all 3 conditions)
    df['Mental_Health_Score'] = df['Depression'] + df['Anxiety'] + df['Panic_Attack']

    # Age Group
    df['Age_Group'] = pd.cut(
        df['Age'], bins=[17, 19, 21, 25],
        labels=['Early(18-19)', 'Mid(20-21)', 'Senior(22+)']
    ).astype(str)

    # CGPA Numeric midpoint
    cgpa_map = {
        '0 - 1.99'   : 1.00,
        '2.00 - 2.49': 2.25,
        '2.50 - 2.99': 2.75,
        '3.00 - 3.49': 3.25,
        '3.50 - 4.00': 3.75,
    }
    df['CGPA_Numeric'] = df['CGPA'].map(cgpa_map)

    return df
