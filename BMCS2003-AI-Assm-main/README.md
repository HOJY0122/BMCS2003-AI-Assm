# Student Mental Health Prediction System
### BMCS2003 Artificial Intelligence | 202605 Session | Tutorial Group 3

---

## Project Overview

This project develops a **Supervised Machine Learning** system to predict mental health conditions among university students. The system compares three classification algorithms — **K-Nearest Neighbor (KNN)**, **Decision Tree**, and **Support Vector Machine (SVM)** — and deploys an interactive web application using **Streamlit**.


## Dataset

- **Source:** Kaggle — [Student Mental Health](https://www.kaggle.com/datasets/shariful07/student-mental-health) (Shariful07, 2020)
- **Total Records:** 600 students
- **Original Features:** 11 columns
- **University:** IIUM Malaysia
- **Type:** Classification (Supervised Learning)

### Features

| Feature | Description |
|---------|-------------|
| Gender | Student gender (Male / Female) |
| Age | Student age |
| Course | Field of study |
| Year of Study | Current year (Year 1–4) |
| CGPA | Cumulative GPA range |
| Marital Status | Marital status |
| Depression | Target — KNN |
| Anxiety | Mental health indicator |
| Panic Attack | Target — SVM |
| Seek Treatment | Whether student sought help |

---

## Project Structure

```
BMCS2003-AI-Assm/
│
├── Main.py                        # Home page (Streamlit entry point)
├── requirements.txt               # Python dependencies
│
├── dataset/
│   └── Student_Mental_health.csv  # Dataset (600 records)
│
├── models/
│   └── knn_model.pkl              # Saved KNN model
│
├── pages/
│   ├── 1_EDA.py                   # Exploratory Data Analysis
│   ├── 2_KNN.py                   # KNN model (Member 1 - Ho Jun Yon)
│   ├── 3_Decision_Tree.py         # Decision Tree (Member 2 - Irvin)
│   ├── 4_SVM.py                   # SVM model (Member 3 - Chiang)
│   └── 5_Comparison.py            # Model comparison page
│
└── utils/
    └── preprocessing.py           # Shared data preprocessing
```

---

## Algorithm Summary

| Member | Algorithm | Encoding | Scaling | Train/Test Split |
|--------|-----------|----------|---------|-----------------|
| Ho Jun Yon | KNN | Label Encoding | MinMax Scaler | 80% / 20% |
| Irvin Tan Wei Shen | Decision Tree | Custom Rules | No Scaling | Full Dataset |
| Chiang Jun Hang | SVM | Label Encoding | Standard Scaler | 75% / 25% |

---

## Evaluation Metrics

Each model is evaluated using:
- **Accuracy** — Overall correct predictions
- **Precision** — How reliable positive predictions are
- **Recall** — How many actual positives were caught
- **F1 Score** — Harmonic mean of Precision and Recall
- **Confusion Matrix** — Breakdown of TP, TN, FP, FN
- **Cross Validation** — 5-Fold CV (KNN & SVM)

---

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/HOJY0122/BMCS2003-AI-Assm.git
cd BMCS2003-AI-Assm
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run Main.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1
```

---

## App Pages

| Page | Description |
|------|-------------|
| **Main** | Project overview, dataset info, group members |
| **EDA** | Data visualization and exploration |
| **KNN** | KNN model training, evaluation, prediction |
| **Decision Tree** | Custom Decision Tree with visual path |
| **SVM** | SVM model training, evaluation, prediction |
| **Comparison** | Side-by-side comparison of all 3 models |

---

## System Pipeline

```
Data Collection → Preprocessing → Model Training → Evaluation → Deployment
      ↓                ↓                ↓               ↓            ↓
  Kaggle CSV     Clean & Encode      KNN / DT / SVM   Accuracy    Streamlit
  600 records    Feature Engineer    Train & Test      F1 Score    Web App
```

---


## References

- Shariful07. (2020). *Student Mental Health* [Dataset]. Kaggle. https://www.kaggle.com/datasets/shariful07/student-mental-health
- Sau, A., & Bhakta, I. (2019). Predicting anxiety and depression in elderly patients using machine learning technology. *Healthcare Technology Letters, 6*(3), 60–65.
- Dhar, E., Alyami, M., Alotaibi, M., & Islam, M. S. (2021). A machine learning approach for mental health assessment. *IEEE Access, 9*, 127002–127012.
