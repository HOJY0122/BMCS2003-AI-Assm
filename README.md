# Student Mental Health Prediction System
### BMCS2003 Artificial Intelligence | 202605 Session | Tutorial Group 3

---

## Project Overview

This project develops a **Supervised Machine Learning** system to predict depression risk among university students. The system compares three classification algorithms — **K-Nearest Neighbor (KNN)**, **Decision Tree**, and **Support Vector Machine (SVM)** — trained on a consistent, shared feature representation, and deploys an interactive web application using **Streamlit**.

All three models are retrained live, in memory, every time the app runs (see `utils/models.py`) rather than loaded from a pre-saved file — this guarantees every page's numbers always match the current dataset and code, with no hardcoded results anywhere in the application.

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
| Anxiety | Mental health indicator |
| Panic Attack | Mental health indicator |
| Seek Treatment | Whether student sought specialist help |
| **Depression** | **Target for all three models** |

---

## Project Structure

```
BMCS2003-AI-Assm/
│
├── Home.py                        # Home page (Streamlit entry point)
├── requirements.txt               # Python dependencies
│
├── dataset/
│   └── Student_Mental_health.csv  # Dataset (600 records)
│
├── pages/
│   ├── 1_EDA.py                   # Exploratory Data Analysis
│   ├── 2_KNN.py                   # KNN predictor (Member 1 — Ho Jun Yon)
│   ├── 3_Decision_Tree.py         # Decision Tree predictor (Member 2 — Irvin)
│   ├── 4_SVM.py                   # SVM predictor (Member 3 — Chiang)
│   ├── 5_Comparison.py            # Live model comparison + Auto-Selector
│   ├── 6_Dataset.py                # Dataset explorer
│   ├── 7_About.py                  # System architecture & pipeline overview
│   ├── 8_FAQ.py                    # Algorithm & metric explanations
│   ├── 9_Feature_Selection.py      # Correlation / t-test / Chi2 / MI analysis
│   ├── 11_Live_Stats.py            # Live dataset statistics & comparator
│   └── 12_Train_Test_Split.py      # Train/test split visualization
│
└── utils/
    ├── models.py                  # Shared model training — single source of truth
    ├── preprocessing.py           # Shared data cleaning & feature engineering
    ├── pdf_report.py              # Per-prediction PDF report generator
    └── sidebar.py                 # Shared navigation sidebar
```

---

## Algorithm Summary

| Member | Algorithm | Features | Encoding | Scaling | Train/Test Split |
|--------|-----------|----------|----------|---------|-------------------|
| Ho Jun Yon | KNN | 7 | Label Encoding | Min-Max Scaler | 80% / 20% |
| Irvin Tan Wei Shen | Decision Tree | 8 (+ Marital Status) | Label Encoding | None (tree-based) | 70% / 30% |
| Chiang Jun Hang | SVM | 8 (+ Marital Status) | Label Encoding | Standard Scaler | 75% / 25% |

All three models share the same encoded feature representation (`Course_Enc`, `Year_Enc` via the same `LabelEncoder` instances) for a directly comparable, consistent basis across algorithms — no model uses a separately-derived or differently-encoded feature set.

---

## Evaluation Metrics

Each model is evaluated using:
- **Accuracy** — Overall correct predictions
- **Precision** — How reliable positive predictions are
- **Recall** — How many actual positives were caught
- **F1 Score** — Harmonic mean of Precision and Recall
- **Confusion Matrix** — Breakdown of TP, TN, FP, FN
- **5-Fold Cross-Validation** — KNN only, as a supplementary robustness check alongside its held-out test-set accuracy (not used for K selection or as the primary metric for any model)

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
streamlit run Home.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## Dependencies

```
streamlit>=1.32.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.1
plotly>=5.0.0
scipy>=1.10.0
joblib>=1.3.0
reportlab>=4.0.0
```

---

## App Pages

| Page | Description |
|------|-------------|
| **Home** | Project overview, live dataset stats, group members |
| **EDA** | Data visualization and exploration |
| **Feature Selection** | Pearson correlation, t-test, Chi², mutual information |
| **Train/Test Split** | Visualizes each model's split configuration |
| **Dataset** | Full dataset explorer with live statistics |
| **Live Statistics** | Live demographic breakdowns & student comparator |
| **KNN** | KNN training, evaluation, single/batch prediction |
| **Decision Tree** | Decision Tree with live decision-path visualization |
| **SVM** | SVM training, evaluation, single/batch prediction |
| **Comparison** | Side-by-side comparison of all 3 models + Auto-Selector |
| **About** | System architecture and preprocessing pipeline |
| **FAQ** | Explanations of each algorithm and evaluation metric |

---

## System Pipeline

```
Data Collection → Preprocessing → Feature Selection → Model Training → Evaluation → Deployment
      ↓                ↓                  ↓                  ↓              ↓            ↓
  Kaggle CSV     Clean & Encode    Pearson/t-test/    KNN / DT / SVM   Accuracy /    Streamlit
  600 records    Feature Engineer   Chi2 / MI           (live, shared)   F1 / CV      Web App
```

---

## References

- Chowdhury, A. H., Rad, D., & Rahman, M. S. (2024). Predicting anxiety, depression, and insomnia among Bangladeshi university students using tree-based machine learning models. *Health Science Reports, 7*(4), e2037.
- Dodake, S., & Sudake, J. (2025). Machine learning-based analysis of student mental health: From clustering profiles to depression risk prediction. *International Journal of Engineering Development and Research, 13*(4), 453–455.
- Hasan, M. E., Arif, M., Hasan, S. M. R., Muwanguzi, M., Abaatyo, J., Kaggwa, M. M., ALmerab, M. M., Atroszko, P. A., Muhit, M., Al-Mamun, F., & Mamun, M. A. (2025). Prevalence, associated factors, and machine learning-based prediction of depression, anxiety, and stress among university students: A cross-sectional study from Bangladesh. *Journal of Health, Population and Nutrition, 44*, Article 361.
- Li, W., Zhao, Z., Chen, D., Peng, Y., & Lu, Z. (2022). Prevalence and associated factors of depression and anxiety symptoms among college students: A systematic review and meta-analysis. *Journal of Child Psychology and Psychiatry, 63*(11), 1222–1230.
- Mohamad, N. E., Sidik, S. M., Akhtari-Zavare, M., & Gani, N. A. (2021). The prevalence risk of anxiety and its associated factors among university students in Malaysia: A national cross-sectional study. *BMC Public Health, 21*, 1–12.
- Nath, M. D., Ahamed, M. K. U., Ahmed, O., Ahmed, T., Roy, S., & Uddin, M. N. (2025). Smart web interface for student mental health prediction using machine learning with blockchain technology. *Intelligent Systems with Applications* (ScienceDirect).
- Qiang, Q., Hu, J., Chen, X., Guo, W., Yang, Q., Wang, Z., Liu, Z., Zhang, Y., & Li, Q. (2025). Identifying risk factors for depression and positive/negative mood changes in college students using machine learning. *Frontiers in Public Health, 13*, 1606947.
- Shariful07. (2020). *Student Mental Health* [Dataset]. Kaggle. https://www.kaggle.com/datasets/shariful07/student-mental-health
