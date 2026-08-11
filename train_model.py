"""
Student Mental Health Prediction - Support Vector Machine (SVM)
=================================================================
Supervised learning system that predicts whether a student is likely to
have DEPRESSION based on demographic / academic survey answers.

Dataset: Student_Mental_health.csv (Kaggle, shariful07)
Algorithm: Support Vector Machine (sklearn.svm.SVC)

Run:
    python3 train_model.py
Outputs:
    model.joblib            -> trained pipeline (preprocessing + SVM)
    metadata.json           -> encoders / options needed by the GUI
    results.json            -> evaluation metrics (for the report)
    confusion_matrix.png    -> confusion matrix plot
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, precision_score,
                              recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "/mnt/user-data/uploads/Student_Mental_health.csv"

# ---------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Timestamp", "What is your course?"])  # ID / high-cardinality

# Standardise messy text fields
df["Your current year of Study"] = (
    df["Your current year of Study"].str.strip().str.lower()
)
df["What is your CGPA?"] = df["What is your CGPA?"].str.strip()

# Impute the single missing Age with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Target
TARGET = "Do you have Depression?"
y = (df[TARGET] == "Yes").astype(int)
X = df.drop(columns=[TARGET])

CGPA_ORDER = ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"]
YEAR_ORDER = ["year 1", "year 2", "year 3", "year 4"]

ordinal_cols = ["What is your CGPA?", "Your current year of Study"]
nominal_cols = [
    "Choose your gender",
    "Marital status",
    "Do you have Anxiety?",
    "Do you have Panic attack?",
    "Did you seek any specialist for a treatment?",
]
numeric_cols = ["Age"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        (
            "ord",
            OrdinalEncoder(categories=[CGPA_ORDER, YEAR_ORDER]),
            ordinal_cols,
        ),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal_cols),
    ]
)

# ---------------------------------------------------------------------------
# 2. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. PIPELINE + GRID SEARCH (SVM)
# ---------------------------------------------------------------------------
pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("svm", None),  # placeholder, set by GridSearch below
    ]
)

from sklearn.svm import SVC

pipe.set_params(svm=SVC(probability=True, random_state=RANDOM_STATE))

param_grid = {
    "svm__kernel": ["linear", "rbf", "poly"],
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": ["scale", "auto"],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best hyperparameters:", grid.best_params_)

# ---------------------------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------------------------
y_pred = best_model.predict(X_test)

metrics = {
    "best_params": grid.best_params_,
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "classification_report": classification_report(
        y_test, y_pred, target_names=["No Depression", "Depression"], zero_division=0
    ),
}

cm = confusion_matrix(y_test, y_pred)
metrics["confusion_matrix"] = cm.tolist()

print("\n=== RESULTS ===")
print(f"Accuracy : {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall   : {metrics['recall']:.3f}")
print(f"F1-score : {metrics['f1_score']:.3f}")
print("\nConfusion matrix:\n", cm)
print("\n", metrics["classification_report"])

with open("results.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------------------
# 5. PLOT CONFUSION MATRIX
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["No Depression", "Depression"])
ax.set_yticklabels(["No Depression", "Depression"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("SVM Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im)
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

# ---------------------------------------------------------------------------
# 6. SAVE MODEL + METADATA (for the GUI)
# ---------------------------------------------------------------------------
joblib.dump(best_model, "model.joblib")

metadata = {
    "target": TARGET,
    "feature_columns": list(X.columns),
    "options": {
        "Choose your gender": sorted(X["Choose your gender"].unique().tolist()),
        "Marital status": sorted(X["Marital status"].unique().tolist()),
        "Do you have Anxiety?": sorted(X["Do you have Anxiety?"].unique().tolist()),
        "Do you have Panic attack?": sorted(X["Do you have Panic attack?"].unique().tolist()),
        "Did you seek any specialist for a treatment?": sorted(
            X["Did you seek any specialist for a treatment?"].unique().tolist()
        ),
        "What is your CGPA?": CGPA_ORDER,
        "Your current year of Study": YEAR_ORDER,
    },
    "age_range": [int(X["Age"].min()), int(X["Age"].max())],
}
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved model.joblib and metadata.json")
