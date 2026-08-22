# Methodology & Results — Student Mental Health Prediction (SVM)

## 1. System Flow
1. Load the raw survey dataset (`Student_Mental_health.csv`, 101 responses).
2. Clean and standardise the data (fix inconsistent text case, strip whitespace, impute one missing Age value with the median).
3. Encode features: numeric scaling for Age, ordinal encoding for CGPA band and Year of Study (since these have a natural order), one-hot encoding for nominal fields (Gender, Marital status, Anxiety, Panic attack, Treatment history).
4. Split data into training (75%) and test (25%) sets, stratified by the target so both classes are represented proportionally.
5. Train a Support Vector Machine classifier inside a `GridSearchCV` (5-fold cross-validation) that searches kernel type (linear/RBF/polynomial), `C`, and `gamma` to find the best-performing configuration.
6. Evaluate the tuned model on the held-out test set.
7. Persist the trained pipeline (`model.joblib`) and a `metadata.json` describing valid input options, so a separate GUI can load the model without retraining.

## 2. Dataset
- **Source:** Kaggle — "Student Mental Health" (shariful07), a self-reported survey of university students.
- **Size:** 101 records, 11 raw columns.
- **Target variable:** `Do you have Depression?` (Yes/No), converted to binary (1 = Yes).
- **Features used:** Gender, Age, Year of Study, CGPA band, Marital status, Anxiety, Panic attack, prior treatment-seeking.
- **Excluded:** `Timestamp` (identifier, no predictive value) and `What is your course?` (too many distinct values relative to only 101 rows — would cause severe overfitting/sparsity).

## 3. Algorithm — Support Vector Machine (SVM)
An SVM was chosen because it performs well on small, high-dimensional tabular datasets (after one-hot encoding, the feature space grows quickly relative to the number of rows) and because the maximum-margin decision boundary tends to generalise better than more complex models when data is limited. `GridSearchCV` was used to select the best kernel and regularisation strength rather than guessing hyperparameters manually.

**Search space:**
| Hyperparameter | Values tried |
|---|---|
| kernel | linear, rbf, poly |
| C | 0.1, 1, 10, 100 |
| gamma | scale, auto |

**Best configuration found:** `kernel = linear`, `C = 10`, `gamma = scale`.

## 4. Evaluation Metrics
| Metric | Score |
|---|---|
| Accuracy | 0.769 |
| Precision (Depression) | 0.800 |
| Recall (Depression) | 0.444 |
| F1-score (Depression) | 0.571 |

**Confusion matrix (test set, n = 26):**

|  | Predicted: No | Predicted: Yes |
|---|---|---|
| **Actual: No** | 16 | 1 |
| **Actual: Yes** | 5 | 4 |

(see `confusion_matrix.png` for the plotted version)

## 5. Discussion
The model achieves ~77% overall accuracy and high precision (80%) for predicting Depression, meaning that when it flags a student as at-risk, it is usually correct. However, recall is lower (44%), meaning it currently misses more than half of the true Depression cases in this small test set — largely a consequence of the dataset's small size (only 101 responses total, with just 35 positive Depression cases) rather than a flaw in the SVM itself. The linear kernel outperforming RBF/polynomial suggests the classes are close to linearly separable once features are encoded, which fits the intuition that co-occurring symptoms (Anxiety, Panic attack) are the dominant signal for Depression status.

## 6. Limitations & Future Work
- **Small sample size** (101 rows) limits statistical reliability; results would benefit from a larger, more diverse dataset.
- **Self-reported labels** (survey responses) may carry response bias.
- **Course field** was dropped due to sparsity; with more data it could be grouped (e.g. by faculty) and reintroduced.
- Future work could explore SMOTE/class-weighting to address class imbalance and improve recall, or compare SVM against other classifiers (Random Forest, Logistic Regression) as a benchmark.

## 7. Tools & Dataset Acknowledgement
- Dataset: Kaggle, "Student Mental Health" by shariful07 — https://www.kaggle.com/datasets/shariful07/student-mental-health
- Libraries: scikit-learn (SVM, preprocessing, model selection), pandas, matplotlib, joblib, Tkinter (GUI).
