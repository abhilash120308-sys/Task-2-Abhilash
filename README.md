# DecodeLabs AI Project 2: Data Classification Using AI — Iris Flower Classification

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/pytest-22%20passed-brightgreen.svg)](tests/)

> **Industrial Training Kit | Batch 2026 | Powered by DecodeLabs**  
> *Track: Supervised Machine Learning & Algorithmic Classification*

---

## 📌 Project Overview

**DecodeLabs Project 2** marks the transition from heuristic, rule-based systems to **Supervised Machine Learning**. Instead of manually crafting hardcoded conditional rules (`if/else`), this project implements a rigorous, leak-free machine learning pipeline that trains an algorithm to discover mathematical decision boundaries directly from empirical data.

Using the gold-standard **Fisher's Iris dataset**, the system classifies unknown floral specimens into one of three biological species (*Iris setosa*, *Iris versicolor*, and *Iris virginica*) based on four morphological measurements.

---

## 📊 Dataset Specifications

The benchmark Iris dataset contains:
- **Total Samples:** 150 balanced botanical specimens
- **Samples Per Class:** Exactly 50 specimens per species (33.3% / 33.3% / 33.3%)
- **Target Classes (3):**
  1. `setosa` (Class ID `0`) — Morphologically distinct, completely linearly separable.
  2. `versicolor` (Class ID `1`) — Intermediate dimensions, adjacent to Virginica.
  3. `virginica` (Class ID `2`) — Largest petals, shares subtle boundary with Versicolor.
- **Morphological Features (4 Dimensions):**
  1. `sepal length (cm)`: Outermost protective floral whorl length (Mean: 5.84 cm, Range: 4.3 – 7.9 cm).
  2. `sepal width (cm)`: Outermost whorl base width (Mean: 3.06 cm, Range: 2.0 – 4.4 cm).
  3. `petal length (cm)`: Inner vibrant petal length (Mean: 3.76 cm, Range: 1.0 – 6.9 cm).
  4. `petal width (cm)`: Inner floral petal width (Mean: 1.20 cm, Range: 0.1 – 2.5 cm).

---

## 🏗️ Architectural Framework: The Master Blueprint (IPO)

This project strictly follows the DecodeLabs **IPO (Input-Process-Output)** framework:

```
[ INPUT ]                          [ PROCESS ]                       [ OUTPUT ]
• 150 Iris Samples        ───►  • Stratified 80/20 Split    ───►  • Confusion Matrix (3x3)
• 4 Morphological Feats        • StandardScaler (Train only)     • Macro Precision: 0.9444
• StandardScaler                • KNN Classifier (Baseline K=5)   • Macro Recall: 0.9333
                                • 5-Fold Cross-Validation CV      • Macro F1-Score: 0.9327
```

### The 7-Stage Machine Learning Lifecycle
1. **Load:** Ingest the benchmark dataset with standard feature definitions.
2. **Explore:** Analyze statistical distributions, bivariate scatter plots, and correlation matrices.
3. **Split:** Partition into 80% Training (120 samples) and 20% Testing (30 samples) using a fixed random seed (`random_state=42`) with **stratification** to preserve 10 test samples per class.
4. **Scale:** Enforce the **Gatekeeper Rule**: Fit `StandardScaler` strictly on the training partition ($\mu=0, \sigma^2=1$) and apply `.transform()` to testing data, completely preventing data leakage.
5. **Train & Tune:**
   - Fit a baseline `KNeighborsClassifier(n_neighbors=5)` as mandated.
   - Perform 5-fold cross-validation on the training set across $K \in [1, 25]$ to identify the optimal elbow without touching the test set. Retrain optimal model ($K=3$).
6. **Predict:** Infer species class, calculate exact $K$-neighbour vote proportions, and inspect the nearest neighbors in standardized Euclidean space.
7. **Evaluate:** Verify generalization on the locked 30-sample held-out test set using Accuracy, Precision, Recall, F1-Score, and 3x3 Confusion Matrices for both models.

---

## ⚠️ The "99% Accuracy Mirage"

In balanced datasets like Iris, accuracy gives a reasonable summary. However, **accuracy alone can be dangerous and deceptive** in real-world scenarios:

- **Imbalance Example:** In a cancer screening cohort with 990 healthy patients and 10 cancer patients, a naive dummy model predicting "Healthy" for everyone achieves **99.0% Accuracy**. Yet, its **Recall is 0.0%**—it misses 100% of cancer cases!
- **Diagnostic Metrics:** To prevent this failure mode, this project calculates:
  - **Precision:** $\frac{TP}{TP + FP}$ — Minimizes false alarms (critical for spam filters and user trust).
  - **Recall:** $\frac{TP}{TP + FN}$ — Minimizes missed detections (critical for medical diagnoses).
  - **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ — The harmonic mean balancing both trade-offs.

---

## 📈 Actual Evaluation Results (Locked Held-Out Test Set: 30 Samples)

Both the **Baseline Model ($K=5$)** and the **CV-Tuned Model ($K=3$)** were retrained on the 120 training samples and evaluated on the exact same locked held-out test set (`random_state=42`, `stratify=y`):

| Evaluation Metric | Baseline Model ($K=5$) | CV-Tuned Model ($K=3$) | Determination & Rationale |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **93.33%** | **93.33%** | 28 out of 30 unseen test specimens classified correctly |
| **Macro Precision** | **0.9444** | **0.9444** | Unweighted average precision across all 3 species |
| **Macro Recall** | **0.9333** | **0.9333** | Unweighted average sensitivity across all 3 species |
| **Macro F1-Score** | **0.9327** | **0.9327** | Harmonic mean reflecting balanced multi-class performance |
| **Weighted F1-Score**| **0.9327** | **0.9327** | Class-weighted harmonic mean |
| **Hyperparameter Selection**| Assignment Core Mandate | 5-Fold CV on Training Set | Held-out test set remained strictly untouched during selection |

### Confusion Matrix Breakdown (Both Models)

```
                  Predicted Setosa   Predicted Versicolor   Predicted Virginica
Actual Setosa            10                   0                      0
Actual Versicolor         0                  10                      0
Actual Virginica          0                   2                      8
```

- **Iris setosa:** 10 / 10 correct (Precision: 1.0000, Recall: 1.0000, F1: 1.0000)
- **Iris versicolor:** 10 / 10 actual detected, 2 false alarms from Virginica (Precision: 0.8333, Recall: 1.0000, F1: 0.9091)
- **Iris virginica:** 8 / 10 correct, 2 misclassified as Versicolor due to borderline petal dimensions (Precision: 1.0000, Recall: 0.8000, F1: 0.8889)

### Cross-Validation Hyperparameter Tuning (Training Data Only)
- **$K=1$ (Overfitting risk / high variance):** CV Error = 0.0583, CV Acc = 94.17%
- **$K=3$ (Optimal CV Elbow):** CV Error = 0.0333, CV Acc = 96.67%
- **$K=5$ (Baseline Assignment Specification):** CV Error = 0.0417, CV Acc = 95.83%
- **$K=25$ (Underfitting risk / high bias):** CV Error = 0.1083, CV Acc = 89.17%

---

## 🌸 Sample Predictions

| Specimen Description | Sepal L (cm) | Sepal W (cm) | Petal L (cm) | Petal W (cm) | Predicted Species | Confidence (Vote Proportion) | Morphological Rationale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clear Setosa Archetype** | 5.1 | 3.5 | 1.4 | 0.2 | `setosa` | **100.0% (5/5 votes)** | Short, narrow petals uniquely isolate Setosa in feature space. |
| **Clear Versicolor Archetype**| 5.7 | 2.8 | 4.1 | 1.3 | `versicolor` | **100.0% (5/5 votes)** | Moderate petal dimensions match historical Versicolor centroid. |
| **Clear Virginica Archetype** | 6.3 | 3.3 | 6.0 | 2.5 | `virginica` | **100.0% (5/5 votes)** | Long, broad petals place flower deeply into Virginica cluster. |
| **Boundary Specimen** | 6.0 | 2.7 | 5.1 | 1.6 | `versicolor` | **60.0% (3/5 votes)** | Sits on the adjacent Versicolor/Virginica boundary (3 vs 2 votes). |

---

## 🛠️ Installation & Setup Instructions

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git (optional)

### 2. Clone or Navigate to Directory
```bash
cd "c:\Users\rajan\OneDrive\Desktop\T-2-AB"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Commands to Run

### Run Unit Tests
Execute the 22 comprehensive unit tests verifying data integrity, zero leakage, model performance, and error handling:
```bash
pytest -v
```

### Run Model Training Pipeline
Execute the end-to-end 7-stage training pipeline, serialize both pipeline artifacts, and export `metrics.json`:
```bash
python train_model.py
```

### Launch Interactive Streamlit Application
Launch the responsive web interface for live predictions, EDA, tuning curves, and metrics visualization:
```bash
streamlit run app.py
```
*The app will automatically open at `http://localhost:8501` (or next available port like `http://localhost:8502`).*

---

## 📂 Project Structure

```
c:\Users\rajan\OneDrive\Desktop\T-2-AB\
│
├── app.py                         # Interactive Streamlit application (UI, EDA, live inference, metrics)
├── model.py                       # Modular ML logic (loading, scaling pipeline, KNN, CV tuning, inference)
├── train_model.py                 # Standalone 7-stage training & evaluation script
├── requirements.txt               # Project dependencies
├── README.md                      # Comprehensive documentation and technical report
├── iris_knn_pipeline.joblib       # Serialized Baseline (K=5) scikit-learn pipeline
├── iris_knn_pipeline_tuned.joblib # Serialized CV-Tuned (K=3) scikit-learn pipeline
├── metrics.json                   # Actual computed evaluation metrics and CV results
│
└── tests/
    ├── __init__.py
    └── test_model.py              # 22 unit tests (loading, split, leakage check, CV, metrics, inputs)
```

---

## ⚠️ Known Limitations & Future Horizons

1. **Dimensional Overlap:** *Iris versicolor* and *Iris virginica* exhibit slight natural morphological overlap in petal dimensions, which causes 2 out of 30 test samples to be misclassified by distance-based metrics.
2. **Local Density Sensitivity:** KNN calculates distances to all training points; for massive datasets ($>10^7$ rows), inference latency scales linearly $O(N \cdot D)$. Approximate Nearest Neighbors (ANN) or Tree-based models (e.g. Random Forest / XGBoost) are preferred for large-scale production.
3. **From Tabular to Computer Vision:** As highlighted in DecodeLabs Slide 18, tabular flower classification provides the conceptual foundation for **Deep Learning and Convolutional Neural Networks (CNNs)**, where pixel matrices replace manual caliper measurements.
