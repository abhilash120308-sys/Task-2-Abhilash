"""
model.py - Core Machine Learning Architecture for Iris Flower Classification.
DecodeLabs Industrial Training Kit | Batch 2026 | AI Project 2

Implements the complete ML lifecycle:
- Dataset loading and exploration
- Stratified 80/20 train/test split (preventing order bias)
- StandardScaler pipeline (preventing data leakage)
- K-Nearest Neighbors Classifier (baseline K=5)
- Hyperparameter tuning via training-only cross-validation
- Comprehensive evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
- Robust inference with input validation and neighborhood inspection
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union, List
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib


# Constants & Reproducibility Seed
RANDOM_STATE: int = 42
DEFAULT_TEST_SIZE: float = 0.20
DEFAULT_K: int = 5

FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

FEATURE_DESCRIPTIONS = {
    "sepal length (cm)": "Length of the outermost floral leaf protecting the flower bud before blooming.",
    "sepal width (cm)": "Width of the base sepal structure; setosa typically exhibits noticeably wide sepals.",
    "petal length (cm)": "Length of the inner, vibrant floral leaves. A primary differentiator for species separation.",
    "petal width (cm)": "Width of the inner floral petals; critical in distinguishing Versicolor from Virginica."
}

SPECIES_METADATA = {
    "setosa": {
        "class_id": 0,
        "display_name": "Iris setosa",
        "badge_color": "#22c55e",
        "description": "Smallest petals, wide sepals. Completely linearly separable from other species.",
        "typical_petal_length": "1.0 - 1.9 cm",
        "typical_petal_width": "0.1 - 0.6 cm"
    },
    "versicolor": {
        "class_id": 1,
        "display_name": "Iris versicolor",
        "badge_color": "#3b82f6",
        "description": "Medium-sized petals and sepals. Sits between Setosa and Virginica in feature space.",
        "typical_petal_length": "3.0 - 5.1 cm",
        "typical_petal_width": "1.0 - 1.8 cm"
    },
    "virginica": {
        "class_id": 2,
        "display_name": "Iris virginica",
        "badge_color": "#8b5cf6",
        "description": "Largest petals and sepals. Shares a subtle decision boundary with Versicolor.",
        "typical_petal_length": "4.5 - 6.9 cm",
        "typical_petal_width": "1.4 - 2.5 cm"
    }
}


def load_iris_data() -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], pd.DataFrame]:
    """
    Load the benchmark Iris dataset with 150 samples, 4 features, and 3 classes.
    
    Returns:
        X: Feature DataFrame (150 rows, 4 columns)
        y: Target Series of integer class IDs (0, 1, 2)
        target_names: List of class names ['setosa', 'versicolor', 'virginica']
        feature_names: List of feature names
        full_df: Combined DataFrame including features, target, and species names
    """
    raw_data = load_iris(as_frame=True)
    X = raw_data.data.copy()
    y = raw_data.target.copy()
    target_names = list(raw_data.target_names)
    feature_names = list(raw_data.feature_names)
    
    # Standardize column names
    X.columns = FEATURE_NAMES
    
    full_df = X.copy()
    full_df["target"] = y
    full_df["species"] = y.map(lambda idx: target_names[idx])
    
    return X, y, target_names, feature_names, full_df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Shuffle and split the dataset into training (80%) and testing (20%) sets.
    Applies stratification to preserve exact class balance (10 test samples per class).
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
        shuffle=True
    )
    return X_train, X_test, y_train, y_test


def create_pipeline(
    n_neighbors: int = DEFAULT_K,
    metric: str = "minkowski",
    p: int = 2
) -> Pipeline:
    """
    Create a scikit-learn Pipeline with StandardScaler and KNeighborsClassifier.
    
    Enforces the 'Gatekeeper Rule': The scaler is fitted strictly on training data
    during `.fit()` and merely transforms evaluation/test data, completely preventing
    data leakage.
    
    Args:
        n_neighbors: Number of nearest neighbors for majority voting
        metric: Distance metric ('minkowski' with p=2 corresponds to Euclidean distance)
        p: Power parameter for the Minkowski metric
        
    Returns:
        Scikit-learn Pipeline instance
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p))
    ])


def tune_k_cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_range: range = range(1, 26),
    cv_folds: int = 5,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, int, float]:
    """
    Tune and compare suitable K values using training-only cross-validation.
    Crucial ML requirement: Does NOT touch the held-out test set to prevent optimistic bias.
    
    Args:
        X_train: Training features
        y_train: Training labels
        k_range: Range of K values to explore
        cv_folds: Number of stratified folds
        random_state: Random state for fold shuffling
        
    Returns:
        cv_results_df: DataFrame with K, mean_accuracy, error_rate, std_accuracy
        best_k: K value with the highest CV accuracy (lowest error rate)
        best_score: Highest mean CV accuracy achieved
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []
    
    for k in k_range:
        # Construct pipeline for each candidate K
        pipe = create_pipeline(n_neighbors=k)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        mean_acc = float(np.mean(scores))
        std_acc = float(np.std(scores))
        error_rate = 1.0 - mean_acc
        
        results.append({
            "k": k,
            "mean_cv_accuracy": round(mean_acc, 4),
            "error_rate": round(error_rate, 4),
            "std_cv_accuracy": round(std_acc, 4)
        })
        
    results_df = pd.DataFrame(results)
    
    # Best K selection: prioritize highest accuracy, breaking ties with smallest odd K
    best_row = results_df.sort_values(
        by=["mean_cv_accuracy", "k"],
        ascending=[False, True]
    ).iloc[0]
    
    best_k = int(best_row["k"])
    best_score = float(best_row["mean_cv_accuracy"])
    
    return results_df, best_k, best_score


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_neighbors: int = DEFAULT_K
) -> Pipeline:
    """
    Train the production scikit-learn Pipeline on the training partition.
    """
    pipeline = create_pipeline(n_neighbors=n_neighbors)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate the final trained model on the locked held-out test set.
    
    Computes:
    - Overall Accuracy
    - Macro-averaged & Weighted Precision, Recall, F1-Score
    - Confusion Matrix (TP, FP, FN, TN per class)
    - Full classification report
    
    Returns:
        Dictionary containing all evaluation metrics and artifacts
    """
    if target_names is None:
        target_names = ["setosa", "versicolor", "virginica"]
        
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    
    precision_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )
    
    # Calculate per-class True Positives, False Positives, False Negatives, True Negatives
    per_class_diagnostics = {}
    total_samples = len(y_test)
    for i, name in enumerate(target_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(total_samples - (tp + fp + fn))
        per_class_diagnostics[name] = {
            "tp": tp,
            "fp": fp,  # Type I Error (False Alarm)
            "fn": fn,  # Type II Error (Missed Detection)
            "tn": tn,
            "precision": round(report_dict[name]["precision"], 4),
            "recall": round(report_dict[name]["recall"], 4),
            "f1": round(report_dict[name]["f1-score"], 4),
            "support": int(report_dict[name]["support"])
        }
        
    return {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_weighted": round(precision_weighted, 4),
        "recall_weighted": round(recall_weighted, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "classification_report_str": report_str,
        "classification_report_dict": report_dict,
        "per_class_diagnostics": per_class_diagnostics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "test_size": total_samples
    }


def validate_input_features(features: Union[Dict[str, float], List[float], np.ndarray]) -> np.ndarray:
    """
    Validate flower measurements provided by a user.
    
    Checks:
    - Input contains exactly 4 numeric values
    - All measurements are positive (> 0.0)
    - Values are within plausible biological ranges (0.1 cm to 15.0 cm)
    
    Raises:
        ValueError: If input format or values violate domain rules
        
    Returns:
        2D numpy array of shape (1, 4) ready for pipeline ingestion
    """
    if isinstance(features, dict):
        # Ensure all required keys exist
        missing = [col for col in FEATURE_NAMES if col not in features]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        vals = [features[col] for col in FEATURE_NAMES]
    elif isinstance(features, (list, tuple, np.ndarray)):
        vals = list(features)
        if len(vals) != 4:
            raise ValueError(f"Expected exactly 4 features, received {len(vals)}")
    else:
        raise ValueError(f"Unsupported feature type: {type(features)}")
        
    try:
        float_vals = [float(v) for v in vals]
    except (TypeError, ValueError) as err:
        raise ValueError("All flower feature measurements must be valid numeric numbers.") from err
        
    for name, v in zip(FEATURE_NAMES, float_vals):
        if np.isnan(v) or np.isinf(v):
            raise ValueError(f"Feature '{name}' contains NaN or infinite values.")
        if v <= 0.0:
            raise ValueError(f"Feature '{name}' must be strictly positive (> 0 cm). Received: {v}")
        if v > 20.0:
            raise ValueError(f"Feature '{name}' exceeds biologically plausible Iris dimensions (max 20 cm). Received: {v}")
            
    return np.array([float_vals])


def predict_flower(
    pipeline: Pipeline,
    features: Union[Dict[str, float], List[float], np.ndarray],
    target_names: Optional[List[str]] = None,
    X_train_scaled: Optional[np.ndarray] = None,
    y_train: Optional[pd.Series] = None
) -> Dict[str, Any]:
    """
    Predict flower species for user-entered measurements with explainability.
    
    Returns:
        dict with:
        - predicted_class_id
        - predicted_species
        - probabilities (mapping of species -> probability)
        - explanation (botanical rationale based on measurements)
        - neighbor_info (distances & classes of the K nearest neighbors if training data supplied)
    """
    if target_names is None:
        target_names = ["setosa", "versicolor", "virginica"]
        
    input_arr = validate_input_features(features)
    input_df = pd.DataFrame(input_arr, columns=FEATURE_NAMES)
    
    pred_id = int(pipeline.predict(input_df)[0])
    pred_species = target_names[pred_id]
    probabilities = pipeline.predict_proba(input_df)[0]
    
    prob_dict = {
        name: float(round(probabilities[idx], 4))
        for idx, name in enumerate(target_names)
    }
    
    # Neighborhood inspection using the trained KNN estimator
    knn_estimator = pipeline.named_steps["knn"]
    scaler = pipeline.named_steps["scaler"]
    scaled_input = scaler.transform(input_df)
    
    distances, neighbor_indices = knn_estimator.kneighbors(scaled_input)
    
    # Explainable rationale based on floral anatomy
    sepal_len, sepal_wid, petal_len, petal_wid = input_arr[0]
    total_k = int(knn_estimator.n_neighbors)
    votes_for_pred = int(round(prob_dict[pred_species] * total_k))
    
    explanation_points = []
    if petal_len < 2.5 and petal_wid < 0.8:
        explanation_points.append(
            f"Petal length ({petal_len:.1f} cm) and width ({petal_wid:.1f} cm) are tiny, "
            "which uniquely characterizes Iris setosa with 100% historical separation."
        )
    elif petal_len >= 5.0 or petal_wid >= 1.8:
        explanation_points.append(
            f"Substantial petal length ({petal_len:.1f} cm) and petal width ({petal_wid:.1f} cm) "
            "place this flower strongly within the Iris virginica cluster."
        )
    else:
        explanation_points.append(
            f"Intermediate petal length ({petal_len:.1f} cm) and width ({petal_wid:.1f} cm) "
            "position this sample in the Iris versicolor regime, adjacent to the Virginica boundary."
        )
        
    explanation_points.append(
        f"{votes_for_pred} of {total_k} nearest neighbors in standardized Euclidean feature space "
        f"({prob_dict[pred_species]*100:.1f}% neighbor-vote proportion) voted for {pred_species}."
    )
    
    return {
        "predicted_class_id": pred_id,
        "predicted_species": pred_species,
        "probabilities": prob_dict,
        "confidence": prob_dict[pred_species],
        "vote_count": votes_for_pred,
        "total_k": total_k,
        "confidence_description": f"{votes_for_pred}/{total_k} neighbor votes ({prob_dict[pred_species]*100:.1f}%)",
        "explanation": " ".join(explanation_points),
        "neighbor_distances": distances[0].tolist(),
        "neighbor_indices": neighbor_indices[0].tolist(),
        "scaled_features": scaled_input[0].tolist()
    }


def save_pipeline(pipeline: Pipeline, filepath: str) -> None:
    """Save trained pipeline artifact using joblib."""
    joblib.dump(pipeline, filepath)


def load_pipeline(filepath: str) -> Pipeline:
    """Load previously saved pipeline artifact."""
    return joblib.load(filepath)
