"""
train_model.py - End-to-End Training & Evaluation Script.
DecodeLabs Industrial Training Kit | Batch 2026 | AI Project 2

Demonstrates the Complete 7-Stage Machine Learning Pipeline:
Stage 1: Load Iris Dataset (150 samples, 4 features, 3 classes)
Stage 2: Explore Dataset Distributions and Class Balance
Stage 3: Stratified 80/20 Train-Test Split (120 train, 30 test)
Stage 4: Feature Scaling via StandardScaler (Fit on Train only, prevent leakage)
Stage 5: Train KNeighborsClassifier (Baseline K=5) & Cross-Validation Tuning (K=1 to 25)
Stage 6: Final Evaluation on Locked Test Set (Accuracy, Precision, Recall, F1, Confusion Matrix)
Stage 7: Interactive Sample Predictions & Pipeline Serialization
"""

import json
import os
import sys
import numpy as np
import pandas as pd

from model import (
    load_iris_data,
    split_data,
    create_pipeline,
    tune_k_cross_validation,
    train_final_model,
    evaluate_model,
    predict_flower,
    save_pipeline,
    FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    SPECIES_METADATA,
    RANDOM_STATE,
    DEFAULT_K
)


def run_training_pipeline(model_path: str = "iris_knn_pipeline.joblib", metrics_path: str = "metrics.json"):
    print("=" * 80)
    print("DECODELABS ARTIFICIAL INTELLIGENCE PROJECT 2")
    print("Project: Data Classification Using AI — Iris Flower Classification")
    print("Industrial Training Kit | Batch 2026 | Powered by DecodeLabs")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # STAGE 1: LOAD THE DATASET
    # ---------------------------------------------------------
    print("\n[STAGE 1/7] LOADING BENCHMARK DATASET...")
    X, y, target_names, feature_names, full_df = load_iris_data()
    print(f"-> Successfully loaded Iris dataset: {X.shape[0]} samples, {X.shape[1]} features.")
    print(f"-> Features: {', '.join(feature_names)}")
    print(f"-> Target Classes: {', '.join(target_names)} (IDs: 0, 1, 2)")
    
    # ---------------------------------------------------------
    # STAGE 2: EXPLORE DATASET MORPHOLOGY & CHARACTERISTICS
    # ---------------------------------------------------------
    print("\n[STAGE 2/7] EXPLORATORY DATA ANALYSIS (EDA)...")
    print("Class Distribution:")
    class_counts = full_df["species"].value_counts()
    for species_name, count in class_counts.items():
        print(f"   • {species_name.capitalize():<12}: {count} samples ({count/len(full_df)*100:.1f}%) [Perfect Balance]")
        
    print("\nMorphological Features Explained:")
    for feat in FEATURE_NAMES:
        print(f"   • {feat:<20}: {FEATURE_DESCRIPTIONS[feat]}")
        
    print("\nFeature Summary Statistics (Mean ± Std [Min - Max]):")
    stats = X.describe().T
    for col in FEATURE_NAMES:
        mean_val = stats.loc[col, "mean"]
        std_val = stats.loc[col, "std"]
        min_val = stats.loc[col, "min"]
        max_val = stats.loc[col, "max"]
        print(f"   • {col:<20}: {mean_val:.2f} ± {std_val:.2f} cm [{min_val:.1f} - {max_val:.1f} cm]")

    # ---------------------------------------------------------
    # STAGE 3: SHUFFLE & STRATIFIED TRAIN/TEST SPLIT
    # ---------------------------------------------------------
    print("\n[STAGE 3/7] STRUCTURAL INTEGRITY: THE SPLIT...")
    print(f"-> Splitting strategy: 80% Training, 20% Testing with Stratification (random_state={RANDOM_STATE}).")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=RANDOM_STATE)
    print(f"-> Training partition size: {X_train.shape[0]} samples (40 per class)")
    print(f"-> Held-out test partition : {X_test.shape[0]} samples (10 per class [LOCKED FOR FINAL VALIDATION])")
    
    # ---------------------------------------------------------
    # STAGE 4 & 5: SCALING & K-NN TRAINING / HYPERPARAMETER TUNING
    # ---------------------------------------------------------
    print("\n[STAGE 4/7] THE GATEKEEPER RULE: SCALING & LEAKAGE PREVENTION...")
    print("-> Wrapping StandardScaler & KNeighborsClassifier inside a scikit-learn Pipeline.")
    print("-> Guarantee: StandardScaler is fitted strictly on the training folds/set.")
    
    print("\n[STAGE 5/7] HYPERPARAMETER TUNING: EXPLORING 'K' VIA 5-FOLD CV...")
    print("-> Evaluating K values from 1 to 25 using 5-Fold Stratified Cross-Validation on TRAINING SET ONLY.")
    cv_results, best_k, best_score = tune_k_cross_validation(X_train, y_train, k_range=range(1, 26), cv_folds=5)
    
    print("Cross-Validation Error Curve Highlights:")
    print("   K=1  (High variance / Overfitting risk): Error Rate = {:.4f}, CV Acc = {:.4f}".format(
        cv_results.loc[cv_results['k'] == 1, 'error_rate'].values[0],
        cv_results.loc[cv_results['k'] == 1, 'mean_cv_accuracy'].values[0]
    ))
    print(f"   K={DEFAULT_K}  (Baseline specification)         : Error Rate = {cv_results.loc[cv_results['k'] == DEFAULT_K, 'error_rate'].values[0]:.4f}, CV Acc = {cv_results.loc[cv_results['k'] == DEFAULT_K, 'mean_cv_accuracy'].values[0]:.4f}")
    print(f"   K={best_k} (Optimal CV Elbow candidate)     : Error Rate = {cv_results.loc[cv_results['k'] == best_k, 'error_rate'].values[0]:.4f}, CV Acc = {best_score:.4f}")
    print("   K=25 (High bias / Underfitting risk)    : Error Rate = {:.4f}, CV Acc = {:.4f}".format(
        cv_results.loc[cv_results['k'] == 25, 'error_rate'].values[0],
        cv_results.loc[cv_results['k'] == 25, 'mean_cv_accuracy'].values[0]
    ))
    
    # Fit baseline pipeline using standard K=5 (mandated baseline)
    print(f"\n[STAGE 5A] TRAINING BASELINE PRODUCTION MODEL (K={DEFAULT_K})...")
    pipeline_k5 = train_final_model(X_train, y_train, n_neighbors=DEFAULT_K)
    print("-> Baseline pipeline successfully fitted on training data.")
    
    # Fit tuned pipeline using best K from training cross-validation (optimal K=3)
    print(f"\n[STAGE 5B] TRAINING CV-TUNED MODEL (Optimal K={best_k})...")
    pipeline_tuned = train_final_model(X_train, y_train, n_neighbors=best_k)
    print(f"-> Tuned pipeline (K={best_k}) successfully fitted on training data.")

    # ---------------------------------------------------------
    # STAGE 6: EVALUATION ON LOCKED TEST SET
    # ---------------------------------------------------------
    print("\n[STAGE 6/7] EVALUATION ON HELD-OUT TEST SET (30 SAMPLES)...")
    eval_k5 = evaluate_model(pipeline_k5, X_test, y_test, target_names=target_names)
    eval_tuned = evaluate_model(pipeline_tuned, X_test, y_test, target_names=target_names)
    
    print("\n" + "-" * 70)
    print(f"{'METRIC':<25} | {'BASELINE (K=5)':<20} | {'CV-TUNED (K=' + str(best_k) + ')':<20}")
    print("-" * 70)
    print(f"{'Test Accuracy':<25} | {eval_k5['accuracy']*100:>18.2f}% | {eval_tuned['accuracy']*100:>18.2f}%")
    print(f"{'Macro Precision':<25} | {eval_k5['precision_macro']:>20.4f} | {eval_tuned['precision_macro']:>20.4f}")
    print(f"{'Macro Recall':<25} | {eval_k5['recall_macro']:>20.4f} | {eval_tuned['recall_macro']:>20.4f}")
    print(f"{'Macro F1-Score':<25} | {eval_k5['f1_macro']:>20.4f} | {eval_tuned['f1_macro']:>20.4f}")
    print(f"{'Weighted F1-Score':<25} | {eval_k5['f1_weighted']:>20.4f} | {eval_tuned['f1_weighted']:>20.4f}")
    print("-" * 70)
    
    print("\nBaseline (K=5) Confusion Matrix (Rows: Actual, Columns: Predicted):")
    print("                  Predicted Setosa  Versicolor  Virginica")
    for i, name in enumerate(target_names):
        row = eval_k5['confusion_matrix'][i]
        print(f"   Actual {name.capitalize():<11}:       {row[0]:<10} {row[1]:<11} {row[2]:<10}")
        
    print(f"\nCV-Tuned (K={best_k}) Confusion Matrix:")
    print("                  Predicted Setosa  Versicolor  Virginica")
    for i, name in enumerate(target_names):
        row = eval_tuned['confusion_matrix'][i]
        print(f"   Actual {name.capitalize():<11}:       {row[0]:<10} {row[1]:<11} {row[2]:<10}")

    print("\nDiagnostic Breakdown per Class for Baseline K=5 (TP, FP, FN, TN):")
    for name, diag in eval_k5["per_class_diagnostics"].items():
        print(f"   • {name.capitalize():<12}: TP={diag['tp']}, FP={diag['fp']} (False Alarm), FN={diag['fn']} (Missed), TN={diag['tn']} | F1={diag['f1']:.4f}")
        
    print("\nWhy Accuracy Alone is Not Enough (The '99% Accuracy Mirage'):")
    print("   In balanced datasets like Iris, accuracy gives a fair top-level indicator.")
    print("   However, in real-world imbalanced datasets (e.g. medical diagnosis or fraud),")
    print("   a trivial model predicting the majority class can achieve 99% accuracy while")
    print("   having a Recall of 0% for the critical positive class. Thus, evaluating")
    print("   Precision, Recall, F1-Score, and the Confusion Matrix is essential.")

    # ---------------------------------------------------------
    # STAGE 7: SAMPLE PREDICTIONS & SERIALIZATION
    # ---------------------------------------------------------
    print("\n[STAGE 7/7] VALIDATING SAMPLE PREDICTIONS & SAVING ARTIFACTS...")
    sample_tests = [
        {"desc": "Clear Setosa sample", "data": [5.1, 3.5, 1.4, 0.2]},
        {"desc": "Clear Versicolor sample", "data": [5.7, 2.8, 4.1, 1.3]},
        {"desc": "Clear Virginica sample", "data": [6.3, 3.3, 6.0, 2.5]}
    ]
    for sample in sample_tests:
        pred = predict_flower(pipeline_k5, sample["data"], target_names=target_names)
        print(f"-> {sample['desc']} {sample['data']}:")
        print(f"   Prediction: {pred['predicted_species']} (Vote proportion: {pred['confidence_description']})")
        print(f"   Rationale : {pred['explanation']}")
        
    # Serialize both pipelines
    save_pipeline(pipeline_k5, model_path)
    tuned_model_path = model_path.replace(".joblib", "_tuned.joblib")
    save_pipeline(pipeline_tuned, tuned_model_path)
    print(f"\n-> Serialized Baseline Pipeline (K=5) saved to: {os.path.abspath(model_path)}")
    print(f"-> Serialized Tuned Pipeline (K={best_k}) saved to: {os.path.abspath(tuned_model_path)}")
    
    # Save metrics JSON
    metrics_export = {
        "dataset": {
            "samples": int(len(full_df)),
            "features": FEATURE_NAMES,
            "classes": target_names,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "random_state": RANDOM_STATE
        },
        "baseline_k5_test_metrics": {
            "k": DEFAULT_K,
            "accuracy": eval_k5["accuracy"],
            "precision_macro": eval_k5["precision_macro"],
            "recall_macro": eval_k5["recall_macro"],
            "f1_macro": eval_k5["f1_macro"],
            "precision_weighted": eval_k5["precision_weighted"],
            "recall_weighted": eval_k5["recall_weighted"],
            "f1_weighted": eval_k5["f1_weighted"],
            "confusion_matrix": eval_k5["confusion_matrix"].tolist(),
            "per_class_diagnostics": eval_k5["per_class_diagnostics"]
        },
        "cv_tuning": {
            "best_k": best_k,
            "best_cv_accuracy": best_score,
            "cv_table": cv_results.to_dict(orient="records")
        },
        "tuned_test_metrics": {
            "k": best_k,
            "accuracy": eval_tuned["accuracy"],
            "precision_macro": eval_tuned["precision_macro"],
            "recall_macro": eval_tuned["recall_macro"],
            "f1_macro": eval_tuned["f1_macro"],
            "precision_weighted": eval_tuned["precision_weighted"],
            "recall_weighted": eval_tuned["recall_weighted"],
            "f1_weighted": eval_tuned["f1_weighted"],
            "confusion_matrix": eval_tuned["confusion_matrix"].tolist(),
            "per_class_diagnostics": eval_tuned["per_class_diagnostics"]
        }
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"-> Saved evaluation metrics to: {os.path.abspath(metrics_path)}")
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    return pipeline_k5, eval_k5, cv_results


if __name__ == "__main__":
    run_training_pipeline()
