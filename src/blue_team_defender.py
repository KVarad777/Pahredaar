"""
=============================================================================
PROJECT AEGIS: SYNCHRONOUS EDGE DEFENDER MODEL (blue_team_defender.py)
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
This module trains the lightweight Synchronous Edge Model for Project AEGIS:
  1. Ingests processed dataset (data/processed/master_aegis_dataset.csv).
  2. Extracts initial low-latency edge tabular features:
     - TransactionAmt (Spend volume)
     - Biometric_Entropy (Behavioral micro-tremor variance)
  3. Corrects extreme real-world class imbalance via SMOTE oversampling.
  4. Trains an optimized XGBoost Classifier (scale_pos_weight=10, eval_metric='auc').
  5. Evaluates classification report, ROC-AUC score, and attack-vector breakout.
  6. Serializes the trained production edge model to models/xgb_edge_model.json.
=============================================================================
"""

import os
import sys
import argparse
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb


# Default Configuration
DEFAULT_DATA_PATH = os.path.join("data", "processed", "master_aegis_dataset.csv")
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "xgb_edge_model.json")
RANDOM_SEED = 42
TEST_SIZE = 0.20


def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads dataset, extracts edge features and targets, and performs stratified splitting.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at: {data_path}")

    print("=" * 80)
    print("  PROJECT AEGIS : BLUE TEAM EDGE MODEL TRAINING PIPELINE")
    print("  Mastercard Innovation Challenge @ Global Fintech Fest 2026")
    print("=" * 80)
    print(f"[*] Ingesting processed dataset from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"[+] Ingested {len(df):,} transactions x {len(df.columns)} features")

    # Resolve target and feature columns
    target_col = "Fraud_Label" if "Fraud_Label" in df.columns else "isFraud"
    features = ["TransactionAmt", "Biometric_Entropy"]

    for feat in features:
        if feat not in df.columns:
            raise KeyError(f"Required feature '{feat}' missing from dataset columns: {df.columns.tolist()}")

    X = df[features].copy()
    y = df[target_col].astype(int).copy()
    attack_types = df["Attack_Type"] if "Attack_Type" in df.columns else pd.Series(["UNKNOWN"] * len(df))

    # Clean any unexpected NaN values
    X["TransactionAmt"] = X["TransactionAmt"].fillna(X["TransactionAmt"].median())
    X["Biometric_Entropy"] = X["Biometric_Entropy"].fillna(0.65)

    print(f"\n[*] Selected Edge Features ({len(features)}): {features}")
    print(f"  • Feature Matrix Shape:  {X.shape}")
    print(f"  • Class Imbalance:       Legitimate (0): {(y == 0).sum():,} | Fraud (1): {(y == 1).sum():,} ({(y == 1).mean() * 100:.2f}%)")

    # Stratified Train/Test Partition (80/20)
    X_train, X_test, y_train, y_test, attack_train, attack_test = train_test_split(
        X, y, attack_types, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\n[*] Partitioned Dataset (Test Size: {int(TEST_SIZE * 100)}%, Stratified):")
    print(f"  • Train Set: {len(X_train):,} samples (Legit: {(y_train == 0).sum():,}, Fraud: {(y_train == 1).sum():,})")
    print(f"  • Test Set:  {len(X_test):,} samples (Legit: {(y_test == 0).sum():,}, Fraud: {(y_test == 1).sum():,})")

    return X_train, X_test, y_train, y_test, attack_test


def balance_training_data_with_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to the training partition.
    """
    print("\n" + "-" * 80)
    print("1. CLASS IMBALANCE CORRECTION (SMOTE OVERSAMPLING)")
    print("-" * 80)
    print(f"[*] Applying SMOTE on minority class (Fraud_Label=1)...")
    
    smote = SMOTE(random_state=RANDOM_SEED, sampling_strategy="auto")
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"  [+] Resampling Complete:")
    print(f"      - Pre-SMOTE Distribution:  Legitimate: {(y_train == 0).sum():,} | Fraud: {(y_train == 1).sum():,}")
    print(f"      - Post-SMOTE Distribution: Legitimate: {(y_train_resampled == 0).sum():,} | Fraud: {(y_train_resampled == 1).sum():,}")
    print(f"      - Synthetic Vectors Added: {(y_train_resampled == 1).sum() - (y_train == 1).sum():,} synthetic minority samples")

    return X_train_resampled, y_train_resampled


def train_edge_xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> xgb.XGBClassifier:
    """
    Initializes and trains the calibrated edge XGBoost classifier.
    """
    print("\n" + "-" * 80)
    print("2. SYNCHRONOUS EDGE MODEL TRAINING (XGBOOST)")
    print("-" * 80)
    
    # Model parameters as specified by AEGIS architecture
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=10,       # High sensitivity penalty for fraud false declines
        eval_metric="auc",          # Optimize Area Under ROC Curve
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_SEED,
        tree_method="hist",
        n_jobs=-1,
    )

    print(f"[*] Training XGBClassifier on {len(X_train):,} balanced samples...")
    print(f"    - scale_pos_weight : 10")
    print(f"    - eval_metric      : auc")
    print(f"    - max_depth        : 5")
    print(f"    - learning_rate    : 0.08")

    model.fit(X_train, y_train)
    print("  [+] Model training completed successfully.")
    return model


def evaluate_edge_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    attack_test: pd.Series,
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics, ROC-AUC, PR-AUC, and vector breakouts.
    """
    print("\n" + "-" * 80)
    print("3. MODEL EVALUATION & PERFORMANCE BENCHMARK")
    print("-" * 80)

    # Inferences
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_str = classification_report(y_test, y_pred, target_names=["Legitimate (0)", "Fraud (1)"], digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"  • ROC-AUC Score:             {roc_auc:.4f}  (Area Under ROC Curve)")
    print(f"  • PR-AUC (Average Precision): {pr_auc:.4f}  (Precision-Recall AUC)")
    print("\n  • Detailed Classification Report:")
    print(report_str)

    print("  • Confusion Matrix:")
    print(f"      [TN: {cm[0][0]:>5} | FP: {cm[0][1]:>5}]")
    print(f"      [FN: {cm[1][0]:>5} | TP: {cm[1][1]:>5}]")

    # -------------------------------------------------------------------------
    # Zero-Day Attack Vector Recall Breakdown
    # -------------------------------------------------------------------------
    print("\n  • Zero-Day Attack Vector Detection Efficacy Breakdown:")
    test_eval_df = X_test.copy()
    test_eval_df["y_true"] = y_test
    test_eval_df["y_pred"] = y_pred
    test_eval_df["y_prob"] = y_prob
    test_eval_df["Attack_Type"] = attack_test.values

    for attack_name, group in test_eval_df.groupby("Attack_Type"):
        total_attack = len(group)
        caught_attack = (group["y_pred"] == 1).sum()
        recall = (caught_attack / max(total_attack, 1)) * 100.0
        avg_prob = group["y_prob"].mean()
        print(f"    - {attack_name:<26}: {caught_attack:>4}/{total_attack:<4} detected ({recall:>6.2f}%) | Mean Risk: {avg_prob:.4f}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
    }


def save_trained_model(model: xgb.XGBClassifier, output_path: str) -> None:
    """
    Serializes the XGBoost model to standard JSON format for low-latency C++ / Python inference.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 80)
    print("4. MODEL SERIALIZATION & DEPLOYMENT ARTIFACT")
    print("-" * 80)
    print(f"[*] Serializing edge model to: {output_path}")
    
    model.save_model(output_path)
    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"  [+] Production model saved successfully ({file_size_kb:.2f} KB)")
    print("=" * 80 + "\n")


def run_pipeline(data_path: str = DEFAULT_DATA_PATH, model_path: str = DEFAULT_MODEL_PATH):
    """Executes the full end-to-end Blue Team Edge Model pipeline."""
    # 1. Load & Partition Data
    X_train, X_test, y_train, y_test, attack_test = load_and_preprocess_data(data_path)

    # 2. Correct Imbalance with SMOTE
    X_train_resampled, y_train_resampled = balance_training_data_with_smote(X_train, y_train)

    # 3. Train XGBoost Model
    edge_model = train_edge_xgboost_model(X_train_resampled, y_train_resampled)

    # 4. Comprehensive Evaluation
    evaluate_edge_model(edge_model, X_test, y_test, attack_test)

    # 5. Serialize Model
    save_trained_model(edge_model, model_path)


def main():
    parser = argparse.ArgumentParser(description="Project AEGIS: Blue Team Edge Model Pipeline")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to processed master CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH, help="Path to save xgb_edge_model.json")
    args = parser.parse_args()

    run_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
