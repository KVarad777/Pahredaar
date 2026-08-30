"""
=============================================================================
PROJECT AEGIS: XGBoost Fraud Detection Model — Training & Inference
=============================================================================
XGBClassifier with:
  - One-hot encoded categoricals (CardType, MerchantCategory, Location)
  - scale_pos_weight for class imbalance
  - Feature importance tracking
  - Joblib model persistence
  - Real-time single-transaction inference
=============================================================================
"""

import os
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

logger = logging.getLogger("AEGIS.XGBoost")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "xgboost")
DATA_DIR = os.path.join(BASE_DIR, "data", "xgboost")

# Columns used for XGBoost training (raw numeric + categorical to one-hot)
IDENTIFIER_COLS = ["TransactionID", "Timestamp", "PAN", "MerchantID", "DeviceID", "IPAddress", "IsFraud"]
CATEGORICAL_COLS = ["CardType", "MerchantCategory", "Location"]

# Numeric features the model operates on (before one-hot expansion)
NUMERIC_FEATURES = [
    "TransactionAmt", "TimeOfDay", "TransactionSpeed", "DailyTransactionCount",
    "MerchantFraudRate", "DegreeCentrality", "ClosenessCentrality", "PageRank", "UserAge"
]


class XGBoostFraudModel:
    """
    XGBoost-based fraud detection model.
    Handles training, evaluation, persistence, and real-time inference.
    """

    def __init__(self):
        self.model: Optional[xgb.XGBClassifier] = None
        self.train_columns: Optional[List[str]] = None
        self.is_trained: bool = False
        self.metrics: Dict = {}
        self.feature_importances: Dict[str, float] = {}

    def train(self, df: pd.DataFrame) -> Dict:
        """
        Train XGBoost on a DataFrame with IsFraud label.
        Returns metrics dict with accuracy, precision, recall, F1, AUC.
        """
        y = df["IsFraud"]
        X = df.drop(columns=IDENTIFIER_COLS, errors="ignore")

        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)
        self.train_columns = list(X_encoded.columns)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.20, random_state=42, stratify=y
        )

        # Class imbalance weight
        num_neg = int(sum(y_train == 0))
        num_pos = max(1, int(sum(y_train == 1)))
        scale_pos_weight = num_neg / num_pos

        logger.info(f"[XGBoost] Training: {num_neg} negative, {num_pos} positive, "
                     f"scale_pos_weight={scale_pos_weight:.2f}")

        # Train
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall_val = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        self.metrics = {
            "accuracy": round(float(accuracy), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall_val), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "fpr": round(float(fp / max(1, fp + tn)), 4),
        }

        # Feature importances
        importances = self.model.feature_importances_
        self.feature_importances = {
            col: round(float(imp), 6)
            for col, imp in zip(self.train_columns, importances)
        }

        # Log top 10
        sorted_fi = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        logger.info("[XGBoost] Training complete. Metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall_val:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
        logger.info("  Top 10 Features:")
        for feat, imp in sorted_fi[:10]:
            logger.info(f"    {feat}: {imp:.4f}")

        return self.metrics

    def predict_proba_single(self, features: Dict) -> float:
        """
        Predict fraud probability for a single transaction.
        Features dict should contain keys matching the training columns.
        """
        if not self.is_trained or self.model is None:
            return 0.5

        # Build a DataFrame row from the features dict
        row = {}
        for col in self.train_columns:
            row[col] = features.get(col, 0.0)

        df_row = pd.DataFrame([row])
        prob = self.model.predict_proba(df_row)[:, 1][0]
        return float(prob)

    def predict_proba_batch(self, features_list: List[Dict]) -> np.ndarray:
        """
        Predict fraud probabilities for a batch of transactions.
        """
        if not self.is_trained or self.model is None:
            return np.full(len(features_list), 0.5)

        rows = []
        for features in features_list:
            row = {}
            for col in self.train_columns:
                row[col] = features.get(col, 0.0)
            rows.append(row)

        df_batch = pd.DataFrame(rows)
        probs = self.model.predict_proba(df_batch)[:, 1]
        return probs

    def save(self, path: Optional[str] = None) -> str:
        """Save model + metadata to disk."""
        if path is None:
            path = MODEL_DIR
        os.makedirs(path, exist_ok=True)

        model_path = os.path.join(path, "fraud_xgb_model.joblib")
        joblib.dump({
            "model": self.model,
            "train_columns": self.train_columns,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances,
        }, model_path)

        logger.info(f"[XGBoost] Model saved to: {model_path}")
        return model_path

    def load(self, path: Optional[str] = None) -> bool:
        """Load model from disk."""
        if path is None:
            path = os.path.join(MODEL_DIR, "fraud_xgb_model.joblib")

        if not os.path.exists(path):
            logger.warning(f"[XGBoost] Model file not found: {path}")
            return False

        try:
            data = joblib.load(path)
            self.model = data["model"]
            self.train_columns = data["train_columns"]
            self.metrics = data.get("metrics", {})
            self.feature_importances = data.get("feature_importances", {})
            self.is_trained = True
            logger.info(f"[XGBoost] Model loaded from: {path}")
            return True
        except Exception as e:
            logger.error(f"[XGBoost] Failed to load model: {e}")
            return False

    def get_metrics(self) -> Dict:
        return self.metrics

    def get_feature_importances(self) -> Dict[str, float]:
        return self.feature_importances


def map_pipeline_features_to_xgboost(fv: Dict, xgb_columns: List[str]) -> Dict:
    """
    Map AEGIS feature pipeline output to XGBoost model input columns.
    The feature pipeline produces keys like 'amount', 'txn_count_1h', 'graph_degree', etc.
    We map these to XGBoost-expected columns from the training dataset.
    """
    mapped = {}

    # Direct numeric mappings
    mapped["TransactionAmt"] = float(fv.get("amount", 0.0))
    mapped["TimeOfDay"] = float(fv.get("_hour_of_day", 12))
    mapped["TransactionSpeed"] = float(fv.get("mean_inter_txn_seconds", 9999.0)) / 60.0  # seconds to minutes
    mapped["DailyTransactionCount"] = float(fv.get("txn_count_24h", 1))
    mapped["MerchantFraudRate"] = float(fv.get("_merchant_fraud_rate", 0.005))
    mapped["DegreeCentrality"] = float(fv.get("graph_degree", 0.01))
    mapped["ClosenessCentrality"] = float(fv.get("graph_closeness", 0.3))
    mapped["PageRank"] = float(fv.get("_pagerank", 0.01))
    mapped["UserAge"] = float(fv.get("_user_age", 35))

    # One-hot encoded categorical mappings
    channel = fv.get("_channel", "")
    card_type = fv.get("_card_type", "")
    # CardType: Debit is drop_first, so CardType_Debit is baseline, CardType_Credit is 1
    # After drop_first=True with alphabetical: Credit is first, Debit is baseline
    # So CardType_Debit is the one-hot column
    mapped["CardType_Debit"] = 1.0 if card_type == "Debit" or channel in ["ATM"] else 0.0

    # MerchantCategory: after drop_first, "Dining" is baseline (alphabetically first after drop)
    # Alphabetical: Dining, Electronics, Entertainment, Grocery, Retail, Travel, Utility
    # drop_first drops "Dining", so columns are: Electronics, Entertainment, Grocery, Retail, Travel, Utility
    mcc = fv.get("_merchant_category", "Grocery")
    for cat in ["Electronics", "Entertainment", "Grocery", "Retail", "Travel", "Utility"]:
        mapped[f"MerchantCategory_{cat}"] = 1.0 if mcc == cat else 0.0

    # Location: after drop_first, "Ahmedabad" is baseline
    # Remaining: Bengaluru, Chennai, Delhi, Hyderabad, Kolkata, Mumbai, Pune
    location = fv.get("_location", "Mumbai")
    for city in ["Bengaluru", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]:
        mapped[f"Location_{city}"] = 1.0 if location == city else 0.0

    # Fill any missing columns with 0
    for col in xgb_columns:
        if col not in mapped:
            mapped[col] = 0.0

    return mapped


def train_and_save_model() -> Dict:
    """
    Full pipeline: generate data, train XGBoost, save model.
    Called on first server start or manual trigger.
    """
    from backend.xgboost_data_generator import generate_fraud_dataset

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Generate dataset
    csv_path = os.path.join(DATA_DIR, "transactions.csv")
    if not os.path.exists(csv_path):
        df = generate_fraud_dataset()
        df.to_csv(csv_path, index=False)
        print(f"Dataset saved: {csv_path}")
    else:
        df = pd.read_csv(csv_path)
        print(f"Dataset loaded from: {csv_path}")

    # Train model
    model = XGBoostFraudModel()
    metrics = model.train(df)
    model.save()

    print("\n" + "=" * 50)
    print("XGBoost Model Training Complete")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_and_save_model()
