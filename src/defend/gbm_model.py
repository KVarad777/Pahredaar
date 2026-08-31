"""
Defend Engine - GBM baseline (LightGBM).

Input: full flat feature vector (~40-60 features in a real system; fewer here
for the hackathon feature set, see feature_assembler.gbm_feature_columns()).
Output: fraud probability 0-1.

Trains fast, handles nulls natively via learned split directions (raw nulls +
flag columns, per the spec's model-level null-handling rule), and gives
interpretable feature importances - good first thing to get working.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score


@dataclass
class GBMTrainResult:
    model: lgb.Booster
    val_auc: float
    val_f1: float
    best_threshold: float
    feature_importance: pd.Series


class GBMFraudModel:
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns
        self.model: lgb.Booster | None = None
        self.threshold: float = 0.5

    def train(self, df: pd.DataFrame, label_col: str = "is_fraud",
              val_size: float = 0.2, seed: int = 42) -> GBMTrainResult:
        X = df[self.feature_columns].copy()
        y = df[label_col].astype(int)

        # LightGBM handles NaN natively - just make sure boolean/object cols are numeric-friendly
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
            elif X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=seed, stratify=y if y.sum() > 1 else None,
        )

        pos = max(y_train.sum(), 1)
        neg = max(len(y_train) - y_train.sum(), 1)
        scale_pos_weight = neg / pos  # set from actual fraud:legit ratio, per the spec

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": 31,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": scale_pos_weight,
            "verbose": -1,
            "seed": seed,
        }

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=500,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        val_probs = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        val_auc = roc_auc_score(y_val, val_probs) if y_val.nunique() > 1 else float("nan")

        # pick threshold maximizing F1 on validation set (report FPR alongside recall, per spec)
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
        best_idx = int(np.nanargmax(f1s[:-1])) if len(thresholds) > 0 else 0
        self.threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5
        val_f1 = float(f1s[best_idx]) if len(thresholds) > 0 else 0.0

        importance = pd.Series(
            self.model.feature_importance(importance_type="gain"),
            index=self.feature_columns,
        ).sort_values(ascending=False)

        return GBMTrainResult(
            model=self.model, val_auc=val_auc, val_f1=val_f1,
            best_threshold=self.threshold, feature_importance=importance,
        )

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability 0-1 for each row."""
        if self.model is None:
            raise RuntimeError("Call train() before score().")
        X = df[self.feature_columns].copy()
        for col in X.columns:
            if X[col].dtype == bool:
                X[col] = X[col].astype(int)
            elif X[col].dtype == object:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        return self.model.predict(X, num_iteration=self.model.best_iteration)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.generate.orchestrator import GenerateOrchestrator
    from src.features.feature_assembler import FeatureAssembler

    scenarios = [
        {"scenario_id": "s1", "scenario_name": "low-and-slow", "f3_tactic": "Evasion",
         "f3_technique": "Low-and-Slow Velocity Abuse", "mechanism_description": "x",
         "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
         "manipulation_type": "behavioral", "novelty_tag": "t"},
        {"scenario_id": "s2", "scenario_name": "device ring", "f3_tactic": "Monetization",
         "f3_technique": "Mule Network Cash-Out", "mechanism_description": "x",
         "fields_manipulated": ["device_fingerprint", "ip_address_hash"],
         "manipulation_type": "network", "novelty_tag": "t"},
    ]
    orch = GenerateOrchestrator()
    raw_df = orch.run_round(scenarios, n_legit_accounts=400, fraud_txns_per_scenario_range=(15, 25),
                             round_n=97, out_dir="data/generated_smoketest")

    assembler = FeatureAssembler()
    feat_df = assembler.assemble(raw_df)

    gbm = GBMFraudModel(feature_columns=assembler.gbm_feature_columns())
    result = gbm.train(feat_df)
    print(f"Val AUC: {result.val_auc:.4f} | Val F1: {result.val_f1:.4f} | threshold: {result.best_threshold:.4f}")
    print("\nTop 5 features by importance:")
    print(result.feature_importance.head(5))
