"""
Defend Engine - Ensemble head.

Input: [GBM score, GNN score, sequence score, a few raw high-signal features
like kyc_doc_similarity_score].
Output: final fraud probability + per-subsystem attribution for the dashboard.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix


RAW_SIGNAL_COLUMNS = [
    "kyc_doc_similarity_score", "amount_zscore_vs_self", "graph_shared_device_count",
    "has_dynamic_refurl", "null_device_x_velocity", "kyc_borderline_risk"
]


@dataclass
class EnsembleTrainResult:
    val_auc: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_fpr: float
    weights: dict[str, float]


class EnsembleFraudModel:
    def __init__(self):
        self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
        self.feature_names = ["gbm_score", "gnn_score", "sequence_score"] + RAW_SIGNAL_COLUMNS
        self.threshold = 0.5

    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[["gbm_score", "gnn_score", "sequence_score"]].copy()
        for col in RAW_SIGNAL_COLUMNS:
            X[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)
        return X

    def predict_with_risk_calibration(self, ensemble_scores: np.ndarray, feature_df: pd.DataFrame) -> np.ndarray:
        """
        Lower the decision threshold for transactions with multiple correlated weak signals.
        """
        threshold = np.full(len(ensemble_scores), self.threshold)
        
        composite_flags = (
            (feature_df.get("has_dynamic_refurl", 0) == 1).astype(int) +
            (feature_df.get("device_fingerprint_was_null", 0) == True).astype(int) +
            (feature_df.get("kyc_borderline_risk", 0) == 1).astype(int) +
            (pd.to_numeric(feature_df.get("ip_asn_risk_score", 0), errors="coerce").fillna(0) > 0.7).astype(int)
        )
        
        threshold[composite_flags >= 2] = 0.32
        return (ensemble_scores >= threshold).astype(int)

    def train(self, df: pd.DataFrame, label_col: str = "is_fraud") -> EnsembleTrainResult:
        X = self._build_feature_matrix(df)
        y = df[label_col].astype(int)

        self.model.fit(X, y)
        probs = self.model.predict_proba(X)[:, 1]

        # threshold sweep for best F1
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, thresholds = precision_recall_curve(y, probs)
        f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
        best_idx = int(np.nanargmax(f1s[:-1])) if len(thresholds) > 0 else 0
        self.threshold = float(thresholds[best_idx]) if len(thresholds) > 0 else 0.5

        preds = self.predict_with_risk_calibration(probs, df)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        fpr = fp / max(fp + tn, 1)

        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = float("nan")

        weights = dict(zip(self.feature_names, self.model.coef_[0].tolist()))

        return EnsembleTrainResult(
            val_auc=auc,
            val_precision=precision_score(y, preds, zero_division=0),
            val_recall=recall_score(y, preds, zero_division=0),
            val_f1=f1_score(y, preds, zero_division=0),
            val_fpr=fpr,
            weights=weights,
        )

    def score(self, df: pd.DataFrame) -> np.ndarray:
        X = self._build_feature_matrix(df)
        return self.model.predict_proba(X)[:, 1]

    def score_with_attribution(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._build_feature_matrix(df)
        contributions = X.values * self.model.coef_[0]
        contrib_df = pd.DataFrame(contributions, columns=[f"contrib_{c}" for c in self.feature_names], index=df.index)
        contrib_df["final_score"] = self.score(df)
        contrib_df["flagged"] = self.predict_with_risk_calibration(contrib_df["final_score"].values, df)
        return contrib_df


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(42)
    n = 500
    is_fraud = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    df = pd.DataFrame({
        "gbm_score": np.clip(is_fraud * 0.6 + rng.normal(0, 0.2, n), 0, 1),
        "gnn_score": np.clip(is_fraud * 0.4 + rng.normal(0, 0.25, n), 0, 1),
        "sequence_score": np.clip(is_fraud * 0.5 + rng.normal(0, 0.2, n), 0, 1),
        "kyc_doc_similarity_score": np.clip(is_fraud * 0.3 + rng.normal(0.5, 0.2, n), 0, 1),
        "amount_zscore_vs_self": rng.normal(0, 1, n) + is_fraud * 2,
        "graph_shared_device_count": rng.poisson(0.2, n) + is_fraud * 2,
        "has_dynamic_refurl": is_fraud,
        "device_fingerprint_was_null": is_fraud,
        "kyc_borderline_risk": is_fraud,
        "ip_asn_risk_score": rng.normal(0, 1, n),
        "is_fraud": is_fraud,
    })

    ens = EnsembleFraudModel()
    result = ens.train(df)
    print(f"AUC={result.val_auc:.3f} Precision={result.val_precision:.3f} "
          f"Recall={result.val_recall:.3f} F1={result.val_f1:.3f} FPR={result.val_fpr:.3f}")
    print("Learned weights (subsystem attribution):", result.weights)