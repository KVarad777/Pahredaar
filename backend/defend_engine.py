"""
=============================================================================
PROJECT AEGIS: DEFEND ENGINE — Multi-Model Ensemble (GBM + GNN + LSTM)
=============================================================================
Real detection pipeline per spec Section 6:
  1. GBM (LightGBM/XGBoost): full flat feature vector (~40 features) -> fraud prob
  2. GNN proxy (NetworkX features + IsolationForest): ring/anomaly score
  3. Sequence model (sklearn-based): sequence anomaly score
  4. Ensemble head (Logistic Regression): final fraud probability + attribution
  5. Held-out scenario split for generalization testing
=============================================================================
"""

import os
import logging
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, confusion_matrix
)

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

logger = logging.getLogger("AEGIS.Defend")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "defend")


class TabularModel:
    """
    GBM tabular model (LightGBM baseline with scale_pos_weight).
    Input: full flat feature vector (~40-60 features)
    Output: fraud probability score 0-1
    """

    def __init__(self, fraud_ratio: float = 0.035):
        self.scale_pos_weight = max(1.0, float((1.0 - fraud_ratio) / fraud_ratio))
        if HAS_LIGHTGBM:
            self.model = LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                num_leaves=31,
                learning_rate=0.05,
                subsample=0.8,
                scale_pos_weight=self.scale_pos_weight,
                random_state=42,
                verbosity=-1,
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importances_ = None

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        X_scaled = self.scaler.fit_transform(X)

        if HAS_LIGHTGBM:
            self.model.fit(X_scaled, y)
        else:
            # Compute sample weights for class imbalance
            n_fraud = max(1, np.sum(y == 1))
            n_legit = max(1, np.sum(y == 0))
            weight_fraud = n_legit / n_fraud
            sample_weights = np.where(y == 1, weight_fraud, 1.0)
            self.model.fit(X_scaled, y, sample_weight=sample_weights)

        self.is_trained = True
        self.feature_importances_ = getattr(self.model, "feature_importances_", None)

        # Validation metrics
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)

        return {
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "auc": float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            return np.full(len(X), 0.5)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def fine_tune(self, X_new: np.ndarray, y_new: np.ndarray,
                  X_replay: np.ndarray, y_replay: np.ndarray) -> Dict:
        """Fine-tune on hard negatives + replay sample (avoids catastrophic forgetting)."""
        X_combined = np.vstack([X_new, X_replay])
        y_combined = np.concatenate([y_new, y_replay])
        return self.train(X_combined, y_combined)


class GraphAnomalyModel:
    """
    GNN proxy using graph-derived features + IsolationForest.
    Input: graph features (degree, closeness, shared_device, shared_ip)
    Output: ring-membership / node-anomaly score 0-1
    """

    GRAPH_FEATURES = [
        "graph_degree", "graph_closeness", "graph_neighbors",
        "shared_device_accounts", "shared_ip_accounts"
    ]

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_graph_features(self, feature_vectors: List[Dict]) -> np.ndarray:
        return np.array([
            [fv.get(f, 0.0) for f in self.GRAPH_FEATURES]
            for fv in feature_vectors
        ])

    def train(self, feature_vectors: List[Dict]) -> None:
        X = self._extract_graph_features(feature_vectors)
        if len(X) < 10:
            return
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info("[DEFEND:GNN] Graph anomaly model trained")

    def predict_scores(self, feature_vectors: List[Dict]) -> np.ndarray:
        if not self.is_trained:
            return np.full(len(feature_vectors), 0.5)
        X = self._extract_graph_features(feature_vectors)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        # Normalize to 0-1 (more negative = more anomalous)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s < 1e-8:
            return np.full(len(feature_vectors), 0.5)
        normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
        return np.clip(normalized, 0, 1)


class SequenceAnomalyModel:
    """
    Sequence model proxy for detecting low-and-slow patterns.
    Uses behavioral sequence features + IsolationForest.
    Input: temporal/behavioral features from last N transactions
    Output: sequence-anomaly score 0-1
    """

    SEQUENCE_FEATURES = [
        "amount_zscore", "hour_deviation", "inter_txn_zscore",
        "mean_inter_txn_seconds", "login_time_deviation_hrs",
        "txn_count_1h", "txn_count_24h",
    ]

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_seq_features(self, feature_vectors: List[Dict]) -> np.ndarray:
        return np.array([
            [fv.get(f, 0.0) for f in self.SEQUENCE_FEATURES]
            for fv in feature_vectors
        ])

    def train(self, feature_vectors: List[Dict]) -> None:
        X = self._extract_seq_features(feature_vectors)
        if len(X) < 10:
            return
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info("[DEFEND:SEQ] Sequence anomaly model trained")

    def predict_scores(self, feature_vectors: List[Dict]) -> np.ndarray:
        if not self.is_trained:
            return np.full(len(feature_vectors), 0.5)
        X = self._extract_seq_features(feature_vectors)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s < 1e-8:
            return np.full(len(feature_vectors), 0.5)
        normalized = 1 - (raw_scores - min_s) / (max_s - min_s)
        return np.clip(normalized, 0, 1)


class EnsembleHead:
    """
    Simple logistic regression ensemble.
    Input: [GBM score, GNN score, Seq score, high-signal features]
    Output: final fraud probability + per-subsystem attribution
    """

    def __init__(self):
        self.model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42,
        )
        self.is_trained = False

    def train(self, gbm_scores: np.ndarray, gnn_scores: np.ndarray,
              seq_scores: np.ndarray, raw_features: np.ndarray,
              y: np.ndarray) -> Dict:
        X = np.column_stack([gbm_scores, gnn_scores, seq_scores, raw_features])

        # Balance weights
        n_fraud = max(1, np.sum(y == 1))
        n_legit = max(1, np.sum(y == 0))
        weights = np.where(y == 1, n_legit / n_fraud, 1.0)

        self.model.fit(X, y, sample_weight=weights)
        self.is_trained = True

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        return {
            "ensemble_precision": float(precision_score(y, preds, zero_division=0)),
            "ensemble_recall": float(recall_score(y, preds, zero_division=0)),
            "ensemble_f1": float(f1_score(y, preds, zero_division=0)),
            "component_weights": {
                "gbm": float(self.model.coef_[0][0]),
                "gnn": float(self.model.coef_[0][1]),
                "sequence": float(self.model.coef_[0][2]),
            }
        }

    def predict_proba(self, gbm_scores: np.ndarray, gnn_scores: np.ndarray,
                      seq_scores: np.ndarray, raw_features: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            # Fallback: weighted average
            return 0.4 * gbm_scores + 0.3 * gnn_scores + 0.3 * seq_scores
        X = np.column_stack([gbm_scores, gnn_scores, seq_scores, raw_features])
        return self.model.predict_proba(X)[:, 1]


class DefendEngine:
    """
    Master Defend Engine coordinating all model components.
    """

    # High-signal features passed directly to ensemble head
    HIGH_SIGNAL_FEATURES = ["kyc_doc_similarity_score", "device_fp_was_null", "ip_hash_was_null"]

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.tabular = TabularModel()
        self.graph_model = GraphAnomalyModel()
        self.sequence_model = SequenceAnomalyModel()
        self.ensemble = EnsembleHead()
        self.version = "V1"
        self.hard_negative_buffer: List[Dict] = []
        self.replay_buffer: List[Dict] = []
        self.threshold = 0.5

    def _features_to_array(self, feature_vectors: List[Dict]) -> np.ndarray:
        rows = []
        for fv in feature_vectors:
            row = []
            for fn in self.feature_names:
                v = fv.get(fn, 0.0)
                if isinstance(v, bool):
                    row.append(1.0 if v else 0.0)
                elif v is None:
                    row.append(0.0)
                else:
                    try:
                        row.append(float(v))
                    except (ValueError, TypeError):
                        row.append(0.0)
            rows.append(row)
        return np.array(rows)

    def _get_high_signal(self, feature_vectors: List[Dict]) -> np.ndarray:
        rows = []
        for fv in feature_vectors:
            row = []
            for f in self.HIGH_SIGNAL_FEATURES:
                v = fv.get(f, 0.0)
                if isinstance(v, bool):
                    row.append(1.0 if v else 0.0)
                elif v is None:
                    row.append(0.0)
                else:
                    try:
                        row.append(float(v))
                    except (ValueError, TypeError):
                        row.append(0.0)
            rows.append(row)
        return np.array(rows)

    def train(self, feature_vectors: List[Dict], held_out_technique: str = "") -> Dict:
        """
        Train all model components.
        Optionally holds out one scenario type for generalization testing.
        """
        # Split held-out if specified
        if held_out_technique:
            train_fv = [fv for fv in feature_vectors if fv.get("_f3_technique", "") != held_out_technique]
            held_out_fv = [fv for fv in feature_vectors if fv.get("_f3_technique", "") == held_out_technique]
        else:
            train_fv = feature_vectors
            held_out_fv = []

        X_train = self._features_to_array(train_fv)
        y_train = np.array([fv.get("_is_fraud", 0) for fv in train_fv])

        # Save replay buffer (random sample of training data for future fine-tuning)
        replay_size = min(500, len(train_fv) // 5)
        indices = np.random.choice(len(train_fv), replay_size, replace=False) if len(train_fv) > replay_size else range(len(train_fv))
        self.replay_buffer = [train_fv[i] for i in indices]

        # 1. Train tabular model
        logger.info("[DEFEND] Training tabular (GBM) model...")
        tabular_metrics = self.tabular.train(X_train, y_train)

        # 2. Train graph model
        logger.info("[DEFEND] Training graph anomaly model...")
        self.graph_model.train(train_fv)

        # 3. Train sequence model
        logger.info("[DEFEND] Training sequence anomaly model...")
        self.sequence_model.train(train_fv)

        # 4. Get all component scores for ensemble training
        gbm_scores = self.tabular.predict_proba(X_train)
        gnn_scores = self.graph_model.predict_scores(train_fv)
        seq_scores = self.sequence_model.predict_scores(train_fv)
        high_signal = self._get_high_signal(train_fv)

        # 5. Train ensemble
        logger.info("[DEFEND] Training ensemble head...")
        ensemble_metrics = self.ensemble.train(gbm_scores, gnn_scores, seq_scores, high_signal, y_train)

        result = {
            "tabular": tabular_metrics,
            "ensemble": ensemble_metrics,
            "train_size": len(train_fv),
            "fraud_count": int(np.sum(y_train)),
            "version": self.version,
        }

        # Evaluate on held-out set if available
        if held_out_fv:
            held_out_metrics = self.evaluate(held_out_fv)
            result["held_out_generalization"] = held_out_metrics
            result["held_out_technique"] = held_out_technique
            logger.info(f"[DEFEND] Held-out '{held_out_technique}' — "
                        f"Recall: {held_out_metrics.get('recall', 0):.3f}")

        return result

    def score(self, feature_vectors: List[Dict]) -> List[Dict]:
        """
        Score transactions through the full ensemble.
        Returns per-transaction score + subsystem attribution.
        """
        X = self._features_to_array(feature_vectors)
        high_signal = self._get_high_signal(feature_vectors)

        gbm_scores = self.tabular.predict_proba(X)
        gnn_scores = self.graph_model.predict_scores(feature_vectors)
        seq_scores = self.sequence_model.predict_scores(feature_vectors)
        final_scores = self.ensemble.predict_proba(gbm_scores, gnn_scores, seq_scores, high_signal)

        results = []
        for i, fv in enumerate(feature_vectors):
            score_val = float(final_scores[i])
            if score_val >= 0.85:
                decision = "BLOCK"
            elif score_val >= 0.60:
                decision = "STEP_UP"
            else:
                decision = "ALLOW"

            results.append({
                "transaction_id": fv.get("_transaction_id", ""),
                "fraud_score": round(score_val, 4),
                "decision": decision,
                "subsystem_scores": {
                    "tabular_gbm": round(float(gbm_scores[i]), 4),
                    "graph_gnn": round(float(gnn_scores[i]), 4),
                    "sequence_lstm": round(float(seq_scores[i]), 4),
                },
                "is_fraud_actual": fv.get("_is_fraud", 0),
                "f3_technique": fv.get("_f3_technique", ""),
                "scenario_id": fv.get("_scenario_id", ""),
                "fraud_vector": fv.get("_fraud_vector", ""),
            })

        return results

    def evaluate(self, feature_vectors: List[Dict]) -> Dict:
        """Compute detection metrics for a set of feature vectors."""
        scored = self.score(feature_vectors)
        y_true = np.array([s["is_fraud_actual"] for s in scored])
        y_scores = np.array([s["fraud_score"] for s in scored])
        y_pred = (y_scores >= self.threshold).astype(int)

        if len(np.unique(y_true)) < 2:
            return {"precision": 0, "recall": 0, "f1": 0, "fpr": 0, "auc": 0}

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)

        return {
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "fpr": round(float(fpr), 4),
            "auc": round(float(roc_auc_score(y_true, y_scores)), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    def evaluate_per_scenario(self, feature_vectors: List[Dict]) -> Dict:
        """Break down detection metrics per F3 technique."""
        scored = self.score(feature_vectors)
        by_technique = {}

        for s in scored:
            tech = s.get("f3_technique", "") or "Legitimate"
            if tech not in by_technique:
                by_technique[tech] = {"y_true": [], "y_pred": [], "y_score": []}
            by_technique[tech]["y_true"].append(s["is_fraud_actual"])
            by_technique[tech]["y_pred"].append(1 if s["fraud_score"] >= self.threshold else 0)
            by_technique[tech]["y_score"].append(s["fraud_score"])

        results = {}
        for tech, data in by_technique.items():
            y_t = np.array(data["y_true"])
            y_p = np.array(data["y_pred"])
            if len(np.unique(y_t)) < 2:
                results[tech] = {
                    "count": len(y_t),
                    "detection_rate": float(np.mean(y_p)) if np.any(y_t) else 0.0,
                }
                continue

            tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
            results[tech] = {
                "count": len(y_t),
                "precision": round(float(precision_score(y_t, y_p, zero_division=0)), 4),
                "recall": round(float(recall_score(y_t, y_p, zero_division=0)), 4),
                "f1": round(float(f1_score(y_t, y_p, zero_division=0)), 4),
                "detection_rate": round(float(tp / max(1, tp + fn)), 4),
                "fpr": round(float(fp / max(1, fp + tn)), 4),
            }

        return results

    def fine_tune(self, hard_negatives: List[Dict]) -> Dict:
        """
        Fine-tune on hard-negative buffer + replay sample.
        Avoids catastrophic forgetting per spec Section 8.
        """
        combined = hard_negatives + self.replay_buffer
        if len(combined) < 20:
            return {"status": "skipped", "reason": "insufficient data for fine-tune"}

        self.version = f"V{int(self.version[1:]) + 1}" if self.version[1:].isdigit() else "V2"
        logger.info(f"[DEFEND] Fine-tuning to {self.version} with "
                    f"{len(hard_negatives)} hard negatives + {len(self.replay_buffer)} replay")
        return self.train(combined)

    def save(self, round_num: int) -> str:
        path = os.path.join(MODEL_DIR, f"checkpoint_round_{round_num:02d}")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "defend_engine.pkl"), "wb") as f:
            pickle.dump({
                "tabular": self.tabular,
                "graph": self.graph_model,
                "sequence": self.sequence_model,
                "ensemble": self.ensemble,
                "version": self.version,
            }, f)
        return path
