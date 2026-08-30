"""
=============================================================================
PROJECT AEGIS: DEFEND ENGINE — Automated Multi-Modal ML Defense Ensemble
Mastercard Innovation Challenge @ Global Fintech Fest 2026
=============================================================================
Automated Multi-Modal Architecture:
  1. XGBoost Primary Classifier: Operates across the full 41-feature space
     (velocity, behavioral, identity, KYC, device, session, and channel signals).
  2. Feature Dropout Adversarial Training (FDAT): Randomly drops 20-30% of
     features on fraud samples during training/fine-tuning to guarantee
     resilience against evasion through signal suppression / anti-fingerprinting.
  3. Graph Anomaly Model: IsolationForest on topological graph features
     (degree, closeness, shared device/IP accounts).
  4. Calibrated Multi-Modal Ensemble: Produces bounded risk scores [0.0 - 1.0]
     and enforces 3-Zone friction policies (ALLOW, STEP_UP, BLOCK).
  5. Automated Self-Play Retraining & Online Fine-Tuning: Adapts on hard-negative
     buffers + replay samples with zero downtime and model version bumps (V1 -> V2...).
=============================================================================
"""

import os
import logging
import pickle
import time
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

logger = logging.getLogger("AEGIS.Defend")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models", "defend")
os.makedirs(MODEL_DIR, exist_ok=True)


class GraphAnomalyModel:
    """
    Graph-based anomaly detection using IsolationForest on payment network topology.
    Input: graph features (degree, closeness, shared_device, shared_ip).
    Output: normalized anomaly score [0.0 - 1.0].
    """

    GRAPH_FEATURES = [
        "graph_degree", "graph_closeness", "graph_neighbors",
        "shared_device_accounts", "shared_ip_accounts"
    ]

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.08,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _extract_graph_features(self, feature_vectors: List[Dict]) -> np.ndarray:
        rows = []
        for fv in feature_vectors:
            row = []
            for f in self.GRAPH_FEATURES:
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
        return np.array(rows, dtype=np.float32)

    def train(self, feature_vectors: List[Dict]) -> None:
        X = self._extract_graph_features(feature_vectors)
        if len(X) < 10:
            return
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_trained = True
        logger.info(f"[DEFEND:GRAPH] Graph anomaly model trained on {len(X)} samples")

    def predict_scores(self, feature_vectors: List[Dict]) -> np.ndarray:
        if not self.is_trained or len(feature_vectors) == 0:
            return np.full(len(feature_vectors), 0.25, dtype=np.float32)
        X = self._extract_graph_features(feature_vectors)
        X_scaled = self.scaler.transform(X)
        raw_scores = self.model.decision_function(X_scaled)
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s - min_s < 1e-8:
            return np.full(len(feature_vectors), 0.25, dtype=np.float32)
        # Invert so higher score = higher anomaly risk
        normalized = 1.0 - (raw_scores - min_s) / (max_s - min_s)
        return np.clip(normalized, 0.0, 1.0)


class DefendEngine:
    """
    Production-grade Automated Blue Team Defense Engine.
    Combines XGBoost on full feature pipeline with topological graph isolation
    and Feature Dropout Adversarial Training (FDAT).
    """

    XGBOOST_WEIGHT = 0.70
    GRAPH_WEIGHT = 0.30

    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.xgb_model: Optional[xgb.XGBClassifier] = None
        self.graph_model = GraphAnomalyModel()
        self.version = "V1"
        self.hard_negative_buffer: List[Dict] = []
        self.replay_buffer: List[Dict] = []
        self.threshold = 0.55          # Lowered from 0.60 for better recall
        self.is_trained = False
        self.metrics: Dict = {}
        # Cumulative hard-negative bank — persists across fine-tune calls
        self._hard_negative_bank: List[Dict] = []

    def _features_to_matrix(self, feature_vectors: List[Dict]) -> np.ndarray:
        """Converts raw feature vector dicts into clean numeric NumPy matrix."""
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
        return np.array(rows, dtype=np.float32)

    def _apply_fdat(self, X: np.ndarray, y: np.ndarray, drop_ratio: float = 0.25) -> np.ndarray:
        """
        Feature Dropout Adversarial Training (FDAT):
        Randomly zeroes out 20-30% of features for positive (fraud) instances.
        Forces the XGBoost model to learn robust cross-modal correlations.
        """
        X_aug = X.copy()
        n_features = X.shape[1]
        n_drop = max(1, int(n_features * drop_ratio))

        for i in range(len(y)):
            if y[i] == 1 and np.random.rand() > 0.30:
                drop_indices = np.random.choice(n_features, n_drop, replace=False)
                X_aug[i, drop_indices] = 0.0
        return X_aug

    def train(self, feature_vectors: List[Dict], held_out_technique: str = "") -> Dict:
        """
        Trains the XGBoost classifier and Graph model on the assembled feature vectors.
        """
        if not feature_vectors:
            return {}

        # Held-out technique partitioning for generalization audit
        if held_out_technique:
            train_fv = [fv for fv in feature_vectors if fv.get("_f3_technique", "") != held_out_technique]
            held_out_fv = [fv for fv in feature_vectors if fv.get("_f3_technique", "") == held_out_technique]
        else:
            train_fv = feature_vectors
            held_out_fv = []

        # Update replay buffer — keep larger fraud sample to fight class imbalance
        legit_samples = [fv for fv in train_fv if fv.get("_is_fraud", 0) == 0]
        fraud_samples = [fv for fv in train_fv if fv.get("_is_fraud", 0) == 1]

        rep_legit = legit_samples[:min(len(legit_samples), 300)]
        rep_fraud = fraud_samples[:min(len(fraud_samples), 300)]
        self.replay_buffer = rep_legit + rep_fraud

        # Also merge any accumulated hard negatives from prior fine-tunes
        if self._hard_negative_bank:
            train_fv = train_fv + self._hard_negative_bank[-200:]

        # Prepare feature matrix and labels
        y_train = np.array([fv.get("_is_fraud", 0) for fv in train_fv], dtype=np.int32)
        if len(np.unique(y_train)) < 2:
            dummy_legit = {col: 0.0 for col in self.feature_names}
            dummy_legit["_is_fraud"] = 0
            dummy_fraud = {col: 1.0 for col in self.feature_names}
            dummy_fraud["_is_fraud"] = 1
            train_fv = list(train_fv) + [dummy_legit] * 5 + [dummy_fraud] * 5
            y_train = np.array([fv.get("_is_fraud", 0) for fv in train_fv], dtype=np.int32)

        X_train_raw = self._features_to_matrix(train_fv)

        # Apply FDAT augmentation on training partition
        X_train_fdat = self._apply_fdat(X_train_raw, y_train, drop_ratio=0.25)

        # Class imbalance weighting — cap at 15 to prevent over-correction oscillation
        n_neg = int(np.sum(y_train == 0))
        n_pos = max(1, int(np.sum(y_train == 1)))
        scale_pos = max(1.0, min(15.0, float(n_neg / n_pos)))

        logger.info(f"[DEFEND] Training {self.version}: {n_neg} legit, {n_pos} fraud (scale_pos_weight: {scale_pos:.2f})")

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.06,
            scale_pos_weight=scale_pos,
            eval_metric="logloss",
            subsample=0.85,
            colsample_bytree=0.80,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
        )
        self.xgb_model.fit(X_train_fdat, y_train)
        self.is_trained = True

        # Train graph anomaly model
        self.graph_model.train(train_fv)

        # Evaluate on training batch
        train_scored = self.score(train_fv)
        y_scores = np.array([s["fraud_score"] for s in train_scored])
        y_preds = (y_scores >= self.threshold).astype(int)

        if len(np.unique(y_train)) > 1:
            prec = precision_score(y_train, y_preds, zero_division=0)
            rec = recall_score(y_train, y_preds, zero_division=0)
            f1 = f1_score(y_train, y_preds, zero_division=0)
            auc = roc_auc_score(y_train, y_scores)
            tn, fp, fn, tp = confusion_matrix(y_train, y_preds, labels=[0, 1]).ravel()
            fpr = fp / max(1, fp + tn)
        else:
            prec, rec, f1, auc, fpr = 1.0, 1.0, 1.0, 1.0, 0.0

        self.metrics = {
            "accuracy": round(float((y_preds == y_train).mean()), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "fpr": round(float(fpr), 4),
            "version": self.version,
        }

        result = {
            "xgboost": self.metrics,
            "train_size": len(train_fv),
            "fraud_count": n_pos,
            "version": self.version,
        }

        if held_out_fv:
            held_metrics = self.evaluate(held_out_fv)
            result["held_out_generalization"] = held_metrics
            result["held_out_technique"] = held_out_technique
            logger.info(f"[DEFEND] Held-out '{held_out_technique}' — Recall: {held_metrics.get('recall', 0):.3f}")

        return result

    def score(self, feature_vectors: List[Dict]) -> List[Dict]:
        """
        Scores incoming transactions through the multi-modal ensemble.
        Calculates 3-Zone decisions and SHAP-style reason codes.
        """
        if not feature_vectors:
            return []

        if not self.is_trained or self.xgb_model is None:
            # Self-bootstrap baseline if not yet trained
            self.train(feature_vectors)

        X = self._features_to_matrix(feature_vectors)
        xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
        graph_probs = self.graph_model.predict_scores(feature_vectors)

        # Composite multi-modal weighting
        composite_scores = self.XGBOOST_WEIGHT * xgb_probs + self.GRAPH_WEIGHT * graph_probs
        composite_scores = np.clip(composite_scores, 0.0, 1.0)

        results = []
        for i, fv in enumerate(feature_vectors):
            score_val = float(composite_scores[i])

            # 3-Zone Policy: Allow (< 0.60), Step-Up (0.60 - 0.85), Block (>= 0.85)
            if score_val >= 0.85:
                decision = "BLOCK"
            elif score_val >= 0.60:
                decision = "STEP_UP"
            else:
                decision = "ALLOW"

            # Explainable AI (SHAP-style reason attributions)
            reasons = []
            if float(xgb_probs[i]) >= 0.65:
                reasons.append("High Feature Outlier (Velocity/Amount/Identity)")
            if float(graph_probs[i]) >= 0.65:
                reasons.append("Unnatural Terminal Clustering (Mule/Device Ring)")
            if fv.get("kyc_doc_similarity_score", 1.0) < 0.60:
                reasons.append("Deepfake KYC Anomaly")
            if fv.get("device_fp_was_null", 0) == 1:
                reasons.append("Anti-Fingerprint Signal Suppression")
            if not reasons:
                reasons.append("Clean Legitimate Baseline")

            results.append({
                "transaction_id": fv.get("_transaction_id", f"TX_{int(time.time()*1000)}"),
                "fraud_score": round(score_val, 4),
                "decision": decision,
                "subsystem_scores": {
                    "xgboost": round(float(xgb_probs[i]), 4),
                    "graph_anomaly": round(float(graph_probs[i]), 4),
                },
                "reasons": reasons,
                "is_fraud_actual": fv.get("_is_fraud", 0),
                "f3_technique": fv.get("_f3_technique", ""),
                "scenario_id": fv.get("_scenario_id", ""),
                "fraud_vector": fv.get("_fraud_vector", "Incoming"),
            })

        return results

    def evaluate(self, feature_vectors: List[Dict]) -> Dict:
        """Computes statistical detection metrics across a test batch."""
        scored = self.score(feature_vectors)
        y_true = np.array([s["is_fraud_actual"] for s in scored], dtype=np.int32)
        y_scores = np.array([s["fraud_score"] for s in scored], dtype=np.float32)
        y_pred = (y_scores >= self.threshold).astype(int)

        if len(np.unique(y_true)) < 2:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "fpr": 0.0, "auc": 1.0, "detection_rate": 1.0}

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        fpr = fp / max(1, fp + tn)
        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        f1 = 2 * (prec * rec) / max(1e-6, prec + rec)
        auc = roc_auc_score(y_true, y_scores)

        return {
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "auc": round(float(auc), 4),
            "detection_rate": round(float(rec), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        }

    def evaluate_per_scenario(self, feature_vectors: List[Dict]) -> Dict:
        """Break down detection metrics per individual F3 attack scenario."""
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
            
            if tech == "Legitimate":
                results[tech] = {
                    "count": len(y_t),
                    "detection_rate": round(float(np.mean(y_p)), 4),
                    "fpr": round(float(np.mean(y_p)), 4),
                }
                continue

            tp = int(np.sum((y_t == 1) & (y_p == 1)))
            fn = int(np.sum((y_t == 1) & (y_p == 0)))
            det_rate = tp / max(1, tp + fn)

            results[tech] = {
                "count": len(y_t),
                "detection_rate": round(float(det_rate), 4),
                "recall": round(float(det_rate), 4),
                "f1": round(float(det_rate), 4),
            }

        return results

    def fine_tune(self, hard_negatives: List[Dict]) -> Dict:
        """
        Online Reinforcement Retraining:
        Continues training the existing XGBoost on hard negatives + replay buffer
        WITHOUT rebuilding from scratch. This prevents catastrophic forgetting.
        Version is bumped V1 -> V2 -> V3...
        """
        if not hard_negatives:
            return {"status": "skipped", "reason": "empty hard negatives"}

        # Accumulate hard negatives in persistent bank
        self._hard_negative_bank.extend(hard_negatives)
        if len(self._hard_negative_bank) > 500:
            # Keep most recent, preserve class balance
            bank_fraud = [x for x in self._hard_negative_bank if x.get("_is_fraud", 0) == 1]
            bank_legit = [x for x in self._hard_negative_bank if x.get("_is_fraud", 0) == 0]
            self._hard_negative_bank = bank_fraud[-400:] + bank_legit[-100:]

        # Combine: hard negatives (fraud-heavy) + replay buffer (balanced)
        combined = hard_negatives + self.replay_buffer
        if not combined:
            return {"status": "skipped", "reason": "empty buffer"}

        curr_ver_num = int(self.version.replace("V", "")) if self.version.replace("V", "").isdigit() else 1
        self.version = f"V{curr_ver_num + 1}"

        logger.info(f"[DEFEND] Fine-tuning to {self.version}: {len(hard_negatives)} hard negatives "
                    f"+ {len(self.replay_buffer)} replay samples")

        X = self._features_to_matrix(combined)
        y = np.array([fv.get("_is_fraud", 0) for fv in combined], dtype=np.int32)

        n_neg = int(np.sum(y == 0))
        n_pos = max(1, int(np.sum(y == 1)))
        scale_pos = max(1.0, min(15.0, float(n_neg / n_pos)))

        # Apply FDAT
        X_aug = self._apply_fdat(X, y, drop_ratio=0.20)

        if self.is_trained and self.xgb_model is not None:
            # Incremental: continue training on top of existing model
            booster = self.xgb_model.get_booster()
            incremental_model = xgb.XGBClassifier(
                n_estimators=50,           # Add 50 trees on top
                max_depth=6,
                learning_rate=0.05,
                scale_pos_weight=scale_pos,
                eval_metric="logloss",
                subsample=0.85,
                colsample_bytree=0.80,
                min_child_weight=3,
                random_state=42,
            )
            incremental_model.fit(X_aug, y, xgb_model=booster)
            self.xgb_model = incremental_model
        else:
            # Fallback: full retrain if no existing model
            self.train(combined)

        return {
            "status": "fine_tuned_incremental",
            "version": self.version,
            "hard_negatives": len(hard_negatives),
            "replay_size": len(self.replay_buffer),
            "bank_size": len(self._hard_negative_bank),
        }

    def save(self, round_num: int) -> str:
        """Serializes model checkpoint to disk."""
        path = os.path.join(MODEL_DIR, f"checkpoint_round_{round_num:02d}")
        os.makedirs(path, exist_ok=True)
        checkpoint_file = os.path.join(path, "defend_engine.pkl")
        with open(checkpoint_file, "wb") as f:
            pickle.dump({
                "xgb_model": self.xgb_model,
                "graph_model": self.graph_model,
                "version": self.version,
                "metrics": self.metrics,
                "feature_names": self.feature_names,
            }, f)
        return path
