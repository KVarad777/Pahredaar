"""
=============================================================================
PROJECT AEGIS: FEEDBACK ENGINE — Miss Explainer
=============================================================================
Generates plain-language explanations for missed fraud transactions.
Rule-based (fast, controllable, demo-safe) per spec Section 7 Step 5.
Fed back into Identify engine's context for next round.
=============================================================================
"""

import logging
from typing import Dict, List

logger = logging.getLogger("AEGIS.Feedback")


class FeedbackEngine:
    """
    Explains WHY missed transactions evaded detection.
    Uses rule-based analysis of feature values and model subscores.
    """

    def explain_misses(self, scored_results: List[Dict],
                       feature_vectors: List[Dict]) -> List[Dict]:
        """
        For each missed fraud transaction, generate a plain-language explanation.
        A 'miss' = is_fraud_actual==1 but decision != 'BLOCK'
        """
        explanations = []
        fv_by_id = {fv.get("_transaction_id", ""): fv for fv in feature_vectors}

        for result in scored_results:
            if result.get("is_fraud_actual", 0) != 1:
                continue
            if result.get("decision") == "BLOCK":
                continue

            txn_id = result.get("transaction_id", "")
            fv = fv_by_id.get(txn_id, {})
            subscores = result.get("subsystem_scores", {})

            reasons = self._analyze_miss(fv, subscores, result)

            explanations.append({
                "transaction_id": txn_id,
                "fraud_score": result.get("fraud_score", 0),
                "decision": result.get("decision", ""),
                "f3_technique": result.get("f3_technique", ""),
                "scenario_id": result.get("scenario_id", ""),
                "fraud_vector": result.get("fraud_vector", ""),
                "explanation": "; ".join(reasons) if reasons else "No specific weakness identified",
                "subsystem_scores": subscores,
                "key_features": self._extract_key_features(fv),
            })

        logger.info(f"[FEEDBACK] Generated {len(explanations)} miss explanations")
        return explanations

    def _analyze_miss(self, fv: Dict, subscores: Dict, result: Dict) -> List[str]:
        """Analyze feature values to determine why this fraud was missed."""
        reasons = []
        technique = result.get("f3_technique", "")

        # Amount-based evasion
        amt = fv.get("amount", 0)
        amt_z = fv.get("amount_zscore", 0)
        if abs(amt_z) < 1.5 and amt < 5000:
            reasons.append(
                f"Amount ({amt:.2f}) stayed within normal bounds (z-score: {amt_z:.2f}), "
                f"making individual transaction indistinguishable from legitimate traffic"
            )

        # Graph signals missed
        graph_degree = fv.get("graph_degree", 0)
        shared_devices = fv.get("shared_device_accounts", 0)
        shared_ips = fv.get("shared_ip_accounts", 0)
        xgb_score = subscores.get("xgboost", 0)
        graph_score = subscores.get("graph_anomaly", 0)

        if shared_devices > 0 and graph_score < 0.5:
            reasons.append(
                f"Shared device signal ({shared_devices} accounts on same device) "
                f"was present but graph model scored low ({graph_score:.3f}), "
                f"indicating ring detection threshold needs lowering"
            )

        if shared_ips > 0 and graph_score < 0.5:
            reasons.append(
                f"Shared IP signal ({shared_ips} accounts) was not weighted "
                f"sufficiently by graph anomaly model"
            )

        # Velocity signals missed
        txn_count_1h = fv.get("txn_count_1h", 0)
        txn_count_24h = fv.get("txn_count_24h", 0)
        if txn_count_1h <= 3 and txn_count_24h <= 10:
            reasons.append(
                f"Velocity stayed below thresholds (1h: {txn_count_1h}, 24h: {txn_count_24h}), "
                f"consistent with low-and-slow evasion pattern"
            )

        # Device fingerprint missing (MNAR signal)
        if fv.get("device_fp_was_null", 0) == 1:
            if xgb_score < 0.6:
                reasons.append(
                    "Device fingerprint was null (possible anti-fingerprinting tool) "
                    "but XGBoost model did not weight this MNAR signal strongly enough"
                )

        # Identity signals
        kyc_sim = fv.get("kyc_doc_similarity_score", 0)
        acct_age = fv.get("account_age_days", 365)
        if kyc_sim > 0.8 and acct_age < 30:
            if xgb_score < 0.7:
                reasons.append(
                    f"High KYC document similarity ({kyc_sim:.2f}) on new account "
                    f"({acct_age} days) — potential deepfake/template reuse not flagged"
                )

        # Behavioral timing weakness
        inter_txn = fv.get("mean_inter_txn_seconds", 86400)
        if inter_txn < 500 and xgb_score < 0.5:
            reasons.append(
                f"Unusually rapid inter-transaction timing ({inter_txn:.0f}s) "
                f"not detected by XGBoost model (score: {xgb_score:.3f})"
            )

        # Behavioral deviation not caught
        hour_dev = fv.get("hour_deviation", 0)
        login_dev = fv.get("login_time_deviation_hrs", 0)
        if hour_dev > 6 or login_dev > 6:
            if xgb_score < 0.5:
                reasons.append(
                    f"Login time deviation ({login_dev:.1f}h from baseline) "
                    f"was not weighted sufficiently by behavioral analysis"
                )

        # Failed auth count
        failed = fv.get("failed_auth_count_24h", 0)
        if failed > 5 and xgb_score < 0.6:
            reasons.append(
                f"High failed auth count ({failed} in 24h) suggests credential stuffing "
                f"but was not weighted strongly in XGBoost model"
            )

        if not reasons:
            reasons.append(
                f"Transaction features closely mimicked legitimate patterns across "
                f"all model subsystems — adversarial evasion was effective"
            )

        return reasons

    def _extract_key_features(self, fv: Dict) -> Dict:
        """Extract the most relevant features for the explanation."""
        return {
            "amount": fv.get("amount", 0),
            "amount_zscore": fv.get("amount_zscore", 0),
            "account_age_days": fv.get("account_age_days", 0),
            "kyc_doc_similarity": fv.get("kyc_doc_similarity_score", 0),
            "device_fp_null": fv.get("device_fp_was_null", 0),
            "failed_auth_24h": fv.get("failed_auth_count_24h", 0),
            "txn_count_1h": fv.get("txn_count_1h", 0),
            "shared_devices": fv.get("shared_device_accounts", 0),
            "geo_velocity": fv.get("geo_velocity_kmh", 0),
        }

    def summarize_round_weaknesses(self, explanations: List[Dict]) -> Dict:
        """Aggregate miss explanations into per-technique summaries."""
        by_technique = {}
        for exp in explanations:
            tech = exp.get("f3_technique", "Unknown")
            if tech not in by_technique:
                by_technique[tech] = {
                    "count": 0,
                    "common_reasons": [],
                    "avg_fraud_score": 0,
                }
            by_technique[tech]["count"] += 1
            by_technique[tech]["common_reasons"].append(exp.get("explanation", ""))
            by_technique[tech]["avg_fraud_score"] += exp.get("fraud_score", 0)

        for tech, data in by_technique.items():
            data["avg_fraud_score"] = round(data["avg_fraud_score"] / max(1, data["count"]), 4)
            # Deduplicate common reasons
            seen = set()
            unique_reasons = []
            for r in data["common_reasons"]:
                short = r[:80]
                if short not in seen:
                    seen.add(short)
                    unique_reasons.append(r)
            data["common_reasons"] = unique_reasons[:3]  # Top 3

        return by_technique
