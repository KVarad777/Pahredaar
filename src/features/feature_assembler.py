"""
Feature Pipeline - Assembler.

Processes a round's transactions IN TIMESTAMP ORDER (critical - velocity/graph/
behavioral state must only reflect what happened BEFORE each transaction, or
you leak future information into the features and get unrealistically good
offline metrics that won't hold up in a real online setting).

Output: one flat feature vector per transaction, ready for the GBM baseline
and the ensemble head. The GNN and sequence model consume a subset/different
view of this (graph subgraph, last-N sequence) - see src/defend/*.
"""

import pandas as pd

from .velocity_store import VelocityStore
from .graph_state import GraphState
from .behavioral_baseline import BehavioralBaseline


class FeatureAssembler:
    def __init__(self):
        self.velocity = VelocityStore()
        self.graph = GraphState()
        self.behavioral = BehavioralBaseline()

    def assemble(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("timestamp").reset_index(drop=True)
        rows = []

        for _, txn in df.iterrows():
            account_id = txn.get("account_id") or txn.get("payerVpa", "unknown")
            ts = txn["timestamp"]
            amount = float(txn["amount"])

            device_details = txn.get("deviceDetails") or {}
            if not isinstance(device_details, dict):
                device_details = {}
            device_id = device_details.get("deviceId")
            ip_address = device_details.get("ipAddress")

            # --- lookup BEFORE update ---
            velocity_feats = self.velocity.lookup(account_id, ts)
            graph_feats = self.graph.account_features(account_id)
            behavioral_feats = self.behavioral.lookup(account_id, ts, amount)

            # --- now update state ---
            self.velocity.update(account_id, ts, amount)
            self.graph.add_transaction(account_id, device_id, ip_address)
            self.behavioral.update(account_id, ts, amount)

            row = dict(txn)
            row.update({f"vel_{k}": v for k, v in velocity_feats.items()})
            row.update(graph_feats)
            row.update(behavioral_feats)

            row["device_fingerprint_was_null"] = row.get("device_fingerprint_was_null", device_id is None)
            row["ip_asn_risk_score_was_null"] = row.get("ip_asn_risk_score_was_null", False)

            rows.append(row)

        final_df = pd.DataFrame(rows)

        # --- AI-TARGETED FEATURE EXTRACTORS ---
        # 1. RefURL / Dynamic Campaign Anomaly
        final_df["has_dynamic_refurl"] = final_df.get("refUrl", "").fillna("").apply(
            lambda x: 1 if ("?" in str(x) or len(str(x).split("/")) > 4 or "claim" in str(x).lower() or "promo" in str(x).lower()) else 0
        )
        
        # 2. Multi-Signal Missingness & Velocity Interaction
        final_df["null_device_x_velocity"] = final_df["device_fingerprint_was_null"].astype(int) * final_df.get("vel_txn_count_1h", 0).fillna(0)
        
        # 3. Biometric / KYC Friction Anomaly
        if "kyc_doc_similarity_score" in final_df.columns:
            final_df["kyc_borderline_risk"] = pd.to_numeric(
                final_df["kyc_doc_similarity_score"], errors="coerce"
            ).fillna(0).apply(lambda x: 1 if (0.45 <= float(x) <= 0.65) else 0)
        else:
            final_df["kyc_borderline_risk"] = 0
            
        return final_df
  
    def gbm_feature_columns(self) -> list[str]:
        """The flat feature set the GBM baseline and ensemble raw-feature slot consume."""
        return [
            "amount",
            "vel_txn_count_1h", "vel_txn_sum_1h",
            "vel_txn_count_24h", "vel_txn_sum_24h",
            "vel_txn_count_7d", "vel_txn_sum_7d",
            "graph_degree", "graph_shared_device_count",
            "graph_shared_ip_count", "graph_2hop_account_count",
            "amount_zscore_vs_self", "login_time_deviation_hrs", "prior_txn_count",
            "device_fingerprint_was_null", "ip_asn_risk_score_was_null",
            "is_new_account",
            "has_dynamic_refurl", "null_device_x_velocity", "kyc_borderline_risk"
        ]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.generate.orchestrator import GenerateOrchestrator

    demo_scenarios = [{
        "scenario_id": "demo01", "scenario_name": "Low-and-slow velocity abuse",
        "f3_tactic": "Evasion", "f3_technique": "Low-and-Slow Velocity Abuse",
        "mechanism_description": "test", "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
        "manipulation_type": "behavioral", "novelty_tag": "test",
    }]
    orch = GenerateOrchestrator()
    raw_df = orch.run_round(demo_scenarios, n_legit_accounts=30, round_n=98, out_dir="data/generated_smoketest")

    assembler = FeatureAssembler()
    feat_df = assembler.assemble(raw_df)
    print(feat_df[["account_id", "amount", "vel_txn_count_1h", "graph_degree",
                    "amount_zscore_vs_self", "is_fraud"]].head(10))
    print(f"\nAssembled {len(feat_df)} rows with {len(assembler.gbm_feature_columns())} GBM features.")