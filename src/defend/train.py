"""
Defend Engine - full training orchestrator.

Trains GBM -> GNN -> Sequence -> joins their scores -> trains Ensemble on top.
Call this once per round from the reward loop orchestrator (src/reward/loop_orchestrator.py).
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.feature_assembler import FeatureAssembler
from src.defend.gbm_model import GBMFraudModel
from src.defend.gnn_model import GNNFraudModel
from src.defend.sequence_model import SequenceFraudModel, build_account_sequences
from src.defend.ensemble import EnsembleFraudModel, RAW_SIGNAL_COLUMNS


@dataclass
class DefendBundle:
    feature_assembler: FeatureAssembler
    gbm: GBMFraudModel
    gnn: GNNFraudModel
    sequence: SequenceFraudModel
    ensemble: EnsembleFraudModel
    scored_df: pd.DataFrame
    metrics: dict = field(default_factory=dict)


def train_defend_bundle(raw_df: pd.DataFrame, feature_assembler: FeatureAssembler | None = None) -> DefendBundle:
    """
    raw_df: one round's transactions from the Generate engine's orchestrator
            (data/generated/round_XX/transactions.csv equivalent, already in memory).
    feature_assembler: pass an existing one to carry forward cumulative velocity/graph/
            behavioral state across rounds (recommended for the real self-play loop -
            see loop_orchestrator.py). Pass None to start fresh (fine for a one-off run).
    """
    assembler = feature_assembler or FeatureAssembler()
    feat_df = assembler.assemble(raw_df)

    # ---- 1. GBM ----
    gbm = GBMFraudModel(feature_columns=assembler.gbm_feature_columns())
    gbm_result = gbm.train(feat_df)
    feat_df["gbm_score"] = gbm.score(feat_df)

    # ---- 2. GNN ----
    account_ids = feat_df["account_id"].dropna().unique().tolist()
    labels = feat_df.groupby("account_id")["is_fraud"].max().astype(int).to_dict()
    gnn = GNNFraudModel()
    gnn_result = gnn.train(assembler.graph, account_ids, labels)
    gnn_scores = gnn.score(assembler.graph, account_ids)
    feat_df["gnn_score"] = feat_df["account_id"].map(gnn_scores).fillna(0.0)

    # ---- 3. Sequence model ----
    sequences = build_account_sequences(feat_df)
    seq_model = SequenceFraudModel()
    seq_result = seq_model.train(sequences, labels)
    seq_scores = seq_model.score(sequences)
    feat_df["sequence_score"] = feat_df["account_id"].map(seq_scores).fillna(0.0)

    # ensure raw-signal columns the ensemble expects exist even if not present this round
    for col in RAW_SIGNAL_COLUMNS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0

    # ---- 4. Ensemble ----
    ensemble = EnsembleFraudModel()
    ens_result = ensemble.train(feat_df)
    attribution = ensemble.score_with_attribution(feat_df)
    feat_df["final_score"] = attribution["final_score"]
    feat_df["flagged"] = attribution["flagged"]
    feat_df["caught_by_model"] = ((feat_df["flagged"] == 1) & (feat_df["is_fraud"] == True))

    metrics = {
        "gbm_val_auc": gbm_result.val_auc,
        "gnn_val_auc": gnn_result.val_auc,
        "gnn_backend": "torch_geometric" if gnn_result.epochs_trained > 0 else "fallback",
        "sequence_val_auc": seq_result.val_auc,
        "sequence_backend": seq_result.backend,
        "ensemble_auc": ens_result.val_auc,
        "ensemble_precision": ens_result.val_precision,
        "ensemble_recall": ens_result.val_recall,
        "ensemble_f1": ens_result.val_f1,
        "ensemble_fpr": ens_result.val_fpr,
        "ensemble_weights": ens_result.weights,
    }

    return DefendBundle(
        feature_assembler=assembler, gbm=gbm, gnn=gnn, sequence=seq_model,
        ensemble=ensemble, scored_df=feat_df, metrics=metrics,
    )


def per_scenario_metrics(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Breaks efficacy down per scenario type, not just aggregate - per the
    requirements doc: a single blended number hides which techniques are
    actually caught vs missed.
    """
    fraud_only = scored_df[scored_df["is_fraud"] == True]
    if fraud_only.empty:
        return pd.DataFrame(columns=["scenario_name", "n_txns", "n_caught", "detection_rate"])

    grouped = fraud_only.groupby("scenario_name").agg(
        n_txns=("is_fraud", "count"),
        n_caught=("caught_by_model", "sum"),
    ).reset_index()
    grouped["detection_rate"] = grouped["n_caught"] / grouped["n_txns"]
    return grouped


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.generate.orchestrator import GenerateOrchestrator

    scenarios = [
        {"scenario_id": "s1", "scenario_name": "low-and-slow", "f3_tactic": "Evasion",
         "f3_technique": "Low-and-Slow Velocity Abuse", "mechanism_description": "x",
         "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
         "manipulation_type": "behavioral", "novelty_tag": "t"},
        {"scenario_id": "s2", "scenario_name": "device ring", "f3_tactic": "Monetization",
         "f3_technique": "Mule Network Cash-Out", "mechanism_description": "x",
         "fields_manipulated": ["device_fingerprint", "ip_address_hash"],
         "manipulation_type": "network", "novelty_tag": "t"},
        {"scenario_id": "s3", "scenario_name": "card testing", "f3_tactic": "Monetization",
         "f3_technique": "Card-Not-Present Abuse", "mechanism_description": "x",
         "fields_manipulated": ["amount", "channel"],
         "manipulation_type": "channel", "novelty_tag": "t"},
    ]
    orch = GenerateOrchestrator()
    raw_df = orch.run_round(scenarios, n_legit_accounts=400, fraud_txns_per_scenario_range=(15, 25),
                             round_n=94, out_dir="data/generated_smoketest")

    bundle = train_defend_bundle(raw_df)
    print("\n=== Round metrics ===")
    for k, v in bundle.metrics.items():
        print(f"{k}: {v}")

    print("\n=== Per-scenario detection ===")
    print(per_scenario_metrics(bundle.scored_df))
