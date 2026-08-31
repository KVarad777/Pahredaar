"""
Run with: pytest tests/test_defend_pipeline.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.features.velocity_store import VelocityStore
from src.features.graph_state import GraphState
from src.features.behavioral_baseline import BehavioralBaseline
from src.features.feature_assembler import FeatureAssembler
from src.generate.orchestrator import GenerateOrchestrator
from src.defend.train import train_defend_bundle, per_scenario_metrics
from src.reward.blue_reward import compute_blue_reward
from src.reward.red_reward import compute_red_rewards
from src.feedback.miss_explainer import generate_round_feedback


DEMO_SCENARIOS = [
    {
        "scenario_id": "t1", "scenario_name": "test low-and-slow", "f3_tactic": "Evasion",
        "f3_technique": "Low-and-Slow Velocity Abuse", "mechanism_description": "test",
        "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
        "manipulation_type": "behavioral", "novelty_tag": "test",
    },
    {
        "scenario_id": "t2", "scenario_name": "test device ring", "f3_tactic": "Monetization",
        "f3_technique": "Mule Network Cash-Out", "mechanism_description": "test",
        "fields_manipulated": ["device_fingerprint", "ip_address_hash"],
        "manipulation_type": "network", "novelty_tag": "test",
    },
]


def test_velocity_store_no_future_leakage():
    """lookup() before update() must never see the current transaction."""
    from datetime import datetime
    store = VelocityStore()
    ts = datetime(2026, 1, 1, 10, 0, 0)
    feats_before_any_txn = store.lookup("acct", ts)
    assert feats_before_any_txn["txn_count_1h"] == 0
    store.update("acct", ts, 100.0)
    feats_after = store.lookup("acct", ts)  # same instant, but this txn already recorded
    # lookup uses strict < current_ts, so a same-timestamp txn shouldn't count either
    assert feats_after["txn_count_1h"] == 0


def test_graph_state_ring_detection():
    gs = GraphState()
    for i in range(4):
        gs.add_transaction(f"mule_{i}", device_id="shared_device", ip_address=f"ip_{i}")
    feats = gs.account_features("mule_0")
    assert feats["graph_shared_device_count"] == 3  # 3 OTHER accounts on the same device


def test_behavioral_baseline_cold_start():
    bb = BehavioralBaseline()
    from datetime import datetime
    feats = bb.lookup("new_acct", datetime(2026, 1, 1), 100.0)
    assert feats["is_new_account"] is True
    assert feats["prior_txn_count"] == 0


def test_feature_assembler_produces_all_gbm_columns():
    orch = GenerateOrchestrator()
    raw_df = orch.run_round(DEMO_SCENARIOS, n_legit_accounts=20, round_n=90, out_dir="data/generated_test_defend")
    assembler = FeatureAssembler()
    feat_df = assembler.assemble(raw_df)
    for col in assembler.gbm_feature_columns():
        assert col in feat_df.columns, f"missing GBM feature column: {col}"


def test_defend_bundle_end_to_end_and_per_scenario_metrics():
    orch = GenerateOrchestrator()
    raw_df = orch.run_round(DEMO_SCENARIOS, n_legit_accounts=150,
                             fraud_txns_per_scenario_range=(10, 15),
                             round_n=91, out_dir="data/generated_test_defend")
    bundle = train_defend_bundle(raw_df)

    assert "final_score" in bundle.scored_df.columns
    assert "caught_by_model" in bundle.scored_df.columns
    assert 0.0 <= bundle.metrics["ensemble_fpr"] <= 1.0

    scenario_metrics = per_scenario_metrics(bundle.scored_df)
    assert set(scenario_metrics["scenario_name"]) == {"test low-and-slow", "test device ring"}
    assert (scenario_metrics["detection_rate"] >= 0).all()
    assert (scenario_metrics["detection_rate"] <= 1).all()


def test_blue_reward_penalizes_high_fpr():
    df_low_fpr = pd.DataFrame({
        "is_fraud": [True, True, False, False, False, False],
        "caught_by_model": [True, True, False, False, False, False],
        "flagged": [True, True, False, False, False, False],
        "scenario_name": ["s1", "s1", None, None, None, None],
    })
    df_high_fpr = pd.DataFrame({
        "is_fraud": [True, True, False, False, False, False],
        "caught_by_model": [True, True, False, False, False, False],
        "flagged": [True, True, True, True, True, True],  # flags everything
        "scenario_name": ["s1", "s1", None, None, None, None],
    })
    r_low = compute_blue_reward(df_low_fpr, new_scenario_names=["s1"])
    r_high = compute_blue_reward(df_high_fpr, new_scenario_names=["s1"])
    assert r_low.reward > r_high.reward, "high-FPR run should score worse than low-FPR run"


def test_red_reward_higher_for_missed_scenarios():
    metrics = pd.DataFrame({
        "scenario_name": ["easy_scenario", "hard_scenario"],
        "n_txns": [10, 10],
        "n_caught": [10, 1],
        "detection_rate": [1.0, 0.1],
    })
    scenarios = [
        {"scenario_name": "easy_scenario", "fields_manipulated": ["amount"], "manipulation_type": "channel"},
        {"scenario_name": "hard_scenario", "fields_manipulated": ["device_fingerprint"], "manipulation_type": "network"},
    ]
    rewards = compute_red_rewards(metrics, scenarios, scenarios)
    reward_by_name = {r.scenario_name: r.reward for r in rewards}
    assert reward_by_name["hard_scenario"] > reward_by_name["easy_scenario"], \
        "a scenario Blue mostly misses should score a higher red_reward"


def test_miss_explainer_returns_nonempty_for_missed_fraud():
    df = pd.DataFrame([{
        "scenario_name": "device ring", "is_fraud": True, "caught_by_model": False,
        "amount_zscore_vs_self": 0.2, "graph_shared_device_count": 3, "graph_shared_ip_count": 0,
        "vel_txn_count_1h": 1, "device_fingerprint_was_null": False,
        "contrib_gbm_score": 0.8, "contrib_gnn_score": 0.1, "contrib_sequence_score": 0.05,
    }])
    explanations = generate_round_feedback(df)
    assert len(explanations) == 1
    assert "device ring" in explanations[0]


if __name__ == "__main__":
    test_velocity_store_no_future_leakage()
    test_graph_state_ring_detection()
    test_behavioral_baseline_cold_start()
    test_feature_assembler_produces_all_gbm_columns()
    test_defend_bundle_end_to_end_and_per_scenario_metrics()
    test_blue_reward_penalizes_high_fpr()
    test_red_reward_higher_for_missed_scenarios()
    test_miss_explainer_returns_nonempty_for_missed_fraud()
    print("All Blue-team tests passed.")
