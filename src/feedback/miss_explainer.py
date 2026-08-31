"""
Feedback Engine - Miss Explainer.

For every missed fraud transaction, generates a plain-language reason. This
closes the loop from "here's a metric" to "here's a reason," fed back into
the Identify engine's next prompt as context (see reward/loop_orchestrator.py).

Deliberately rule-based, not LLM-based: fast, deterministic, and demo-safe.
A rule-based explainer that's correct beats an LLM one that hallucinates a
reason on stage. Extend the rules below as you add more scenario categories.
"""

import pandas as pd


def explain_miss(row: pd.Series) -> str:
    """Given one missed-fraud transaction row (scored_df with all assembled
    features), return a plain-language explanation of why it likely evaded
    the model. Checked in order - most specific rule wins."""

    amount_z = row.get("amount_zscore_vs_self", 0) or 0
    shared_device = row.get("graph_shared_device_count", 0) or 0
    shared_ip = row.get("graph_shared_ip_count", 0) or 0
    vel_count_1h = row.get("vel_txn_count_1h", 0) or 0
    was_null = row.get("device_fingerprint_was_null", False)
    contrib_gbm = row.get("contrib_gbm_score", 0) or 0
    contrib_gnn = row.get("contrib_gnn_score", 0) or 0
    contrib_seq = row.get("contrib_sequence_score", 0) or 0

    if abs(amount_z) < 1.0 and (shared_device > 0 or shared_ip > 0) and contrib_gnn < contrib_gbm:
        return ("amount stayed within normal bounds, but a graph ring signal "
                "(shared device/IP) wasn't weighted heavily enough by the ensemble")

    if abs(amount_z) < 1.0 and vel_count_1h <= 3 and contrib_seq < contrib_gbm:
        return ("individual transaction amounts looked normal and velocity stayed low - "
                "the sequence model likely needed more history to catch the slow drain pattern")

    if was_null and contrib_gbm < 0.1:
        return ("device fingerprint was null (a real fraud signal for this scenario type), "
                "but the model didn't weight the missingness flag strongly enough")

    if abs(amount_z) < 0.5:
        return ("this transaction's amount was statistically unremarkable for the account - "
                "the anomaly here is likely only visible at an aggregation level the current "
                "feature set doesn't fully capture yet")

    return ("transaction had moderate signal across subsystems but stayed under the "
            "ensemble's decision threshold - consider whether the threshold is tuned "
            "too conservatively for this scenario type")


def generate_round_feedback(scored_df: pd.DataFrame, max_examples: int = 15) -> list[str]:
    """
    Returns a list of plain-language miss explanations for this round, ready
    to pass as `miss_explanations` into ScenarioProposer.propose() next round.
    """
    missed = scored_df[(scored_df["is_fraud"] == True) & (scored_df["caught_by_model"] == False)]
    if missed.empty:
        return []

    explanations = []
    for scenario_name, group in missed.groupby("scenario_name"):
        sample = group.sample(min(3, len(group)), random_state=1)
        for _, row in sample.iterrows():
            reason = explain_miss(row)
            explanations.append(f"[{scenario_name}] {reason}")

    return explanations[:max_examples]


if __name__ == "__main__":
    df = pd.DataFrame([
        {"scenario_name": "device ring", "is_fraud": True, "caught_by_model": False,
         "amount_zscore_vs_self": 0.2, "graph_shared_device_count": 3, "graph_shared_ip_count": 0,
         "vel_txn_count_1h": 1, "device_fingerprint_was_null": False,
         "contrib_gbm_score": 0.8, "contrib_gnn_score": 0.1, "contrib_sequence_score": 0.05},
        {"scenario_name": "low-and-slow", "is_fraud": True, "caught_by_model": False,
         "amount_zscore_vs_self": 0.1, "graph_shared_device_count": 0, "graph_shared_ip_count": 0,
         "vel_txn_count_1h": 2, "device_fingerprint_was_null": False,
         "contrib_gbm_score": 0.5, "contrib_gnn_score": 0.05, "contrib_sequence_score": 0.1},
    ])
    for exp in generate_round_feedback(df):
        print("-", exp)
