"""
Reward System - Blue (Defend) reward.

blue_reward = w1 * recall_on_new_techniques - w2 * false_positive_rate - w3 * latency_penalty

Rewarded for catching NEWLY introduced scenario types specifically (not just
overall accuracy - this stops Blue from "winning" by only getting good at
easy, already-seen fraud). Penalized for FPR so it can't win by flagging
everything.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BlueRewardResult:
    reward: float
    recall_on_new_techniques: float
    fpr: float
    latency_penalty: float


def compute_blue_reward(
    scored_df: pd.DataFrame,
    new_scenario_names: list[str],
    latency_seconds: float = 0.0,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 0.3,
    latency_budget_seconds: float = 1.0,
) -> BlueRewardResult:
    """
    scored_df: this round's fully-scored transactions (must have is_fraud,
        caught_by_model, flagged, scenario_name columns - i.e. train.py's output).
    new_scenario_names: scenario names introduced THIS round (from the Identify
        engine) - recall is measured specifically on these, not overall.
    latency_seconds: mean scoring latency for this round's online path (GBM +
        ensemble only - the online-scoreable subset). Penalized if it exceeds budget.
    """
    new_fraud = scored_df[(scored_df["is_fraud"] == True) & (scored_df["scenario_name"].isin(new_scenario_names))]
    if len(new_fraud) > 0:
        recall_new = new_fraud["caught_by_model"].mean()
    else:
        # no new scenarios this round (e.g. round 0 before Identify has run) - neutral score
        recall_new = float("nan")

    legit = scored_df[scored_df["is_fraud"] == False]
    fpr = legit["flagged"].mean() if len(legit) > 0 else 0.0

    latency_penalty = max(0.0, latency_seconds - latency_budget_seconds) / max(latency_budget_seconds, 1e-6)

    recall_term = recall_new if not pd.isna(recall_new) else 0.0
    reward = w1 * recall_term - w2 * fpr - w3 * latency_penalty

    return BlueRewardResult(
        reward=reward, recall_on_new_techniques=recall_new, fpr=fpr, latency_penalty=latency_penalty,
    )


if __name__ == "__main__":
    df = pd.DataFrame({
        "is_fraud": [True, True, True, False, False, False, False],
        "caught_by_model": [True, False, True, False, False, False, True],  # last False->flagged wrongly
        "flagged": [True, False, True, False, False, False, True],
        "scenario_name": ["new_scen", "new_scen", "old_scen", None, None, None, None],
    })
    result = compute_blue_reward(df, new_scenario_names=["new_scen"], latency_seconds=0.3)
    print(result)
