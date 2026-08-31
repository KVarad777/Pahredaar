"""
Reward System - Red (Generate/Identify) reward.

red_reward = w1 * (1 - blue_detection_rate_on_this_scenario) + w2 * structural_novelty_score

A scenario Blue currently misses scores high - this is what drives the
Feedback engine to propose harder variants specifically of missed techniques
rather than random new ones. structural_novelty_score (from the validator)
prevents Red from resubmitting near-duplicates that happen to evade the
current model without adding real diversity.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class RedRewardResult:
    scenario_name: str
    reward: float
    detection_rate: float
    novelty_score: float


def _novelty_score(scenario: dict, all_scenarios: list[dict]) -> float:
    """
    Simple structural novelty proxy: fraction of (field, manipulation_type)
    pairs in this scenario that DON'T appear in any other accepted scenario.
    The validator already guarantees no exact duplicates get this far - this
    just scores HOW distinct an accepted scenario is, for reward-weighting.
    """
    this_fields = set(scenario["fields_manipulated"])
    this_type = scenario["manipulation_type"]

    overlap_scores = []
    for other in all_scenarios:
        if other["scenario_name"] == scenario["scenario_name"]:
            continue
        other_fields = set(other["fields_manipulated"])
        if not this_fields or not other_fields:
            continue
        jaccard = len(this_fields & other_fields) / len(this_fields | other_fields)
        type_match = 1.0 if other["manipulation_type"] == this_type else 0.0
        overlap_scores.append(0.5 * jaccard + 0.5 * type_match)

    if not overlap_scores:
        return 1.0  # first scenario of its kind - maximally novel
    max_overlap = max(overlap_scores)
    return round(1.0 - max_overlap, 4)


def compute_red_rewards(
    per_scenario_metrics: pd.DataFrame,
    scenarios_this_round: list[dict],
    all_accepted_scenarios: list[dict],
    w1: float = 1.0,
    w2: float = 0.5,
) -> list[RedRewardResult]:
    """
    per_scenario_metrics: output of src.defend.train.per_scenario_metrics()
        (scenario_name, n_txns, n_caught, detection_rate).
    scenarios_this_round: the scenario dicts proposed/generated this round.
    all_accepted_scenarios: full coverage-matrix history, used for novelty scoring.
    """
    results = []
    detection_by_name = dict(zip(per_scenario_metrics["scenario_name"], per_scenario_metrics["detection_rate"]))

    for scenario in scenarios_this_round:
        name = scenario["scenario_name"]
        detection_rate = detection_by_name.get(name, 0.0)
        novelty = _novelty_score(scenario, all_accepted_scenarios)
        reward = w1 * (1 - detection_rate) + w2 * novelty
        results.append(RedRewardResult(
            scenario_name=name, reward=reward, detection_rate=detection_rate, novelty_score=novelty,
        ))

    return results


if __name__ == "__main__":
    metrics = pd.DataFrame({
        "scenario_name": ["low-and-slow", "device ring", "card testing"],
        "n_txns": [17, 16, 21],
        "n_caught": [17, 15, 5],
        "detection_rate": [1.0, 0.9375, 0.238],
    })
    scenarios = [
        {"scenario_name": "low-and-slow", "fields_manipulated": ["amount", "mean_inter_txn_seconds"], "manipulation_type": "behavioral"},
        {"scenario_name": "device ring", "fields_manipulated": ["device_fingerprint", "ip_address_hash"], "manipulation_type": "network"},
        {"scenario_name": "card testing", "fields_manipulated": ["amount", "channel"], "manipulation_type": "channel"},
    ]
    rewards = compute_red_rewards(metrics, scenarios, scenarios)
    for r in rewards:
        print(r)
