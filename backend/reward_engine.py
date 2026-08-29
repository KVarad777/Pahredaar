"""
=============================================================================
PROJECT AEGIS: REWARD ENGINE — Red vs Blue Self-Play Scoring
=============================================================================
Implements the reward-scored selection loop from spec Section 7:
  - blue_reward: w1 * recall_new_techniques - w2 * FPR - w3 * latency_penalty
  - red_reward: w1 * (1 - blue_detection_rate) + w2 * structural_novelty_score
  - Fine-tune trigger logic
=============================================================================
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger("AEGIS.Reward")


def blue_reward(recall_new_techniques: float, fpr: float,
                latency_penalty: float = 0.0,
                w1: float = 1.0, w2: float = 1.0, w3: float = 0.3) -> float:
    """
    Blue Team (Defend) reward per round.
    - Rewarded for catching newly introduced scenario types
    - Penalized for FPR (can't win by flagging everything)
    - Penalized for latency (optional)
    """
    reward = w1 * recall_new_techniques - w2 * fpr - w3 * latency_penalty
    return round(float(reward), 4)


def red_reward(blue_detection_rate_on_scenario: float,
               structural_novelty_score: float,
               w1: float = 1.0, w2: float = 0.5) -> float:
    """
    Red Team (Generate/Identify) reward per scenario.
    - Higher when Blue misses more of this scenario
    - Boosted by structural novelty (prevents near-duplicate submissions)
    """
    reward = w1 * (1 - blue_detection_rate_on_scenario) + w2 * structural_novelty_score
    return round(float(reward), 4)


class RewardEngine:
    """
    Coordinates the full reward computation for a round.
    Determines:
      - Whether Blue should fine-tune this round
      - Which Red scenarios get flagged for harder variants
    """

    def __init__(self, fine_tune_threshold: float = 0.5):
        self.fine_tune_threshold = fine_tune_threshold
        self.round_history: List[Dict] = []

    def compute_round_rewards(self, per_scenario_metrics: Dict,
                              overall_metrics: Dict,
                              scenarios: List[Dict],
                              latency_ms: float = 50.0) -> Dict:
        """
        Compute rewards for an entire round.
        Returns decision on whether to fine-tune Blue, and which scenarios to flag Red.
        """
        # Compute Blue reward
        # recall_new = average recall across newly-introduced techniques
        new_recalls = []
        for tech, metrics in per_scenario_metrics.items():
            if tech == "Legitimate":
                continue
            detection_rate = metrics.get("detection_rate", metrics.get("recall", 0))
            new_recalls.append(detection_rate)

        avg_recall_new = float(sum(new_recalls) / max(1, len(new_recalls)))
        fpr = overall_metrics.get("fpr", 0)
        latency_penalty = max(0, (latency_ms - 100) / 1000)  # Penalty starts at >100ms

        b_reward = blue_reward(avg_recall_new, fpr, latency_penalty)

        # Should Blue fine-tune?
        should_fine_tune = b_reward < self.fine_tune_threshold

        # Compute Red rewards per scenario
        flagged_scenarios = []
        scenario_rewards = {}

        for scenario in scenarios:
            scenario_name = scenario.get("scenario_name", "") or scenario.get("f3_technique", "")
            scenario_id = scenario.get("scenario_id", "")
            technique = scenario.get("f3_technique", "")

            metrics = per_scenario_metrics.get(technique, {})
            detection_rate = metrics.get("detection_rate", metrics.get("recall", 0))

            # Structural novelty: higher for ai_specific and adversarial_variant tags
            novelty_tag = scenario.get("novelty_tag", "baseline")
            if novelty_tag == "ai_specific":
                novelty_score = 0.8
            elif novelty_tag == "adversarial_variant":
                novelty_score = 0.9
            else:
                novelty_score = 0.3

            r_reward = red_reward(detection_rate, novelty_score)
            scenario_rewards[scenario_id] = {
                "scenario_name": scenario_name,
                "technique": technique,
                "detection_rate": detection_rate,
                "novelty_score": novelty_score,
                "red_reward": r_reward,
            }

            # Flag high-reward (= high-miss) scenarios for harder variants
            if r_reward > 0.7:
                flagged_scenarios.append({
                    "scenario_id": scenario_id,
                    "scenario_name": scenario_name,
                    "f3_technique": technique,
                    "f3_tactic": scenario.get("f3_tactic", ""),
                    "fields_manipulated": scenario.get("fields_manipulated", []),
                    "manipulation_type": scenario.get("manipulation_type", ""),
                    "detection_rate": detection_rate,
                    "red_reward": r_reward,
                })

        round_result = {
            "blue_reward": b_reward,
            "avg_recall_new": avg_recall_new,
            "fpr": fpr,
            "should_fine_tune": should_fine_tune,
            "scenario_rewards": scenario_rewards,
            "flagged_for_harder_variants": flagged_scenarios,
            "n_scenarios_total": len(scenarios),
            "n_scenarios_flagged": len(flagged_scenarios),
        }

        self.round_history.append(round_result)
        logger.info(f"[REWARD] Blue reward: {b_reward:.3f} | Fine-tune: {should_fine_tune} | "
                    f"Flagged: {len(flagged_scenarios)}/{len(scenarios)} scenarios")

        return round_result

    def get_history(self) -> List[Dict]:
        return self.round_history
