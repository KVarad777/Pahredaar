"""
=============================================================================
PROJECT AEGIS: LOOP ORCHESTRATOR — Full Closed-Loop Controller
=============================================================================
Coordinates the complete 8-stage pipeline per spec Section 8:
  1. IDENTIFY → new/harder scenarios from F3
  2. GENERATE → legit + fraud + null injection
  3. FEATURE PIPELINE → velocity/graph/behavioral
  4. DEFEND → GBM + GNN + LSTM → ensemble
  5. SCORING → precision/recall/F1/FPR per scenario
  6. REWARD → blue/red reward → fine-tune decision
  7. FEEDBACK → miss explanations → Identify context
  8. DASHBOARD → coverage matrix + chart update
=============================================================================
"""

import os
import time
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.identify_engine import IdentifyEngine
from backend.generate_engine import GenerateEngine
from backend.feature_pipeline import FeaturePipeline
from backend.defend_engine import DefendEngine
from backend.reward_engine import RewardEngine
from backend.feedback_engine import FeedbackEngine
from backend.capability_graph import CapabilityGraph

logger = logging.getLogger("AEGIS.Orchestrator")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class LoopOrchestrator:
    """
    Master controller for the AEGIS closed-loop system.
    Runs N rounds of Red vs Blue self-play.
    """

    def __init__(self, n_transactions_per_round: int = 5000):
        self.n_transactions = n_transactions_per_round
        self.current_round = 0
        self.round_results: List[Dict] = []

        # Initialize all engines
        logger.info("[ORCHESTRATOR] Initializing engines...")
        self.identify = IdentifyEngine()
        self.generate = GenerateEngine()
        self.feature_pipeline = FeaturePipeline()
        self.defend: Optional[DefendEngine] = None
        self.reward = RewardEngine()
        self.feedback = FeedbackEngine()
        self.capability_graph = CapabilityGraph()

        self.is_running = False
        self.status = "initialized"

    def run_round(self, round_num: Optional[int] = None) -> Dict:
        """Execute one complete round of the closed loop."""
        if round_num is not None:
            self.current_round = round_num
        else:
            self.current_round += 1

        round_n = self.current_round
        self.is_running = True
        self.status = f"running_round_{round_n}"
        round_start = time.time()

        logger.info(f"\n{'='*80}")
        logger.info(f"  ROUND {round_n} — Starting")
        logger.info(f"{'='*80}")

        result = {
            "round": round_n,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stages": {},
        }

        try:
            # ── STAGE 1: IDENTIFY ──
            logger.info(f"[ROUND {round_n}] Stage 1: IDENTIFY")
            stage_start = time.time()

            if round_n <= 1:
                # First round: generate from F3 taxonomy
                new_scenarios = self.identify.generate_scenarios_from_taxonomy(round_n)
            else:
                # Subsequent rounds: generate harder variants from misses
                flagged = self.round_results[-1].get("reward", {}).get(
                    "flagged_for_harder_variants", [])
                if flagged:
                    new_scenarios = self.identify.generate_harder_variants(round_n, flagged)
                else:
                    new_scenarios = []

            active_scenarios = self.identify.get_active_scenarios()
            scenario_dicts = [s.to_dict() for s in active_scenarios]
            self.identify.save_coverage_matrix()

            result["stages"]["identify"] = {
                "new_scenarios": len(new_scenarios),
                "total_active": len(active_scenarios),
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── STAGE 2: GENERATE ──
            logger.info(f"[ROUND {round_n}] Stage 2: GENERATE")
            stage_start = time.time()

            transactions = self.generate.generate_round(
                scenario_dicts, self.n_transactions, round_n)
            self.generate.save_round_data(transactions, round_n)

            n_fraud = sum(1 for t in transactions if t.get("labels", {}).get("is_fraud"))
            result["stages"]["generate"] = {
                "total_transactions": len(transactions),
                "fraud_count": n_fraud,
                "legit_count": len(transactions) - n_fraud,
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── STAGE 3: FEATURE PIPELINE ──
            logger.info(f"[ROUND {round_n}] Stage 3: FEATURE PIPELINE")
            stage_start = time.time()

            # Reset pipeline for clean state each round
            self.feature_pipeline = FeaturePipeline()
            feature_vectors = self.feature_pipeline.process_batch(transactions)

            result["stages"]["feature_pipeline"] = {
                "features_extracted": len(feature_vectors),
                "feature_count": len(self.feature_pipeline.get_feature_names()),
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── STAGE 4: DEFEND ──
            logger.info(f"[ROUND {round_n}] Stage 4: DEFEND")
            stage_start = time.time()

            feature_names = self.feature_pipeline.get_feature_names()

            if self.defend is None or round_n <= 1:
                self.defend = DefendEngine(feature_names)

            # Pick a held-out technique for generalization testing
            techniques_in_round = list(set(
                fv.get("_f3_technique", "") for fv in feature_vectors
                if fv.get("_f3_technique")
            ))
            held_out = techniques_in_round[-1] if len(techniques_in_round) > 2 else ""

            train_metrics = self.defend.train(feature_vectors, held_out_technique=held_out)

            result["stages"]["defend"] = {
                "model_version": self.defend.version,
                "train_metrics": train_metrics,
                "held_out_technique": held_out,
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── STAGE 5: SCORING ──
            logger.info(f"[ROUND {round_n}] Stage 5: SCORING")
            stage_start = time.time()

            scored_results = self.defend.score(feature_vectors)
            overall_metrics = self.defend.evaluate(feature_vectors)
            per_scenario_metrics = self.defend.evaluate_per_scenario(feature_vectors)

            result["stages"]["scoring"] = {
                "overall": overall_metrics,
                "per_scenario": per_scenario_metrics,
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── STAGE 6: REWARD ──
            logger.info(f"[ROUND {round_n}] Stage 6: REWARD")
            stage_start = time.time()

            latency = (time.time() - round_start) * 1000 / max(1, len(transactions))
            reward_result = self.reward.compute_round_rewards(
                per_scenario_metrics, overall_metrics, scenario_dicts, latency)

            # Fine-tune if needed
            if reward_result["should_fine_tune"] and round_n > 1:
                hard_negatives = [
                    fv for fv, sr in zip(feature_vectors, scored_results)
                    if sr.get("is_fraud_actual") == 1 and sr.get("decision") != "BLOCK"
                ]
                if hard_negatives:
                    ft_metrics = self.defend.fine_tune(hard_negatives)
                    reward_result["fine_tune_result"] = ft_metrics

            result["stages"]["reward"] = {
                **reward_result,
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── STAGE 7: FEEDBACK ──
            logger.info(f"[ROUND {round_n}] Stage 7: FEEDBACK")
            stage_start = time.time()

            miss_explanations = self.feedback.explain_misses(scored_results, feature_vectors)
            weakness_summary = self.feedback.summarize_round_weaknesses(miss_explanations)
            self.identify.update_miss_explanations(miss_explanations)

            result["stages"]["feedback"] = {
                "total_misses": len(miss_explanations),
                "weakness_summary": weakness_summary,
                "sample_explanations": miss_explanations[:5],
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── STAGE 8: DASHBOARD UPDATE ──
            logger.info(f"[ROUND {round_n}] Stage 8: DASHBOARD UPDATE")
            stage_start = time.time()

            # Update capability graph
            missed_techniques = list(set(
                e.get("f3_technique", "") for e in miss_explanations if e.get("f3_technique")
            ))
            detected_techniques = list(set(
                sr.get("f3_technique", "") for sr in scored_results
                if sr.get("is_fraud_actual") == 1 and sr.get("decision") == "BLOCK"
                and sr.get("f3_technique")
            ))
            self.capability_graph.update_after_round(
                round_n, missed_techniques, detected_techniques)

            # Get predictions for next round
            defense_status = self.capability_graph.get_defense_status(round_n)

            result["stages"]["dashboard"] = {
                "coverage_matrix": self.identify.get_coverage_summary(),
                "capability_graph": self.capability_graph.get_graph_data(),
                "defense_predictions": defense_status,
                "duration_ms": round((time.time() - stage_start) * 1000, 1),
            }

            # ── ROUND SUMMARY ──
            total_time = time.time() - round_start
            result["total_duration_seconds"] = round(total_time, 2)
            result["summary"] = {
                "round": round_n,
                "model_version": self.defend.version,
                "overall_f1": overall_metrics.get("f1", 0),
                "overall_recall": overall_metrics.get("recall", 0),
                "overall_fpr": overall_metrics.get("fpr", 0),
                "blue_reward": reward_result.get("blue_reward", 0),
                "scenarios_flagged": reward_result.get("n_scenarios_flagged", 0),
                "total_misses": len(miss_explanations),
                "next_predicted_attacks": [
                    p.get("predicted_attack", "") for p in
                    defense_status.get("predicted_next_attacks", [])[:3]
                ],
            }

            self.round_results.append(result)
            self.is_running = False
            self.status = f"completed_round_{round_n}"

            logger.info(f"\n{'='*80}")
            logger.info(f"  ROUND {round_n} COMPLETE — F1: {overall_metrics.get('f1', 0):.3f} | "
                        f"Recall: {overall_metrics.get('recall', 0):.3f} | "
                        f"FPR: {overall_metrics.get('fpr', 0):.3f} | "
                        f"Duration: {total_time:.1f}s")
            logger.info(f"{'='*80}\n")

            return result

        except Exception as e:
            logger.error(f"[ROUND {round_n}] Error: {e}", exc_info=True)
            self.is_running = False
            self.status = f"error_round_{round_n}"
            result["error"] = str(e)
            self.round_results.append(result)
            return result

    def run_multiple_rounds(self, n_rounds: int = 3) -> List[Dict]:
        """Run N rounds of the closed loop."""
        results = []
        for i in range(n_rounds):
            result = self.run_round()
            results.append(result)
        return results

    def get_round_over_round_metrics(self) -> List[Dict]:
        """Return detection metrics across all rounds for the dashboard chart."""
        metrics = []
        for r in self.round_results:
            scoring = r.get("stages", {}).get("scoring", {})
            summary = r.get("summary", {})
            metrics.append({
                "round": r.get("round", 0),
                "overall_f1": summary.get("overall_f1", 0),
                "overall_recall": summary.get("overall_recall", 0),
                "overall_fpr": summary.get("overall_fpr", 0),
                "blue_reward": summary.get("blue_reward", 0),
                "model_version": summary.get("model_version", ""),
                "per_scenario": scoring.get("per_scenario", {}),
            })
        return metrics

    def get_status(self) -> Dict:
        return {
            "is_running": self.is_running,
            "status": self.status,
            "current_round": self.current_round,
            "total_rounds_completed": len(self.round_results),
            "model_version": self.defend.version if self.defend else "N/A",
        }
