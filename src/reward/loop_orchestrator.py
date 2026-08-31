"""
The complete closed loop, end to end:

  IDENTIFY -> GENERATE -> FEATURE PIPELINE -> DEFEND -> SCORING -> REWARD -> FEEDBACK -> DASHBOARD
  ^_____________________________________________________________________________________________|
                                          (loop back for N rounds)

This is the single entry point for the whole project. Run this from a Colab
notebook (see notebooks/04_full_loop.ipynb) after config/distribution_params.yaml
has been fitted (notebook 00).
"""

import json
from pathlib import Path

import pandas as pd

from src.identify.llm_scenario_proposer import ScenarioProposer
from src.identify.validator import ScenarioValidator
from src.identify.coverage_matrix import CoverageMatrix
from src.generate.orchestrator import GenerateOrchestrator
from src.defend.train import train_defend_bundle, per_scenario_metrics
from src.reward.blue_reward import compute_blue_reward
from src.reward.red_reward import compute_red_rewards
from src.feedback.miss_explainer import generate_round_feedback


class FraudRedTeamLoop:
    def __init__(
        self,
        f3_taxonomy_path: str = "config/f3_taxonomy.json",
        schema_path: str = "config/upi_schema.json",
        distribution_params_path: str = "config/distribution_params.yaml",
        null_config_path: str = "config/null_injection_rates.yaml",
        coverage_matrix_path: str = "data/coverage_matrix.csv",
        dashboard_log_path: str = "data/dashboard_log.csv",
        model: str = None,
    ):
        if model:
            self.proposer = ScenarioProposer(f3_taxonomy_path, schema_path, model=model)
        else:
            self.proposer = ScenarioProposer(f3_taxonomy_path, schema_path)
        self.validator = ScenarioValidator()
        self.coverage = CoverageMatrix(coverage_matrix_path)
        self.validator.load_existing(self.coverage.as_scenario_dicts())

        self.generate_orch = GenerateOrchestrator(distribution_params_path, null_config_path)
        self.dashboard_log_path = Path(dashboard_log_path)
        self.dashboard_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.miss_explanations: list[str] = []
        self.feature_assembler = None  # set fresh each round below; see note in run_round

    def propose_and_validate_scenarios(self, target_count: int, round_n: int, max_attempts_multiplier: int = 4) -> list[dict]:
        accepted = []
        attempts = 0
        max_attempts = target_count * max_attempts_multiplier

        while len(accepted) < target_count and attempts < max_attempts:
            attempts += 1
            try:
                proposal = self.proposer.propose(
                    existing_scenario_ids=self.coverage.all_scenario_names(),
                    miss_explanations=self.miss_explanations,
                )
            except ValueError as e:
                print(f"  [attempt {attempts}] LLM output error, skipping: {e}")
                continue

            result = self.validator.validate(proposal)
            if result.accepted:
                self.validator.commit(proposal)
                scenario_id = self.coverage.add(proposal, round_n=round_n)
                proposal["scenario_id"] = scenario_id
                accepted.append(proposal)
                print(f"  [attempt {attempts}] ACCEPTED: {proposal['scenario_name']} "
                      f"({proposal['manipulation_type']})")
            else:
                print(f"  [attempt {attempts}] REJECTED: {proposal.get('scenario_name','?')} - {result.reason}")

        return accepted

    def run_round(
        self,
        round_n: int,
        n_new_scenarios: int = 3,
        n_legit_accounts: int = 400,
        fraud_txns_per_scenario_range=(10, 20),
        reuse_state_across_rounds: bool = False,
    ) -> dict:
        print(f"\n{'='*60}\nROUND {round_n}\n{'='*60}")

        # ---- 1. IDENTIFY ----
        print("\n[1/7] Identify: proposing new scenarios...")
        new_scenarios = self.propose_and_validate_scenarios(n_new_scenarios, round_n)
        new_scenario_names = [s["scenario_name"] for s in new_scenarios]

        if not new_scenarios:
            print("  No new scenarios accepted this round (taxonomy may be exhausted "
                  "at current validator strictness) - reusing prior accepted scenarios.")
            new_scenarios = self.coverage.as_scenario_dicts()
            for i, s in enumerate(new_scenarios):
                s.setdefault("scenario_id", f"prior_{i}")

        # ---- 2. GENERATE ----
        print("\n[2/7] Generate: producing synthetic transactions...")
        raw_df = self.generate_orch.run_round(
            scenarios=new_scenarios, n_legit_accounts=n_legit_accounts,
            fraud_txns_per_scenario_range=fraud_txns_per_scenario_range,
            round_n=round_n,
        )

        # ---- 3. FEATURE PIPELINE + 4. DEFEND ----
        print("\n[3-4/7] Feature pipeline + Defend: training GBM/GNN/Sequence/Ensemble...")
        assembler = self.feature_assembler if reuse_state_across_rounds else None
        bundle = train_defend_bundle(raw_df, feature_assembler=assembler)
        if reuse_state_across_rounds:
            self.feature_assembler = bundle.feature_assembler

        # ---- 5. SCORING ----
        print("\n[5/7] Scoring: computing per-scenario detection rates...")
        scenario_metrics = per_scenario_metrics(bundle.scored_df)
        print(scenario_metrics.to_string(index=False))

        # ---- 6. REWARD ----
        print("\n[6/7] Reward: computing blue_reward and red_reward...")
        blue_result = compute_blue_reward(bundle.scored_df, new_scenario_names=new_scenario_names)
        print(f"  blue_reward: {blue_result.reward:.4f} "
              f"(recall_new={blue_result.recall_on_new_techniques:.3f}, fpr={blue_result.fpr:.4f})")

        red_results = compute_red_rewards(
            scenario_metrics, new_scenarios, self.coverage.as_scenario_dicts(),
        )
        for r in red_results:
            self.coverage.update_scoring(
                scenario_id=next((s["scenario_id"] for s in new_scenarios if s["scenario_name"] == r.scenario_name), ""),
                detection_rate=r.detection_rate, red_reward=r.reward,
            )
            print(f"  red_reward [{r.scenario_name}]: {r.reward:.4f} "
                  f"(detection_rate={r.detection_rate:.3f}, novelty={r.novelty_score:.3f})")

        # ---- 7. FEEDBACK ----
        print("\n[7/7] Feedback: generating miss explanations for next round...")
        self.miss_explanations = generate_round_feedback(bundle.scored_df)
        for exp in self.miss_explanations[:5]:
            print(f"  - {exp}")
            
        # Save feedback to JSON for the dashboard
        feedback_path = self.dashboard_log_path.parent / "latest_miss_explanations.json"
        with open(feedback_path, "w") as f:
            json.dump(self.miss_explanations, f, indent=2)

        # ---- DASHBOARD LOG ----
        round_summary = {
            "round": round_n,
            "n_scenarios_this_round": len(new_scenarios),
            "n_total_txns": len(bundle.scored_df),
            "n_fraud_txns": int(bundle.scored_df["is_fraud"].sum()),
            "blue_reward": blue_result.reward,
            "blue_recall_on_new": blue_result.recall_on_new_techniques,
            "blue_fpr": blue_result.fpr,
            "ensemble_auc": bundle.metrics["ensemble_auc"],
            "ensemble_precision": bundle.metrics["ensemble_precision"],
            "ensemble_recall": bundle.metrics["ensemble_recall"],
            "ensemble_f1": bundle.metrics["ensemble_f1"],
            "mean_red_reward": sum(r.reward for r in red_results) / max(len(red_results), 1),
        }
        self._append_dashboard_log(round_summary)

        return {
            "round_summary": round_summary,
            "scenario_metrics": scenario_metrics,
            "bundle": bundle,
            "new_scenarios": new_scenarios,
        }

    def _append_dashboard_log(self, round_summary: dict) -> None:
        df_row = pd.DataFrame([round_summary])
        if self.dashboard_log_path.exists():
            existing = pd.read_csv(self.dashboard_log_path)
            combined = pd.concat([existing, df_row], ignore_index=True)
        else:
            combined = df_row
        combined.to_csv(self.dashboard_log_path, index=False)

    def run(self, n_rounds: int = 5, **round_kwargs) -> pd.DataFrame:
        for round_n in range(n_rounds):
            self.run_round(round_n=round_n, **round_kwargs)
        return pd.read_csv(self.dashboard_log_path)


if __name__ == "__main__":
    # Requires ANTHROPIC_API_KEY set in the environment (Identify engine calls the LLM).
    loop = FraudRedTeamLoop()
    history = loop.run(n_rounds=3, n_new_scenarios=3, n_legit_accounts=300)
    print("\n\n=== FULL LOOP DASHBOARD LOG ===")
    print(history.to_string(index=False))
