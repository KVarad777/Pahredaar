"""
Generate Engine orchestrator - Phase 1's main entry point.

Takes a list of approved scenarios (from Identify + Validator), generates a
legit-traffic base, injects fraud per scenario using the right injector class
(dispatched by manipulation_type, NOT a big if/else), applies controlled
null injection, and returns/saves the full round's labeled dataset.
"""

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from .legit_traffic_sim import LegitTrafficSimulator
from .null_injector import NullInjector
from .injectors.identity_injector import IdentityInjector
from .injectors.behavioral_injector import BehavioralInjector
from .injectors.network_injector import NetworkInjector
from .injectors.channel_injector import ChannelInjector
from .injectors.ai_specific_injector import AISpecificInjector


INJECTOR_REGISTRY = {
    "identity": IdentityInjector(),
    "behavioral": BehavioralInjector(),
    "network": NetworkInjector(),
    "channel": ChannelInjector(),
    "ai_specific": AISpecificInjector(),
}


class GenerateOrchestrator:
    def __init__(
        self,
        distribution_params_path: str = "config/distribution_params.yaml",
        null_config_path: str = "config/null_injection_rates.yaml",
        seed: int = 42,
    ):
        self.legit_sim = LegitTrafficSimulator(distribution_params_path, seed=seed)
        self.null_injector = NullInjector(null_config_path, seed=seed)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def run_round(
        self,
        scenarios: list[dict],
        n_legit_accounts: int = 300,
        fraud_txns_per_scenario_range=(3, 10),
        round_n: int = 0,
        out_dir: str = "data/generated",
    ) -> pd.DataFrame:
        # 1. legitimate baseline
        legit_batch = self.legit_sim.generate_batch(n_accounts=n_legit_accounts)
        for t in legit_batch:
            t["is_fraud"] = False
            t["f3_tactic"] = None
            t["f3_technique"] = None
            t["scenario_id"] = None
            t["scenario_name"] = None
            t["caught_by_model"] = None

        # 2. fraud injection, one scenario at a time, dispatched to the right injector
        fraud_batch = []
        for scenario in scenarios:
            injector = INJECTOR_REGISTRY.get(scenario["manipulation_type"])
            if injector is None:
                raise ValueError(f"No injector registered for manipulation_type="
                                  f"{scenario['manipulation_type']!r} (scenario={scenario['scenario_name']!r})")

            # seed each scenario's fraud from a small slice of legit accounts as the "base"
            n_seed = int(self.rng.integers(*fraud_txns_per_scenario_range))
            seed_txns = random.sample(legit_batch, k=min(n_seed, len(legit_batch)))

            injected = injector.inject(seed_txns, scenario, self.rng)
            fraud_batch.extend(injected)

        # 3. controlled null injection applied to EVERYTHING (legit and fraud alike -
        # real production data has nulls in legit traffic too, not just fraud)
        all_txns = legit_batch + fraud_batch
        scenario_by_id = {s.get("scenario_id"): s for s in scenarios}

        final = []
        for t in all_txns:
            matching_scenario = scenario_by_id.get(t.get("scenario_id")) if t.get("is_fraud") else None
            final.append(self.null_injector.apply(t, matching_scenario))

        df = pd.DataFrame(final)

        out_path = Path(out_dir) / f"round_{round_n:02d}"
        out_path.mkdir(parents=True, exist_ok=True)
        df.to_json(out_path / "transactions.jsonl", orient="records", lines=True)
        df.to_csv(out_path / "transactions.csv", index=False)

        summary = {
            "round": round_n,
            "n_total": len(df),
            "n_fraud": int(df["is_fraud"].sum()),
            "n_legit": int((~df["is_fraud"]).sum()),
            "fraud_ratio": float(df["is_fraud"].mean()),
            "scenarios_this_round": [s["scenario_name"] for s in scenarios],
        }
        with open(out_path / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[round {round_n}] {summary['n_total']} txns "
              f"({summary['n_fraud']} fraud, {summary['fraud_ratio']:.3%}) "
              f"-> saved to {out_path}")
        return df


if __name__ == "__main__":
    demo_scenarios = [
        {
            "scenario_id": "demo01",
            "scenario_name": "Low-and-slow velocity abuse",
            "f3_tactic": "Evasion",
            "f3_technique": "Low-and-Slow Velocity Abuse",
            "mechanism_description": "many small txns just under threshold",
            "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
            "manipulation_type": "behavioral",
            "novelty_tag": "baseline",
        },
        {
            "scenario_id": "demo02",
            "scenario_name": "Device/IP ring",
            "f3_tactic": "Monetization",
            "f3_technique": "Mule Network Cash-Out",
            "mechanism_description": "shared device/IP across many accounts",
            "fields_manipulated": ["device_fingerprint", "ip_address_hash"],
            "manipulation_type": "network",
            "novelty_tag": "baseline",
        },
    ]
    orch = GenerateOrchestrator()
    df = orch.run_round(demo_scenarios, n_legit_accounts=50, round_n=0)
    print(df[["txnId", "amount", "is_fraud", "scenario_name"]].head(10))
