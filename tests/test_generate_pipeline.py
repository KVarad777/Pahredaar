"""
Run with: pytest tests/ -v   (from the project root, after `pip install -e .`
or with src/ on PYTHONPATH)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.identify.validator import ScenarioValidator
from src.generate.legit_traffic_sim import LegitTrafficSimulator
from src.generate.null_injector import NullInjector
from src.generate.orchestrator import GenerateOrchestrator


def test_validator_rejects_exact_duplicate():
    v = ScenarioValidator()
    s = {
        "scenario_name": "A", "f3_tactic": "Evasion", "f3_technique": "Low-and-Slow",
        "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
        "manipulation_type": "behavioral", "novelty_tag": "x",
    }
    r1 = v.validate(s)
    assert r1.accepted
    v.commit(s)
    r2 = v.validate(dict(s, scenario_name="B"))
    assert not r2.accepted


def test_validator_rejects_single_field_scenario():
    v = ScenarioValidator()
    s = {
        "scenario_name": "trivial", "f3_tactic": "X", "f3_technique": "Y",
        "fields_manipulated": ["amount"], "manipulation_type": "channel", "novelty_tag": "x",
    }
    r = v.validate(s)
    assert not r.accepted


def test_legit_traffic_amounts_within_realistic_bounds():
    sim = LegitTrafficSimulator()
    batch = sim.generate_batch(n_accounts=20, txns_per_account_range=(2, 4))
    amounts = [t["amount"] for t in batch]
    p = sim.params["amount_distribution"]
    assert all(p["min_realistic"] <= a <= p["max_realistic"] for a in amounts)
    assert len(set(amounts)) > 1, "amounts should not all be identical (uniform/degenerate generator)"


def test_null_injector_produces_was_null_flags():
    sim = LegitTrafficSimulator()
    injector = NullInjector()
    batch = sim.generate_batch(n_accounts=30, txns_per_account_range=(2, 3))
    injected = injector.apply_batch(batch)
    assert all("device_fingerprint_was_null" in t for t in injected)
    null_rate = sum(t["device_fingerprint_was_null"] for t in injected) / len(injected)
    # loose bounds check - not exact since it's random, just confirms it's not 0% or 100%
    assert 0.0 < null_rate < 0.5


def test_orchestrator_end_to_end_produces_labeled_fraud_and_legit():
    scenarios = [{
        "scenario_id": "test01",
        "scenario_name": "test low-and-slow",
        "f3_tactic": "Evasion",
        "f3_technique": "Low-and-Slow Velocity Abuse",
        "mechanism_description": "test",
        "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
        "manipulation_type": "behavioral",
        "novelty_tag": "test",
    }]
    orch = GenerateOrchestrator()
    df = orch.run_round(scenarios, n_legit_accounts=10, round_n=99, out_dir="data/generated_test")
    assert (~df["is_fraud"]).sum() > 0, "should contain legit transactions"
    assert df["is_fraud"].sum() > 0, "should contain injected fraud transactions"
    assert df["txnId"].is_unique, "every transaction must have a unique ID"


if __name__ == "__main__":
    test_validator_rejects_exact_duplicate()
    test_validator_rejects_single_field_scenario()
    test_legit_traffic_amounts_within_realistic_bounds()
    test_null_injector_produces_was_null_flags()
    test_orchestrator_end_to_end_produces_labeled_fraud_and_legit()
    print("All tests passed.")
