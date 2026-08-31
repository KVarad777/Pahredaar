"""
Identify Engine - Phase 1, Step 3 (Validation gate).

This is deliberately NOT an LLM call - it's a plain code check that stops the
Identify engine from producing 30 cosmetic variants of the same 2 ideas
(the exact pitfall the project requirements doc calls out).
"""

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    accepted: bool
    reason: str


class ScenarioValidator:
    def __init__(self):
        # each entry: {"fields_manipulated": frozenset, "manipulation_type": str, "f3_technique": str}
        self._accepted_signatures: list[dict] = []

    def _signature(self, scenario: dict) -> frozenset:
        return frozenset(scenario["fields_manipulated"])

    def validate(self, scenario: dict) -> ValidationResult:
        sig = self._signature(scenario)
        m_type = scenario["manipulation_type"]
        technique = scenario["f3_technique"]

        # Rule 1: exact (fields_manipulated + manipulation_type) combo already exists -> reject.
        for existing in self._accepted_signatures:
            if existing["fields"] == sig and existing["manipulation_type"] == m_type:
                return ValidationResult(
                    accepted=False,
                    reason=(
                        f"Duplicate signature: fields={sorted(sig)} + "
                        f"manipulation_type={m_type} already covered by an existing scenario."
                    ),
                )

        # Rule 2: same F3 technique already covered -> only allow if manipulation_type
        # genuinely differs from every existing scenario that used that technique.
        same_technique = [s for s in self._accepted_signatures if s["f3_technique"] == technique]
        if same_technique:
            same_type_too = [s for s in same_technique if s["manipulation_type"] == m_type]
            if same_type_too:
                return ValidationResult(
                    accepted=False,
                    reason=(
                        f"F3 technique '{technique}' already covered with the same "
                        f"manipulation_type='{m_type}'. A genuinely distinct implementation "
                        f"(e.g. different manipulation_type) is required to reuse this technique."
                    ),
                )

        # Rule 3: minimum field-manipulation richness - reject single-field "anomalies"
        # (aggregation-level anomaly requirement from the prompt spec).
        if len(sig) < 2:
            return ValidationResult(
                accepted=False,
                reason="Scenario manipulates fewer than 2 fields - too simple to be an "
                       "aggregation-level anomaly; likely a trivial single-field spike.",
            )

        return ValidationResult(accepted=True, reason="Passed all distinctness checks.")

    def commit(self, scenario: dict) -> None:
        """Call only after validate() returned accepted=True."""
        self._accepted_signatures.append({
            "fields": self._signature(scenario),
            "manipulation_type": scenario["manipulation_type"],
            "f3_technique": scenario["f3_technique"],
        })

    def load_existing(self, scenarios: list[dict]) -> None:
        """Rehydrate validator state from a previously saved coverage matrix."""
        for s in scenarios:
            self.commit(s)


if __name__ == "__main__":
    v = ScenarioValidator()

    s1 = {
        "scenario_name": "Low-and-slow velocity abuse",
        "f3_tactic": "Evasion",
        "f3_technique": "Low-and-Slow Velocity Abuse",
        "mechanism_description": "many small txns just under threshold",
        "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
        "manipulation_type": "behavioral",
        "novelty_tag": "baseline",
    }
    r1 = v.validate(s1)
    print("s1:", r1)
    assert r1.accepted
    v.commit(s1)

    # near-duplicate: same fields, same manipulation_type -> should be REJECTED
    s2 = dict(s1, scenario_name="Low-and-slow variant B")
    r2 = v.validate(s2)
    print("s2 (expected rejected):", r2)
    assert not r2.accepted

    # genuinely different: same technique, different manipulation_type -> ACCEPTED
    s3 = dict(
        s1,
        scenario_name="Low-and-slow via device rotation",
        fields_manipulated=["device_fingerprint", "mean_inter_txn_seconds"],
        manipulation_type="network",
    )
    r3 = v.validate(s3)
    print("s3 (expected accepted):", r3)
    assert r3.accepted
    print("All smoke tests passed.")
