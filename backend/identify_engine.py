"""
=============================================================================
PROJECT AEGIS: IDENTIFY ENGINE — F3-Driven Scenario Generation & Validation
=============================================================================
Real implementation of the Identify phase:
  1. Loads MITRE F3 taxonomy from config/f3_taxonomy.json
  2. Proposes new fraud scenarios mapped to F3 tactics/techniques
  3. Validates structural distinctness (rejects duplicate field+type combos)
  4. Writes approved scenarios to data/coverage_matrix.csv
  5. Accepts miss-explanations from previous rounds as context
=============================================================================
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple
import csv

logger = logging.getLogger("AEGIS.Identify")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
F3_TAXONOMY_PATH = os.path.join(BASE_DIR, "config", "f3_taxonomy.json")
COVERAGE_MATRIX_PATH = os.path.join(BASE_DIR, "data", "coverage_matrix.csv")


class FraudScenario:
    """A single fraud scenario mapped to an F3 technique."""

    def __init__(self, scenario_name: str, f3_tactic: str, f3_technique: str,
                 technique_id: str, mechanism_description: str,
                 fields_manipulated: List[str], manipulation_type: str,
                 novelty_tag: str, round_introduced: int = 0,
                 miss_context: str = ""):
        self.scenario_id = self._generate_id(scenario_name, technique_id)
        self.scenario_name = scenario_name
        self.f3_tactic = f3_tactic
        self.f3_technique = f3_technique
        self.technique_id = technique_id
        self.mechanism_description = mechanism_description
        self.fields_manipulated = sorted(fields_manipulated)
        self.manipulation_type = manipulation_type
        self.novelty_tag = novelty_tag
        self.round_introduced = round_introduced
        self.miss_context = miss_context
        self.detection_rate = 0.0
        self.red_reward = 0.0
        self.is_active = True

    def _generate_id(self, name: str, tech_id: str) -> str:
        raw = f"{name}_{tech_id}"
        return f"SC_{hashlib.md5(raw.encode()).hexdigest()[:8].upper()}"

    def signature(self) -> str:
        """Unique structural signature for distinctness checking."""
        return f"{self.manipulation_type}::{','.join(self.fields_manipulated)}"

    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "f3_tactic": self.f3_tactic,
            "f3_technique": self.f3_technique,
            "technique_id": self.technique_id,
            "mechanism_description": self.mechanism_description,
            "fields_manipulated": self.fields_manipulated,
            "manipulation_type": self.manipulation_type,
            "novelty_tag": self.novelty_tag,
            "round_introduced": self.round_introduced,
            "detection_rate": self.detection_rate,
            "red_reward": self.red_reward,
            "is_active": self.is_active,
        }


class ScenarioValidator:
    """
    Structural distinctness gate (Section 4 Step 3).
    Rejects scenarios whose (fields_manipulated, manipulation_type) combo already exists,
    UNLESS the manipulation_type genuinely differs (e.g., same technique but CP vs CNP).
    """

    def __init__(self):
        self.existing_signatures: Dict[str, str] = {}

    def register(self, scenario: FraudScenario) -> None:
        sig = scenario.signature()
        self.existing_signatures[sig] = scenario.scenario_id

    def validate(self, scenario: FraudScenario) -> Tuple[bool, str]:
        sig = scenario.signature()

        if sig in self.existing_signatures:
            existing_id = self.existing_signatures[sig]
            return False, (
                f"REJECTED: Duplicate signature '{sig}' — "
                f"matches existing scenario {existing_id}. "
                f"Same fields_manipulated + manipulation_type already covered."
            )

        # Check for near-duplicates (>80% field overlap with same type)
        for existing_sig, existing_id in self.existing_signatures.items():
            existing_type, existing_fields_str = existing_sig.split("::", 1)
            if existing_type != scenario.manipulation_type:
                continue
            existing_fields = set(existing_fields_str.split(","))
            new_fields = set(scenario.fields_manipulated)
            overlap = len(existing_fields & new_fields) / max(len(existing_fields | new_fields), 1)
            if overlap > 0.80:
                return False, (
                    f"REJECTED: Near-duplicate ({overlap:.0%} field overlap) with "
                    f"scenario {existing_id}. Need more structural diversity."
                )

        return True, "APPROVED: Structurally distinct scenario."


class IdentifyEngine:
    """
    The Identify Engine: generates fraud scenarios from the F3 taxonomy.
    Produces scenarios grounded in the real MITRE F3 framework, validated
    for structural distinctness.
    """

    def __init__(self):
        self.taxonomy = self._load_taxonomy()
        self.scenarios: List[FraudScenario] = []
        self.validator = ScenarioValidator()
        self.miss_explanations: List[Dict] = []
        self._load_existing_coverage()

    def _load_taxonomy(self) -> Dict:
        if not os.path.exists(F3_TAXONOMY_PATH):
            logger.error(f"F3 taxonomy not found at {F3_TAXONOMY_PATH}")
            return {"tactics": []}
        with open(F3_TAXONOMY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_existing_coverage(self) -> None:
        """Load any previously approved scenarios from coverage matrix."""
        if not os.path.exists(COVERAGE_MATRIX_PATH):
            return
        try:
            with open(COVERAGE_MATRIX_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fields = row.get("fields_manipulated", "").split(";")
                    scenario = FraudScenario(
                        scenario_name=row["scenario_name"],
                        f3_tactic=row["f3_tactic"],
                        f3_technique=row["f3_technique"],
                        technique_id=row["technique_id"],
                        mechanism_description=row.get("mechanism_description", ""),
                        fields_manipulated=fields,
                        manipulation_type=row["manipulation_type"],
                        novelty_tag=row.get("novelty_tag", "baseline"),
                        round_introduced=int(row.get("round_introduced", 0)),
                    )
                    scenario.detection_rate = float(row.get("detection_rate", 0))
                    self.scenarios.append(scenario)
                    self.validator.register(scenario)
            logger.info(f"Loaded {len(self.scenarios)} existing scenarios from coverage matrix")
        except Exception as e:
            logger.warning(f"Could not load coverage matrix: {e}")

    def generate_scenarios_from_taxonomy(self, round_num: int = 0) -> List[FraudScenario]:
        """
        Generate scenarios directly from the F3 taxonomy.
        Each technique becomes a scenario, validated for distinctness.
        """
        new_scenarios = []

        for tactic in self.taxonomy.get("tactics", []):
            tactic_name = tactic["name"]
            for technique in tactic.get("techniques", []):
                # Build scenario from taxonomy entry
                scenario = FraudScenario(
                    scenario_name=technique["name"],
                    f3_tactic=tactic_name,
                    f3_technique=technique["name"],
                    technique_id=technique["id"],
                    mechanism_description=technique["description"],
                    fields_manipulated=technique["fields_manipulated"],
                    manipulation_type=technique["manipulation_type"],
                    novelty_tag=technique.get("novelty_tag", "baseline"),
                    round_introduced=round_num,
                )

                # Validate distinctness
                is_valid, reason = self.validator.validate(scenario)
                if is_valid:
                    self.validator.register(scenario)
                    self.scenarios.append(scenario)
                    new_scenarios.append(scenario)
                    logger.info(f"[IDENTIFY] {reason} — {scenario.scenario_name}")
                else:
                    logger.debug(f"[IDENTIFY] {reason}")

        return new_scenarios

    def generate_harder_variants(self, round_num: int,
                                 missed_scenarios: List[Dict]) -> List[FraudScenario]:
        """
        Given scenarios that Blue missed, generate harder variants.
        Uses miss_explanations from Feedback engine to target weaknesses.
        """
        harder_variants = []

        for missed in missed_scenarios:
            original_technique = missed.get("f3_technique", "")
            miss_reason = missed.get("explanation", "")

            # Create a variant that targets the identified weakness
            variant_fields = list(missed.get("fields_manipulated", []))

            # If the miss was due to amount staying in bounds but graph spiking,
            # create a variant that also normalizes graph features
            if "graph" in miss_reason.lower() or "centrality" in miss_reason.lower():
                if "device_fingerprint" not in variant_fields:
                    variant_fields.append("device_fingerprint")
                variant_type = "network"
            elif "velocity" in miss_reason.lower() or "timing" in miss_reason.lower():
                if "mean_inter_txn_seconds" not in variant_fields:
                    variant_fields.append("mean_inter_txn_seconds")
                variant_type = "behavioral"
            elif "amount" in miss_reason.lower():
                if "amount" not in variant_fields:
                    variant_fields.append("amount")
                variant_type = missed.get("manipulation_type", "behavioral")
            else:
                variant_type = missed.get("manipulation_type", "behavioral")
                if "login_time_deviation_hrs" not in variant_fields:
                    variant_fields.append("login_time_deviation_hrs")

            variant = FraudScenario(
                scenario_name=f"{original_technique} (Hardened v{round_num})",
                f3_tactic=missed.get("f3_tactic", "Evasion"),
                f3_technique=original_technique,
                technique_id=f"T_VAR_{round_num}_{len(harder_variants)}",
                mechanism_description=(
                    f"Hardened variant targeting Blue weakness: {miss_reason[:100]}. "
                    f"Additional evasion applied to {', '.join(variant_fields)}."
                ),
                fields_manipulated=variant_fields,
                manipulation_type=variant_type,
                novelty_tag="adversarial_variant",
                round_introduced=round_num,
                miss_context=miss_reason,
            )

            is_valid, reason = self.validator.validate(variant)
            if is_valid:
                self.validator.register(variant)
                self.scenarios.append(variant)
                harder_variants.append(variant)
                logger.info(f"[IDENTIFY] Generated harder variant: {variant.scenario_name}")
            else:
                logger.debug(f"[IDENTIFY] Variant rejected: {reason}")

        return harder_variants

    def update_scenario_detection_rates(self, per_scenario_metrics: Dict) -> None:
        """Update scenarios with their actual measured detection rates from Defend."""
        for scenario in self.scenarios:
            tech_name = scenario.f3_technique
            scen_name = scenario.scenario_name
            metrics = per_scenario_metrics.get(tech_name, per_scenario_metrics.get(scen_name, {}))
            if metrics:
                rate = metrics.get("detection_rate", metrics.get("recall", 0.0))
                scenario.detection_rate = round(float(rate), 4)
        self.save_coverage_matrix()

    def update_miss_explanations(self, explanations: List[Dict]) -> None:
        """Store miss explanations for next round's context."""
        self.miss_explanations = explanations

    def save_coverage_matrix(self) -> str:
        """Write all approved scenarios to coverage_matrix.csv."""
        os.makedirs(os.path.dirname(COVERAGE_MATRIX_PATH), exist_ok=True)

        fieldnames = [
            "scenario_id", "scenario_name", "f3_tactic", "f3_technique",
            "technique_id", "mechanism_description", "fields_manipulated",
            "manipulation_type", "novelty_tag", "round_introduced",
            "detection_rate", "red_reward", "is_active"
        ]

        with open(COVERAGE_MATRIX_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in self.scenarios:
                row = s.to_dict()
                row["fields_manipulated"] = ";".join(row["fields_manipulated"])
                writer.writerow(row)

        logger.info(f"[IDENTIFY] Coverage matrix saved: {len(self.scenarios)} scenarios")
        return COVERAGE_MATRIX_PATH

    def get_coverage_summary(self) -> Dict:
        """Return summary statistics for the dashboard."""
        tactics = {}
        types = {}
        novelty = {}

        for s in self.scenarios:
            tactics[s.f3_tactic] = tactics.get(s.f3_tactic, 0) + 1
            types[s.manipulation_type] = types.get(s.manipulation_type, 0) + 1
            novelty[s.novelty_tag] = novelty.get(s.novelty_tag, 0) + 1

        return {
            "total_scenarios": len(self.scenarios),
            "by_tactic": tactics,
            "by_manipulation_type": types,
            "by_novelty": novelty,
            "scenarios": [s.to_dict() for s in self.scenarios],
        }

    def get_active_scenarios(self) -> List[FraudScenario]:
        return [s for s in self.scenarios if s.is_active]
