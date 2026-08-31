"""
Identify Engine - Phase 1, Step 4 (Coverage matrix write).

Appends approved scenarios to a CSV that becomes the spine of your demo dashboard:
scenario_id x category x mechanism x novelty_tag x (later) detection_rate x red_reward.
"""

import csv
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


COLUMNS = [
    "scenario_id",
    "scenario_name",
    "f3_tactic",
    "f3_technique",
    "manipulation_type",
    "fields_manipulated",
    "mechanism_description",
    "novelty_tag",
    "round_added",
    "created_at",
    # filled in later rounds by the reward/scoring phase - blank at creation time
    "detection_rate",
    "red_reward",
    "times_missed",
]


class CoverageMatrix:
    def __init__(self, path: str = "data/coverage_matrix.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=COLUMNS)
                writer.writeheader()

    def add(self, scenario: dict, round_n: int) -> str:
        scenario_id = str(uuid.uuid4())[:8]
        row = {
            "scenario_id": scenario_id,
            "scenario_name": scenario["scenario_name"],
            "f3_tactic": scenario["f3_tactic"],
            "f3_technique": scenario["f3_technique"],
            "manipulation_type": scenario["manipulation_type"],
            "fields_manipulated": "|".join(scenario["fields_manipulated"]),
            "mechanism_description": scenario["mechanism_description"],
            "novelty_tag": scenario["novelty_tag"],
            "round_added": round_n,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "detection_rate": "",
            "red_reward": "",
            "times_missed": 0,
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writerow(row)
        return scenario_id

    def all_scenario_ids(self) -> list[str]:
        return [r["scenario_id"] for r in self._read_all()]

    def all_scenario_names(self) -> list[str]:
        return [r["scenario_name"] for r in self._read_all()]

    def as_scenario_dicts(self) -> list[dict]:
        """Rehydrate rows back into the shape validator.load_existing() expects."""
        out = []
        for r in self._read_all():
            out.append({
                "scenario_name": r["scenario_name"],
                "f3_tactic": r["f3_tactic"],
                "f3_technique": r["f3_technique"],
                "fields_manipulated": r["fields_manipulated"].split("|") if r["fields_manipulated"] else [],
                "manipulation_type": r["manipulation_type"],
                "novelty_tag": r["novelty_tag"],
            })
        return out

    def update_scoring(self, scenario_id: str, detection_rate: float, red_reward: float) -> None:
        """Called from the Phase-2 reward loop after a scoring round - not used in Phase 1."""
        rows = self._read_all()
        for r in rows:
            if r["scenario_id"] == scenario_id:
                r["detection_rate"] = detection_rate
                r["red_reward"] = red_reward
                r["times_missed"] = int(r["times_missed"] or 0) + (1 if detection_rate < 0.5 else 0)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def _read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f))


if __name__ == "__main__":
    cm = CoverageMatrix(path="data/coverage_matrix_smoketest.csv")
    sid = cm.add({
        "scenario_name": "Low-and-slow velocity abuse",
        "f3_tactic": "Evasion",
        "f3_technique": "Low-and-Slow Velocity Abuse",
        "mechanism_description": "many small txns just under threshold",
        "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
        "manipulation_type": "behavioral",
        "novelty_tag": "baseline",
    }, round_n=0)
    print("added scenario:", sid)
    print("all names:", cm.all_scenario_names())
    os.remove("data/coverage_matrix_smoketest.csv")
