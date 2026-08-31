"""
Shared base class for all category injectors.

Design principle (directly answers the "diversity" pitfall in the project
requirements doc): each manipulation_type gets its OWN file with genuinely
different logic, parameterized by scenario['fields_manipulated'] - not one
big if/else block with a config number changing.
"""

import uuid
from abc import ABC, abstractmethod


class BaseInjector(ABC):
    manipulation_type: str = "override_me"

    @abstractmethod
    def inject(self, legit_txns: list[dict], scenario: dict, rng) -> list[dict]:
        """
        Takes a slice of legitimate transactions (usually for ONE account, or a
        related group of accounts for network scenarios) and returns a list of
        transactions manipulated according to `scenario`, each labeled with
        ground-truth fields.

        Must NOT mutate legit_txns in place - always work on copies.
        """
        raise NotImplementedError

    def _label(self, txn: dict, scenario: dict) -> dict:
        txn = dict(txn)
        txn["is_fraud"] = True
        txn["f3_tactic"] = scenario["f3_tactic"]
        txn["f3_technique"] = scenario["f3_technique"]
        txn["scenario_id"] = scenario.get("scenario_id", "unassigned")
        txn["scenario_name"] = scenario["scenario_name"]
        txn["caught_by_model"] = None  # filled later by Defend scoring, Phase 2
        return txn

    def _new_txn_id(self) -> str:
        return str(uuid.uuid4())
