"""
Identity-based scenarios: synthetic identity creation, account takeover,
deepfake-assisted KYC bypass. Manipulates identity/KYC-join fields, not
raw transaction amounts - the anomaly lives in WHO is transacting, not
what they're transacting.
"""

import random

from .base_injector import BaseInjector


class IdentityInjector(BaseInjector):
    manipulation_type = "identity"

    def inject(self, legit_txns: list[dict], scenario: dict, rng) -> list[dict]:
        fields = set(scenario["fields_manipulated"])
        out = []

        for txn in legit_txns:
            t = self._label(dict(txn), scenario)
            t["txnId"] = self._new_txn_id()

            if "kyc_doc_similarity_score" in fields:
                # near-1.0 = suspicious template reuse across many synthetic identities
                t["kyc_doc_similarity_score"] = round(rng.uniform(0.92, 0.999), 4)

            if "account_age_days" in fields:
                # synthetic identities cluster near account creation
                t["account_age_days"] = int(rng.integers(0, 5))
                t["is_new_account"] = True

            if "email_domain_risk_score" in fields:
                t["email_domain_risk_score"] = round(rng.uniform(0.7, 1.0), 3)  # disposable-domain proxy

            if "failed_auth_count_24h" in fields:
                # account-takeover signature: burst of failed logins then a success
                t["failed_auth_count_24h"] = int(rng.integers(4, 12))

            if "kyc_verification_method" in fields:
                # deepfake-KYC-bypass targets automated/biometric verification specifically
                t["kyc_verification_method"] = "biometric"

            out.append(t)

        return out
