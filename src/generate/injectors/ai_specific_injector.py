"""
AI-specific scenarios: deepfake-assisted KYC bypass, device emulation, AI-generated
phishing leading to a compromised-credential transaction. This is the category your
project's novelty argument leans on most - it targets attack mechanisms that are
newer than the formal F3 framework itself (per the spec doc's own framing).
"""

import random

from .base_injector import BaseInjector

_PHISHING_REMARKS = [
    "Refund initiated - confirm to receive INR {amt}",
    "KYC update required - complete payment to verify account",
    "You have won a cashback of INR {amt}, claim now",
    "Urgent: pending bill payment to avoid service suspension",
]


class AISpecificInjector(BaseInjector):
    manipulation_type = "ai_specific"

    def inject(self, legit_txns: list[dict], scenario: dict, rng) -> list[dict]:
        fields = set(scenario["fields_manipulated"])
        out = []

        if "deviceDetails" in fields or "geocode" in scenario["fields_manipulated"]:
            # Device emulation: geocode/OS internally inconsistent with the account's
            # established pattern (e.g. sudden OS version jump + geocode teleport).
            for txn in legit_txns:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                t["deviceDetails"] = dict(t.get("deviceDetails") or {})
                t["deviceDetails"]["os"] = "Android 9"  # implausibly outdated for an emulator image
                # geocode teleport: swap to a far-away plausible-looking coordinate
                t["deviceDetails"]["geocode"] = f"{round(rng.uniform(8, 34), 4)},{round(rng.uniform(69, 97), 4)}"
                t["geo_velocity_kmh"] = round(float(rng.uniform(900, 2500)), 1)  # impossible travel
                out.append(t)
            return out

        if "kyc_doc_similarity_score" in fields and "kyc_verification_method" in fields:
            # Deepfake-assisted KYC bypass: near-perfect biometric match score
            # (synthetic face/voice), submitted via the automated/biometric path.
            for txn in legit_txns:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                t["kyc_doc_similarity_score"] = round(rng.uniform(0.985, 0.9999), 4)
                t["kyc_verification_method"] = "biometric"
                out.append(t)
            return out

        if "remarks" in fields or "refUrl" in fields:
            # AI-generated phishing: plausible, personalized-looking payment-request
            # text driving a victim-authorized (not stolen-credential) transfer.
            for txn in legit_txns:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                template = random.choice(_PHISHING_REMARKS)
                t["remarks"] = template.format(amt=t.get("amount", 0))
                t["refUrl"] = f"https://secure-{random.choice(['upi-verify', 'kyc-update', 'refund-portal'])}.example-phish.tld/{rng.integers(1000,9999)}"
                out.append(t)
            return out

        # fallback
        for txn in legit_txns:
            t = self._label(dict(txn), scenario)
            t["txnId"] = self._new_txn_id()
            out.append(t)
        return out
