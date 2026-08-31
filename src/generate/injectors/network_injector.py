"""
Network-based scenarios: mule networks, device/IP sharing rings, collusive merchants.
The anomaly here is RELATIONAL - it only shows up when you look at the graph of
which accounts share a device fingerprint / IP / rapid transfer chain, not at any
single account's transaction history in isolation.
"""

import hashlib

from .base_injector import BaseInjector


def _hash(v: str) -> str:
    return hashlib.sha256(v.encode()).hexdigest()[:16]


class NetworkInjector(BaseInjector):
    manipulation_type = "network"

    def inject(self, legit_txns: list[dict], scenario: dict, rng) -> list[dict]:
        fields = set(scenario["fields_manipulated"])
        out = []

        if "device_fingerprint" in fields and "ip_address_hash" in fields:
            # Ring signal: force a shared device/IP across what look like DIFFERENT accounts.
            shared_device = _hash(f"ring_device_{rng.integers(0, 1_000_000)}")
            shared_ip = _hash(f"ring_ip_{rng.integers(0, 1_000_000)}")

            for i, txn in enumerate(legit_txns):
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                t["deviceDetails"] = dict(t.get("deviceDetails") or {})
                t["deviceDetails"]["deviceId"] = shared_device
                t["deviceDetails"]["ipAddress"] = shared_ip
                # different account_id / payerVpa per txn simulates "different" identities
                t["account_id"] = f"mule_{i:03d}_{rng.integers(0, 9999)}"
                out.append(t)
            return out

        if "device_fingerprint" in fields and "Anti-Fingerprinting" in scenario.get("f3_technique", ""):
            # Handled primarily by null_injector's MNAR override, but we still mark the
            # scenario mechanism here so the label/coverage matrix reflects it.
            for txn in legit_txns:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                out.append(t)
            return out

        if "amount" in fields and "timestamp" in fields:
            # Mule-network cash-out: rapid layered transfer chain, amount decaying
            # slightly at each hop (a cut taken at each layer) within a short window.
            from datetime import datetime, timedelta
            try:
                base_ts = datetime.fromisoformat(legit_txns[0]["timestamp"])
            except (ValueError, IndexError):
                base_ts = None
            amount = float(rng.uniform(20000, 90000))
            for hop in range(int(rng.integers(3, 6))):
                t = self._label(dict(legit_txns[0] if legit_txns else {}), scenario)
                t["txnId"] = self._new_txn_id()
                t["amount"] = round(amount, 2)
                if base_ts:
                    t["timestamp"] = (base_ts + timedelta(minutes=hop * rng.uniform(2, 8))).isoformat()
                t["account_id"] = f"mule_hop_{hop:02d}_{rng.integers(0, 9999)}"
                amount *= rng.uniform(0.85, 0.95)  # cut taken at each layer
                out.append(t)
            return out

        # fallback
        for txn in legit_txns:
            t = self._label(dict(txn), scenario)
            t["txnId"] = self._new_txn_id()
            out.append(t)
        return out
