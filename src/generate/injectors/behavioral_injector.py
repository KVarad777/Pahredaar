"""
Behavioral scenarios: low-and-slow velocity abuse, unusual login-hour patterns.
The anomaly here is an AGGREGATION across multiple transactions - no single
transaction looks suspicious alone, which is the whole point (fidelity requirement:
per-transaction, it must look normal).
"""

from datetime import timedelta

from .base_injector import BaseInjector


class BehavioralInjector(BaseInjector):
    manipulation_type = "behavioral"

    def inject(self, legit_txns: list[dict], scenario: dict, rng) -> list[dict]:
        fields = set(scenario["fields_manipulated"])
        out = []

        if not legit_txns:
            return out

        base_ts_str = legit_txns[0]["timestamp"]

        if "mean_inter_txn_seconds" in fields and "amount" in fields:
            # Low-and-slow: many small, individually-unremarkable transactions,
            # spaced with unnaturally STABLE (low-variance) gaps - a real user's
            # timing has more natural jitter than a scripted drain.
            n_txns = int(rng.integers(8, 20))
            small_amount = float(rng.uniform(50, 400))  # stays well under any obvious threshold
            stable_gap_seconds = float(rng.uniform(55, 65))  # suspiciously consistent, low variance

            from datetime import datetime
            try:
                base_ts = datetime.fromisoformat(base_ts_str)
            except ValueError:
                base_ts = datetime.now()

            for i in range(n_txns):
                t = self._label(dict(legit_txns[0]), scenario)
                t["txnId"] = self._new_txn_id()
                t["amount"] = round(small_amount + rng.normal(0, 3), 2)  # tiny natural jitter only
                t["timestamp"] = (base_ts + timedelta(seconds=stable_gap_seconds * i)).isoformat()
                t["mean_inter_txn_seconds"] = stable_gap_seconds
                out.append(t)
            return out

        if "login_time_deviation_hrs" in fields:
            for txn in legit_txns:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                # login far outside this account's normal hour baseline
                t["login_time_deviation_hrs"] = round(float(rng.uniform(6, 14)), 2)
                out.append(t)
            return out

        # fallback: generic behavioral deviation if fields don't match a known pattern above
        for txn in legit_txns:
            t = self._label(dict(txn), scenario)
            t["txnId"] = self._new_txn_id()
            out.append(t)
        return out
