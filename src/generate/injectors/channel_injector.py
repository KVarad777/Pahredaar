"""
Channel-based scenarios: card-not-present abuse, promo/refund abuse, chargeback fraud,
MCC-mismatch fraud. The anomaly here lives in channel/MCC/refund-flag combinations
that are individually valid but collectively unusual for how that account normally
transacts.
"""

from .base_injector import BaseInjector


class ChannelInjector(BaseInjector):
    manipulation_type = "channel"

    def inject(self, legit_txns: list[dict], scenario: dict, rng) -> list[dict]:
        fields = set(scenario["fields_manipulated"])
        out = []

        if "amount" in fields and "channel" in fields:
            # Card-testing: burst of very small CNP probes, then one large charge.
            for txn in legit_txns[:-1]:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                t["channel"] = "CNP"
                t["amount"] = round(float(rng.uniform(1, 15)), 2)  # tiny test charges
                t["auth_result"] = rng.choice(["approved", "declined"], p=[0.3, 0.7])
                out.append(t)
            if legit_txns:
                t = self._label(dict(legit_txns[-1]), scenario)
                t["txnId"] = self._new_txn_id()
                t["channel"] = "CNP"
                t["amount"] = round(float(rng.uniform(15000, 60000)), 2)  # the real charge
                t["auth_result"] = "approved"
                out.append(t)
            return out

        if "is_refund" in fields:
            # Refund-abuse: repeated refund claims clustered close together in time
            # for the same account, which real legitimate refund patterns rarely show.
            for txn in legit_txns:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                t["is_refund"] = True
                t["processing_code"] = "refund"
                out.append(t)
            return out

        if "mcc" in fields:
            # MCC-mismatch: transaction routed under a low-risk MCC while behaving
            # like a high-risk category (e.g. gambling disguised as retail).
            for txn in legit_txns:
                t = self._label(dict(txn), scenario)
                t["txnId"] = self._new_txn_id()
                t["mcc"] = "5999"          # disguised as generic retail
                t["true_mcc_estimate"] = "7995"  # gambling - what the behavior actually looks like
                out.append(t)
            return out

        # fallback
        for txn in legit_txns:
            t = self._label(dict(txn), scenario)
            t["txnId"] = self._new_txn_id()
            out.append(t)
        return out
