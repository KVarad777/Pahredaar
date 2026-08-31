"""
Generate Engine - legitimate traffic simulator.

Produces realistic-looking NON-FRAUD UPI transactions using distribution
parameters fitted from real public datasets (see notebooks/00_fit_distributions.ipynb).

This must be validated against real data (notebooks/../fidelity checks) BEFORE
any fraud injection is layered on top - if this baseline is unrealistic, every
downstream Defend result is meaningless.
"""

import hashlib
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import yaml
from faker import Faker

fake = Faker("en_IN")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


class LegitTrafficSimulator:
    def __init__(self, distribution_params_path: str = "config/distribution_params.yaml", seed: int = 42):
        with open(distribution_params_path) as f:
            self.params = yaml.safe_load(f)
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        mcc_w = self.params["mcc_frequency"]["weights"]
        self._mcc_codes = list(mcc_w.keys())
        self._mcc_probs = np.array(list(mcc_w.values()))
        self._mcc_probs = self._mcc_probs / self._mcc_probs.sum()

        ch_w = self.params["channel_mix"]["weights"]
        self._channels = list(ch_w.keys())
        self._channel_probs = np.array(list(ch_w.values()))
        self._channel_probs = self._channel_probs / self._channel_probs.sum()

        hour_w = np.array(self.params["diurnal_hour_weights"]["weights"], dtype=float)
        self._hour_probs = hour_w / hour_w.sum()

    # ---- individual field generators, exposed so injectors can call them too ----

    def sample_amount(self) -> float:
        p = self.params["amount_distribution"]
        val = self.rng.lognormal(mean=np.log(max(p["scale"], 1e-6)), sigma=p["shape"])
        return float(np.clip(val, p["min_realistic"], p["max_realistic"]))

    def sample_hour(self) -> int:
        return int(self.rng.choice(24, p=self._hour_probs))

    def sample_mcc(self) -> str:
        return str(self.rng.choice(self._mcc_codes, p=self._mcc_probs)).replace("_luxury", "")

    def sample_channel(self) -> str:
        return str(self.rng.choice(self._channels, p=self._channel_probs))

    def sample_timestamp(self, base_date: Optional[datetime] = None) -> str:
        base_date = base_date or datetime.now(timezone.utc)
        hour = self.sample_hour()
        minute = self.rng.integers(0, 60)
        second = self.rng.integers(0, 60)
        ts = base_date.replace(hour=hour, minute=int(minute), second=int(second), microsecond=0)
        return ts.isoformat()

    def sample_device_details(self) -> dict:
        os_choices = ["Android 13", "Android 14", "iOS 17", "iOS 18", "Android 12"]
        app_choices = ["com.phonepe.app", "com.google.android.apps.nbu.paisa.user",
                       "net.one97.paytm", "in.org.npci.upiapp"]
        lat, lon = fake.local_latlng(country_code="IN")[:2]
        return {
            "deviceId": _hash(str(uuid.uuid4())),
            "os": random.choice(os_choices),
            "geocode": f"{lat},{lon}",
            "ipAddress": _hash(fake.ipv4()),
            "appId": random.choice(app_choices),
        }

    # ---- full transaction assembly ----

    def generate_one(self, account_id: str, base_date: Optional[datetime] = None) -> dict:
        amount = round(self.sample_amount(), 2)
        return {
            "txnId": str(uuid.uuid4()),
            "account_id": account_id,  # internal join key, stripped before emitting raw UPI JSON if needed
            "payerVpa": f"{account_id}@{random.choice(['oksbi', 'okhdfcbank', 'okicici', 'ybl', 'paytm'])}",
            "payeeVpa": f"{fake.user_name()}@{random.choice(['paytm', 'ybl', 'okaxis'])}",
            "amount": amount,
            "currency": "INR",
            "timestamp": self.sample_timestamp(base_date),
            "deviceDetails": self.sample_device_details(),
            "remarks": random.choice(["Payment for order", "UPI Payment", "Bill payment", ""]),
            "refUrl": "",
            "mcc": self.sample_mcc(),
            "channel": self.sample_channel(),
            "initiationMode": random.choice(["00", "01", "05"]),
            "purposeCode": "00",
            "auth_result": random.choices(["approved", "declined", "retried"], weights=[0.94, 0.04, 0.02])[0],
            "is_refund": random.random() < 0.03,
        }

    def generate_batch(self, n_accounts: int = 200, txns_per_account_range=(1, 15),
                        base_date: Optional[datetime] = None) -> list[dict]:
        base_date = base_date or datetime.now(timezone.utc)
        out = []
        for i in range(n_accounts):
            account_id = f"user{i:05d}"
            n_txns = self.rng.integers(txns_per_account_range[0], txns_per_account_range[1] + 1)
            for _ in range(n_txns):
                out.append(self.generate_one(account_id, base_date))
        return out


if __name__ == "__main__":
    sim = LegitTrafficSimulator()
    batch = sim.generate_batch(n_accounts=5, txns_per_account_range=(1, 3))
    import json
    print(json.dumps(batch[0], indent=2))
    print(f"\nGenerated {len(batch)} legitimate transactions.")
