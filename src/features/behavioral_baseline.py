"""
Feature Pipeline - Behavioral Baseline.

Maintains precomputed rolling statistics per account (typical login hour,
typical amount range), updated after each transaction. Deviation from an
account's OWN history is often a stronger fraud signal than any absolute
threshold.
"""

from collections import defaultdict
from datetime import datetime


class BehavioralBaseline:
    def __init__(self):
        # per account: running stats
        self._amount_sum: dict[str, float] = defaultdict(float)
        self._amount_sq_sum: dict[str, float] = defaultdict(float)
        self._amount_count: dict[str, int] = defaultdict(int)
        self._hour_history: dict[str, list[int]] = defaultdict(list)

    @staticmethod
    def _parse_hour(ts) -> int:
        if isinstance(ts, datetime):
            return ts.hour
        return datetime.fromisoformat(str(ts)).hour

    def lookup(self, account_id: str, current_ts, current_amount: float) -> dict:
        """Compute deviation of THIS transaction from the account's prior baseline."""
        n = self._amount_count[account_id]
        hour = self._parse_hour(current_ts)

        if n == 0:
            # cold start - no history yet. Use population defaults, tag explicitly.
            return {
                "amount_zscore_vs_self": 0.0,
                "login_time_deviation_hrs": 0.0,
                "is_new_account": True,
                "prior_txn_count": 0,
            }

        mean = self._amount_sum[account_id] / n
        variance = max(self._amount_sq_sum[account_id] / n - mean ** 2, 1e-6)
        std = variance ** 0.5
        zscore = (current_amount - mean) / std if std > 0 else 0.0

        prior_hours = self._hour_history[account_id]
        typical_hour = sum(prior_hours) / len(prior_hours)
        # circular hour distance (23 and 0 are only 1hr apart, not 23hr apart)
        raw_diff = abs(hour - typical_hour)
        hour_deviation = min(raw_diff, 24 - raw_diff)

        return {
            "amount_zscore_vs_self": round(zscore, 3),
            "login_time_deviation_hrs": round(hour_deviation, 2),
            "is_new_account": False,
            "prior_txn_count": n,
        }

    def update(self, account_id: str, ts, amount: float) -> None:
        self._amount_sum[account_id] += amount
        self._amount_sq_sum[account_id] += amount ** 2
        self._amount_count[account_id] += 1
        self._hour_history[account_id].append(self._parse_hour(ts))
        # cap history length so memory doesn't grow unbounded over many rounds
        if len(self._hour_history[account_id]) > 500:
            self._hour_history[account_id] = self._hour_history[account_id][-500:]

    def lookup_and_update(self, account_id: str, ts, amount: float) -> dict:
        features = self.lookup(account_id, ts, amount)
        self.update(account_id, ts, amount)
        return features


if __name__ == "__main__":
    from datetime import datetime, timedelta

    bb = BehavioralBaseline()
    base = datetime(2026, 1, 1, 14, 0, 0)  # 2pm baseline

    for i in range(5):
        ts = base + timedelta(days=i)
        feats = bb.lookup_and_update("acct_001", ts, amount=100.0 + i)
        print(f"txn {i}: {feats}")

    # now a suspicious 3am login with a huge amount
    odd_ts = base + timedelta(days=6, hours=-11)  # 3am
    feats = bb.lookup("acct_001", odd_ts, current_amount=50000.0)
    print("\nOdd transaction (3am, huge amount):", feats)
