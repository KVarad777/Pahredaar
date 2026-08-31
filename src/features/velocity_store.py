"""
Feature Pipeline - Velocity Store.

Maintains running per-account transaction counters (count/sum in last 1h/24h/7d)
INCREMENTALLY - updated as each transaction is processed, not recomputed by
scanning full history each time. This mirrors a real-time payment processor's
online feature store.

Usage:
    store = VelocityStore()
    for txn in sorted_by_timestamp(transactions):
        features = store.lookup(txn["account_id"], txn["timestamp"])  # BEFORE this txn
        store.update(txn["account_id"], txn["timestamp"], txn["amount"])  # AFTER using it
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta


WINDOWS = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
}


class VelocityStore:
    def __init__(self):
        # per account: deque of (timestamp, amount), oldest first
        self._history: dict[str, deque] = defaultdict(deque)

    @staticmethod
    def _parse_ts(ts) -> datetime:
        if isinstance(ts, datetime):
            return ts
        return datetime.fromisoformat(str(ts))

    def lookup(self, account_id: str, current_ts) -> dict:
        """
        Returns velocity features computed from history STRICTLY BEFORE current_ts.
        Call this before update() for the current transaction, so the current
        transaction never leaks into its own velocity features.
        """
        current_ts = self._parse_ts(current_ts)
        hist = self._history[account_id]

        features = {}
        for label, window in WINDOWS.items():
            cutoff = current_ts - window
            count = 0
            total = 0.0
            for ts, amount in hist:
                if ts >= cutoff and ts < current_ts:
                    count += 1
                    total += amount
            features[f"txn_count_{label}"] = count
            features[f"txn_sum_{label}"] = round(total, 2)

        return features

    def update(self, account_id: str, ts, amount: float) -> None:
        """Call after lookup(), to record this transaction into the account's history."""
        ts = self._parse_ts(ts)
        hist = self._history[account_id]
        hist.append((ts, float(amount)))

        # prune anything older than the largest window to bound memory growth
        max_window = max(WINDOWS.values())
        cutoff = ts - max_window
        while hist and hist[0][0] < cutoff:
            hist.popleft()

    def lookup_and_update(self, account_id: str, ts, amount: float) -> dict:
        """Convenience: compute features from prior history, then record this txn."""
        features = self.lookup(account_id, ts)
        self.update(account_id, ts, amount)
        return features


if __name__ == "__main__":
    from datetime import datetime, timedelta

    store = VelocityStore()
    base = datetime(2026, 1, 1, 12, 0, 0)

    # simulate 5 transactions, 10 minutes apart, for the same account
    for i in range(5):
        ts = base + timedelta(minutes=10 * i)
        feats = store.lookup_and_update("acct_001", ts, amount=100.0)
        print(f"txn {i} @ {ts.time()} -> velocity BEFORE this txn: {feats}")
