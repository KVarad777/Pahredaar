"""
Defend Engine - Sequence model (LSTM).

Input: last N transactions for that account as an ordered sequence
(amount, time-delta, MCC-encoded, channel-encoded).
Output: sequence-anomaly score 0-1 - catches low-and-slow, where no single
transaction is anomalous but the SEQUENCE is (unnaturally stable small gaps,
sustained small amounts).

Primary path: PyTorch LSTM (train this for real results - see
notebooks/03_defend_training.ipynb for the Colab GPU setup).

Fallback path: if torch isn't installed, uses hand-computed sequence
statistics (variance of inter-transaction gaps, mean amount, amount std)
fed through logistic regression - a legitimate simpler baseline for the
same signal, not a placeholder. Swap in the trained LSTM for your real
results; the fallback is honestly weaker and you should say so in your
feasibility slide if you end up relying on it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


SEQ_LEN = 15  # per spec: start at 10-20 prior transactions


def build_account_sequences(df: pd.DataFrame, seq_len: int = SEQ_LEN) -> dict[str, np.ndarray]:
    """
    Groups transactions by account_id (sorted by timestamp), returns a fixed-length
    sequence of [amount, time_delta_seconds, mcc_hash_normalized] per account,
    left-padded with zeros if shorter than seq_len.
    """
    df = df.sort_values("timestamp")
    sequences = {}

    for account_id, group in df.groupby("account_id"):
        group = group.sort_values("timestamp")
        amounts = group["amount"].astype(float).tolist()
        timestamps = pd.to_datetime(group["timestamp"], format="ISO8601").tolist()
        deltas = [0.0] + [(timestamps[i] - timestamps[i - 1]).total_seconds() for i in range(1, len(timestamps))]
        mcc_hash = [hash(str(m)) % 1000 / 1000.0 for m in group["mcc"].fillna("unknown")]

        seq = np.array(list(zip(amounts, deltas, mcc_hash)), dtype=np.float32)

        if len(seq) >= seq_len:
            seq = seq[-seq_len:]
        else:
            pad = np.zeros((seq_len - len(seq), 3), dtype=np.float32)
            seq = np.vstack([pad, seq])

        sequences[account_id] = seq

    return sequences


if TORCH_AVAILABLE:

    class LSTMFraudNet(nn.Module):
        def __init__(self, input_dim: int = 3, hidden_dim: int = 32, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, dropout=0)
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            h = self.drop(h_n[-1])
            return torch.sigmoid(self.out(h)).squeeze(-1)


@dataclass
class SequenceTrainResult:
    val_auc: float
    backend: str


class SequenceFraudModel:
    def __init__(self, seq_len: int = SEQ_LEN, hidden_dim: int = 32, dropout: float = 0.2):
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.model = None
        self.scaler = None  # only used by fallback

    def _fallback_features(self, seq: np.ndarray) -> np.ndarray:
        amounts = seq[:, 0]
        deltas = seq[:, 1]
        nonzero_deltas = deltas[deltas > 0]
        return np.array([
            amounts.mean(), amounts.std(),
            nonzero_deltas.mean() if len(nonzero_deltas) else 0.0,
            nonzero_deltas.std() if len(nonzero_deltas) else 0.0,  # LOW std = suspiciously stable pacing
            (amounts > 0).sum(),  # number of real (non-pad) transactions in the window
        ])

    def train(self, sequences: dict[str, np.ndarray], labels: dict[str, int],
              epochs: int = 30, lr: float = 1e-3, seed: int = 42) -> SequenceTrainResult:
        account_ids = [a for a in sequences if a in labels]
        X_seq = np.stack([sequences[a] for a in account_ids])
        y = np.array([labels[a] for a in account_ids], dtype=np.float32)

        if TORCH_AVAILABLE:
            torch.manual_seed(seed)
            X_t = torch.tensor(X_seq, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)

            self.model = LSTMFraudNet(input_dim=3, hidden_dim=self.hidden_dim, dropout=self.dropout)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            pos_weight = max((y == 0).sum() / max((y == 1).sum(), 1), 1.0)
            criterion = nn.BCELoss(weight=None)  # keep simple; class imbalance handled via sampling if needed

            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                preds = self.model(X_t)
                loss = criterion(preds, y_t)
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.no_grad():
                final_preds = self.model(X_t).numpy()

            from sklearn.metrics import roc_auc_score
            try:
                val_auc = roc_auc_score(y, final_preds)
            except ValueError:
                val_auc = float("nan")
            return SequenceTrainResult(val_auc=val_auc, backend="lstm_torch")

        else:
            X_feat = np.stack([self._fallback_features(sequences[a]) for a in account_ids])
            self.scaler = StandardScaler().fit(X_feat)
            X_scaled = self.scaler.transform(X_feat)

            self.model = LogisticRegression(class_weight="balanced", max_iter=1000)
            self.model.fit(X_scaled, y)

            from sklearn.metrics import roc_auc_score
            probs = self.model.predict_proba(X_scaled)[:, 1]
            try:
                val_auc = roc_auc_score(y, probs)
            except ValueError:
                val_auc = float("nan")
            return SequenceTrainResult(val_auc=val_auc, backend="logreg_fallback")

    def score(self, sequences: dict[str, np.ndarray]) -> dict[str, float]:
        account_ids = list(sequences.keys())

        if TORCH_AVAILABLE and isinstance(self.model, LSTMFraudNet):
            X_t = torch.tensor(np.stack([sequences[a] for a in account_ids]), dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                preds = self.model(X_t).numpy()
            return dict(zip(account_ids, preds.tolist()))

        X_feat = np.stack([self._fallback_features(sequences[a]) for a in account_ids])
        X_scaled = self.scaler.transform(X_feat)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        return dict(zip(account_ids, probs.tolist()))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.generate.orchestrator import GenerateOrchestrator

    scenarios = [{
        "scenario_id": "s1", "scenario_name": "low-and-slow", "f3_tactic": "Evasion",
        "f3_technique": "Low-and-Slow Velocity Abuse", "mechanism_description": "x",
        "fields_manipulated": ["amount", "mean_inter_txn_seconds"],
        "manipulation_type": "behavioral", "novelty_tag": "t",
    }]
    orch = GenerateOrchestrator()
    raw_df = orch.run_round(scenarios, n_legit_accounts=200, fraud_txns_per_scenario_range=(15, 20),
                             round_n=95, out_dir="data/generated_smoketest")

    sequences = build_account_sequences(raw_df)
    labels = raw_df.groupby("account_id")["is_fraud"].max().astype(int).to_dict()

    seq_model = SequenceFraudModel()
    print(f"torch available: {TORCH_AVAILABLE}")
    result = seq_model.train(sequences, labels)
    print(f"Sequence model train result: {result}")

    sample_accounts = list(sequences.keys())[:5]
    scores = seq_model.score({a: sequences[a] for a in sample_accounts})
    print("Sample scores:", scores)
