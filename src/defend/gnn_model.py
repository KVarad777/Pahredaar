"""
Defend Engine - GNN component (GraphSAGE).

Input: 2-hop subgraph around the transacting account (device/IP/account nodes),
node features = per-account aggregate stats.
Output: ring-membership / node-anomaly score 0-1.

Primary path: PyTorch Geometric GraphSAGE (recommended - install in Colab with
the matching torch/CUDA build, see notebooks/03_defend_training.ipynb).

Fallback path: if torch_geometric isn't installed, falls back to a hand-rolled
2-layer mean-aggregation GraphSAGE-style forward pass using plain numpy, so
the pipeline still runs end-to-end for local testing / CPU-only environments.
The fallback is NOT trained via backprop - it's a fixed random-projection
aggregator used only so the ensemble always has a graph-based score to
combine, even before you've set up the full torch_geometric training loop.
Swap in the real trained GraphSAGE for your actual results.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import from_networkx
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


NODE_FEATURE_DIM = 4  # [degree, shared_device_count, shared_ip_count, 2hop_account_count]


def _node_features_for_account(graph_state, account_id: str) -> np.ndarray:
    f = graph_state.account_features(account_id)
    return np.array([
        f["graph_degree"], f["graph_shared_device_count"],
        f["graph_shared_ip_count"], f["graph_2hop_account_count"],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Primary path: real GraphSAGE via torch_geometric (use this in Colab)
# ---------------------------------------------------------------------------

if TORCH_GEOMETRIC_AVAILABLE:

    class GraphSAGENet(torch.nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.out = torch.nn.Linear(hidden_dim, 1)

        def forward(self, x, edge_index):
            for conv in self.convs[:-1]:
                x = F.relu(conv(x, edge_index))
            x = self.convs[-1](x, edge_index)
            return torch.sigmoid(self.out(x)).squeeze(-1)


@dataclass
class GNNTrainResult:
    val_auc: float
    epochs_trained: int


class GNNFraudModel:
    """
    account_ids: ordered list of account_ids this model was trained/scored on.
    Node features come from GraphState.account_features() per account, at the
    time the full graph is built (call graph_state.add_transaction for every
    txn BEFORE building the model - i.e. reuse the same GraphState your
    FeatureAssembler already populated).
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self._account_to_idx: dict[str, int] = {}

    def _build_pyg_graph(self, graph_state, account_ids: list[str]):
        # subgraph limited to account nodes + their 1-hop device/ip neighbors, unioned
        nodes_to_keep = set()
        for acct in account_ids:
            node = f"account:{acct}"
            if node in graph_state.graph:
                nodes_to_keep.add(node)
                nodes_to_keep.update(graph_state.graph.neighbors(node))
        sub = graph_state.graph.subgraph(nodes_to_keep).copy()

        for n in sub.nodes:
            kind = sub.nodes[n].get("kind", "account")
            if kind == "account":
                acct_id = n.split(":", 1)[1]
                feats = _node_features_for_account(graph_state, acct_id)
            else:
                feats = np.zeros(NODE_FEATURE_DIM, dtype=np.float32)  # device/ip nodes get zero feature vec
            sub.nodes[n]["x"] = feats

        return sub

    def train(self, graph_state, account_ids: list[str], labels: dict[str, int],
              epochs: int = 100, lr: float = 0.01, seed: int = 42) -> GNNTrainResult:
        if not TORCH_GEOMETRIC_AVAILABLE:
            print("[GNNFraudModel] torch_geometric not installed - skipping real training. "
                  "score() will use the fallback aggregator. Install torch_geometric in "
                  "Colab (see notebooks/03_defend_training.ipynb) for real GNN training.")
            return GNNTrainResult(val_auc=float("nan"), epochs_trained=0)

        torch.manual_seed(seed)
        sub = self._build_pyg_graph(graph_state, account_ids)
        node_list = list(sub.nodes)
        self._account_to_idx = {n.split(":", 1)[1]: i for i, n in enumerate(node_list) if n.startswith("account:")}

        data = from_networkx(sub, group_node_attrs=["x"])
        data.x = data.x.float()

        y = torch.zeros(len(node_list))
        train_mask = torch.zeros(len(node_list), dtype=torch.bool)
        for acct, idx in self._account_to_idx.items():
            if acct in labels:
                y[idx] = float(labels[acct])
                train_mask[idx] = True

        self.model = GraphSAGENet(in_dim=NODE_FEATURE_DIM, hidden_dim=self.hidden_dim, num_layers=self.num_layers)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.binary_cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

        self.model.eval()
        with torch.no_grad():
            preds = self.model(data.x, data.edge_index)
        from sklearn.metrics import roc_auc_score
        try:
            val_auc = roc_auc_score(y[train_mask].numpy(), preds[train_mask].numpy())
        except ValueError:
            val_auc = float("nan")

        self._last_graph = sub
        self._last_preds = preds.detach().numpy()
        return GNNTrainResult(val_auc=val_auc, epochs_trained=epochs)

    def score(self, graph_state, account_ids: list[str]) -> dict[str, float]:
        """Returns {account_id: ring_anomaly_score} for the given accounts."""
        if TORCH_GEOMETRIC_AVAILABLE and self.model is not None:
            scores = {}
            for acct in account_ids:
                idx = self._account_to_idx.get(acct)
                scores[acct] = float(self._last_preds[idx]) if idx is not None else 0.0
            return scores

        # --- fallback: fixed random-projection 2-hop mean aggregator ---
        rng = np.random.default_rng(42)
        W1 = rng.normal(0, 1, (NODE_FEATURE_DIM, 16))
        W2 = rng.normal(0, 1, (16, 1))
        scores = {}
        for acct in account_ids:
            self_feat = _node_features_for_account(graph_state, acct)
            neighbor_ids = list(graph_state.graph.neighbors(f"account:{acct}")) if f"account:{acct}" in graph_state.graph else []
            neighbor_feats = [self_feat]  # include self
            for n in neighbor_ids:
                for nn in graph_state.graph.neighbors(n):
                    if nn.startswith("account:") and nn != f"account:{acct}":
                        neighbor_feats.append(_node_features_for_account(graph_state, nn.split(":", 1)[1]))
            agg = np.mean(neighbor_feats, axis=0)
            h1 = np.tanh(agg @ W1)
            raw_score = float(1 / (1 + np.exp(-(h1 @ W2)[0])))
            scores[acct] = raw_score
        return scores


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.generate.orchestrator import GenerateOrchestrator
    from src.features.feature_assembler import FeatureAssembler

    scenarios = [{
        "scenario_id": "s1", "scenario_name": "device ring", "f3_tactic": "Monetization",
        "f3_technique": "Mule Network Cash-Out", "mechanism_description": "x",
        "fields_manipulated": ["device_fingerprint", "ip_address_hash"],
        "manipulation_type": "network", "novelty_tag": "t",
    }]
    orch = GenerateOrchestrator()
    raw_df = orch.run_round(scenarios, n_legit_accounts=200, fraud_txns_per_scenario_range=(10, 15),
                             round_n=96, out_dir="data/generated_smoketest")

    assembler = FeatureAssembler()
    feat_df = assembler.assemble(raw_df)  # this populates assembler.graph as a side effect

    account_ids = feat_df["account_id"].dropna().unique().tolist()
    labels = dict(zip(feat_df["account_id"], feat_df["is_fraud"].astype(int)))

    gnn = GNNFraudModel()
    print(f"torch_geometric available: {TORCH_GEOMETRIC_AVAILABLE}")
    result = gnn.train(assembler.graph, account_ids, labels)
    print(f"GNN train result: {result}")

    scores = gnn.score(assembler.graph, account_ids[:5])
    print("Sample scores:", scores)
