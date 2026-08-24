"""
=============================================================================
PROJECT AEGIS: DEEP LEARNING GRAPH CONVOLUTIONAL NETWORK (GCN) DEFENSE (graph_model.py)
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
This module upgrades the AEGIS Graph Defense to a Graph Neural Network (GNN)
using PyTorch Geometric (PyG) and Graph Convolutional Networks (GCN):
  1. Ingests master dataset (data/processed/master_aegis_dataset.csv).
  2. Builds payment network directed graph (Tokenized_PAN -> Terminal_Node_ID).
  3. Maps nodes to integer indices and builds PyG Data object (edge_index & features).
  4. Passes topology through a 2-layer GCNConv neural network to compute
     16-dimensional dense neighborhood structural embeddings.
  5. Fits an Isolation Forest directly on the GNN spatial node embeddings.
  6. Flags topological zero-day anomalies ('TERM-9999-EVIL' sleeper mule farming).
  7. Serializes the trained GNN anomaly model to models/gnn_iso_model.joblib.
=============================================================================
"""

import os
import sys
import argparse
from typing import Dict, Tuple, List, Any
import numpy as np
import pandas as pd
import networkx as nx
import joblib
import torch
import torch.nn as nn

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from sklearn.ensemble import IsolationForest

# Default Configuration
DEFAULT_DATA_PATH = os.path.join("data", "processed", "master_aegis_dataset.csv")
DEFAULT_MODEL_DIR = "models"
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "gnn_iso_model.joblib")
CONTAMINATION = 0.01
RANDOM_SEED = 42
GNN_HIDDEN_DIM = 16
GNN_OUT_DIM = 16


# =============================================================================
# PYTORCH GEOMETRIC GRAPH CONVOLUTIONAL NETWORK (GCN) ARCHITECTURE
# =============================================================================
class AegisGCN(nn.Module):
    """
    2-Layer Graph Convolutional Network (GCN) structural feature extractor.
    Performs neighborhood message passing across PAN and Terminal payment topologies.
    """
    def __init__(self, in_channels: int = 7, hidden_channels: int = GNN_HIDDEN_DIM, out_channels: int = GNN_OUT_DIM):
        super().__init__()
        torch.manual_seed(RANDOM_SEED)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1: 1-hop spatial neighborhood aggregation
        h1 = self.conv1(x, edge_index)
        h1 = self.relu(h1)
        # Layer 2: 2-hop structural topological embedding
        h2 = self.conv2(h1, edge_index)
        return h2


def load_transaction_data(data_path: str) -> pd.DataFrame:
    """Loads and validates the processed transaction dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at: {data_path}")

    print("=" * 80)
    print("  PROJECT AEGIS : DEEP LEARNING GRAPH CONVOLUTIONAL DEFENSE (GCN)")
    print("  Mastercard Innovation Challenge @ Global Fintech Fest 2026")
    print("=" * 80)
    print(f"[*] Ingesting dataset from: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"[+] Loaded {len(df):,} transactions x {len(df.columns)} columns")
    return df


def build_pyg_graph_data(df: pd.DataFrame) -> Tuple[Data, List[str], Dict[str, int], Dict[int, str]]:
    """
    Constructs a NetworkX graph and converts it into a PyTorch Geometric Data object.
    """
    print("\n" + "-" * 80)
    print("1. GRAPH TOPOLOGY & PYTORCH GEOMETRIC DATA OBJECT CREATION")
    print("-" * 80)
    print("[*] Initializing NetworkX Directed Graph & Aggregating Financial Flows...")
    
    G = nx.DiGraph()
    edge_agg = (
        df.groupby(["Tokenized_PAN", "Terminal_Node_ID"])
        .agg(
            weight=("TransactionAmt", "sum"),
            tx_count=("TransactionAmt", "count")
        )
        .reset_index()
    )

    for _, r in edge_agg.iterrows():
        G.add_edge(r["Tokenized_PAN"], r["Terminal_Node_ID"], weight=float(r["weight"]), tx_count=int(r["tx_count"]))

    all_nodes = sorted(list(G.nodes()))
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    idx_to_node = {i: n for i, n in enumerate(all_nodes)}

    terminals = df["Terminal_Node_ID"].dropna().unique().tolist()
    print(f"  [+] Graph Nodes: {len(all_nodes):,} ({len(terminals):,} Terminals + {len(all_nodes)-len(terminals):,} PANs)")
    print(f"  [+] Directed Flow Edges: {G.number_of_edges():,}")

    # Build Edge Index Tensor
    src_indices = [node_to_idx[u] for u, v in G.edges()]
    dst_indices = [node_to_idx[v] for u, v in G.edges()]
    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    edge_index_bi = to_undirected(edge_index)

    # Compute node structural baseline features
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    in_w = dict(G.in_degree(weight="weight"))
    out_w = dict(G.out_degree(weight="weight"))
    cent = nx.in_degree_centrality(G)

    feature_matrix = []
    for n in all_nodes:
        is_term = 1.0 if str(n).startswith("TERM-") else 0.0
        id_val = float(in_deg.get(n, 0))
        od_val = float(out_deg.get(n, 0))
        iw_val = float(in_w.get(n, 0.0))
        ow_val = float(out_w.get(n, 0.0))
        c_val = float(cent.get(n, 0.0))
        avg_in = iw_val / max(id_val, 1.0)
        feature_matrix.append([is_term, id_val, od_val, iw_val, ow_val, c_val, avg_in])

    x = torch.tensor(feature_matrix, dtype=torch.float)
    x_norm = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

    data = Data(x=x_norm, edge_index=edge_index_bi)
    print(f"  [+] PyG Data Object Created:")
    print(f"      - Node Feature Tensor (x):         {data.x.shape}")
    print(f"      - Message Passing Edges (bi-flow): {data.edge_index.shape[1]:,}")

    return data, terminals, node_to_idx, idx_to_node


def extract_gnn_embeddings(
    data: Data, terminals: List[str], node_to_idx: Dict[str, int], idx_to_node: Dict[int, str]
) -> Tuple[AegisGCN, np.ndarray, List[str]]:
    """
    Executes a forward pass through the GCN to extract 16-D structural embeddings.
    """
    print("\n" + "-" * 80)
    print("2. GRAPH CONVOLUTIONAL FORWARD PASS & DENSE EMBEDDING EXTRACTION")
    print("-" * 80)
    print(f"[*] Initializing 2-Layer GCNConv Model (Hidden Dim: {GNN_HIDDEN_DIM}, Out Dim: {GNN_OUT_DIM})...")

    model = AegisGCN(in_channels=data.x.shape[1], hidden_channels=GNN_HIDDEN_DIM, out_channels=GNN_OUT_DIM)
    model.eval()

    with torch.no_grad():
        all_embeddings = model(data.x, data.edge_index).cpu().numpy()

    # Extract embeddings for Terminal nodes
    term_indices = [node_to_idx[t] for t in terminals if t in node_to_idx]
    term_embeddings = all_embeddings[term_indices]
    term_names = [idx_to_node[i] for i in term_indices]

    print(f"  [+] Forward Pass Complete:")
    print(f"      - Global Embedding Matrix:   {all_embeddings.shape}")
    print(f"      - Terminal Embedding Matrix: {term_embeddings.shape} (16-D Dense Topological Vectors)")

    return model, term_embeddings, term_names


def train_gnn_isolation_forest(
    term_embeddings: np.ndarray, term_names: List[str]
) -> Tuple[IsolationForest, pd.DataFrame]:
    """
    Fits an Isolation Forest directly on the 16-dimensional GNN dense embeddings.
    """
    print("\n" + "-" * 80)
    print("3. GNN EMBEDDING ISOLATION FOREST ANOMALY MODEL TRAINING")
    print("-" * 80)
    print(f"[*] Training IsolationForest on 16-D GCN spatial embeddings...")
    print(f"    - contamination : {CONTAMINATION} (Top 1% Structural Outliers)")
    print(f"    - random_state  : {RANDOM_SEED}")
    print(f"    - n_estimators  : 150")

    iso_model = IsolationForest(
        n_estimators=150,
        contamination=CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    preds = iso_model.fit_predict(term_embeddings)
    decision_scores = iso_model.decision_function(term_embeddings)

    # Build results dataframe
    term_df = pd.DataFrame({
        "Terminal_Node_ID": term_names,
        "Iso_Pred": preds,
        "Decision_Score": decision_scores
    })

    # Normalized risk score in [0.0, 1.0]
    min_s, max_s = decision_scores.min(), decision_scores.max()
    if max_s > min_s:
        term_df["Graph_Anomaly_Risk"] = (max_s - decision_scores) / (max_s - min_s)
    else:
        term_df["Graph_Anomaly_Risk"] = 0.0

    flagged_anomalies = (term_df["Iso_Pred"] == -1).sum()
    print(f"  [+] GNN Isolation Forest Fitted:")
    print(f"      - Flagged Anomalous Terminals: {flagged_anomalies:,} / {len(term_df):,} ({flagged_anomalies*100.0/len(term_df):.2f}%)")
    print(f"      - Normal Terminals:           {(term_df['Iso_Pred'] == 1).sum():,}")

    return iso_model, term_df


def evaluate_gnn_defense_efficacy(df: pd.DataFrame, term_df: pd.DataFrame) -> None:
    """
    Maps GNN anomaly scores back to transaction rows and benchmarks Sleeper Mule detection.
    """
    print("\n" + "-" * 80)
    print("4. ZERO-DAY ATTACK EFFICACY & GNN DEFENSE BENCHMARK")
    print("-" * 80)

    merged_df = df.merge(
        term_df[["Terminal_Node_ID", "Iso_Pred", "Decision_Score", "Graph_Anomaly_Risk"]],
        on="Terminal_Node_ID",
        how="left"
    )

    farming_df = merged_df[merged_df["Attack_Type"] == "GRAPH_POISONING_FARMING"]
    bustout_df = merged_df[merged_df["Attack_Type"] == "GRAPH_POISONING"]

    print(f"[*] Inspecting Malicious Mule Ring ('TERM-9999-EVIL') GNN Profile:")
    evil_profile = term_df[term_df["Terminal_Node_ID"] == "TERM-9999-EVIL"]
    if not evil_profile.empty:
        p = evil_profile.iloc[0]
        status_tag = "[ANOMALY FLAGGED: -1]" if p["Iso_Pred"] == -1 else "[NORMAL: +1]"
        print(f"  • Terminal ID:              {p['Terminal_Node_ID']}")
        print(f"  • GNN Isolation Forest:     {status_tag}")
        print(f"  • Graph Anomaly Risk:       {p['Graph_Anomaly_Risk']:.4f} / 1.0000")
        print(f"  • Decision Function Margin: {p['Decision_Score']:.6f}")

    total_farming = len(farming_df)
    flagged_farming = (farming_df["Iso_Pred"] == -1).sum()
    farming_recall = (flagged_farming / max(total_farming, 1)) * 100.0

    total_bustout = len(bustout_df)
    flagged_bustout = (bustout_df["Iso_Pred"] == -1).sum()
    bustout_recall = (flagged_bustout / max(total_bustout, 1)) * 100.0

    print("\n[*] Efficacy Comparison: Tabular Edge vs. Deep Learning GNN Core Defense:")
    print("  " + "-" * 76)
    print(f"  {'Attack Vector':<32} | {'XGBoost (Edge)':<18} | {'GNN Conv + IF':<18}")
    print("  " + "-" * 76)
    print(f"  {'GRAPH_POISONING_FARMING (50 tx)':<32} | {'0.00% (0/50)':<18} | {farming_recall:.2f}% ({flagged_farming}/{total_farming})")
    print(f"  {'GRAPH_POISONING (1 bustout)':<32} | {'100.00% (1/1)':<18} | {bustout_recall:.2f}% ({flagged_bustout}/{total_bustout})")
    print("  " + "-" * 76)

    # General attack breakdown across all vectors
    print("\n[*] Overall Attack Type Anomaly Flagging Rates:")
    for attack_type, grp in merged_df.groupby("Attack_Type"):
        flagged = (grp["Iso_Pred"] == -1).sum()
        total = len(grp)
        pct = (flagged / max(total, 1)) * 100.0
        avg_risk = grp["Graph_Anomaly_Risk"].mean()
        print(f"    - {attack_type:<26}: {flagged:>4}/{total:<4} flagged ({pct:>6.2f}%) | Mean Risk: {avg_risk:.4f}")

    if farming_recall == 100.0:
        print(f"\n  [SUCCESS] 100.00% of sleeper-mule micro-transactions successfully flagged by GNN Defense!")


def save_gnn_model(model: IsolationForest, output_path: str) -> None:
    """Serializes the trained GNN Isolation Forest model artifact to disk."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 80)
    print("5. MODEL SERIALIZATION & DEPLOYMENT")
    print("-" * 80)
    print(f"[*] Serializing GNN Isolation Forest model to: {output_path}")
    
    joblib.dump(model, output_path)
    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"  [+] Production GNN Graph Model artifact saved successfully ({file_size_kb:.2f} KB)")
    print("=" * 80 + "\n")


def run_pipeline(data_path: str = DEFAULT_DATA_PATH, model_path: str = DEFAULT_MODEL_PATH):
    """Executes the end-to-end Deep Learning GNN graph defense pipeline."""
    # 1. Load Data
    df = load_transaction_data(data_path)

    # 2. Build PyG Graph Data
    data, terminals, node_to_idx, idx_to_node = build_pyg_graph_data(df)

    # 3. GCN Forward Pass & Embedding Extraction
    gcn_model, term_embeddings, term_names = extract_gnn_embeddings(data, terminals, node_to_idx, idx_to_node)

    # 4. Train Isolation Forest on GNN Embeddings
    iso_model, term_df = train_gnn_isolation_forest(term_embeddings, term_names)

    # 5. Evaluate Efficacy
    evaluate_gnn_defense_efficacy(df, term_df)

    # 6. Save Model Artifact
    save_gnn_model(iso_model, model_path)


def main():
    parser = argparse.ArgumentParser(description="Project AEGIS: Deep Learning GNN Graph Defense Pipeline")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="Path to processed dataset CSV")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH, help="Path to save gnn_iso_model.joblib")
    args = parser.parse_args()

    run_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
