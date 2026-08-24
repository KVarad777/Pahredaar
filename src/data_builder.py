"""
=============================================================================
PROJECT AEGIS: MODULAR RED-TEAM SYNTHETIC DATA GENERATION PIPELINE
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
This module loads a sampled subset of the base IEEE-CIS dataset and injects
three specialized AEGIS Zero-Day attack layers:
  1. Baseline Feature Synthesis: Tokenized_PAN, Terminal_Node_ID,
     Biometric_Entropy (0.400 - 0.900), and Remittance_Metadata.
  2. Vector E (Generative Graph Poisoning): Sleeper-mule farming on
     'TERM-9999-EVIL' (50 micro-txs $1.50-$4.00) followed by a $10,000 bust-out.
  3. Vector F (Biometric Latent Diffusion Mimicry): Overwrites 20 fraud rows
     with deterministic entropy = 0.50001 (unnatural GenAI smoothness).
  4. Vector G (Agentic Semantic Smuggling): Overwrites 20 high-value fraud
     rows with innocent B2B memo "Q3 Enterprise Software Subscription Invoice - Rack 4B".
=============================================================================
"""

import os
import sys
import argparse
import random
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Domain Configuration
DEFAULT_SAMPLE_SIZE = 50_000
RANDOM_SEED = 42

# Legitimate Remittance Text Templates by Domain
DEFAULT_MEMO_TEMPLATES = [
    "Standard Point of Sale Settlement",
    "Authorized Customer Checkout",
    "Verified Retail Purchase",
    "Electronic Payment Clearance",
    "Weekly Grocery Store Checkout",
    "Organic Produce and Pantry Supplies",
    "Cafe Espresso and Breakfast",
    "Express Luncheon Order",
    "Enterprise SaaS Software Subscription",
    "Monthly Cloud Server Hosting Fee",
    "Corporate Strategy Advisory Retainer",
    "Commercial Contract Review Fee",
    "Intermodal Freight Transport Invoice",
    "Supply Chain Warehouse Distribution",
    "Telecommunication Monthly Bandwidth Plan",
    "Digital Streaming Service Renewal",
]


def load_raw_dataset(raw_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """Loads and samples the raw base transaction dataset."""
    if not os.path.exists(raw_path):
        candidates = [
            os.path.join("data", "raw", "train_transaction.csv"),
            os.path.join("data", "raw", "train_transactions.csv"),
            os.path.join("data", "train_transactions.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                raw_path = c
                break

    print(f"[*] Loading raw base dataset from: {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"[+] Loaded raw dataset: {len(df):,} rows x {len(df.columns)} columns")

    # Sample to target subset size
    if len(df) > sample_size:
        print(f"[*] Sampling {sample_size:,} rows (Random Seed: {RANDOM_SEED})...")
        df = df.sample(n=sample_size, random_state=RANDOM_SEED).copy()
        # Re-sort by transaction timestamp/ID to preserve chronological sequencing
        if "TransactionDT" in df.columns:
            df = df.sort_values(by="TransactionDT").reset_index(drop=True)
        elif "Timestamp" in df.columns:
            df = df.sort_values(by="Timestamp").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
    else:
        df = df.copy().reset_index(drop=True)

    return df


def apply_baseline_injection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies baseline column standardization and features:
      - Renames card1 -> Tokenized_PAN, isFraud -> Fraud_Label
      - Generates Terminal_Node_ID
      - Generates Biometric_Entropy [0.400, 0.900]
      - Generates Remittance_Metadata text
      - Initializes Attack_Type to 'BENIGN'
    """
    print("\n[*] Applying AEGIS Baseline Feature Synthesis...")
    
    # 1. Column Renaming
    rename_map = {}
    if "card1" in df.columns:
        rename_map["card1"] = "Tokenized_PAN"
    elif "PAN" in df.columns:
        rename_map["PAN"] = "Tokenized_PAN"

    if "isFraud" in df.columns:
        rename_map["isFraud"] = "Fraud_Label"
    elif "IsFraud" in df.columns:
        rename_map["IsFraud"] = "Fraud_Label"

    df = df.rename(columns=rename_map)

    # Ensure Tokenized_PAN exists
    if "Tokenized_PAN" not in df.columns:
        df["Tokenized_PAN"] = [f"CARD_LEGIT_{i:06d}" for i in range(len(df))]

    # Ensure Fraud_Label exists
    if "Fraud_Label" not in df.columns:
        df["Fraud_Label"] = 0

    # Ensure TransactionAmt exists
    if "TransactionAmt" not in df.columns:
        if "amount" in df.columns:
            df = df.rename(columns={"amount": "TransactionAmt"})
        else:
            df["TransactionAmt"] = np.random.lognormal(mean=3.8, sigma=0.8, size=len(df)).round(2)

    # 2. Terminal_Node_ID Generation (e.g., TERM-1000 through TERM-2500)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    n_terminals = 500
    terminal_pool = [f"TERM-{1000 + i:04d}" for i in range(n_terminals)]
    df["Terminal_Node_ID"] = np.random.choice(terminal_pool, size=len(df))

    # 3. Biometric_Entropy Generation (Normal human variance is random uniform between 0.400 and 0.900)
    # Continuous human micro-tremors produce realistic continuous entropy
    human_entropy = np.random.uniform(0.400, 0.900, size=len(df))
    df["Biometric_Entropy"] = np.round(human_entropy, 5)

    # 4. Remittance_Metadata Generation
    if "TextMemo" in df.columns:
        df["Remittance_Metadata"] = df["TextMemo"].fillna("Standard Point of Sale Settlement")
    else:
        df["Remittance_Metadata"] = np.random.choice(DEFAULT_MEMO_TEMPLATES, size=len(df))

    # 5. Attack_Type tag initialization
    df["Attack_Type"] = "BENIGN"

    print(f"  [+] Standardized Tokenized_PAN & Fraud_Label")
    print(f"  [+] Synthesized Terminal_Node_ID across {n_terminals} merchant terminals")
    print(f"  [+] Injected Human Biometric_Entropy (Mean: {df['Biometric_Entropy'].mean():.4f}, Range: [{df['Biometric_Entropy'].min():.3f}, {df['Biometric_Entropy'].max():.3f}])")
    print(f"  [+] Injected Remittance_Metadata descriptions")
    return df


def inject_vector_e_graph_poisoning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector E: Generative Graph Poisoning (Sleeper Mule)
      - Farming Phase: 50 legitimate rows routed to 'TERM-9999-EVIL' with micro-values ($1.50 - $4.00)
      - Bust-Out Phase: 1 high-value fraud row ($10,000.00) on 'TERM-9999-EVIL'
    """
    print("\n[*] Injecting Vector E: Generative Graph Poisoning (Sleeper Mule)...")
    evil_terminal = "TERM-9999-EVIL"

    # Select 50 legitimate rows for farming
    legit_indices = df[df["Fraud_Label"] == 0].index
    if len(legit_indices) < 50:
        raise ValueError("Insufficient legitimate rows to perform graph poisoning farming.")
    
    farming_indices = np.random.choice(legit_indices, size=50, replace=False)
    
    # Farming phase updates
    micro_amounts = np.round(np.random.uniform(1.50, 4.00, size=50), 2)
    df.loc[farming_indices, "Terminal_Node_ID"] = evil_terminal
    df.loc[farming_indices, "TransactionAmt"] = micro_amounts
    df.loc[farming_indices, "Attack_Type"] = "GRAPH_POISONING_FARMING"

    # Bust-Out Phase: Synthesize 1 high-value bust-out transaction
    max_dt = df["TransactionDT"].max() if "TransactionDT" in df.columns else 100000
    last_tx_id = df["TransactionID"].iloc[-1] if "TransactionID" in df.columns else 2999999
    try:
        new_tx_id = int(last_tx_id) + 1
    except Exception:
        new_tx_id = f"TX_BUSTOUT_0001"

    bust_out_row = {
        "TransactionID": new_tx_id,
        "TransactionDT": max_dt + 15 if "TransactionDT" in df.columns else None,
        "TransactionAmt": 10000.00,
        "Tokenized_PAN": "CARD_MULE_BUSTOUT_001",
        "Terminal_Node_ID": evil_terminal,
        "Biometric_Entropy": round(float(np.random.uniform(0.400, 0.900)), 5),
        "Remittance_Metadata": "Bulk Terminal High-Value Settlement",
        "Fraud_Label": 1,
        "Attack_Type": "GRAPH_POISONING",
    }

    # Fill any remaining columns with defaults or NaN
    for col in df.columns:
        if col not in bust_out_row:
            bust_out_row[col] = df[col].iloc[0] if not df[col].empty else None

    # Append the bust-out row
    df_bust = pd.DataFrame([bust_out_row])
    df = pd.concat([df, df_bust], ignore_index=True)

    print(f"  [+] Farming Phase: Injected 50 trust-building micro-transactions ($1.50 - $4.00) on '{evil_terminal}'")
    print(f"  [+] Bust-Out Phase: Injected 1 coordinated $10,000.00 attack row on '{evil_terminal}' tagged 'GRAPH_POISONING'")
    return df


def inject_vector_f_biometric_mimicry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector F: Biometric Latent Diffusion Mimicry
      - Overwrite 20 existing fraud rows with exact Biometric_Entropy = 0.50001
      - Simulates GenAI spoof that is mathematically too smooth and devoid of human micro-tremors.
    """
    print("\n[*] Injecting Vector F: Biometric Latent Diffusion Mimicry...")
    
    # Target existing fraud rows not already modified by Vector E
    candidate_indices = df[(df["Fraud_Label"] == 1) & (df["Attack_Type"] == "BENIGN")].index
    if len(candidate_indices) < 20:
        # Fallback to any fraud row if subset is small
        candidate_indices = df[df["Fraud_Label"] == 1].index

    sample_size = min(20, len(candidate_indices))
    mimicry_indices = np.random.choice(candidate_indices, size=sample_size, replace=False)

    df.loc[mimicry_indices, "Biometric_Entropy"] = 0.50001
    df.loc[mimicry_indices, "Attack_Type"] = "BIOMETRIC_MIMICRY"

    print(f"  [+] Overwrote {sample_size} fraud rows with exact Biometric_Entropy = 0.50001 (Zero-Jitter Bot Signature)")
    print(f"  [+] Tagged Attack_Type = 'BIOMETRIC_MIMICRY'")
    return df


def inject_vector_g_semantic_smuggling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector G: Agentic Semantic Smuggling
      - Overwrite 20 high-value fraud rows with innocent B2B invoice description:
        'Q3 Enterprise Software Subscription Invoice - Rack 4B'
    """
    print("\n[*] Injecting Vector G: Agentic Semantic Smuggling...")
    
    smuggle_memo = "Q3 Enterprise Software Subscription Invoice - Rack 4B"
    
    # Target high-value fraud rows
    fraud_df = df[(df["Fraud_Label"] == 1) & (df["Attack_Type"] == "BENIGN")]
    if len(fraud_df) >= 20:
        high_val_indices = fraud_df.sort_values(by="TransactionAmt", ascending=False).head(20).index
    else:
        high_val_indices = df[df["Fraud_Label"] == 1].sort_values(by="TransactionAmt", ascending=False).head(20).index

    df.loc[high_val_indices, "Remittance_Metadata"] = smuggle_memo
    df.loc[high_val_indices, "Attack_Type"] = "SEMANTIC_SMUGGLING"

    print(f"  [+] Overwrote {len(high_val_indices)} high-value fraud rows with sanitized memo: '{smuggle_memo}'")
    print(f"  [+] Tagged Attack_Type = 'SEMANTIC_SMUGGLING'")
    return df


def generate_master_aegis_dataset(raw_path: str, output_path: str, sample_size: int = DEFAULT_SAMPLE_SIZE) -> pd.DataFrame:
    """End-to-end orchestration pipeline for AEGIS synthetic data generation."""
    print("=" * 80)
    print("  PROJECT AEGIS : RED-TEAM ZERO-DAY SYNTHETIC DATA GENERATION PIPELINE")
    print("  Mastercard Innovation Challenge @ Global Fintech Fest 2026")
    print("=" * 80)

    # 1. Load & Sample
    df = load_raw_dataset(raw_path, sample_size)

    # 2. Baseline Standardization & Features
    df = apply_baseline_injection(df)

    # 3. Vector E: Generative Graph Poisoning
    df = inject_vector_e_graph_poisoning(df)

    # 4. Vector F: Biometric Latent Diffusion Mimicry
    df = inject_vector_f_biometric_mimicry(df)

    # 5. Vector G: Agentic Semantic Smuggling
    df = inject_vector_g_semantic_smuggling(df)

    # 6. Save Processed Dataset
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n[*] Exporting master dataset to: {output_path}...")
    df.to_csv(output_path, index=False)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # 7. Summary Report
    print("\n" + "=" * 80)
    print("  MASTER AEGIS DATASET GENERATION SUMMARY")
    print("=" * 80)
    print(f"  • Total Dataset Records:     {len(df):,} rows x {len(df.columns)} columns")
    print(f"  • Output File Size:          {file_size_mb:.2f} MB")
    print(f"  • Target File Destination:   {output_path}")
    print("\n  • Attack Type Distribution Breakdown:")
    attack_counts = df["Attack_Type"].value_counts()
    for attack, count in attack_counts.items():
        pct = (count / len(df)) * 100.0
        print(f"    - {attack:<26}: {count:>6,} ({pct:.3f}%)")

    fraud_total = (df["Fraud_Label"] == 1).sum()
    fraud_pct = (fraud_total / len(df)) * 100.0
    print(f"\n  • Total Fraudulent Records:  {fraud_total:,} ({fraud_pct:.2f}%)")
    print(f"  • Total Legitimate Records:  {(len(df) - fraud_total):,} ({100.0 - fraud_pct:.2f}%)")
    print("=" * 80 + "\n")

    return df


def main():
    parser = argparse.ArgumentParser(description="Project AEGIS: Red-Team Synthetic Data Builder")
    parser.add_argument("--raw", type=str, default="data/raw/train_transaction.csv", help="Path to raw input CSV")
    parser.add_argument("--output", type=str, default="data/processed/master_aegis_dataset.csv", help="Destination path")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of baseline rows to sample")
    args = parser.parse_args()

    generate_master_aegis_dataset(args.raw, args.output, args.sample_size)


if __name__ == "__main__":
    main()
