"""
=============================================================================
PROJECT AEGIS: MODULAR RED-TEAM SYNTHETIC DATA GENERATION PIPELINE (data_builder.py)
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
This module loads a sampled subset of the base IEEE-CIS dataset and injects
comprehensive AEGIS Zero-Day attack vectors and Cyber primitives:
  1. Baseline Feature Synthesis: Tokenized_PAN, Terminal_Node_ID,
     Biometric_Entropy (0.400 - 0.900), and Remittance_Metadata.
  2. Zero-Trust Token Primitives: Token_ID ('AUTH-XXXX') & Token_Status ('ACTIVE').
  3. Vector E (Generative Graph Poisoning): Sleeper-mule farming on
     'TERM-9999-EVIL' (50 micro-txs $1.50-$4.00) followed by a $10,000 bust-out.
  4. Vector F (Biometric Latent Diffusion Mimicry): Overwrites 20 fraud rows
     with deterministic entropy = 0.50001 (unnatural GenAI smoothness).
  5. Vector G (Agentic Semantic Smuggling): Overwrites 20 high-value fraud
     rows with innocent B2B memo "Q3 Enterprise Software Subscription Invoice - Rack 4B".
  6. Canary Honeypot Probe Injection: 10 automated bot reconnaissance probes
     targeting 5 decoy terminals ('CANARY-NODE-01' through 'CANARY-NODE-05').
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

CANARY_TERMINALS = [
    "CANARY-NODE-01",
    "CANARY-NODE-02",
    "CANARY-NODE-03",
    "CANARY-NODE-04",
    "CANARY-NODE-05",
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
    Applies baseline column standardization, Zero-Trust tokens, and synthetic features.
    """
    print("\n[*] Applying AEGIS Baseline Feature Synthesis & Zero-Trust Primaries...")
    
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

    if "Tokenized_PAN" not in df.columns:
        df["Tokenized_PAN"] = [f"CARD_LEGIT_{i:06d}" for i in range(len(df))]

    if "Fraud_Label" not in df.columns:
        df["Fraud_Label"] = 0

    if "TransactionAmt" not in df.columns:
        if "amount" in df.columns:
            df = df.rename(columns={"amount": "TransactionAmt"})
        else:
            df["TransactionAmt"] = np.random.lognormal(mean=3.8, sigma=0.8, size=len(df)).round(2)

    # 2. Terminal_Node_ID Generation (TERM-1000 through TERM-1499)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    n_terminals = 500
    terminal_pool = [f"TERM-{1000 + i:04d}" for i in range(n_terminals)]
    df["Terminal_Node_ID"] = np.random.choice(terminal_pool, size=len(df))

    # 3. Biometric_Entropy Generation
    human_entropy = np.random.uniform(0.400, 0.900, size=len(df))
    df["Biometric_Entropy"] = np.round(human_entropy, 5)

    # 4. Remittance_Metadata Generation
    if "TextMemo" in df.columns:
        df["Remittance_Metadata"] = df["TextMemo"].fillna("Standard Point of Sale Settlement")
    else:
        df["Remittance_Metadata"] = np.random.choice(DEFAULT_MEMO_TEMPLATES, size=len(df))

    # 5. Zero-Trust Delegated Auth Token Primitive
    token_ids = [f"AUTH-{1000 + (i % 9000):04d}" for i in range(len(df))]
    df["Token_ID"] = token_ids
    df["Token_Status"] = "ACTIVE"

    # 6. Attack_Type tag initialization
    df["Attack_Type"] = "BENIGN"

    print(f"  [+] Standardized Tokenized_PAN & Fraud_Label")
    print(f"  [+] Synthesized Terminal_Node_ID across {n_terminals} merchant terminals")
    print(f"  [+] Injected Zero-Trust delegated Web Bot tokens (Token_ID: AUTH-XXXX, Token_Status: ACTIVE)")
    print(f"  [+] Injected Human Biometric_Entropy & Remittance_Metadata")
    return df


def inject_vector_e_graph_poisoning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector E: Generative Graph Poisoning (Sleeper Mule)
      - Farming: 50 micro-txs ($1.50 - $4.00) into 'TERM-9999-EVIL'
      - Bust-Out: 1 high-value fraud row ($10,000.00) on 'TERM-9999-EVIL'
    """
    print("\n[*] Injecting Vector E: Generative Graph Poisoning (Sleeper Mule)...")
    evil_terminal = "TERM-9999-EVIL"

    legit_indices = df[df["Fraud_Label"] == 0].index
    farming_indices = np.random.choice(legit_indices, size=50, replace=False)
    
    micro_amounts = np.round(np.random.uniform(1.50, 4.00, size=50), 2)
    df.loc[farming_indices, "Terminal_Node_ID"] = evil_terminal
    df.loc[farming_indices, "TransactionAmt"] = micro_amounts
    df.loc[farming_indices, "Attack_Type"] = "GRAPH_POISONING_FARMING"

    # Bust-Out Phase
    max_dt = df["TransactionDT"].max() if "TransactionDT" in df.columns else 100000
    last_tx_id = df["TransactionID"].iloc[-1] if "TransactionID" in df.columns else 2999999
    try:
        new_tx_id = int(last_tx_id) + 1
    except Exception:
        new_tx_id = "TX_BUSTOUT_0001"

    bust_out_row = {
        "TransactionID": new_tx_id,
        "TransactionDT": max_dt + 15 if "TransactionDT" in df.columns else None,
        "TransactionAmt": 10000.00,
        "Tokenized_PAN": "CARD_MULE_BUSTOUT_001",
        "Terminal_Node_ID": evil_terminal,
        "Biometric_Entropy": round(float(np.random.uniform(0.400, 0.900)), 5),
        "Remittance_Metadata": "Bulk Terminal High-Value Settlement",
        "Token_ID": "AUTH-9999",
        "Token_Status": "ACTIVE",
        "Fraud_Label": 1,
        "Attack_Type": "GRAPH_POISONING",
    }

    for col in df.columns:
        if col not in bust_out_row:
            bust_out_row[col] = df[col].iloc[0] if not df[col].empty else None

    df = pd.concat([df, pd.DataFrame([bust_out_row])], ignore_index=True)
    print(f"  [+] Injected 50 farming micro-transactions ($1.50-$4.00) & 1 $10,000 bust-out on '{evil_terminal}'")
    return df


def inject_vector_f_biometric_mimicry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector F: Biometric Latent Diffusion Mimicry (Entropy = 0.50001)
    """
    print("\n[*] Injecting Vector F: Biometric Latent Diffusion Mimicry...")
    candidate_indices = df[(df["Fraud_Label"] == 1) & (df["Attack_Type"] == "BENIGN")].index
    sample_size = min(20, len(candidate_indices))
    mimicry_indices = np.random.choice(candidate_indices, size=sample_size, replace=False)

    df.loc[mimicry_indices, "Biometric_Entropy"] = 0.50001
    df.loc[mimicry_indices, "Attack_Type"] = "BIOMETRIC_MIMICRY"
    print(f"  [+] Overwrote {sample_size} fraud rows with exact Biometric_Entropy = 0.50001")
    return df


def inject_vector_g_semantic_smuggling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vector G: Agentic Semantic Smuggling (B2B Invoice Disguise)
    """
    print("\n[*] Injecting Vector G: Agentic Semantic Smuggling...")
    smuggle_memo = "Q3 Enterprise Software Subscription Invoice - Rack 4B"
    high_val_candidates = df[(df["Fraud_Label"] == 1) & (df["TransactionAmt"] > 1000.0) & (df["Attack_Type"] == "BENIGN")].index
    
    if len(high_val_candidates) < 20:
        high_val_candidates = df[(df["Fraud_Label"] == 1) & (df["Attack_Type"] == "BENIGN")].index

    sample_size = min(20, len(high_val_candidates))
    smuggle_indices = np.random.choice(high_val_candidates, size=sample_size, replace=False)

    df.loc[smuggle_indices, "Remittance_Metadata"] = smuggle_memo
    df.loc[smuggle_indices, "Attack_Type"] = "SEMANTIC_SMUGGLING"
    print(f"  [+] Injected {sample_size} high-value smuggled B2B remittance descriptions")
    return df


def inject_canary_honeypot_probes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Honeypot Decoy Nodes:
      - Injects 10 reconnaissance probe transactions targeting 5 Canary Terminals.
      - Simulates automated botnet port scanning and endpoint enumeration.
    """
    print("\n[*] Injecting Canary Honeypot Decoy Probes (Zero-Trust Deception)...")
    
    max_dt = df["TransactionDT"].max() if "TransactionDT" in df.columns else 100000
    last_tx_id = df["TransactionID"].iloc[-1] if "TransactionID" in df.columns else 3000000

    canary_rows = []
    for i in range(10):
        try:
            curr_tx_id = int(last_tx_id) + 1 + i
        except Exception:
            curr_tx_id = f"TX_CANARY_{i+1:04d}"

        canary_target = CANARY_TERMINALS[i % len(CANARY_TERMINALS)]
        bot_pan = f"CARD_BOTNET_PROBE_{100 + i}"
        bot_token = f"AUTH-BOT-{800 + i}"

        row = {
            "TransactionID": curr_tx_id,
            "TransactionDT": max_dt + 30 + (i * 5) if "TransactionDT" in df.columns else None,
            "TransactionAmt": round(float(np.random.uniform(0.50, 12.00)), 2),
            "Tokenized_PAN": bot_pan,
            "Terminal_Node_ID": canary_target,
            "Biometric_Entropy": 0.50001,  # Bot signature
            "Remittance_Metadata": "Automated Endpoint Health Check & Port Probe",
            "Token_ID": bot_token,
            "Token_Status": "ACTIVE",
            "MerchantCategory": "Decoy Honeypot Terminal",
            "MCC": 9999,
            "card5": 999,
            "Fraud_Label": 1,
            "Attack_Type": "RECON_PROBE",
        }

        for col in df.columns:
            if col not in row:
                row[col] = df[col].iloc[0] if not df[col].empty else None

        canary_rows.append(row)

    df_canary = pd.DataFrame(canary_rows)
    df = pd.concat([df, df_canary], ignore_index=True)

    print(f"  [+] Injected 10 Recon Probe transactions across 5 Canary Decoy Nodes ({CANARY_TERMINALS})")
    print(f"  [+] Tagged Attack_Type = 'RECON_PROBE'")
    return df


def validate_and_export_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Validates the synthetic master dataset and saves to processed CSV."""
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("  PROJECT AEGIS : SYNTHETIC DATASET GENERATION SUMMARY")
    print("=" * 80)
    print(f"  • Total Transactions:         {len(df):,}")
    print(f"  • Legitimate Transactions:    {(df['Fraud_Label'] == 0).sum():,} ({(df['Fraud_Label'] == 0).mean()*100:.2f}%)")
    print(f"  • Fraud Transactions:         {(df['Fraud_Label'] == 1).sum():,} ({(df['Fraud_Label'] == 1).mean()*100:.2f}%)")
    print(f"  • Zero-Trust Bot Tokens:      {df['Token_ID'].nunique():,} unique tokens")
    print(f"  • Canary Decoy Terminals:     {len(CANARY_TERMINALS)} active honeypots")

    print("\n[*] Attack Type Distribution Breakdown:")
    for attack, count in df["Attack_Type"].value_counts().items():
        pct = (count / len(df)) * 100.0
        print(f"    - {attack:<26}: {count:>6,} ({pct:.2f}%)")

    print(f"\n[*] Exporting cyber-augmented dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"[+] Dataset successfully exported ({len(df):,} rows x {len(df.columns)} columns)")
    print("=" * 80 + "\n")


def build_pipeline(raw_path: str = "data/raw/train_transaction.csv", output_path: str = "data/processed/master_aegis_dataset.csv"):
    """Orchestrates end-to-end dataset creation pipeline."""
    df_raw = load_raw_dataset(raw_path, sample_size=DEFAULT_SAMPLE_SIZE)
    df = apply_baseline_injection(df_raw)
    df = inject_vector_e_graph_poisoning(df)
    df = inject_vector_f_biometric_mimicry(df)
    df = inject_vector_g_semantic_smuggling(df)
    df = inject_canary_honeypot_probes(df)
    validate_and_export_dataset(df, output_path)


def main():
    parser = argparse.ArgumentParser(description="Project AEGIS: Red-Team Cyber-Augmented Synthetic Data Generator")
    parser.add_argument("--raw", type=str, default="data/raw/train_transaction.csv", help="Path to raw dataset")
    parser.add_argument("--output", type=str, default="data/processed/master_aegis_dataset.csv", help="Output path")
    args = parser.parse_args()

    build_pipeline(args.raw, args.output)


if __name__ == "__main__":
    main()
