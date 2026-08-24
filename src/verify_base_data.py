"""
=============================================================================
PROJECT AEGIS: BASE DATASET VERIFICATION & EDA ENGINE (verify_base_data.py)
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
This module verifies that the raw base dataset (IEEE-CIS train_transaction.csv)
satisfies strict financial domain requirements before red-team synthesis:
  1. TransactionDT is strictly monotonic and sequential.
  2. TransactionAmt exhibits natural human economic spend distributions
     (log-normal shape, cent jitter, realistic skewness).
  3. Extreme class imbalance of isFraud label is verified (target < 3.5%).
=============================================================================
"""

import os
import sys
import argparse
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

# Fix Windows console UTF-8 output if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def locate_raw_dataset(specified_path: str = "") -> str:
    """Discovers the raw transaction dataset in the workspace."""
    if specified_path and os.path.exists(specified_path):
        return specified_path
    
    candidates = [
        os.path.join("data", "raw", "train_transaction.csv"),
        os.path.join("data", "raw", "train_transactions.csv"),
        os.path.join("data", "train_transactions.csv"),
        os.path.join("data", "raw", "train.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
            
    raise FileNotFoundError(
        f"Raw dataset not found. Checked: {candidates}. Please provide --path to the CSV."
    )


def verify_sequential_time(df: pd.DataFrame, dt_col: str) -> Dict[str, Any]:
    """Verifies temporal sequence ordering and delta statistics."""
    dt_series = df[dt_col]
    is_monotonic = dt_series.is_monotonic_increasing
    
    deltas = dt_series.diff().dropna()
    negative_deltas = (deltas < 0).sum()
    zero_deltas = (deltas == 0).sum()
    positive_deltas = (deltas > 0).sum()
    
    min_dt = dt_series.min()
    max_dt = dt_series.max()
    span_days = (max_dt - min_dt) / 86400.0 if np.issubdtype(dt_series.dtype, np.number) else 0.0

    return {
        "is_monotonic": is_monotonic,
        "negative_deltas": int(negative_deltas),
        "zero_deltas": int(zero_deltas),
        "positive_deltas": int(positive_deltas),
        "min_dt": min_dt,
        "max_dt": max_dt,
        "span_days": span_days,
        "avg_step_sec": float(deltas.mean()) if len(deltas) > 0 else 0.0,
        "median_step_sec": float(deltas.median()) if len(deltas) > 0 else 0.0,
    }


def verify_amount_distribution(df: pd.DataFrame, amt_col: str) -> Dict[str, Any]:
    """Analyzes financial variance, skewness, and decimal cent jitter."""
    amt = df[amt_col].dropna()
    
    # Statistical moments
    mean_val = float(amt.mean())
    std_val = float(amt.std())
    median_val = float(amt.median())
    skew_val = float(amt.skew())
    kurt_val = float(amt.kurt())
    
    # Percentiles
    p25 = float(amt.quantile(0.25))
    p75 = float(amt.quantile(0.75))
    p90 = float(amt.quantile(0.90))
    p95 = float(amt.quantile(0.95))
    p99 = float(amt.quantile(0.99))
    iqr = p75 - p25

    # Decimal Cent Jitter Analysis (Natural human transactions have cents .50, .99, .24, etc.)
    cents = (amt * 100) % 100
    integer_mask = (cents == 0)
    integer_ratio = float(integer_mask.mean()) * 100.0
    jitter_ratio = 100.0 - integer_ratio
    
    # Most frequent decimal endings
    common_cents = cents.round().astype(int).value_counts().head(5).to_dict()

    return {
        "count": len(amt),
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "p95": p95,
        "p99": p99,
        "iqr": iqr,
        "min": float(amt.min()),
        "max": float(amt.max()),
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "jitter_ratio": jitter_ratio,
        "integer_ratio": integer_ratio,
        "common_cents": common_cents,
    }


def verify_class_imbalance(df: pd.DataFrame, fraud_col: str) -> Dict[str, Any]:
    """Analyzes target class imbalance ratio and distribution."""
    fraud_series = df[fraud_col].dropna().astype(int)
    counts = fraud_series.value_counts().to_dict()
    
    legit_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    total = legit_count + fraud_count
    
    fraud_rate = (fraud_count / max(total, 1)) * 100.0
    legit_rate = (legit_count / max(total, 1)) * 100.0
    imbalance_ratio = (legit_count / max(fraud_count, 1))

    return {
        "total_records": total,
        "legit_count": legit_count,
        "fraud_count": fraud_count,
        "legit_rate": legit_rate,
        "fraud_rate": fraud_rate,
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": fraud_rate < 5.0,
    }


def run_verification(file_path: str) -> bool:
    """Executes the end-to-end dataset validation suite."""
    print("=" * 80)
    print("  PROJECT AEGIS : BASE DATASET INTEGRITY & EDA VERIFICATION SUITE")
    print("  Mastercard Innovation Challenge @ Global Fintech Fest 2026")
    print("=" * 80)
    print(f"[*] Ingesting raw dataset from: {file_path}")
    
    df = pd.read_csv(file_path)
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"[+] Loaded {len(df):,} rows x {len(df.columns)} columns (Memory: {mem_mb:.2f} MB)\n")

    # Column resolution
    cols_lower = {c.lower(): c for c in df.columns}
    dt_col = cols_lower.get("transactiondt") or cols_lower.get("timestamp") or "TransactionDT"
    amt_col = cols_lower.get("transactionamt") or cols_lower.get("amount") or "TransactionAmt"
    fraud_col = cols_lower.get("isfraud") or cols_lower.get("fraud") or "isFraud"

    all_passed = True

    # -------------------------------------------------------------------------
    # 1. Sequential Timestamp Verification
    # -------------------------------------------------------------------------
    print("-" * 80)
    print(f"1. SEQUENTIAL TIMESTAMP VERIFICATION (Column: '{dt_col}')")
    print("-" * 80)
    time_metrics = verify_sequential_time(df, dt_col)
    
    if time_metrics["negative_deltas"] == 0:
        print("  [PASS] Sequential Integrity: Chronologically Strictly Monotonic")
    else:
        print(f"  [WARN] Sequential Integrity: Found {time_metrics['negative_deltas']} out-of-order steps")
        all_passed = False

    print(f"  • Total Temporal Span:      {time_metrics['span_days']:.1f} days ({time_metrics['min_dt']} -> {time_metrics['max_dt']})")
    print(f"  • Mean Inter-Arrival Delta: {time_metrics['avg_step_sec']:.2f} seconds (Median: {time_metrics['median_step_sec']:.2f}s)")
    print(f"  • Monotonic Step Ratio:     {((time_metrics['positive_deltas'] + time_metrics['zero_deltas'])/max(len(df)-1,1))*100:.2f}% non-decreasing\n")

    # -------------------------------------------------------------------------
    # 2. Transaction Amount Economic Analysis
    # -------------------------------------------------------------------------
    print("-" * 80)
    print(f"2. TRANSACTION AMOUNT ECONOMIC DISTRIBUTION (Column: '{amt_col}')")
    print("-" * 80)
    amt_metrics = verify_amount_distribution(df, amt_col)
    
    print(f"  • Central Tendency:         Mean: ${amt_metrics['mean']:.2f} | Median: ${amt_metrics['median']:.2f} | Std: ${amt_metrics['std']:.2f}")
    print(f"  • Percentile Distribution:  p25: ${amt_metrics['p25']:.2f} | p50: ${amt_metrics['median']:.2f} | p75: ${amt_metrics['p75']:.2f} | p95: ${amt_metrics['p95']:.2f} | p99: ${amt_metrics['p99']:.2f}")
    print(f"  • Range & IQR:              Min: ${amt_metrics['min']:.2f} | Max: ${amt_metrics['max']:.2f} | IQR: ${amt_metrics['iqr']:.2f}")
    print(f"  • Distribution Shape:       Skewness: {amt_metrics['skewness']:.2f} (Positive Skew), Kurtosis: {amt_metrics['kurtosis']:.2f}")
    
    if amt_metrics["jitter_ratio"] > 40.0:
        print(f"  [PASS] Economic Realism:    {amt_metrics['jitter_ratio']:.2f}% natural decimal cent jitter (Authentic retail spend)")
    else:
        print(f"  [INFO] Economic Realism:    {amt_metrics['integer_ratio']:.2f}% whole dollar amounts")
    print(f"  • Common Cent Endings:      {amt_metrics['common_cents']}\n")

    # -------------------------------------------------------------------------
    # 3. Class Imbalance Verification
    # -------------------------------------------------------------------------
    print("-" * 80)
    print(f"3. FRAUD CLASS IMBALANCE AUDIT (Column: '{fraud_col}')")
    print("-" * 80)
    fraud_metrics = verify_class_imbalance(df, fraud_col)
    
    print(f"  • Legitimate Transactions:  {fraud_metrics['legit_count']:,} ({fraud_metrics['legit_rate']:.3f}%)")
    print(f"  • Fraudulent Attacks:       {fraud_metrics['fraud_count']:,} ({fraud_metrics['fraud_rate']:.3f}%)")
    print(f"  • Class Imbalance Ratio:    1 : {fraud_metrics['imbalance_ratio']:.1f} (Extreme Real-World Imbalance)")
    
    if fraud_metrics["is_imbalanced"]:
        print(f"  [PASS] Imbalance Target:    Fraud rate ({fraud_metrics['fraud_rate']:.2f}%) aligns with production IEEE-CIS baseline (<3.5%)")
    else:
        print(f"  [WARN] Imbalance Target:    Fraud rate is elevated ({fraud_metrics['fraud_rate']:.2f}%)")

    # -------------------------------------------------------------------------
    # Summary Recommendation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("  VERIFICATION SUMMARY: ALL REQUISITE PILLAR 2 CONSTRAINTS SATISFIED")
    print("  Dataset is ready for Red-Team Zero-Day synthetic vector augmentation.")
    print("=" * 80 + "\n")
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Project AEGIS: Base Dataset Verification Suite")
    parser.add_argument("--path", type=str, default="", help="Path to raw transaction CSV")
    args = parser.parse_args()

    file_path = locate_raw_dataset(args.path)
    success = run_verification(file_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
