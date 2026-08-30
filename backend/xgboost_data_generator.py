"""
=============================================================================
PROJECT AEGIS: XGBoost Data Generator — Realistic Transaction Dataset
=============================================================================
Generates 15,662 realistic Indian digital payment transactions with
probabilistic fraud labels based on multiple risk factors. Not hardcoded.
Ported from Requirements/Fraud_Detection_XGBoost/generate_data.py
=============================================================================
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "xgboost")


def generate_fraud_dataset(n_rows: int = 15662, seed: int = 42) -> pd.DataFrame:
    """Generate realistic transaction data with probabilistic fraud labels."""
    np.random.seed(seed)
    random.seed(seed)

    print(f"Generating {n_rows} rows of realistic transaction data...")

    # 1. TransactionID
    tx_ids = [f"TX{100000 + i}" for i in range(n_rows)]

    # 2. Timestamp
    start_date = datetime(2026, 8, 1)
    timestamps = [start_date + timedelta(
        days=random.randint(0, 28),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    ) for _ in range(n_rows)]
    timestamps.sort()

    # 3. PAN (Card Number) — masked
    unique_pans = [f"4532{random.randint(1000, 9999)}XXXX{random.randint(1000, 9999)}" for _ in range(3000)]
    pans = [random.choice(unique_pans) for _ in range(n_rows)]

    # 4. MerchantID
    unique_merchants = [f"M_{random.randint(10000, 99999)}" for _ in range(500)]
    merchant_ids = [random.choice(unique_merchants) for _ in range(n_rows)]

    # 5. TransactionAmt — log-normal distribution (realistic spend pattern)
    tx_amts = np.random.lognormal(mean=4.2, sigma=1.2, size=n_rows)
    tx_amts = np.clip(tx_amts, 1, 10000).round(2)

    # 6. CardType
    card_types = [random.choice(["Credit", "Debit"]) for _ in range(n_rows)]

    # 7. MerchantCategory — weighted by real-world frequency
    categories = ["Grocery", "Retail", "Electronics", "Travel", "Dining", "Entertainment", "Utility"]
    cat_weights = [0.35, 0.25, 0.10, 0.05, 0.15, 0.05, 0.05]
    merchant_cats = np.random.choice(categories, size=n_rows, p=cat_weights)

    # 8. Location — Indian cities weighted by population density
    cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata", "Ahmedabad"]
    city_weights = [0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.08, 0.04]
    locations = np.random.choice(cities, size=n_rows, p=city_weights)

    # 9. DeviceID
    unique_devices = [f"D_{random.randint(100000, 999999)}" for _ in range(4000)]
    device_ids = [random.choice(unique_devices) for _ in range(n_rows)]

    # 10. IPAddress
    unique_ips = [f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}" for _ in range(4000)]
    ips = [random.choice(unique_ips) for _ in range(n_rows)]

    # 11. TimeOfDay (Hour of the day)
    hours = [ts.hour for ts in timestamps]

    # 12. TransactionSpeed (minutes since last transaction on same card)
    temp_df = pd.DataFrame({"PAN": pans, "Timestamp": timestamps})
    temp_df["Speed"] = 9999.0
    temp_df = temp_df.sort_values(by=["PAN", "Timestamp"])
    temp_df["Speed"] = temp_df.groupby("PAN")["Timestamp"].diff().dt.total_seconds() / 60.0
    temp_df["Speed"] = temp_df["Speed"].fillna(9999.0)
    tx_speed = temp_df.sort_index()["Speed"].values

    # 13. DailyTransactionCount
    temp_df["Date"] = temp_df["Timestamp"].dt.date
    temp_df["DailyCount"] = temp_df.groupby(["PAN", "Date"]).cumcount() + 1
    daily_count = temp_df.sort_index()["DailyCount"].values

    # 14. MerchantFraudRate (simulated historic merchant risk)
    merchant_risk_map = {m: (0.15 if random.random() < 0.03 else random.uniform(0.001, 0.015)) for m in unique_merchants}
    merchant_fraud_rate = [merchant_risk_map[m] for m in merchant_ids]

    # 15. DegreeCentrality
    pan_merchant_counts = temp_df.groupby("PAN")["PAN"].transform("count")
    degree_centrality = (pan_merchant_counts / 100.0).clip(0.01, 1.0).values

    # 16. ClosenessCentrality (simulated graph closeness)
    closeness = np.random.beta(a=3, b=5, size=n_rows)

    # 17. PageRank
    pagerank = np.random.exponential(scale=0.01, size=n_rows)
    pagerank = np.clip(pagerank, 0.0001, 0.1)

    # 18. UserAge
    user_ages = np.random.randint(18, 75, size=n_rows)

    # 19. IsFraud — probabilistic labels based on realistic risk factors
    prob = np.zeros(n_rows)
    for i in range(n_rows):
        p = 0.002  # baseline risk

        # Amount risk
        if tx_amts[i] > 2000:
            p += 0.25
        elif tx_amts[i] > 800:
            p += 0.08

        # Velocity risk
        if tx_speed[i] < 3.0:
            p += 0.35
        elif tx_speed[i] < 15.0:
            p += 0.12

        # Time risk (late night / early morning)
        if hours[i] >= 1 and hours[i] <= 4:
            p += 0.05

        # Merchant risk
        p += merchant_fraud_rate[i] * 5.0

        # Daily transaction count risk
        if daily_count[i] > 6:
            p += 0.20
        elif daily_count[i] > 3:
            p += 0.05

        # Graph features correlation
        if pagerank[i] > 0.03:
            p += 0.08

        prob[i] = p

    prob = np.clip(prob, 0, 0.99)
    is_fraud = np.random.binomial(1, prob)

    n_fraud = np.sum(is_fraud)
    print(f"Total fraud transactions generated: {n_fraud} ({n_fraud/n_rows*100:.2f}%)")

    df = pd.DataFrame({
        "TransactionID": tx_ids,
        "Timestamp": [ts.isoformat() for ts in timestamps],
        "PAN": pans,
        "MerchantID": merchant_ids,
        "TransactionAmt": tx_amts,
        "CardType": card_types,
        "MerchantCategory": merchant_cats,
        "Location": locations,
        "DeviceID": device_ids,
        "IPAddress": ips,
        "TimeOfDay": hours,
        "TransactionSpeed": tx_speed,
        "DailyTransactionCount": daily_count,
        "MerchantFraudRate": merchant_fraud_rate,
        "DegreeCentrality": degree_centrality,
        "ClosenessCentrality": closeness,
        "PageRank": pagerank,
        "UserAge": user_ages,
        "IsFraud": is_fraud
    })

    return df


def generate_and_save() -> str:
    """Generate dataset and save to data/xgboost/transactions.csv."""
    os.makedirs(DATA_DIR, exist_ok=True)
    df = generate_fraud_dataset()
    path = os.path.join(DATA_DIR, "transactions.csv")
    df.to_csv(path, index=False)
    print(f"Dataset saved to: {path}")
    return path


if __name__ == "__main__":
    generate_and_save()
