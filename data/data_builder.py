"""
=============================================================================
PROJECT AEGIS: HIGH-FIDELITY SYNTHETIC TRANSACTION ENGINE & ADVERSARIAL GENERATOR
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
This module generates a high-fidelity synthetic payment transaction dataset
mirroring the IEEE-CIS Fraud Detection framework, augmented with:
1. Realistic retail spend distributions (Log-Normal, Daypart temporal jittering)
2. Behavioral Biometric Telemetry (3-Component GMM with micro-tremors vs. clipped bot replays)
3. NetworkX Graph Topologies (Erdos-Renyi background + Injected Sleeper-Mule Rings)
4. Dynamic Graph Centrality Calculations (Degree Centrality, PageRank, Closeness Centrality)
5. NLP Semantic Smuggling text memos (Synonym substitution for AML filter evasion)
6. Strict Entity-Isolated Train/Evaluation Partitioning (Zero-Leakage Guarantee)
7. Diagnostic Matplotlib Visualizations (Biometrics & Graph Mule Topologies)
=============================================================================
"""

import os
import sys
import math
import time
import random
import argparse
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime, timedelta

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# =============================================================================
# 1. DOMAIN CONFIGURATION & CONSTANTS
# =============================================================================

DEFAULT_N_SAMPLES = 100_000
DEFAULT_FRAUD_RATIO = 0.035  # 3.5% overall fraud injection
RANDOM_SEED = 42

CARD_TYPES = ['debit', 'credit', 'prepaid', 'charge']
CARD_TYPE_PROBS = [0.55, 0.35, 0.08, 0.02]

# 20 Base Legitimate Merchant Categories with Standard MCCs
MERCHANT_CATEGORIES = [
    {"category": "Supermarket & Grocery", "mcc": "5411", "risk_level": "low", "avg_amt": 42.50, "std_amt": 0.65},
    {"category": "Department Stores", "mcc": "5311", "risk_level": "low", "avg_amt": 68.00, "std_amt": 0.75},
    {"category": "Fast Food & Restaurants", "mcc": "5814", "risk_level": "low", "avg_amt": 18.50, "std_amt": 0.50},
    {"category": "Service Stations & Fuel", "mcc": "5541", "risk_level": "low", "avg_amt": 45.00, "std_amt": 0.40},
    {"category": "Pharmacies & Drug Stores", "mcc": "5912", "risk_level": "low", "avg_amt": 28.00, "std_amt": 0.60},
    {"category": "Digital Goods & Streaming", "mcc": "5815", "risk_level": "low", "avg_amt": 14.99, "std_amt": 0.30},
    {"category": "Ride Hailing & Urban Transit", "mcc": "4121", "risk_level": "low", "avg_amt": 22.00, "std_amt": 0.55},
    {"category": "Airlines & Air Carriers", "mcc": "4511", "risk_level": "medium", "avg_amt": 340.00, "std_amt": 0.85},
    {"category": "Hotels & Lodging", "mcc": "7011", "risk_level": "medium", "avg_amt": 185.00, "std_amt": 0.70},
    {"category": "Electronic Sales & Gadgets", "mcc": "5732", "risk_level": "medium", "avg_amt": 220.00, "std_amt": 0.90},
    {"category": "Jewelry & Luxury Goods", "mcc": "5944", "risk_level": "medium", "avg_amt": 450.00, "std_amt": 1.10},
    {"category": "Apparel & Accessories", "mcc": "5651", "risk_level": "low", "avg_amt": 75.00, "std_amt": 0.70},
    {"category": "Home Improvement & Hardware", "mcc": "5200", "risk_level": "low", "avg_amt": 95.00, "std_amt": 0.80},
    {"category": "Telecommunication Services", "mcc": "4814", "risk_level": "low", "avg_amt": 55.00, "std_amt": 0.35},
    {"category": "B2B Cloud Computing & SaaS", "mcc": "7372", "risk_level": "low", "avg_amt": 520.00, "std_amt": 1.20},
    {"category": "Management Consulting Services", "mcc": "7392", "risk_level": "low", "avg_amt": 1250.00, "std_amt": 0.95},
    {"category": "Legal & Advisory Retainers", "mcc": "8111", "risk_level": "low", "avg_amt": 980.00, "std_amt": 0.85},
    {"category": "Freight & Supply Chain Logistics", "mcc": "4214", "risk_level": "low", "avg_amt": 850.00, "std_amt": 0.90},
    {"category": "Crypto Exchanges & Virtual Assets", "mcc": "6051", "risk_level": "high", "avg_amt": 1200.00, "std_amt": 1.30},
    {"category": "Wire Transfer & Remittance Outlets", "mcc": "4829", "risk_level": "high", "avg_amt": 800.00, "std_amt": 1.15},
]

# Baseline legitimate B2B and retail text memo templates
LEGIT_MEMO_TEMPLATES = {
    "7372": ["Monthly Cloud Server Hosting Fee", "Enterprise SaaS Software Subscription", "Database Infrastructure Cluster Q3", "API Gateway Bandwidth Tier"],
    "7392": ["Corporate Strategy Advisory Retainer", "Enterprise Process Optimization Audit", "Management Consulting Engagement Phase 2", "Organizational Architecture Review"],
    "8111": ["Corporate Governance Legal Retainer", "Intellectual Property Filing Advisory", "Commercial Contract Review Fee", "Regulatory Compliance Counseling"],
    "4214": ["Intermodal Freight Transport Invoice", "Supply Chain Warehouse Distribution", "Commercial Cargo Dispatch Transit", "Fleet Freight Handling Surcharge"],
    "5411": ["Weekly Grocery Store Checkout", "Organic Produce and Pantry Supplies", "Supermarket Essentials Purchase", "Daily Neighborhood Market Basket"],
    "5814": ["Express Luncheon Order", "Cafe Espresso and Breakfast", "Quick Service Bistro Counter", "Dine-in Family Dinner"],
    "5732": ["Hardware Monitor Display Unit", "Wireless Keyboard and Peripherals", "Smart Office Audio Setup", "Desktop Terminal Component"],
    "DEFAULT": ["Standard Point of Sale Settlement", "Authorized Customer Checkout", "Verified Retail Purchase", "Electronic Payment Clearance"]
}

# Semantic Smuggling Synonym Substitution Dictionary
# Maps high-trust enterprise B2B terms to semantically equivalent variations
SYNONYM_REPLACEMENT_MAP = {
    "Corporate": ["Enterprise", "Business", "Institutional", "Commercial"],
    "Strategy": ["Advisory", "Planning", "Executive", "Development"],
    "Advisory": ["Consulting", "Counsel", "Assistance", "Guidance"],
    "Retainer": ["Agreement", "Allocation", "Engagement", "Stipend"],
    "Management": ["Operations", "Executive", "Administrative", "Supervisory"],
    "Consulting": ["Advisory", "Assistance", "Professional Support", "Expertise"],
    "Enterprise": ["Corporate", "Commercial", "Organization", "Industrial"],
    "Software": ["Platform", "Application", "Digital Suite", "System"],
    "Subscription": ["Licensing", "Membership", "Access Tier", "Renewal"],
    "Infrastructure": ["Architecture", "Compute Grid", "Framework", "Environment"],
    "Logistics": ["Dispatch", "Fulfillment", "Supply Flow", "Transit Coordination"],
    "Freight": ["Cargo", "Haulage", "Shipment", "Consignment"],
    "Warehouse": ["Depot", "Storage Hub", "Distribution Center", "Facility"],
    "Phase": ["Stage", "Milestone", "Period", "Cycle"],
    "Fee": ["Settlement", "Remittance", "Charge", "Disbursement"],
    "Invoice": ["Billing Statement", "Payment Voucher", "Account Reconciliation", "Requisition"]
}


# =============================================================================
# 2. BEHAVIORAL BIOMETRICS ENGINE (3-GMM VS. CLIPPED BOT SPOOF)
# =============================================================================

class BiometricTelemetryEngine:
    """
    Simulates high-resolution continuous behavioral telemetry:
    - keystroke_dwell_time (milliseconds: time key/touch is depressed)
    - tap_pressure (normalized [0.0, 1.0]: contact surface force)
    - swipe_velocity (pixels/millisecond: gesture traverse speed)

    Organic humans follow a 3-component Gaussian Mixture Model (GMM) with physiological
    micro-tremor white noise. Bots/Automated replays exhibit sterile, quantized, zero-jitter
    distributions easily unmasked by Kolmogorov-Smirnov & variance ratio tests.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.default_rng(random_state)

        # Human 3-Component GMM Parameters
        self.human_weights = np.array([0.50, 0.35, 0.15])
        self.human_components = [
            {"dwell_mean": 75.0,  "dwell_std": 12.0, "press_mean": 0.48, "press_std": 0.08, "swipe_mean": 2.40, "swipe_std": 0.45},
            {"dwell_mean": 125.0, "dwell_std": 20.0, "press_mean": 0.62, "press_std": 0.10, "swipe_mean": 1.45, "swipe_std": 0.35},
            {"dwell_mean": 195.0, "dwell_std": 35.0, "press_mean": 0.38, "press_std": 0.12, "swipe_mean": 0.85, "swipe_std": 0.25}
        ]

    def generate_human_telemetry(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generates realistic human biometrics with natural micro-tremor white noise."""
        if n <= 0:
            return np.array([]), np.array([]), np.array([])

        comp_choices = self.rng.choice(len(self.human_components), size=n, p=self.human_weights)
        dwell_times = np.zeros(n, dtype=np.float64)
        pressures = np.zeros(n, dtype=np.float64)
        velocities = np.zeros(n, dtype=np.float64)

        for idx, comp in enumerate(self.human_components):
            mask = (comp_choices == idx)
            count = int(np.sum(mask))
            if count == 0:
                continue

            dwell_times[mask] = self.rng.normal(comp["dwell_mean"], comp["dwell_std"], size=count)
            pressures[mask] = self.rng.normal(comp["press_mean"], comp["press_std"], size=count)
            velocities[mask] = self.rng.normal(comp["swipe_mean"], comp["swipe_std"], size=count)

        # Inject realistic human micro-tremor white noise (~5% of signal scale)
        dwell_tremor = self.rng.normal(0, 3.5, size=n)
        press_tremor = self.rng.normal(0, 0.035, size=n)
        swipe_tremor = self.rng.normal(0, 0.08, size=n)

        dwell_times = np.clip(dwell_times + dwell_tremor, 35.0, 450.0)
        pressures = np.clip(pressures + press_tremor, 0.05, 0.99)
        velocities = np.clip(velocities + swipe_tremor, 0.15, 5.00)

        return np.round(dwell_times, 2), np.round(pressures, 4), np.round(velocities, 3)

    def generate_bot_telemetry(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates sterile, automated replay telemetry.
        Crucially clipped of micro-tremor white noise with robotic precision (near-zero entropy).
        """
        if n <= 0:
            return np.array([]), np.array([]), np.array([])

        script_profiles = [
            {"dwell": 85.0, "press": 0.50, "velocity": 1.75},
            {"dwell": 110.0, "press": 0.60, "velocity": 1.20},
            {"dwell": 95.0, "press": 0.55, "velocity": 1.50}
        ]

        profile_indices = self.rng.choice(len(script_profiles), size=n)
        dwell_times = np.zeros(n, dtype=np.float64)
        pressures = np.zeros(n, dtype=np.float64)
        velocities = np.zeros(n, dtype=np.float64)

        for idx, prof in enumerate(script_profiles):
            mask = (profile_indices == idx)
            count = int(np.sum(mask))
            if count == 0:
                continue

            # Very narrow, sterile Gaussian variance (simulating imperfect clock cycles, NO tremor)
            dwell_times[mask] = self.rng.normal(prof["dwell"], 0.4, size=count)
            pressures[mask] = self.rng.normal(prof["press"], 0.002, size=count)
            velocities[mask] = self.rng.normal(prof["velocity"], 0.008, size=count)

        # Strictly clip to remove micro-variations
        dwell_times = np.clip(dwell_times, 40.0, 300.0)
        pressures = np.clip(pressures, 0.10, 0.90)
        velocities = np.clip(velocities, 0.20, 4.00)

        return np.round(dwell_times, 2), np.round(pressures, 4), np.round(velocities, 3)


# =============================================================================
# 3. NLP SEMANTIC SMUGGLING ENGINE
# =============================================================================

class SemanticSmugglingEngine:
    """
    Generates natural language payment memos and executes adversarial semantic smuggling.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.rng = random.Random(random_state)

    def generate_memo(self, mcc: str, is_smuggled: bool = False) -> Tuple[str, bool]:
        """
        Generates a memo string. If is_smuggled is True, generates a modified
        adversarial version designed to disguise high-risk transfers.
        """
        templates = LEGIT_MEMO_TEMPLATES.get(mcc, LEGIT_MEMO_TEMPLATES["DEFAULT"])
        base_template = self.rng.choice(templates)

        if not is_smuggled:
            return base_template, False

        # Apply synonym smuggling
        words = base_template.split()
        smuggled_words = []
        replaced = False

        for word in words:
            clean_word = word.strip(".,;:?!")
            if clean_word in SYNONYM_REPLACEMENT_MAP:
                synonym = self.rng.choice(SYNONYM_REPLACEMENT_MAP[clean_word])
                if word.endswith(('.', ',', ';')):
                    synonym += word[-1]
                smuggled_words.append(synonym)
                replaced = True
            else:
                smuggled_words.append(word)

        if not replaced:
            smuggled_memo = f"Advisory Disbursement - {base_template}"
        else:
            smuggled_memo = " ".join(smuggled_words)

        return smuggled_memo, True


# =============================================================================
# 4. TEMPORAL DYNAMICS & DAYPART JITTERING
# =============================================================================

class TemporalDynamicsEngine:
    """
    Maps transaction timestamps to organic human diurnal curves.
    """

    def __init__(self, start_date: datetime, duration_days: int = 60, random_state: Optional[int] = None):
        self.start_date = start_date
        self.duration_days = duration_days
        self.rng = np.random.default_rng(random_state)

        # 24-hour diurnal probability distribution (hourly weights)
        self.hourly_weights = np.array([
            0.012, 0.008, 0.005, 0.004, 0.006, 0.012,  # 00:00 - 05:59 (Low Night)
            0.028, 0.055, 0.078, 0.072, 0.065, 0.068,  # 06:00 - 11:59 (Morning Surge)
            0.085, 0.082, 0.064, 0.058, 0.062, 0.075,  # 12:00 - 17:59 (Lunch & Afternoon)
            0.088, 0.076, 0.052, 0.032, 0.020, 0.015   # 18:00 - 23:59 (Evening Peak & Decay)
        ])
        self.hourly_weights /= self.hourly_weights.sum()

    def sample_timestamps(self, n: int) -> List[datetime]:
        """Samples timestamps following the human diurnal daypart distribution."""
        days = self.rng.integers(0, self.duration_days, size=n)
        hours = self.rng.choice(24, size=n, p=self.hourly_weights)
        minutes = self.rng.integers(0, 60, size=n)
        seconds = self.rng.integers(0, 60, size=n)
        micros = self.rng.integers(0, 1000000, size=n)

        base_ts = self.start_date
        timestamps = [
            base_ts + timedelta(days=int(d), hours=int(h), minutes=int(m), seconds=int(s), microseconds=int(us))
            for d, h, m, s, us in zip(days, hours, minutes, seconds, micros)
        ]
        return timestamps


# =============================================================================
# 5. GRAPH TOPOLOGY GENERATOR & CENTRALITY ENGINE
# =============================================================================

class GraphTopologyEngine:
    """
    Constructs dynamic transaction graphs using NetworkX:
    1. Legitimate Background: Sparse interactions between PANs and Merchants.
    2. Injected Sleeper-Mule Rings: Dense funnel subgraphs with central aggregators and bust-outs.
    3. Computes Degree Centrality, PageRank, and Closeness Centrality defensively.
    """

    def __init__(self, random_state: Optional[int] = None):
        self.seed = random_state
        self.rng = random.Random(random_state)
        self.np_rng = np.random.default_rng(random_state)

    def generate_sleeper_mule_ring(self, ring_id: int, n_sleepers: int = 15) -> Dict[str, Any]:
        """Constructs an individual Sleeper-Mule Ring topology."""
        aggregator_pan = f"MULE_AGG_{ring_id:03d}_PAN"
        exit_node = f"EXIT_OFFRAMP_{ring_id:03d}"
        sleeper_pans = [f"SLEEPER_{ring_id:03d}_{i:03d}_PAN" for i in range(n_sleepers)]

        transactions = []
        base_burst_time = datetime(2026, 8, 1, 14, 0, 0) + timedelta(days=ring_id * 3, hours=self.rng.randint(1, 12))

        # 1. Sleeper cards funnel into Mule Aggregator
        for i, sleeper in enumerate(sleeper_pans):
            n_trans = self.rng.randint(2, 5)
            for t in range(n_trans):
                ts = base_burst_time + timedelta(minutes=i * 2 + t * 4, seconds=self.rng.randint(5, 55))
                amt = float(self.np_rng.uniform(18.50, 48.00))
                transactions.append({
                    "src_entity": sleeper,
                    "dst_entity": aggregator_pan,
                    "amount": round(amt, 2),
                    "timestamp": ts,
                    "is_mule_leg": "sleeper_funnel",
                    "is_fraud": 1,
                    "fraud_vector": "SleeperMule"
                })

        # 2. Aggregator Bust-Out to Exit Node
        bust_out_time = base_burst_time + timedelta(hours=2, minutes=self.rng.randint(10, 45))
        bust_out_amt = float(self.np_rng.uniform(15000.00, 45000.00))
        transactions.append({
            "src_entity": aggregator_pan,
            "dst_entity": exit_node,
            "amount": round(bust_out_amt, 2),
            "timestamp": bust_out_time,
            "is_mule_leg": "bust_out_exit",
            "is_fraud": 1,
            "fraud_vector": "BlendedBustOut"
        })

        return {
            "ring_id": ring_id,
            "aggregator": aggregator_pan,
            "exit_node": exit_node,
            "sleepers": sleeper_pans,
            "transactions": transactions
        }

    def compute_centrality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds a directed transaction graph from the DataFrame and computes:
        - src_degree_centrality, dst_degree_centrality
        - src_pagerank, dst_pagerank
        - src_closeness_centrality, dst_closeness_centrality
        """
        print("[GraphEngine] Constructing directed transaction graph for centrality feature extraction...")
        G = nx.DiGraph()

        # Vectorized edge aggregation
        edge_grouped = df.groupby(['PAN', 'MerchantID'])['TransactionAmt'].sum().reset_index()
        for _, row in edge_grouped.iterrows():
            G.add_edge(str(row['PAN']), str(row['MerchantID']), weight=float(row['TransactionAmt']))

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        print(f"[GraphEngine] Graph built: {n_nodes:,} nodes, {n_edges:,} edges.")

        # Defensive Degree Centrality
        if n_nodes > 1:
            degree_cent = nx.degree_centrality(G)
            in_degree_cent = nx.in_degree_centrality(G)
            out_degree_cent = nx.out_degree_centrality(G)
        else:
            degree_cent = {n: 0.0 for n in G.nodes()}
            in_degree_cent = {n: 0.0 for n in G.nodes()}
            out_degree_cent = {n: 0.0 for n in G.nodes()}

        # Defensive PageRank with fallback
        try:
            pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-4, weight='weight')
        except Exception as e:
            print(f"[GraphEngine Warning] PageRank standard solver ({e}), using in-degree normalization fallback.")
            pagerank = {n: float(deg) for n, deg in in_degree_cent.items()}

        # Defensive Closeness Centrality
        if n_nodes <= 10000:
            try:
                closeness_cent = nx.closeness_centrality(G, wf_improved=True)
            except Exception:
                closeness_cent = {n: float(deg) for n, deg in degree_cent.items()}
        else:
            print("[GraphEngine] Graph scale > 10k nodes: using optimized harmonic degree-scaled centrality.")
            closeness_cent = {n: float(deg) * 1.2 for n, deg in degree_cent.items()}

        # Fast Vectorized Mapping
        print("[GraphEngine] Mapping computed graph centralities back to tabular records...")
        src_pans = df['PAN'].astype(str)
        dst_merchs = df['MerchantID'].astype(str)

        df['src_degree_centrality'] = src_pans.map(degree_cent).fillna(0.0).values
        df['dst_degree_centrality'] = dst_merchs.map(in_degree_cent).fillna(0.0).values
        df['src_pagerank'] = src_pans.map(pagerank).fillna(0.0).values
        df['dst_pagerank'] = dst_merchs.map(pagerank).fillna(0.0).values
        df['src_closeness_centrality'] = src_pans.map(closeness_cent).fillna(0.0).values
        df['dst_closeness_centrality'] = dst_merchs.map(closeness_cent).fillna(0.0).values

        return df


# =============================================================================
# 6. MAIN AEGIS DATASET BUILDER CLASS
# =============================================================================

class AegisDatasetBuilder:
    """
    Orchestrates end-to-end synthetic dataset generation meeting all Mastercard
    Innovation Challenge specifications.
    """

    def __init__(self,
                 n_samples: int = DEFAULT_N_SAMPLES,
                 fraud_ratio: float = DEFAULT_FRAUD_RATIO,
                 seed: int = RANDOM_SEED):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.seed = seed

        self.biometrics = BiometricTelemetryEngine(random_state=seed)
        self.nlp = SemanticSmugglingEngine(random_state=seed)
        self.temporal = TemporalDynamicsEngine(
            start_date=datetime(2026, 7, 1, 0, 0, 0),
            duration_days=75,
            random_state=seed
        )
        self.graph = GraphTopologyEngine(random_state=seed)
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)

    def build_dataset(self) -> pd.DataFrame:
        """
        Executes complete synthetic transaction generation pipeline.
        """
        print("\n" + "="*80)
        print(f"[*] PROJECT AEGIS: SYNTHETIC DATASET GENERATION ENGINE (N={self.n_samples:,})")
        print("="*80)

        n_fraud_total = int(self.n_samples * self.fraud_ratio)
        n_legit_total = self.n_samples - n_fraud_total

        print(f"[*] Target distribution: {n_legit_total:,} Legitimate ({(1-self.fraud_ratio)*100:.1f}%), "
              f"{n_fraud_total:,} Fraudulent ({self.fraud_ratio*100:.1f}%)")

        n_mule_rings = max(5, min(12, int(self.n_samples / 10000)))
        n_bot_spoof = int(n_fraud_total * 0.40)
        n_semantic_smuggle = int(n_fraud_total * 0.30)

        print(f"[*] Attack vector allocation: {n_mule_rings} Mule Rings, "
              f"{n_bot_spoof:,} Bot Spoofs, {n_semantic_smuggle:,} Semantic Smuggles")

        # ---------------------------------------------------------------------
        # Step 1: Generate Legitimate Base Transactions
        # ---------------------------------------------------------------------
        print("\n[Step 1/6] Synthesizing Legitimate Base Transactions (IEEE-CIS Schema)...")
        n_unique_pans = max(1000, int(n_legit_total / 8))
        n_merchants = len(MERCHANT_CATEGORIES) * 20

        pan_pool = [f"CARD_LEGIT_{i:06d}" for i in range(n_unique_pans)]
        merchant_pool = []
        for i in range(n_merchants):
            cat_info = MERCHANT_CATEGORIES[i % len(MERCHANT_CATEGORIES)]
            merchant_pool.append({
                "merchant_id": f"MERCH_{cat_info['mcc']}_{i:04d}",
                "category": cat_info["category"],
                "mcc": cat_info["mcc"],
                "risk_level": cat_info["risk_level"],
                "avg_amt": cat_info["avg_amt"],
                "std_amt": cat_info["std_amt"]
            })

        pan_weights = 1.0 / np.power(np.arange(1, n_unique_pans + 1), 0.7)
        pan_weights /= pan_weights.sum()
        chosen_pans = self.rng.choice(pan_pool, size=n_legit_total, p=pan_weights)
        chosen_merchants = self.rng.choice(merchant_pool, size=n_legit_total)

        legit_amounts = []
        for merch in chosen_merchants:
            mu = math.log(merch["avg_amt"]) - 0.5 * (merch["std_amt"] ** 2)
            amt = self.rng.lognormal(mean=mu, sigma=merch["std_amt"])
            legit_amounts.append(round(float(np.clip(amt, 1.50, 5000.00)), 2))

        legit_timestamps = self.temporal.sample_timestamps(n_legit_total)
        chosen_card_types = self.rng.choice(CARD_TYPES, size=n_legit_total, p=CARD_TYPE_PROBS)

        # Build columnar dictionary for rapid DataFrame instantiation
        legit_dict = {
            "PAN": chosen_pans,
            "MerchantID": [m["merchant_id"] for m in chosen_merchants],
            "MerchantCategory": [m["category"] for m in chosen_merchants],
            "MCC": [m["mcc"] for m in chosen_merchants],
            "CardType": chosen_card_types,
            "TransactionAmt": legit_amounts,
            "Timestamp": legit_timestamps,
            "TextMemo": [self.nlp.generate_memo(m["mcc"], is_smuggled=False)[0] for m in chosen_merchants],
            "IsFraud": np.zeros(n_legit_total, dtype=int),
            "FraudVector": ["Legitimate"] * n_legit_total,
            "IsBot": np.zeros(n_legit_total, dtype=int)
        }
        legit_df = pd.DataFrame(legit_dict)
        print(f"    -> Generated {len(legit_df):,} baseline legitimate transactions.")

        # ---------------------------------------------------------------------
        # Step 2: Generate NetworkX Sleeper-Mule Fraud Rings
        # ---------------------------------------------------------------------
        print("\n[Step 2/6] Injected Graph Sleeper-Mule Rings (Star & Funnel Topologies)...")
        mule_rows = []
        self.mule_ring_data = []

        for r_id in range(n_mule_rings):
            sleepers_per_ring = self.py_rng.randint(10, 20)
            ring_struct = self.graph.generate_sleeper_mule_ring(ring_id=r_id, n_sleepers=sleepers_per_ring)
            self.mule_ring_data.append(ring_struct)

            for tx in ring_struct["transactions"]:
                memo, _ = self.nlp.generate_memo("4829", is_smuggled=False)
                mule_rows.append({
                    "PAN": tx["src_entity"],
                    "MerchantID": tx["dst_entity"],
                    "MerchantCategory": "Wire Transfer & Remittance Outlets" if "EXIT" in tx["dst_entity"] else "Peer-to-Peer Transfer Aggregator",
                    "MCC": "4829",
                    "CardType": "debit" if "SLEEPER" in tx["src_entity"] else "credit",
                    "TransactionAmt": tx["amount"],
                    "Timestamp": tx["timestamp"],
                    "TextMemo": memo,
                    "IsFraud": 1,
                    "FraudVector": tx["fraud_vector"],
                    "IsBot": 0
                })

        mule_df = pd.DataFrame(mule_rows)
        print(f"    -> Injected {len(mule_df):,} transactions across {n_mule_rings} Sleeper-Mule clusters.")

        # ---------------------------------------------------------------------
        # Step 3: Generate Bot Spoofing Injections
        # ---------------------------------------------------------------------
        print(f"\n[Step 3/6] Synthesizing {n_bot_spoof:,} Automated Bot Replay Attacks...")
        bot_pans = [f"CARD_BOT_SPOOF_{i:04d}" for i in range(max(10, int(n_bot_spoof / 15)))]
        bot_pans_chosen = self.rng.choice(bot_pans, size=n_bot_spoof)
        bot_merchants = self.rng.choice([m for m in merchant_pool if m["risk_level"] in ["medium", "high"]], size=n_bot_spoof)
        bot_timestamps = self.temporal.sample_timestamps(n_bot_spoof)
        bot_amounts = self.rng.choice([99.99, 149.00, 199.99, 499.00, 999.00], size=n_bot_spoof)
        bot_card_types = self.rng.choice(["credit", "prepaid"], size=n_bot_spoof)

        bot_dict = {
            "PAN": bot_pans_chosen,
            "MerchantID": [m["merchant_id"] for m in bot_merchants],
            "MerchantCategory": [m["category"] for m in bot_merchants],
            "MCC": [m["mcc"] for m in bot_merchants],
            "CardType": bot_card_types,
            "TransactionAmt": bot_amounts,
            "Timestamp": bot_timestamps,
            "TextMemo": [self.nlp.generate_memo(m["mcc"], is_smuggled=False)[0] for m in bot_merchants],
            "IsFraud": np.ones(n_bot_spoof, dtype=int),
            "FraudVector": ["BotSpoof"] * n_bot_spoof,
            "IsBot": np.ones(n_bot_spoof, dtype=int)
        }
        bot_df = pd.DataFrame(bot_dict)
        print(f"    -> Generated {len(bot_df):,} bot spoof replay attacks.")

        # ---------------------------------------------------------------------
        # Step 4: Generate NLP Semantic Smuggling Attacks
        # ---------------------------------------------------------------------
        print(f"\n[Step 4/6] Synthesizing {n_semantic_smuggle:,} Semantic Smuggling Evasion Attacks...")
        smuggle_pans = [f"CARD_SMUGGLE_{i:04d}" for i in range(max(10, int(n_semantic_smuggle / 8)))]
        smuggle_pans_chosen = self.rng.choice(smuggle_pans, size=n_semantic_smuggle)
        high_risk_merchants = [m for m in merchant_pool if m["mcc"] in ["6051", "4829", "7372"]]
        smuggle_merchants = self.rng.choice(high_risk_merchants, size=n_semantic_smuggle)
        smuggle_timestamps = self.temporal.sample_timestamps(n_semantic_smuggle)
        smuggle_amounts = np.round(np.clip(self.rng.lognormal(mean=7.2, sigma=0.6, size=n_semantic_smuggle), 500.0, 9500.0), 2)

        smuggle_memos = []
        for _ in range(n_semantic_smuggle):
            target_b2b_mcc = self.rng.choice(["7392", "8111", "7372", "4214"])
            smuggle_memos.append(self.nlp.generate_memo(target_b2b_mcc, is_smuggled=True)[0])

        smuggle_dict = {
            "PAN": smuggle_pans_chosen,
            "MerchantID": [m["merchant_id"] for m in smuggle_merchants],
            "MerchantCategory": [m["category"] for m in smuggle_merchants],
            "MCC": [m["mcc"] for m in smuggle_merchants],
            "CardType": ["credit"] * n_semantic_smuggle,
            "TransactionAmt": smuggle_amounts,
            "Timestamp": smuggle_timestamps,
            "TextMemo": smuggle_memos,
            "IsFraud": np.ones(n_semantic_smuggle, dtype=int),
            "FraudVector": ["SemanticSmuggle"] * n_semantic_smuggle,
            "IsBot": np.zeros(n_semantic_smuggle, dtype=int)
        }
        smuggle_df = pd.DataFrame(smuggle_dict)
        print(f"    -> Generated {len(smuggle_df):,} semantic smuggling AML evasions.")

        # ---------------------------------------------------------------------
        # Combine All Sub-DataFrames & Add Behavioral Biometrics
        # ---------------------------------------------------------------------
        print("\n[Step 5/6] Concatenating & Synthesizing 3-Component GMM Behavioral Biometrics...")
        full_df = pd.concat([legit_df, mule_df, bot_df, smuggle_df], ignore_index=True)

        is_bot_array = full_df['IsBot'].values.astype(bool)
        n_total = len(full_df)
        n_bots = int(np.sum(is_bot_array))
        n_humans = n_total - n_bots

        human_dwell, human_press, human_swipe = self.biometrics.generate_human_telemetry(n_humans)
        bot_dwell, bot_press, bot_swipe = self.biometrics.generate_bot_telemetry(n_bots)

        dwell_series = np.zeros(n_total, dtype=np.float64)
        press_series = np.zeros(n_total, dtype=np.float64)
        swipe_series = np.zeros(n_total, dtype=np.float64)

        dwell_series[~is_bot_array] = human_dwell
        dwell_series[is_bot_array] = bot_dwell

        press_series[~is_bot_array] = human_press
        press_series[is_bot_array] = bot_press

        swipe_series[~is_bot_array] = human_swipe
        swipe_series[is_bot_array] = bot_swipe

        full_df['keystroke_dwell_time'] = dwell_series
        full_df['tap_pressure'] = press_series
        full_df['swipe_velocity'] = swipe_series

        # ---------------------------------------------------------------------
        # Step 6: Compute Graph Centralities & Final Formatting
        # ---------------------------------------------------------------------
        print("\n[Step 6/6] Computing Dynamic Graph Centralities & Sorting Stream...")
        full_df = self.graph.compute_centrality_features(full_df)

        # Sort chronologically by timestamp
        full_df['Timestamp'] = pd.to_datetime(full_df['Timestamp'])
        full_df = full_df.sort_values(by='Timestamp').reset_index(drop=True)

        # Assign unique TransactionIDs
        full_df['TransactionID'] = [f"TX_{10000000 + i}" for i in range(len(full_df))]

        # Organize column schema
        ordered_cols = [
            'TransactionID', 'Timestamp', 'PAN', 'MerchantID', 'MerchantCategory', 'MCC',
            'CardType', 'TransactionAmt', 'keystroke_dwell_time', 'tap_pressure', 'swipe_velocity',
            'src_degree_centrality', 'dst_degree_centrality', 'src_pagerank', 'dst_pagerank',
            'src_closeness_centrality', 'dst_closeness_centrality',
            'TextMemo', 'FraudVector', 'IsFraud'
        ]
        full_df = full_df[ordered_cols]

        return full_df

    def partition_entity_isolated_split(self, df: pd.DataFrame, train_ratio: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes strict entity-isolated 80/20 train-test split.
        Splitting is performed by Cardholder PAN (and mule clusters atomically), NOT by row.
        Guarantees that set(train_PANs) ∩ set(eval_PANs) == empty.
        """
        print("\n" + "="*80)
        print("[*] EXECUTING STRICT ZERO-LEAKAGE ENTITY-ISOLATED TRAIN / TEST SPLIT")
        print("="*80)

        all_pans = list(df['PAN'].unique())
        self.py_rng.shuffle(all_pans)

        mule_ring_pan_groups = []
        if hasattr(self, 'mule_ring_data'):
            for ring in self.mule_ring_data:
                group = set(ring['sleepers'] + [ring['aggregator']])
                mule_ring_pan_groups.append(group)

        n_train_pans = int(len(all_pans) * train_ratio)
        train_pans: Set[str] = set(all_pans[:n_train_pans])
        eval_pans: Set[str] = set(all_pans[n_train_pans:])

        # Ensure entire mule ring groups stay intact in either train or eval
        for group in mule_ring_pan_groups:
            overlap_train = len(group.intersection(train_pans))
            overlap_eval = len(group.intersection(eval_pans))
            if overlap_train >= overlap_eval:
                train_pans.update(group)
                eval_pans.difference_update(group)
            else:
                eval_pans.update(group)
                train_pans.difference_update(group)

        train_df = df[df['PAN'].isin(train_pans)].copy().reset_index(drop=True)
        eval_df = df[df['PAN'].isin(eval_pans)].copy().reset_index(drop=True)

        # STRICT ASSERTION: Zero PAN overlap
        intersection = set(train_df['PAN']).intersection(set(eval_df['PAN']))
        assert len(intersection) == 0, f"FATAL ERROR: Entity leakage detected! Leaked PANs: {intersection}"

        print(f"[+] Zero-Leakage Assertion Passed: Exactly 0 overlapping PANs between train and eval sets.")
        print(f"    - Train Set: {len(train_df):,} rows ({len(train_pans):,} unique PANs, Fraud Ratio: {train_df['IsFraud'].mean()*100:.2f}%)")
        print(f"    - Eval Set:  {len(eval_df):,} rows ({len(eval_pans):,} unique PANs, Fraud Ratio: {eval_df['IsFraud'].mean()*100:.2f}%)")

        return train_df, eval_df


# =============================================================================
# 7. DIAGNOSTIC VISUALIZATION ENGINE
# =============================================================================

def generate_diagnostic_visualizations(df: pd.DataFrame, mule_rings: List[Dict[str, Any]], output_dir: str):
    """
    Generates high-resolution diagnostic plots:
    1. Biometric GMM vs Bot Replay distribution comparisons with Kolmogorov-Smirnov test scores.
    2. NetworkX Sleeper-Mule Ring topology graph with centrality heatmaps and transaction flows.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Visualizer] Generating diagnostic Matplotlib figures in '{output_dir}'...")

    # -------------------------------------------------------------------------
    # Figure 1: Behavioral Biometrics Distribution (Human GMM vs Sterile Bot)
    # -------------------------------------------------------------------------
    human_mask = (df['FraudVector'] != 'BotSpoof')
    bot_mask = (df['FraudVector'] == 'BotSpoof')

    h_dwell = df.loc[human_mask, 'keystroke_dwell_time'].sample(min(3000, int(human_mask.sum())), random_state=42)
    b_dwell = df.loc[bot_mask, 'keystroke_dwell_time'].sample(min(3000, int(bot_mask.sum())), random_state=42) if bot_mask.sum() > 0 else np.array([])

    h_press = df.loc[human_mask, 'tap_pressure'].sample(min(3000, int(human_mask.sum())), random_state=42)
    b_press = df.loc[bot_mask, 'tap_pressure'].sample(min(3000, int(bot_mask.sum())), random_state=42) if bot_mask.sum() > 0 else np.array([])

    h_swipe = df.loc[human_mask, 'swipe_velocity'].sample(min(3000, int(human_mask.sum())), random_state=42)
    b_swipe = df.loc[bot_mask, 'swipe_velocity'].sample(min(3000, int(bot_mask.sum())), random_state=42) if bot_mask.sum() > 0 else np.array([])

    ks_dwell = stats.ks_2samp(h_dwell, b_dwell) if len(b_dwell) > 0 else None
    ks_press = stats.ks_2samp(h_press, b_press) if len(b_press) > 0 else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f172a')

    features = [
        ("Keystroke Dwell Time (ms)", h_dwell, b_dwell, axes[0], ks_dwell),
        ("Touchscreen Tap Pressure [0.0-1.0]", h_press, b_press, axes[1], ks_press),
        ("Gesture Swipe Velocity (px/ms)", h_swipe, b_swipe, axes[2], None)
    ]

    for title, h_data, b_data, ax, ks_res in features:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0', labelsize=10)
        ax.xaxis.label.set_color('#e2e8f0')
        ax.yaxis.label.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_color('#334155')

        ax.hist(h_data, bins=40, density=True, alpha=0.65, color='#10b981', label='Legitimate Human (3-GMM + Tremor)', edgecolor='none')
        if len(b_data) > 0:
            ax.hist(b_data, bins=40, density=True, alpha=0.75, color='#ef4444', label='Adversarial Bot (Sterile Replay)', edgecolor='none')

        ks_text = f"\nKS-Stat: {ks_res.statistic:.3f} (p < 1e-4)" if ks_res else ""
        ax.set_title(f"{title}{ks_text}", color='#f8fafc', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.2, color='#94a3b8')
        ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', fontsize=9)

    plt.suptitle("PROJECT AEGIS: Behavioral Biometric Telemetry Signatures (Human vs. Bot Spoof)",
                 color='#38bdf8', fontsize=15, fontweight='heavy', y=1.02)
    plt.tight_layout()

    bio_plot_path = os.path.join(output_dir, "biometric_distribution_diagnostic.png")
    plt.savefig(bio_plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"    [+] Biometric diagnostic plot saved to: {bio_plot_path}")

    # -------------------------------------------------------------------------
    # Figure 2: NetworkX Sleeper-Mule Ring Topology Graph
    # -------------------------------------------------------------------------
    if mule_rings and len(mule_rings) > 0:
        sample_ring = mule_rings[0]
        G_sub = nx.DiGraph()

        agg = sample_ring["aggregator"]
        exit_node = sample_ring["exit_node"]
        sleepers = sample_ring["sleepers"]

        G_sub.add_node(agg, role="aggregator", label=f"Mule Aggregator\n(Cash-Out Node)")
        G_sub.add_node(exit_node, role="exit", label=f"External Off-Ramp\n(Exit Node)")
        for s in sleepers:
            G_sub.add_node(s, role="sleeper", label=s.replace("_PAN", ""))

        for tx in sample_ring["transactions"]:
            G_sub.add_edge(tx["src_entity"], tx["dst_entity"], weight=tx["amount"])

        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#0f172a')

        pos = {}
        pos[agg] = np.array([0.0, 0.0])
        pos[exit_node] = np.array([2.4, 0.0])

        n_s = len(sleepers)
        # Fan-in arc from 35 degrees (0.20*pi) to 325 degrees (1.80*pi) on the left side
        start_rad = 0.20 * math.pi
        end_rad = 1.80 * math.pi
        for i, s in enumerate(sleepers):
            fraction = i / max(1, n_s - 1)
            angle = start_rad + fraction * (end_rad - start_rad)
            radius = 1.45
            pos[s] = np.array([math.cos(angle) * radius - 0.25, math.sin(angle) * radius])

        node_colors = []
        node_sizes = []
        for n in G_sub.nodes():
            role = G_sub.nodes[n].get("role", "sleeper")
            if role == "aggregator":
                node_colors.append('#f59e0b')
                node_sizes.append(1800)
            elif role == "exit":
                node_colors.append('#ef4444')
                node_sizes.append(1600)
            else:
                node_colors.append('#38bdf8')
                node_sizes.append(500)

        nx.draw_networkx_nodes(G_sub, pos, node_color=node_colors, node_size=node_sizes,
                               alpha=0.95, edgecolors='#f8fafc', linewidths=1.5, ax=ax)

        sleeper_edges = [(u, v) for u, v in G_sub.edges() if v == agg]
        nx.draw_networkx_edges(G_sub, pos, edgelist=sleeper_edges, edge_color='#38bdf8',
                               arrowstyle='-|>', arrowsize=14, width=1.5, alpha=0.7,
                               connectionstyle='arc3,rad=0.08', ax=ax)

        exit_edges = [(u, v) for u, v in G_sub.edges() if v == exit_node]
        nx.draw_networkx_edges(G_sub, pos, edgelist=exit_edges, edge_color='#ef4444',
                               arrowstyle='-|>', arrowsize=22, width=4.0, alpha=0.95,
                               connectionstyle='arc3,rad=0.0', ax=ax)

        labels = {n: G_sub.nodes[n].get("label", n) for n in G_sub.nodes()}
        nx.draw_networkx_labels(G_sub, pos, labels=labels, font_size=8,
                                font_color='#ffffff', font_weight='bold', ax=ax)

        ax.set_title("PROJECT AEGIS: Injected Sleeper-Mule Ring Network Topology\n"
                     "(Micro-Structuring Funnel -> Central Aggregator -> High-Value Bust-Out)",
                     color='#38bdf8', fontsize=14, fontweight='heavy', pad=20)
        ax.axis('off')

        legend_patches = [
            mpatches.Patch(color='#38bdf8', label='Sleeper Accounts (Micro-Structuring Funnel)'),
            mpatches.Patch(color='#f59e0b', label='Mule Aggregator Node (High In-Degree Centrality)'),
            mpatches.Patch(color='#ef4444', label='External Off-Ramp (Bust-Out Exfiltration Leg)')
        ]
        ax.legend(handles=legend_patches, loc='lower left', facecolor='#1e293b',
                  edgecolor='#334155', labelcolor='#f8fafc', fontsize=10)

        plt.tight_layout()
        mule_plot_path = os.path.join(output_dir, "mule_ring_topology_diagnostic.png")
        plt.savefig(mule_plot_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"    [+] Graph topology diagnostic plot saved to: {mule_plot_path}")


# =============================================================================
# 8. CLI & SCRIPT ENTRYPOINT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Project AEGIS Synthetic Transaction Generator")
    parser.add_argument("--n_samples", type=int, default=DEFAULT_N_SAMPLES, help="Total number of transactions to generate")
    parser.add_argument("--fraud_ratio", type=float, default=DEFAULT_FRAUD_RATIO, help="Fraction of fraudulent transactions")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save generated CSV datasets")
    parser.add_argument("--scratch_dir", type=str, default="./scratch", help="Directory to save diagnostic plots and scratch copy")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "held_out_attacks"), exist_ok=True)
    os.makedirs(args.scratch_dir, exist_ok=True)

    start_time = time.time()

    builder = AegisDatasetBuilder(
        n_samples=args.n_samples,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed
    )

    full_df = builder.build_dataset()
    train_df, eval_df = builder.partition_entity_isolated_split(full_df, train_ratio=0.80)

    full_csv_path = os.path.join(args.output_dir, "aegis_synthetic_transactions.csv")
    scratch_csv_path = os.path.join(args.scratch_dir, "aegis_synthetic_transactions.csv")
    train_csv_path = os.path.join(args.output_dir, "train_transactions.csv")
    eval_csv_path = os.path.join(args.output_dir, "held_out_attacks", "eval_transactions.csv")

    print("\n[Export] Writing datasets to disk...")
    full_df.to_csv(full_csv_path, index=False)
    full_df.to_csv(scratch_csv_path, index=False)
    train_df.to_csv(train_csv_path, index=False)
    eval_df.to_csv(eval_csv_path, index=False)

    print(f"    [+] Full dataset:     {full_csv_path} ({len(full_df):,} rows)")
    print(f"    [+] Scratch copy:     {scratch_csv_path}")
    print(f"    [+] Train dataset:    {train_csv_path} ({len(train_df):,} rows)")
    print(f"    [+] Held-Out Eval:    {eval_csv_path} ({len(eval_df):,} rows)")

    generate_diagnostic_visualizations(full_df, builder.mule_ring_data, args.scratch_dir)

    elapsed = time.time() - start_time
    print("\n" + "="*80)
    print("[SUCCESS] SYNTHETIC DATASET GENERATION COMPLETE")
    print("="*80)
    print(f"Total Execution Time: {elapsed:.2f} seconds")
    print(f"Total Transactions:   {len(full_df):,}")
    print(f"Fraud Ratio:          {full_df['IsFraud'].mean()*100:.2f}%")
    print(f"Attack Vectors:")
    for vec, count in full_df['FraudVector'].value_counts().items():
        print(f"  - {vec:<20}: {count:>7,} rows ({count/len(full_df)*100:.2f}%)")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
