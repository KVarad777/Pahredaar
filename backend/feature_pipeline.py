"""
=============================================================================
PROJECT AEGIS: FEATURE PIPELINE — Online Feature Store & Vector Assembly
=============================================================================
Real-time feature pipeline (Section 5 of spec):
  1. Velocity counters (txn count/sum in last 1h/24h/7d) — incremental update
  2. Graph state (account-device-IP adjacency, degree/centrality via NetworkX)
  3. Behavioral baseline (per-account rolling stats)
  4. Missingness flags (*_was_null boolean columns)
  5. Flat feature vector assembly (~40-60 features)
=============================================================================
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None
    import warnings
    warnings.warn(
        "[AEGIS] networkx is not installed. Graph features (degree, closeness, shared_device/ip) "
        "will default to zero, degrading the model. Install it: pip install networkx",
        ImportWarning,
        stacklevel=2,
    )

logger = logging.getLogger("AEGIS.Features")


class VelocityStore:
    """
    Incremental velocity counters per account.
    Maintains running count/sum for 1h, 24h, and 7d windows.
    Updated after each transaction — NOT recomputed by scanning history.
    """

    def __init__(self):
        self.account_txns: Dict[str, List[Dict]] = defaultdict(list)

    def update(self, account_id: str, timestamp: str, amount: float) -> None:
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.utcnow()
        self.account_txns[account_id].append({"ts": ts, "amount": amount})
        # Keep only last 7 days of history
        cutoff = ts - timedelta(days=7)
        self.account_txns[account_id] = [
            t for t in self.account_txns[account_id] if t["ts"] >= cutoff
        ]

    def get_velocity(self, account_id: str, current_ts: str) -> Dict:
        try:
            now = datetime.fromisoformat(current_ts.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            now = datetime.utcnow()

        txns = self.account_txns.get(account_id, [])

        h1 = now - timedelta(hours=1)
        h24 = now - timedelta(hours=24)
        d7 = now - timedelta(days=7)

        txns_1h = [t for t in txns if t["ts"] >= h1]
        txns_24h = [t for t in txns if t["ts"] >= h24]
        txns_7d = [t for t in txns if t["ts"] >= d7]

        return {
            "txn_count_1h": len(txns_1h),
            "txn_sum_1h": sum(t["amount"] for t in txns_1h),
            "txn_count_24h": len(txns_24h),
            "txn_sum_24h": sum(t["amount"] for t in txns_24h),
            "txn_count_7d": len(txns_7d),
            "txn_sum_7d": sum(t["amount"] for t in txns_7d),
        }


class GraphStore:
    """
    Incrementally-updated graph structure for account-device-IP relationships.
    Degree/centrality features are read from current state, not recomputed fully.
    """

    def __init__(self):
        if nx:
            self.graph = nx.Graph()
        else:
            self.graph = None
        self._degree_cache: Dict[str, float] = {}
        self._centrality_cache: Dict[str, float] = {}
        self._dirty = True

    def add_transaction(self, account_id: str, device_fp: Optional[str],
                        ip_hash: Optional[str]) -> None:
        if not self.graph:
            return

        self.graph.add_node(account_id, node_type="account")
        if device_fp:
            self.graph.add_node(f"dev_{device_fp}", node_type="device")
            self.graph.add_edge(account_id, f"dev_{device_fp}")
        if ip_hash:
            self.graph.add_node(f"ip_{ip_hash}", node_type="ip")
            self.graph.add_edge(account_id, f"ip_{ip_hash}")
        self._dirty = True

    def _recompute_if_dirty(self) -> None:
        if not self._dirty or not self.graph or len(self.graph) == 0:
            return
        try:
            self._degree_cache = dict(nx.degree_centrality(self.graph))
            if len(self.graph) > 1 and nx.is_connected(self.graph):
                self._centrality_cache = dict(nx.closeness_centrality(self.graph))
            else:
                # For disconnected graphs, compute per component
                self._centrality_cache = {}
                for component in nx.connected_components(self.graph):
                    subg = self.graph.subgraph(component)
                    if len(subg) > 1:
                        cc = nx.closeness_centrality(subg)
                        self._centrality_cache.update(cc)
                    else:
                        for n in subg:
                            self._centrality_cache[n] = 0.0
        except Exception:
            pass
        self._dirty = False

    def get_features(self, account_id: str) -> Dict:
        if not self.graph:
            return {
                "graph_degree": 0.0,
                "graph_closeness": 0.0,
                "graph_neighbors": 0,
                "shared_device_accounts": 0,
                "shared_ip_accounts": 0,
            }

        self._recompute_if_dirty()

        degree = self._degree_cache.get(account_id, 0.0)
        closeness = self._centrality_cache.get(account_id, 0.0)
        neighbors = list(self.graph.neighbors(account_id)) if self.graph.has_node(account_id) else []

        # Count shared devices/IPs (ring detection signal)
        shared_device = 0
        shared_ip = 0
        for nb in neighbors:
            if nb.startswith("dev_"):
                # How many OTHER accounts share this device?
                device_neighbors = [n for n in self.graph.neighbors(nb) if n != account_id]
                shared_device += len(device_neighbors)
            elif nb.startswith("ip_"):
                ip_neighbors = [n for n in self.graph.neighbors(nb) if n != account_id]
                shared_ip += len(ip_neighbors)

        return {
            "graph_degree": round(degree, 6),
            "graph_closeness": round(closeness, 6),
            "graph_neighbors": len(neighbors),
            "shared_device_accounts": shared_device,
            "shared_ip_accounts": shared_ip,
        }


class BehavioralBaseline:
    """
    Per-account rolling statistics for behavioral deviation detection.
    Updated after each transaction.
    """

    def __init__(self):
        self.account_stats: Dict[str, Dict] = defaultdict(lambda: {
            "amounts": [],
            "hours": [],
            "inter_txn_gaps": [],
            "last_ts": None,
        })

    def update(self, account_id: str, amount: float, timestamp: str) -> None:
        stats = self.account_stats[account_id]
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.utcnow()

        stats["amounts"].append(amount)
        stats["hours"].append(ts.hour + ts.minute / 60)

        if stats["last_ts"]:
            gap = (ts - stats["last_ts"]).total_seconds()
            stats["inter_txn_gaps"].append(gap)

        stats["last_ts"] = ts

        # Keep rolling window of last 50
        for key in ["amounts", "hours", "inter_txn_gaps"]:
            if len(stats[key]) > 50:
                stats[key] = stats[key][-50:]

    def get_deviation(self, account_id: str, amount: float, timestamp: str) -> Dict:
        stats = self.account_stats.get(account_id)
        is_new = stats is None or len(stats.get("amounts", [])) < 3

        if is_new:
            return {
                "amount_zscore": 0.0,
                "hour_deviation": 0.0,
                "inter_txn_zscore": 0.0,
                "is_new_account": 1.0,
            }

        amounts = stats["amounts"]
        hours = stats["hours"]

        # Amount z-score
        mean_amt = np.mean(amounts) if amounts else amount
        std_amt = max(np.std(amounts), 1.0) if amounts else 1.0
        amt_z = (amount - mean_amt) / std_amt

        # Hour deviation
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            ts = datetime.utcnow()
        current_hour = ts.hour + ts.minute / 60
        mean_hour = np.mean(hours) if hours else current_hour
        hour_dev = abs(current_hour - mean_hour)

        # Inter-txn gap z-score
        gaps = stats.get("inter_txn_gaps", [])
        if gaps and stats["last_ts"]:
            current_gap = (ts - stats["last_ts"]).total_seconds()
            mean_gap = np.mean(gaps)
            std_gap = max(np.std(gaps), 1.0)
            gap_z = (current_gap - mean_gap) / std_gap
        else:
            gap_z = 0.0

        return {
            "amount_zscore": round(float(amt_z), 4),
            "hour_deviation": round(float(hour_dev), 4),
            "inter_txn_zscore": round(float(gap_z), 4),
            "is_new_account": 0.0,
        }


class FeaturePipeline:
    """
    Master feature pipeline that assembles the full flat feature vector.
    Coordinates velocity store, graph store, and behavioral baseline.
    """

    def __init__(self):
        self.velocity_store = VelocityStore()
        self.graph_store = GraphStore()
        self.behavioral = BehavioralBaseline()

    def process_transaction(self, txn: Dict) -> Dict:
        """
        Process a single transaction and return its flat feature vector.
        Also updates all stateful stores.
        """
        account_id = txn.get("account_id", "")
        timestamp = txn.get("timestamp", "")
        amount = txn.get("amount", 0)
        device = txn.get("device_details", {})
        session = txn.get("session", {})
        identity = txn.get("identity", {})

        # Parse timestamp for hour_of_day
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            hour_of_day = ts.hour
        except (ValueError, AttributeError):
            hour_of_day = 12

        # 1. Get current state BEFORE updating (for this transaction's features)
        velocity = self.velocity_store.get_velocity(account_id, timestamp)
        graph_feats = self.graph_store.get_features(account_id)
        behavioral = self.behavioral.get_deviation(account_id, amount, timestamp)

        # 2. Update stores with this transaction
        self.velocity_store.update(account_id, timestamp, amount)
        self.graph_store.add_transaction(
            account_id,
            device.get("device_fingerprint"),
            device.get("ip_address_hash")
        )
        self.behavioral.update(account_id, amount, timestamp)

        # 3. Derive location from geocode
        geocode = device.get("geocode", "")
        location = self._geocode_to_city(geocode)

        # 4. Derive merchant category from MCC code
        merchant_category = self._mcc_to_category(txn.get("merchant_category_code", ""))

        # 5. Assemble flat feature vector with missingness flags
        features = {
            # Transaction core
            "amount": amount,
            "channel_CNP": 1 if txn.get("channel") == "CNP" else 0,
            "channel_CP": 1 if txn.get("channel") == "CP" else 0,
            "channel_P2P": 1 if txn.get("channel") == "P2P" else 0,
            "channel_ATM": 1 if txn.get("channel") == "ATM" else 0,
            "is_refund": 1 if txn.get("is_refund") else 0,
            "auth_declined": 1 if txn.get("auth_result") == "declined" else 0,
            "auth_retried": 1 if txn.get("auth_result") == "retried" else 0,
            "mcc_high_risk": 1 if txn.get("merchant_category_code") in ["6051", "4829"] else 0,

            # Identity features
            "account_age_days": identity.get("account_age_days", 365),
            "kyc_doc_similarity_score": identity.get("kyc_doc_similarity_score") or 0.0,
            "kyc_doc_similarity_was_null": 1 if identity.get("kyc_doc_similarity_score") is None else 0,
            "kyc_method_automated": 1 if identity.get("kyc_verification_method") == "automated" else 0,
            "kyc_method_biometric": 1 if identity.get("kyc_verification_method") == "biometric" else 0,
            "email_domain_risk": identity.get("email_domain_risk_score") or 0.0,

            # Device features + missingness flags
            "device_fp_was_null": 1 if device.get("device_fingerprint") is None else 0,
            "ip_hash_was_null": 1 if device.get("ip_address_hash") is None else 0,
            "ip_asn_risk_score": device.get("ip_asn_risk_score") or 0.0,
            "ip_asn_was_null": 1 if device.get("ip_asn_risk_score") is None else 0,
            "geo_velocity_kmh": device.get("geo_velocity_kmh") or 0.0,
            "geo_velocity_was_null": 1 if device.get("geo_velocity_kmh") is None else 0,

            # Session / behavioral features
            "login_time_deviation_hrs": session.get("login_time_deviation_hrs") or 0.0,
            "mean_inter_txn_seconds": session.get("mean_inter_txn_seconds") or 86400.0,
            "failed_auth_count_24h": session.get("failed_auth_count_24h") or 0,
            "typing_cadence_variance": session.get("typing_cadence_variance") or 0.15,
            "typing_cadence_was_null": 1 if session.get("typing_cadence_variance") is None else 0,

            # Velocity features (from incremental store)
            **velocity,

            # Graph features
            **graph_feats,

            # Behavioral deviation
            **behavioral,

            # ── XGBoost metadata (prefixed with _ so they pass through to XGBoost mapper) ──
            "_hour_of_day": hour_of_day,
            "_channel": txn.get("channel", "CNP"),
            "_card_type": txn.get("card_type", "debit").capitalize(),
            "_merchant_category": merchant_category,
            "_location": location,
            "_merchant_fraud_rate": 0.005,  # derived from merchant history if available
            "_pagerank": graph_feats.get("graph_degree", 0.01) * 0.5,
            "_user_age": max(18, min(75, int(identity.get("account_age_days", 365) / 14))),
        }

        return features

    @staticmethod
    def _geocode_to_city(geocode: str) -> str:
        """Map geocode string to Indian city name for XGBoost Location feature."""
        city_map = {
            "19.0760": "Mumbai", "28.6139": "Delhi", "12.9716": "Bengaluru",
            "13.0827": "Chennai", "17.3850": "Hyderabad", "22.5726": "Kolkata",
            "23.0225": "Ahmedabad", "18.5204": "Pune", "26.9124": "Mumbai",
            "21.1702": "Mumbai",
        }
        if geocode:
            lat = geocode.split(",")[0] if "," in geocode else ""
            for prefix, city in city_map.items():
                if lat.startswith(prefix[:7]):
                    return city
        return "Mumbai"  # default

    @staticmethod
    def _mcc_to_category(mcc: str) -> str:
        """Map MCC code to merchant category for XGBoost MerchantCategory feature."""
        mcc_map = {
            "5411": "Grocery", "5412": "Grocery", "5311": "Retail", "5331": "Retail",
            "5732": "Electronics", "5734": "Electronics", "4511": "Travel", "4722": "Travel",
            "5812": "Dining", "5813": "Dining", "7832": "Entertainment", "7841": "Entertainment",
            "4900": "Utility", "4814": "Utility", "6051": "Retail", "4829": "Retail",
            "5999": "Retail",
        }
        return mcc_map.get(mcc, "Retail")

    def process_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Process a batch of transactions and return feature vectors."""
        feature_vectors = []
        for txn in transactions:
            features = self.process_transaction(txn)
            # Attach labels for training
            labels = txn.get("labels", {})
            features["_is_fraud"] = 1 if labels.get("is_fraud") else 0
            features["_f3_tactic"] = labels.get("f3_tactic", "")
            features["_f3_technique"] = labels.get("f3_technique", "")
            features["_scenario_id"] = labels.get("scenario_id", "")
            features["_fraud_vector"] = labels.get("fraud_vector", "Legitimate")
            features["_transaction_id"] = txn.get("transaction_id", "")
            feature_vectors.append(features)
        return feature_vectors

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of model input feature names (excluding labels)."""
        return [
            "amount", "channel_CNP", "channel_CP", "channel_P2P", "channel_ATM",
            "is_refund", "auth_declined", "auth_retried", "mcc_high_risk",
            "account_age_days", "kyc_doc_similarity_score", "kyc_doc_similarity_was_null",
            "kyc_method_automated", "kyc_method_biometric", "email_domain_risk",
            "device_fp_was_null", "ip_hash_was_null", "ip_asn_risk_score", "ip_asn_was_null",
            "geo_velocity_kmh", "geo_velocity_was_null",
            "login_time_deviation_hrs", "mean_inter_txn_seconds", "failed_auth_count_24h",
            "typing_cadence_variance", "typing_cadence_was_null",
            "txn_count_1h", "txn_sum_1h", "txn_count_24h", "txn_sum_24h",
            "txn_count_7d", "txn_sum_7d",
            "graph_degree", "graph_closeness", "graph_neighbors",
            "shared_device_accounts", "shared_ip_accounts",
            "amount_zscore", "hour_deviation", "inter_txn_zscore", "is_new_account",
        ]
