"""
=============================================================================
PROJECT AEGIS: CAPABILITY GRAPH — USP Attack Prediction System
=============================================================================
The core USP differentiator:
  - Attack techniques as graph nodes
  - Directed weighted edges represent capability-transfer relationships
  - After a miss, predicts the NEXT most likely attack
  - Preemptively generates and trains against predicted attacks
  - Implements the "time machine" demo flow
=============================================================================
"""

import json
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger("AEGIS.CapabilityGraph")


# Initial capability-transfer priors (kill-chain relationships)
INITIAL_EDGES = [
    # Reconnaissance -> Access Acquisition
    ("Account Enumeration", "Credential Stuffing / Account Takeover", 0.65),
    ("BIN Attack / Card Testing", "CNP Rapid Cashout", 0.70),

    # Access -> Positioning
    ("Credential Stuffing / Account Takeover", "Mule Network Setup", 0.60),
    ("Credential Stuffing / Account Takeover", "Sleeper Mule Activation", 0.55),
    ("Deepfake KYC Bypass", "Synthetic Identity Creation", 0.73),
    ("Synthetic Identity Creation", "Mule Network Setup", 0.68),
    ("Synthetic Identity Creation", "Sleeper Mule Activation", 0.62),

    # Positioning -> Monetization
    ("Mule Network Setup", "Mule Network Cash-Out", 0.80),
    ("Mule Network Setup", "CNP Rapid Cashout", 0.55),
    ("Sleeper Mule Activation", "Mule Network Cash-Out", 0.73),
    ("Sleeper Mule Activation", "Low-and-Slow Drain", 0.65),
    ("Device Farm / Emulator Ring", "CNP Rapid Cashout", 0.70),
    ("Device Farm / Emulator Ring", "BIN Attack / Card Testing", 0.60),
    ("Collusive Merchant Setup", "Refund / Chargeback Abuse", 0.75),
    ("Collusive Merchant Setup", "Mule Network Cash-Out", 0.50),

    # Evasion chains
    ("Anti-Fingerprinting / Signal Suppression", "Device Farm / Emulator Ring", 0.72),
    ("Anti-Fingerprinting / Signal Suppression", "Impossible Travel Masking", 0.60),
    ("Semantic Smuggling / NLP Evasion", "Low-and-Slow Drain", 0.55),
    ("Impossible Travel Masking", "Credential Stuffing / Account Takeover", 0.50),

    # Agentic attack chains
    ("Agentic Token Hijacking", "Prompt Injection Payment Redirect", 0.78),
    ("Prompt Injection Payment Redirect", "CNP Rapid Cashout", 0.65),

    # AI-specific escalation paths
    ("Deepfake KYC Bypass", "Device Farm / Emulator Ring", 0.58),
    ("Deepfake KYC Bypass", "Agentic Token Hijacking", 0.45),
]


class CapabilityGraph:
    """
    Directed weighted graph of attack capabilities.
    Nodes = attack techniques, edges = capability-transfer relationships.
    """

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: Dict[str, Dict[str, Dict]] = defaultdict(dict)
        self.event_log: List[Dict] = []
        self._initialize_from_priors()

    def _initialize_from_priors(self):
        """Build initial graph from kill-chain priors."""
        all_techniques = set()
        for src, tgt, weight in INITIAL_EDGES:
            all_techniques.add(src)
            all_techniques.add(tgt)

        for tech in all_techniques:
            self.nodes[tech] = {
                "name": tech,
                "times_observed": 0,
                "times_missed": 0,
                "last_round_seen": -1,
            }

        for src, tgt, weight in INITIAL_EDGES:
            self.edges[src][tgt] = {
                "weight": weight,
                "evidence_count": 1,  # prior = 1 observation
                "source": "kill_chain_prior",
            }

        logger.info(f"[CAPABILITY] Initialized with {len(self.nodes)} nodes, "
                    f"{sum(len(v) for v in self.edges.values())} edges")

    def add_node(self, technique: str) -> None:
        if technique not in self.nodes:
            self.nodes[technique] = {
                "name": technique,
                "times_observed": 0,
                "times_missed": 0,
                "last_round_seen": -1,
            }

    def update_after_round(self, round_num: int,
                           missed_techniques: List[str],
                           detected_techniques: List[str]) -> None:
        """
        Update graph weights based on round results.
        Repeated transitions increase their weight.
        """
        # Record observed techniques
        for tech in detected_techniques + missed_techniques:
            if tech in self.nodes:
                self.nodes[tech]["times_observed"] += 1
                self.nodes[tech]["last_round_seen"] = round_num
            else:
                self.add_node(tech)
                self.nodes[tech]["times_observed"] = 1
                self.nodes[tech]["last_round_seen"] = round_num

        # Record misses
        for tech in missed_techniques:
            if tech in self.nodes:
                self.nodes[tech]["times_missed"] += 1

        # Strengthen edges between missed techniques and their successors
        for missed in missed_techniques:
            if missed in self.edges:
                for target, edge_data in self.edges[missed].items():
                    # Increase weight for observed transitions
                    edge_data["evidence_count"] += 1
                    # Weight asymptotically approaches 1.0
                    edge_data["weight"] = min(0.99,
                        edge_data["weight"] + 0.05 * (1 - edge_data["weight"]))

        # Log event
        self.event_log.append({
            "round": round_num,
            "missed": missed_techniques,
            "detected": detected_techniques,
        })

        logger.info(f"[CAPABILITY] Round {round_num}: "
                    f"{len(missed_techniques)} missed, {len(detected_techniques)} detected")

    def predict_next_attacks(self, missed_technique: str,
                             top_k: int = 3) -> List[Dict]:
        """
        Given a missed technique, predict the top-K most likely next attacks.
        Returns sorted by weight (descending).
        """
        if missed_technique not in self.edges:
            return []

        candidates = []
        for target, edge_data in self.edges[missed_technique].items():
            candidates.append({
                "predicted_attack": target,
                "confidence": round(edge_data["weight"], 3),
                "evidence_count": edge_data["evidence_count"],
                "source_technique": missed_technique,
            })

        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates[:top_k]

    def predict_all_next_attacks(self, missed_techniques: List[str],
                                 top_k: int = 5) -> List[Dict]:
        """
        For all missed techniques, return the aggregated top-K predictions.
        """
        all_predictions = {}

        for tech in missed_techniques:
            preds = self.predict_next_attacks(tech, top_k=top_k)
            for p in preds:
                target = p["predicted_attack"]
                if target not in all_predictions or p["confidence"] > all_predictions[target]["confidence"]:
                    all_predictions[target] = p

        sorted_preds = sorted(all_predictions.values(),
                              key=lambda x: x["confidence"], reverse=True)
        return sorted_preds[:top_k]

    def get_graph_data(self) -> Dict:
        """Return full graph for dashboard visualization."""
        nodes_list = []
        for name, data in self.nodes.items():
            nodes_list.append({
                "id": name,
                "label": name,
                "times_observed": data["times_observed"],
                "times_missed": data["times_missed"],
                "last_round": data["last_round_seen"],
            })

        edges_list = []
        for src, targets in self.edges.items():
            for tgt, data in targets.items():
                edges_list.append({
                    "source": src,
                    "target": tgt,
                    "weight": data["weight"],
                    "evidence_count": data["evidence_count"],
                })

        return {
            "nodes": nodes_list,
            "edges": edges_list,
            "total_nodes": len(nodes_list),
            "total_edges": len(edges_list),
        }

    def get_defense_status(self, round_num: int) -> Dict:
        """Summary of predictions and defense readiness."""
        # Get all recently missed techniques
        recent_misses = []
        for event in self.event_log[-3:]:  # last 3 rounds
            recent_misses.extend(event.get("missed", []))

        predictions = self.predict_all_next_attacks(list(set(recent_misses)), top_k=5)

        return {
            "recent_misses": list(set(recent_misses)),
            "predicted_next_attacks": predictions,
            "total_predictions": len(predictions),
            "graph_stats": {
                "nodes": len(self.nodes),
                "edges": sum(len(v) for v in self.edges.values()),
            },
        }
