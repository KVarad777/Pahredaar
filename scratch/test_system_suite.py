"""
=============================================================================
PROJECT AEGIS — FULL SYSTEM AUTOMATED TEST SUITE (10/10 MODULES)
Mastercard Innovation Challenge @ Global Fintech Fest 2026
=============================================================================
"""

import os
import sys
import json
import time
import unittest
from datetime import datetime, timezone

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure project root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.feature_pipeline import FeaturePipeline
from backend.identify_engine import IdentifyEngine
from backend.generate_engine import GenerateEngine
from backend.defend_engine import DefendEngine
from backend.reward_engine import RewardEngine, blue_reward, red_reward
from backend.feedback_engine import FeedbackEngine
from backend.capability_graph import CapabilityGraph
from backend.loop_orchestrator import LoopOrchestrator
from backend.server import app
from starlette.testclient import TestClient

import demo_closed_loop as dcl


class TestAegisSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("  PROJECT AEGIS: EXECUTING FULL SYSTEM VERIFICATION SUITE")
        print("=" * 80)
        cls.client = TestClient(app)

    def test_01_feature_pipeline(self):
        """Test FeaturePipeline state tracking and feature vector extraction."""
        print("\n[TEST 1] Feature Pipeline Engine...")
        fp = FeaturePipeline()

        sample_txn = {
            "transaction_id": "TXN_TEST_001",
            "timestamp": "2026-08-30T10:00:00Z",
            "account_id": "ACC_USER_42",
            "amount": 250.0,
            "currency": "INR",
            "channel": "UPI",
            "merchant": {
                "merchant_id": "MERCH_STORE_1",
                "merchant_category_code": "5411",
                "merchant_country": "IND"
            },
            "device": {
                "device_id": "DEV_PHONE_99",
                "ip_address": "192.168.1.1",
                "ip_country": "IND",
                "is_vpn": False,
                "is_emulator": False,
                "biometrics": {
                    "keystroke_dynamics": {"flight_time_avg_ms": 110.0, "dwell_time_avg_ms": 80.0},
                    "touch_entropy": 0.72,
                    "behavioral_biometric_score": 0.88
                }
            },
            "labels": {
                "is_fraud": False,
                "fraud_vector": "Legitimate",
                "f3_technique": "F3.0"
            }
        }

        fv = fp.process_transaction(sample_txn)
        self.assertIsInstance(fv, dict)
        self.assertIn("amount", fv)
        self.assertIn("txn_count_1h", fv)
        self.assertIn("txn_sum_1h", fv)
        self.assertIn("typing_cadence_variance", fv)
        self.assertAlmostEqual(fv["amount"], 250.0)
        print("  [PASS] Feature Pipeline successfully computed stateful and behavioral features.")

    def test_02_identify_engine(self):
        """Test IdentifyEngine F3 taxonomy coverage matrix."""
        print("\n[TEST 2] Identify Engine & F3 Taxonomy...")
        ie = IdentifyEngine()
        cov = ie.get_coverage_summary()
        self.assertIn("total_scenarios", cov)
        self.assertIn("scenarios", cov)
        self.assertGreaterEqual(cov["total_scenarios"], 8)

        new_scenarios = ie.generate_scenarios_from_taxonomy(round_num=1)
        self.assertIsInstance(new_scenarios, list)
        print(f"  [PASS] Identify Engine loaded {cov['total_scenarios']} F3 scenarios across {len(cov.get('by_tactic', {}))} tactics.")

    def test_03_generate_engine(self):
        """Test GenerateEngine red team attack synthesis."""
        print("\n[TEST 3] Generate Engine (Red Team Attack Synthesis)...")
        ie = IdentifyEngine()
        ge = GenerateEngine()
        scenario_dicts = [s.to_dict() if hasattr(s, 'to_dict') else s for s in ie.scenarios]
        txns = ge.generate_round(round_num=1, n_transactions=100, scenarios=scenario_dicts)
        self.assertGreaterEqual(len(txns), 90)

        fraud_count = sum(1 for t in txns if t.get("labels", {}).get("is_fraud"))
        self.assertGreater(fraud_count, 0)
        print(f"  [PASS] Generate Engine synthesized {len(txns)} transactions ({fraud_count} red-team attacks).")

    def test_04_defend_engine(self):
        """Test DefendEngine multi-modal scoring and decisioning."""
        print("\n[TEST 4] Defend Engine (Blue Team Multi-Modal Scoring)...")
        fp = FeaturePipeline()
        ie = IdentifyEngine()
        ge = GenerateEngine()
        scenario_dicts = [s.to_dict() if hasattr(s, 'to_dict') else s for s in ie.scenarios]
        txns = ge.generate_round(round_num=1, n_transactions=150, scenarios=scenario_dicts)
        fvs = fp.process_batch(txns)

        de = DefendEngine(feature_names=fp.get_feature_names())
        scored = de.score(fvs[:20])
        self.assertEqual(len(scored), 20)
        for s in scored:
            self.assertIn("fraud_score", s)
            self.assertIn("decision", s)
            self.assertIn(s["decision"], ["ALLOW", "STEP_UP", "BLOCK"])
            self.assertGreaterEqual(s["fraud_score"], 0.0)
            self.assertLessEqual(s["fraud_score"], 1.0)
        print("  [PASS] Defend Engine scored batch with multi-modal ensemble (ALLOW/STEP_UP/BLOCK bounds).")

    def test_05_reward_engine(self):
        """Test RewardEngine multi-objective evaluation."""
        print("\n[TEST 5] Reward Engine (Multi-Objective Evaluation)...")
        re = RewardEngine()
        overall = {"precision": 0.88, "recall": 0.85, "f1": 0.865, "fpr": 0.015, "auc": 0.94}
        scenarios = [{"scenario_name": "Mule Network Cash-Out", "detection_rate": 0.82, "novelty_tag": "baseline"}]
        per_scenario = {"Mule Network Cash-Out": {"detection_rate": 0.82}}
        
        round_rewards = re.compute_round_rewards(
            per_scenario_metrics=per_scenario,
            overall_metrics=overall,
            scenarios=scenarios,
            latency_ms=45.0
        )
        self.assertIn("blue_reward", round_rewards)
        self.assertIn("should_fine_tune", round_rewards)
        self.assertIn("flagged_for_harder_variants", round_rewards)
        print(f"  [PASS] Reward Engine: Blue Reward = {round_rewards['blue_reward']:.4f}, Fine-tune needed = {round_rewards['should_fine_tune']}, Flagged = {len(round_rewards['flagged_for_harder_variants'])}")

    def test_06_feedback_engine(self):
        """Test FeedbackEngine explainability and failure analysis."""
        print("\n[TEST 6] Feedback Engine (Explainable AI & Countermeasures)...")
        fe = FeedbackEngine()
        
        scored_txns = [{
            "transaction_id": "TXN_MISS_1",
            "amount": 4999.0,
            "channel": "UPI",
            "fraud_score": 0.42,
            "decision": "ALLOW",
            "is_fraud_actual": 1,
            "fraud_vector": "Credential Stuffing / Account Takeover",
            "f3_technique": "Credential Stuffing / Account Takeover",
            "scenario_id": "SC_TEST_01",
            "subsystem_scores": {"xgboost": 0.45, "graph_anomaly": 0.38}
        }]
        fvs = [{
            "_transaction_id": "TXN_MISS_1",
            "amount": 4999.0,
            "channel_CNP": 1,
            "login_time_deviation_hrs": 3.5,
            "typing_cadence_variance": 0.001
        }]

        explanations = fe.explain_misses(scored_results=scored_txns, feature_vectors=fvs)
        self.assertEqual(len(explanations), 1)
        self.assertIn("explanation", explanations[0])
        print(f"  [PASS] Feedback Engine generated root cause attributions for {len(explanations)} missed vector(s).")

    def test_07_capability_graph(self):
        """Test CapabilityGraph Markov state traversal."""
        print("\n[TEST 7] Capability Graph (Markov Attack Progression)...")
        cg = CapabilityGraph()
        graph_data = cg.get_graph_data()
        self.assertIn("nodes", graph_data)
        self.assertIn("edges", graph_data)
        self.assertGreater(len(graph_data["nodes"]), 0)

        preds = cg.get_defense_status(round_num=1)
        self.assertIsInstance(preds, dict)
        print(f"  [PASS] Capability Graph loaded {len(graph_data['nodes'])} attack nodes & {len(graph_data['edges'])} transitions.")

    def test_08_fastapi_endpoints(self):
        """Test all REST API endpoints exposed by FastAPI."""
        print("\n[TEST 8] FastAPI REST Endpoints via TestClient...")
        
        # GET /api/status
        r = self.client.get("/api/status")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Project AEGIS", r.json().get("system", ""))

        # GET /api/coverage-matrix
        r = self.client.get("/api/coverage-matrix")
        self.assertEqual(r.status_code, 200)

        # GET /api/metrics
        r = self.client.get("/api/metrics")
        self.assertEqual(r.status_code, 200)

        # GET /api/transactions/live
        r = self.client.get("/api/transactions/live")
        self.assertEqual(r.status_code, 200)

        # GET /api/capability-graph
        r = self.client.get("/api/capability-graph")
        self.assertEqual(r.status_code, 200)

        # GET /api/alerts
        r = self.client.get("/api/alerts")
        self.assertEqual(r.status_code, 200)

        # POST /api/defend/score-transaction
        sample_txn = {
            "transaction_id": "API_TEST_TXN",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "account_id": "ACC_TEST",
            "amount": 9500.0,
            "currency": "INR",
            "channel": "UPI",
            "merchant": {"merchant_id": "M_TEST", "merchant_category_code": "6011", "merchant_country": "IND"},
            "device": {
                "device_id": "D_TEST",
                "ip_address": "10.0.0.1",
                "ip_country": "IND",
                "is_vpn": True,
                "is_emulator": False,
                "biometrics": {
                    "keystroke_dynamics": {"flight_time_avg_ms": 15.0, "dwell_time_avg_ms": 10.0},
                    "touch_entropy": 0.12,
                    "behavioral_biometric_score": 0.15
                }
            },
            "labels": {"is_fraud": True, "fraud_vector": "Credential Stuffing / Account Takeover", "f3_technique": "Credential Stuffing / Account Takeover"}
        }
        r = self.client.post("/api/defend/score-transaction", json=sample_txn)
        self.assertEqual(r.status_code, 200)
        res_data = r.json()
        self.assertIn("fraud_score", res_data)
        self.assertIn("decision", res_data)
        print(f"  [PASS] Live scoring API returned: Score={res_data['fraud_score']}, Decision={res_data['decision']}")

    def test_09_orchestrator_closed_loop_round(self):
        """Test LoopOrchestrator running an autonomous round."""
        print("\n[TEST 9] Autonomous Loop Orchestrator (Full Micro-Round)...")
        orch = LoopOrchestrator(n_transactions_per_round=250)
        round_summary = orch.run_round(round_num=1)
        self.assertIsNotNone(round_summary)
        self.assertEqual(round_summary.get("round"), 1)
        
        scoring = round_summary.get("stages", {}).get("scoring", {})
        overall = scoring.get("overall", {})
        self.assertIn("auc", overall)
        self.assertIn("f1", overall)
        self.assertIn("fpr", overall)
        print(f"  [PASS] Orchestrator completed Round 1: AUC={overall.get('auc', 0):.4f}, F1={overall.get('f1', 0):.4f}, FPR={overall.get('fpr', 0)*100:.2f}%")

    def test_10_closed_loop_demo_execution(self):
        """Test the end-to-end multi-modal demo simulation."""
        print("\n[TEST 10] Closed-Loop Terminal Demo Pipeline Execution...")
        dcl.run_closed_loop_demo(delay=0.0)
        print("  [PASS] Closed-loop 5-phase adversarial demo completed successfully.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
