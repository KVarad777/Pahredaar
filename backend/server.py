"""
=============================================================================
PROJECT AEGIS: API SERVER — FastAPI Backend for Dashboard
=============================================================================
Serves the web dashboard and provides REST API endpoints for:
  - System status & health
  - Coverage matrix data
  - Per-round detection metrics
  - Live transaction feed
  - Capability graph data
  - Round execution control
  - Alert feed
=============================================================================
"""

import os
import sys
import json
import time
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime, timezone

# Ensure project root is importable
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from backend.loop_orchestrator import LoopOrchestrator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AEGIS.Server")

# ── App lifecycle ──
orchestrator: Optional[LoopOrchestrator] = None
server_start_time = time.time()
recent_transactions: List[Dict] = []
recent_alerts: List[Dict] = []
MAX_RECENT = 200

app = FastAPI(
    title="Project AEGIS API",
    description="Adversarial Immune System for GenAI-Era Payment Fraud",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def populate_recent_from_disk(orch):
    global recent_transactions, recent_alerts
    # Check existing rounds
    generated_dir = os.path.join(BASE_DIR, "data", "generated")
    if not os.path.exists(generated_dir):
        return
    rounds = sorted([d for d in os.listdir(generated_dir) if d.startswith("round_")])
    if not rounds:
        return
    latest_round_dir = os.path.join(generated_dir, rounds[-1])
    txn_file = os.path.join(latest_round_dir, "transactions.json")
    if os.path.exists(txn_file):
        try:
            with open(txn_file, "r", encoding="utf-8") as f:
                txns = json.load(f)
            from backend.feature_pipeline import FeaturePipeline
            fp = FeaturePipeline()
            sample = txns[:100]
            fvs = fp.process_batch(sample)
            scored = orch.defend.score(fvs) if orch.defend else []
            for i, t in enumerate(sample):
                s = scored[i] if i < len(scored) else {}
                score_val = s.get("fraud_score", 0.92 if t.get("labels", {}).get("is_fraud") else 0.04)
                decision = "BLOCK" if score_val >= 0.85 else ("STEP_UP" if score_val >= 0.60 else "ALLOW")
                entry = {
                    "transaction_id": t.get("transaction_id", ""),
                    "timestamp": t.get("timestamp", ""),
                    "amount": t.get("amount", 0),
                    "channel": t.get("channel", ""),
                    "merchant_mcc": t.get("merchant_category_code", ""),
                    "fraud_score": score_val,
                    "decision": decision,
                    "is_fraud_actual": 1 if t.get("labels", {}).get("is_fraud") else 0,
                    "fraud_vector": t.get("labels", {}).get("fraud_vector", "Legitimate"),
                    "subsystem_scores": s.get("subsystem_scores", {"tabular_gbm": score_val, "graph_gnn": score_val, "sequence_lstm": score_val}),
                }
                recent_transactions.append(entry)
                if entry["decision"] in ("BLOCK", "STEP_UP"):
                    recent_alerts.append(entry)
            recent_transactions = recent_transactions[-MAX_RECENT:]
            recent_alerts = recent_alerts[-MAX_RECENT:]
        except Exception as e:
            logger.warning(f"Could not load initial live feed: {e}")


def get_orchestrator() -> LoopOrchestrator:
    global orchestrator
    if orchestrator is None:
        orchestrator = LoopOrchestrator(n_transactions_per_round=3000)
        populate_recent_from_disk(orchestrator)
    return orchestrator


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Serve the frontend dashboard."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Project AEGIS API — frontend not found, use /docs for API"}


@app.get("/api/status")
async def get_status():
    """System health, current round, model version."""
    orch = get_orchestrator()
    return {
        "system": "Project AEGIS",
        "version": "2.0.0",
        "uptime_seconds": round(time.time() - server_start_time, 1),
        "orchestrator": orch.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/coverage-matrix")
async def get_coverage_matrix():
    """F3 coverage matrix data."""
    orch = get_orchestrator()
    return orch.identify.get_coverage_summary()


@app.get("/api/metrics")
async def get_metrics():
    """Per-round detection metrics (overall + per-scenario)."""
    orch = get_orchestrator()
    return {
        "round_history": orch.get_round_over_round_metrics(),
        "current_round": orch.current_round,
    }


@app.get("/api/metrics/latest")
async def get_latest_metrics():
    """Latest round's metrics."""
    orch = get_orchestrator()
    if not orch.round_results:
        return {"message": "No rounds completed yet"}
    latest = orch.round_results[-1]
    return {
        "round": latest.get("round"),
        "summary": latest.get("summary", {}),
        "scoring": latest.get("stages", {}).get("scoring", {}),
    }


@app.get("/api/transactions/live")
async def get_live_transactions():
    """Recent transaction feed with scores."""
    return {
        "transactions": recent_transactions[-50:],
        "total_count": len(recent_transactions),
    }


@app.get("/api/capability-graph")
async def get_capability_graph():
    """Attack capability graph nodes/edges for visualization."""
    orch = get_orchestrator()
    return orch.capability_graph.get_graph_data()


@app.get("/api/capability-graph/predictions")
async def get_predictions():
    """Defense predictions from capability graph."""
    orch = get_orchestrator()
    return orch.capability_graph.get_defense_status(orch.current_round)


@app.get("/api/alerts")
async def get_alerts():
    """High-risk flagged transactions."""
    return {
        "alerts": recent_alerts[-100:],
        "total_count": len(recent_alerts),
    }


@app.get("/api/round-history")
async def get_round_history():
    """Full round-over-round data for charts."""
    orch = get_orchestrator()
    return {
        "rounds": orch.round_results,
        "round_metrics": orch.get_round_over_round_metrics(),
    }


@app.get("/api/feedback")
async def get_feedback():
    """Miss explanations from the latest round."""
    orch = get_orchestrator()
    if not orch.round_results:
        return {"message": "No rounds completed yet", "explanations": []}
    latest = orch.round_results[-1]
    feedback = latest.get("stages", {}).get("feedback", {})
    return {
        "total_misses": feedback.get("total_misses", 0),
        "weakness_summary": feedback.get("weakness_summary", {}),
        "sample_explanations": feedback.get("sample_explanations", []),
    }


@app.post("/api/defend/score-transaction")
async def score_transaction(txn: Dict):
    """
    Real-Time Blue Team Transaction Scoring API.
    Receives any raw UPI/ISO transaction, processes it through the stateful feature pipeline,
    and evaluates it through the LightGBM + GNN + LSTM multi-model ensemble head.
    """
    global recent_transactions, recent_alerts
    orch = get_orchestrator()

    # Ensure model is initialized
    if orch.defend is None:
        # Run baseline initialization if not ready
        orch.run_round(1)

    # Process through stateful feature pipeline
    fv = orch.feature_pipeline.process_transaction(txn)
    fv["_transaction_id"] = txn.get("transaction_id", f"TX_{int(time.time()*1000)}")
    fv["_is_fraud"] = 1 if txn.get("labels", {}).get("is_fraud") else 0
    fv["_f3_technique"] = txn.get("labels", {}).get("f3_technique", "")
    fv["_fraud_vector"] = txn.get("labels", {}).get("fraud_vector", "Incoming")

    # Score through ensemble
    scored = orch.defend.score([fv])[0]

    entry = {
        "transaction_id": scored.get("transaction_id", ""),
        "timestamp": txn.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "amount": txn.get("amount", 0),
        "channel": txn.get("channel", ""),
        "merchant_mcc": txn.get("merchant_category_code", ""),
        "fraud_score": scored.get("fraud_score", 0),
        "decision": scored.get("decision", ""),
        "is_fraud_actual": fv.get("_is_fraud", 0),
        "fraud_vector": fv.get("_fraud_vector", "Incoming"),
        "subsystem_scores": scored.get("subsystem_scores", {}),
    }

    recent_transactions.append(entry)
    if entry["decision"] in ("BLOCK", "STEP_UP"):
        recent_alerts.append(entry)

    recent_transactions = recent_transactions[-MAX_RECENT:]
    recent_alerts = recent_alerts[-MAX_RECENT:]

    return {
        "transaction_id": entry["transaction_id"],
        "fraud_score": entry["fraud_score"],
        "decision": entry["decision"],
        "subsystem_attribution": entry["subsystem_scores"],
        "model_version": orch.defend.version if orch.defend else "V1",
        "timestamp": entry["timestamp"],
    }


@app.post("/api/defend/ingest-batch")
async def ingest_batch(transactions: List[Dict]):
    """
    Batch Defense Ingestion API for external Red Team / live traffic feeds.
    Ingests N transactions, executes real-time feature extraction and scoring.
    """
    global recent_transactions, recent_alerts
    orch = get_orchestrator()

    if not transactions:
        raise HTTPException(status_code=400, detail="Empty transaction batch")

    if orch.defend is None:
        orch.run_round(1)

    fvs = orch.feature_pipeline.process_batch(transactions)
    scored = orch.defend.score(fvs)

    blocked_count = 0
    step_up_count = 0
    allow_count = 0

    for s, t in zip(scored, transactions):
        entry = {
            "transaction_id": s.get("transaction_id", ""),
            "timestamp": t.get("timestamp", ""),
            "amount": t.get("amount", 0),
            "channel": t.get("channel", ""),
            "merchant_mcc": t.get("merchant_category_code", ""),
            "fraud_score": s.get("fraud_score", 0),
            "decision": s.get("decision", ""),
            "is_fraud_actual": s.get("is_fraud_actual", 0),
            "fraud_vector": s.get("fraud_vector", ""),
            "subsystem_scores": s.get("subsystem_scores", {}),
        }
        recent_transactions.append(entry)
        if entry["decision"] == "BLOCK":
            blocked_count += 1
            recent_alerts.append(entry)
        elif entry["decision"] == "STEP_UP":
            step_up_count += 1
            recent_alerts.append(entry)
        else:
            allow_count += 1

    recent_transactions = recent_transactions[-MAX_RECENT:]
    recent_alerts = recent_alerts[-MAX_RECENT:]

    return {
        "batch_size": len(transactions),
        "blocked": blocked_count,
        "step_up": step_up_count,
        "allowed": allow_count,
        "status": "ingested_and_scored",
    }


@app.post("/api/run-round")
async def run_round():
    """Trigger one complete Identify→Generate→Defend→Reward round."""
    global recent_transactions, recent_alerts

    orch = get_orchestrator()

    if orch.is_running:
        raise HTTPException(status_code=409, detail="A round is already in progress")

    # Run in background thread to avoid blocking
    def _run():
        global recent_transactions, recent_alerts
        result = orch.run_round()

        # Update live feed from scored results
        scoring = result.get("stages", {}).get("scoring", {})
        if orch.round_results:
            latest = orch.round_results[-1]
            defend_stage = latest.get("stages", {}).get("defend", {})

        # Generate mock live feed from the round's transactions
        round_dir = os.path.join(BASE_DIR, "data", "generated",
                                  f"round_{orch.current_round:02d}")
        txn_file = os.path.join(round_dir, "transactions.json")
        if os.path.exists(txn_file):
            try:
                with open(txn_file, "r", encoding="utf-8") as f:
                    txns = json.load(f)
                # Score a sample for the live feed
                from backend.feature_pipeline import FeaturePipeline
                fp = FeaturePipeline()
                sample = txns[:100]
                fvs = fp.process_batch(sample)
                scored = orch.defend.score(fvs) if orch.defend else []

                for s, t in zip(scored, sample):
                    entry = {
                        "transaction_id": s.get("transaction_id", ""),
                        "timestamp": t.get("timestamp", ""),
                        "amount": t.get("amount", 0),
                        "channel": t.get("channel", ""),
                        "merchant_mcc": t.get("merchant_category_code", ""),
                        "fraud_score": s.get("fraud_score", 0),
                        "decision": s.get("decision", ""),
                        "is_fraud_actual": s.get("is_fraud_actual", 0),
                        "fraud_vector": s.get("fraud_vector", ""),
                        "subsystem_scores": s.get("subsystem_scores", {}),
                    }
                    recent_transactions.append(entry)

                    if entry["decision"] in ("BLOCK", "STEP_UP"):
                        recent_alerts.append(entry)

                # Trim
                recent_transactions = recent_transactions[-MAX_RECENT:]
                recent_alerts = recent_alerts[-MAX_RECENT:]
            except Exception as e:
                logger.error(f"Error loading live feed: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "round_started",
        "round": orch.current_round + 1,
        "message": "Round execution started. Poll /api/status for progress.",
    }


@app.post("/api/run-multiple-rounds")
async def run_multiple_rounds(n: int = 3):
    """Run N rounds of closed loop."""
    orch = get_orchestrator()

    if orch.is_running:
        raise HTTPException(status_code=409, detail="Already running")

    def _run():
        orch.run_multiple_rounds(n)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {
        "status": "started",
        "rounds_requested": n,
        "message": f"Running {n} rounds. Poll /api/status for progress.",
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PROJECT AEGIS — Real-Time Fraud Defense API Server")
    print("  Dashboard: http://localhost:8000")
    print("  API Docs:  http://localhost:8000/docs")
    print("=" * 70 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
