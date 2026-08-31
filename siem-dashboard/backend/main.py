from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import json
import os
from pathlib import Path

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

@app.get("/api/metrics")
def get_metrics():
    log_path = DATA_DIR / "dashboard_log.csv"
    if not log_path.exists():
        return []
    df = pd.read_csv(log_path)
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")

@app.get("/api/coverage")
def get_coverage():
    cov_path = DATA_DIR / "coverage_matrix.csv"
    if not cov_path.exists():
        return []
    df = pd.read_csv(cov_path)
    df = df.astype(object).where(pd.notnull(df), None)
    return df.to_dict(orient="records")

@app.get("/api/logs")
def get_logs(round: int = -1, limit: int = 100):
    gen_dir = DATA_DIR / "generated"
    
    if not gen_dir.exists():
        return []
        
    rounds = sorted([d for d in os.listdir(gen_dir) if d.startswith("round_")])
    if not rounds:
        return []
        
    target_round = rounds[-1] if round == -1 else f"round_{round:02d}"
    
    tx_path = gen_dir / target_round / "transactions.csv"
    if not tx_path.exists():
        return []
        
    # Read the CSV
    df = pd.read_csv(tx_path)
    
    # Check if 'is_fraud' exists
    if "is_fraud" in df.columns:
        fraud_df = df[df["is_fraud"] == 1]
        legit_df = df[df["is_fraud"] == 0]
        
        # Mix fraud and legit to show an interesting log
        take_fraud = min(len(fraud_df), limit)
        take_legit = limit - take_fraud
        if take_legit < 0: take_legit = 0
        
        sample = pd.concat([fraud_df.tail(take_fraud), legit_df.tail(take_legit)])
        sample = sample.sort_index() # Keep chronological order roughly based on index
    else:
        sample = df.tail(limit)
        
    # Replace NaN with None so it's valid JSON
    sample = sample.fillna(0) # or replace with None, but fillna is safer for float serialization
    # if you want actual None for JSON null:
    sample = sample.astype(object).where(pd.notnull(sample), None)
    
    return sample.to_dict(orient="records")

@app.get("/api/feedback")
def get_feedback():
    feedback_path = DATA_DIR / "latest_miss_explanations.json"
    if not feedback_path.exists():
        return []
    with open(feedback_path) as f:
        return json.load(f)

import sys
import queue
import threading
import asyncio
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

class StreamCapture:
    def __init__(self):
        self.q = queue.Queue()
        self.original_stdout = sys.stdout

    def write(self, text):
        if text:
            # Push non-empty text to queue
            self.q.put(text)
        self.original_stdout.write(text)

    def flush(self):
        self.original_stdout.flush()

log_capture = StreamCapture()

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

@app.post("/api/attack/generate")
async def generate_attack():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        
    from src.reward.loop_orchestrator import FraudRedTeamLoop
    
    def run_simulation():
        try:
            sys.stdout = log_capture
            loop = FraudRedTeamLoop(
                f3_taxonomy_path=str(PROJECT_ROOT / "config" / "f3_taxonomy.json"),
                schema_path=str(PROJECT_ROOT / "config" / "upi_schema.json"),
                distribution_params_path=str(PROJECT_ROOT / "config" / "distribution_params.yaml"),
                null_config_path=str(PROJECT_ROOT / "config" / "null_injection_rates.yaml"),
                coverage_matrix_path=str(DATA_DIR / "coverage_matrix.csv"),
                dashboard_log_path=str(DATA_DIR / "dashboard_log.csv"),
            )
            # Run one round with small numbers to be interactive but fast enough
            loop.run(n_rounds=1, n_new_scenarios=3, n_legit_accounts=300)
        except Exception as e:
            print(f"Error during simulation: {e}")
        finally:
            sys.stdout = log_capture.original_stdout
            log_capture.q.put("[DONE]")

    thread = threading.Thread(target=run_simulation)
    thread.start()
    return {"status": "started"}

@app.get("/api/attack/stream")
async def stream_attack():
    async def event_generator():
        while True:
            try:
                line = await asyncio.to_thread(log_capture.q.get, True, 1.0)
                if line == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                if line:
                    # SSE format expects `data: ...\n\n`
                    # Replace newlines so SSE doesn't break
                    clean_line = line.replace('\n', '\\n')
                    yield f"data: {clean_line}\n\n"
            except queue.Empty:
                yield "data: \n\n"
                
    return StreamingResponse(event_generator(), media_type="text/event-stream")

