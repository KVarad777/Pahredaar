"""
=============================================================================
PROJECT AEGIS: REAL-TIME TELEMETRY & STRUCTURED METRICS LOGGER (ml/metrics_logger.py)
Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
=============================================================================
High-performance, thread-safe structured telemetry engine. Records real-time
edge latencies, rolling transactions-per-second (TPS), 4-channel multi-modal
risk scores, composite decisions, ground truth labels, and active learning
disagreement telemetry.
=============================================================================
"""

import os
import sys
import csv
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
import pandas as pd

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


class MetricsLogger:
    """
    Thread-safe, low-overhead structured telemetry logger for Project AEGIS.
    Persists real-time streaming evaluations to CSV with atomic flushing
    and maintains an in-memory buffer for real-time statistical queries.
    """

    LOG_FIELDNAMES = [
        "Timestamp",
        "TransactionID",
        "Model_Version",
        "Processing_Latency_ms",
        "TPS",
        "Score_Tabular",
        "Score_Graph",
        "Score_Biometric",
        "Score_Text",
        "Combined_Risk_Score",
        "System_Decision",
        "Ground_Truth",
        "Is_Fuzzed",
        "Disagreement_Flag",
        "Action_Code",
        "Reason_Codes"
    ]

    def __init__(self, log_filepath: str = "scratch/live_system_logs.csv", auto_create_dir: bool = True):
        self.log_filepath = log_filepath
        self._lock = threading.Lock()
        self._total_logged = 0
        self._start_time = time.time()
        self._last_tps_timestamp = time.time()
        self._window_tx_count = 0
        self._current_tps = 0.0
        self._in_memory_records: List[Dict[str, Any]] = []

        if auto_create_dir:
            log_dir = os.path.dirname(self.log_filepath)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

        self._initialize_log_file()

    def _initialize_log_file(self):
        """Creates the log file and writes headers if the file does not exist."""
        with self._lock:
            if not os.path.exists(self.log_filepath) or os.path.getsize(self.log_filepath) == 0:
                with open(self.log_filepath, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.LOG_FIELDNAMES)
                    writer.writeheader()
                    f.flush()

    def _compute_rolling_tps(self) -> float:
        """Computes rolling instantaneous transactions per second."""
        now = time.time()
        self._window_tx_count += 1
        elapsed = now - self._last_tps_timestamp
        
        if elapsed >= 0.5:  # Update TPS window every 500ms
            self._current_tps = self._window_tx_count / elapsed
            self._window_tx_count = 0
            self._last_tps_timestamp = now

        # Fallback to cumulative average if rolling hasn't ticked yet
        if self._current_tps <= 0.0:
            total_elapsed = max(now - self._start_time, 0.001)
            return round(self._total_logged / total_elapsed, 2)

        return round(self._current_tps, 2)

    def log_transaction(
        self,
        transaction_id: str,
        model_version: str,
        processing_latency_ms: float,
        score_tabular: float,
        score_graph: float,
        score_biometric: float,
        score_text: float,
        combined_risk_score: float,
        system_decision: str,
        ground_truth: int,
        is_fuzzed: bool = False,
        disagreement_flag: bool = False,
        action_code: str = "APPROVE_FRICTIONLESS",
        reason_codes: Optional[List[str]] = None,
        custom_timestamp: Optional[str] = None,
        tps: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Thread-safely records a single transaction event into CSV and in-memory cache.
        """
        with self._lock:
            self._total_logged += 1
            calculated_tps = tps if tps is not None else self._compute_rolling_tps()
            ts = custom_timestamp or datetime.now(timezone.utc).isoformat()

            reasons_str = ";".join(reason_codes) if reason_codes else "BASELINE_AUTHORIZED_TRAFFIC"

            record = {
                "Timestamp": ts,
                "TransactionID": str(transaction_id),
                "Model_Version": str(model_version),
                "Processing_Latency_ms": round(float(processing_latency_ms), 3),
                "TPS": round(float(calculated_tps), 2),
                "Score_Tabular": round(float(score_tabular), 4),
                "Score_Graph": round(float(score_graph), 4),
                "Score_Biometric": round(float(score_biometric), 4),
                "Score_Text": round(float(score_text), 4),
                "Combined_Risk_Score": round(float(combined_risk_score), 4),
                "System_Decision": str(system_decision),
                "Ground_Truth": int(ground_truth),
                "Is_Fuzzed": 1 if is_fuzzed else 0,
                "Disagreement_Flag": 1 if disagreement_flag else 0,
                "Action_Code": str(action_code),
                "Reason_Codes": reasons_str
            }

            self._in_memory_records.append(record)

            # Write row and force atomic disk flush
            with open(self.log_filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.LOG_FIELDNAMES)
                writer.writerow(record)
                f.flush()

            return record

    def log_batch(self, batch_records: List[Dict[str, Any]]) -> int:
        """
        High-throughput batch ingestion method for bulk simulation streaming.
        """
        if not batch_records:
            return 0

        with self._lock:
            formatted_records = []
            now_ts = datetime.now(timezone.utc).isoformat()

            for rec in batch_records:
                self._total_logged += 1
                tps_val = rec.get("tps") or rec.get("TPS") or self._compute_rolling_tps()
                
                reasons = rec.get("reason_codes") or rec.get("Reason_Codes")
                if isinstance(reasons, list):
                    reasons_str = ";".join(reasons)
                elif isinstance(reasons, str):
                    reasons_str = reasons
                else:
                    reasons_str = "BASELINE_AUTHORIZED_TRAFFIC"

                raw_record = {
                    "Timestamp": rec.get("timestamp") or rec.get("Timestamp") or now_ts,
                    "TransactionID": str(rec.get("transaction_id") or rec.get("TransactionID") or f"TX_{self._total_logged}"),
                    "Model_Version": str(rec.get("model_version") or rec.get("Model_Version") or "Blue_V1"),
                    "Processing_Latency_ms": round(float(rec.get("processing_latency_ms") or rec.get("Processing_Latency_ms") or rec.get("execution_latency_ms") or 5.0), 3),
                    "TPS": round(float(tps_val), 2),
                    "Score_Tabular": round(float(rec.get("score_tabular") or rec.get("Score_Tabular") or rec.get("model_scores", {}).get("tabular_risk", 0.05)), 4),
                    "Score_Graph": round(float(rec.get("score_graph") or rec.get("Score_Graph") or rec.get("model_scores", {}).get("graph_risk", 0.05)), 4),
                    "Score_Biometric": round(float(rec.get("score_biometric") or rec.get("Score_Biometric") or rec.get("model_scores", {}).get("biometric_risk", 0.05)), 4),
                    "Score_Text": round(float(rec.get("score_text") or rec.get("Score_Text") or rec.get("model_scores", {}).get("text_risk", 0.05)), 4),
                    "Combined_Risk_Score": round(float(rec.get("combined_risk_score") or rec.get("Combined_Risk_Score") or rec.get("total_risk_score", 0.10)), 4),
                    "System_Decision": str(rec.get("system_decision") or rec.get("System_Decision") or rec.get("decision", "ALLOW")),
                    "Ground_Truth": int(rec.get("ground_truth") or rec.get("Ground_Truth") or rec.get("is_fraud") or 0),
                    "Is_Fuzzed": 1 if (rec.get("is_fuzzed") or rec.get("Is_Fuzzed")) else 0,
                    "Disagreement_Flag": 1 if (rec.get("disagreement_flag") or rec.get("Disagreement_Flag")) else 0,
                    "Action_Code": str(rec.get("action_code") or rec.get("Action_Code") or "APPROVE_FRICTIONLESS"),
                    "Reason_Codes": reasons_str
                }
                formatted_records.append(raw_record)
                self._in_memory_records.append(raw_record)

            with open(self.log_filepath, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.LOG_FIELDNAMES)
                writer.writerows(formatted_records)
                f.flush()

            return len(formatted_records)

    def get_dataframe(self) -> pd.DataFrame:
        """Returns the logged records as a pandas DataFrame."""
        with self._lock:
            if os.path.exists(self.log_filepath) and os.path.getsize(self.log_filepath) > 0:
                try:
                    return pd.read_csv(self.log_filepath)
                except Exception:
                    pass
            if self._in_memory_records:
                return pd.DataFrame(self._in_memory_records)
            return pd.DataFrame(columns=self.LOG_FIELDNAMES)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculates live operational telemetry metrics."""
        df = self.get_dataframe()
        if df.empty:
            return {
                "total_transactions": 0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "peak_tps": 0.0,
                "mean_tps": 0.0,
                "hard_blocks": 0,
                "step_ups": 0,
                "allows": 0,
                "false_declines": 0,
                "false_positive_rate_pct": 0.0,
                "fraud_detection_rate_pct": 0.0,
                "v1_evasions_count": 0,
                "v2_blocked_count": 0
            }

        total_tx = len(df)
        avg_lat = float(df["Processing_Latency_ms"].mean())
        p95_lat = float(df["Processing_Latency_ms"].quantile(0.95))
        p99_lat = float(df["Processing_Latency_ms"].quantile(0.99))
        peak_tps = float(df["TPS"].max())
        mean_tps = float(df["TPS"].mean())

        allows = int((df["System_Decision"] == "ALLOW").sum())
        step_ups = int((df["System_Decision"] == "STEP_UP").sum())
        hard_blocks = int((df["System_Decision"] == "HARD_BLOCK").sum())

        # Ground Truth Evaluation
        legit_mask = df["Ground_Truth"] == 0
        fraud_mask = df["Ground_Truth"] == 1
        
        # False Positive (False Decline) = Legitimate flagged as HARD_BLOCK
        false_declines = int(((df["System_Decision"] == "HARD_BLOCK") & legit_mask).sum())
        fpr = (false_declines / max(legit_mask.sum(), 1)) * 100.0

        # Fraud Detection Rate = Fraud caught in STEP_UP or HARD_BLOCK
        caught_fraud = int((df["System_Decision"].isin(["STEP_UP", "HARD_BLOCK"]) & fraud_mask).sum())
        fraud_det_rate = (caught_fraud / max(fraud_mask.sum(), 1)) * 100.0

        # Fuzzed Attack stats
        fuzzed_v1 = df[(df["Is_Fuzzed"] == 1) & (df["Model_Version"] == "Blue_V1")]
        fuzzed_v2 = df[(df["Is_Fuzzed"] == 1) & (df["Model_Version"] == "Blue_V2")]
        
        v1_evasions = int((fuzzed_v1["System_Decision"] == "ALLOW").sum()) if not fuzzed_v1.empty else 0
        v2_blocks = int((fuzzed_v2["System_Decision"] == "HARD_BLOCK").sum()) if not fuzzed_v2.empty else 0

        return {
            "total_transactions": total_tx,
            "avg_latency_ms": round(avg_lat, 3),
            "p95_latency_ms": round(p95_lat, 3),
            "p99_latency_ms": round(p99_lat, 3),
            "peak_tps": round(peak_tps, 2),
            "mean_tps": round(mean_tps, 2),
            "hard_blocks": hard_blocks,
            "step_ups": step_ups,
            "allows": allows,
            "false_declines": false_declines,
            "false_positive_rate_pct": round(fpr, 4),
            "fraud_detection_rate_pct": round(fraud_det_rate, 2),
            "v1_evasions_count": v1_evasions,
            "v2_blocked_count": v2_blocks
        }

    def reset(self):
        """Clears the log file and internal state."""
        with self._lock:
            self._total_logged = 0
            self._start_time = time.time()
            self._last_tps_timestamp = time.time()
            self._window_tx_count = 0
            self._current_tps = 0.0
            self._in_memory_records.clear()
            if os.path.exists(self.log_filepath):
                try:
                    os.remove(self.log_filepath)
                except Exception:
                    pass
            self._initialize_log_file()


# Module-level default singleton
GLOBAL_LOGGER = MetricsLogger()

if __name__ == "__main__":
    print("[*] Testing MetricsLogger...")
    test_logger = MetricsLogger("scratch/test_metrics.csv")
    test_logger.reset()
    
    test_logger.log_transaction(
        transaction_id="TX_TEST_001",
        model_version="Blue_V1",
        processing_latency_ms=4.12,
        score_tabular=0.12,
        score_graph=0.08,
        score_biometric=0.05,
        score_text=0.02,
        combined_risk_score=0.09,
        system_decision="ALLOW",
        ground_truth=0
    )

    stats = test_logger.get_summary_statistics()
    print(f"[+] MetricsLogger verified successfully. Logged 1 record. Stats: {stats}")
    if os.path.exists("scratch/test_metrics.csv"):
        os.remove("scratch/test_metrics.csv")
