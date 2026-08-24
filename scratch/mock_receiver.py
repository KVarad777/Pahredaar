"""
Project AEGIS: High-Speed Mock TCP Receiver for stress testing & validation
"""
import sys
import socket
import json
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

PORT = 8008
TARGET_COUNT = 5000

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', PORT))
    server.listen(1)
    print(f"[*] Mock Receiver listening on 127.0.0.1:{PORT}")

    conn, addr = server.accept()
    print(f"[+] Connected to simulator from {addr}")

    buffer = ""
    received_count = 0
    fraud_count = 0
    start_time = time.time()

    sample_tx = None

    with conn:
        while received_count < TARGET_COUNT:
            data = conn.recv(65536)
            if not data:
                break
            buffer += data.decode('utf-8', errors='replace')
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    received_count += 1
                    if payload.get('IsFraud') == 1:
                        fraud_count += 1
                    if received_count == 1:
                        sample_tx = payload
                except Exception as e:
                    print(f"[!] JSON parse error: {e}")
                
                if received_count >= TARGET_COUNT:
                    break

    elapsed = time.time() - start_time
    tps = received_count / max(elapsed, 0.0001)
    print("\n" + "="*70)
    print("MOCK RECEIVER VALIDATION REPORT")
    print("="*70)
    print(f"Total Transactions Received: {received_count}")
    print(f"Fraud Attacks Detected:      {fraud_count} ({(fraud_count*100.0/max(received_count,1)):.2f}%)")
    print(f"Elapsed Time:                {elapsed:.3f} seconds")
    print(f"Ingestion Throughput:        {tps:.1f} TPS")
    print("\nSample Ingested JSON Schema:")
    print(json.dumps(sample_tx, indent=2))
    print("="*70)
    server.close()

if __name__ == "__main__":
    main()
