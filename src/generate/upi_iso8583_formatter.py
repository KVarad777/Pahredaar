"""
Formats an internal transaction dict into:
  (a) the canonical UPI JSON shape (matches config/upi_schema.json), and
  (b) an annotated ISO 8583-style field mapping, for wire-level fidelity demos.

Not every txn needs (b) - it's there so you can show judges "our simulator can
speak the actual byte-level format NPCI's switch uses," per the reference doc's
Section 2.3 raw-message example, without emitting raw ISO 8583 for every row.
"""

from typing import Optional


# DE = Data Element number in ISO 8583, per the reference doc's field catalog.
ISO8583_FIELD_MAP = {
    "2": "PAN (not applicable to UPI - no card number; VPA used instead)",
    "3": "processing_code",
    "4": "amount",
    "7": "timestamp",
    "11": "txnId (proxy for System Trace Audit Number)",
    "12_13": "timestamp (local time/date components)",
    "18": "mcc",
    "22": "channel (maps to POS entry mode: CNP/CP/ATM/P2P)",
    "32": "payeeVpa (proxy for acquiring institution ID in UPI context)",
    "41": "deviceDetails.deviceId (proxy for terminal ID)",
    "42": "payeeVpa (merchant ID proxy)",
    "43": "deviceDetails.geocode (merchant/acceptor location proxy)",
    "48": "deviceDetails (private field - this is exactly where UPI stuffs device/geo/IP tags)",
    "49": "currency",
    "62_63": "deviceDetails (private/reserved - UPI geocode/IP/OS tags typically live here)",
}


def to_upi_json(txn: dict) -> dict:
    """Strip internal-only fields, emit the canonical UPI JSON shape."""
    internal_only = {
        "account_id", "channel", "auth_result", "scenario_id", "scenario_name",
        "caught_by_model", "f3_tactic", "f3_technique", "is_fraud",
        "device_fingerprint_was_null", "ip_asn_risk_score_was_null", "is_new_account",
        "graph_centrality", "login_time_deviation_hrs", "true_mcc_estimate",
        "mean_inter_txn_seconds", "geo_velocity_kmh", "ip_asn_risk_score",
        "kyc_doc_similarity_score", "kyc_verification_method", "email_domain_risk_score",
        "account_age_days", "failed_auth_count_24h", "processing_code",
    }
    return {k: v for k, v in txn.items() if k not in internal_only}


def to_iso8583_annotation(txn: dict) -> dict:
    """
    Produces a human-readable {DE_number: (field_name, value)} annotation showing
    which ISO 8583 data elements this transaction would populate on a real switch.
    This is documentation-level fidelity, not a byte-packer - sufficient for a
    hackathon demo slide, not for an actual wire simulator.
    """
    annotated = {}
    for de, mapped_field in ISO8583_FIELD_MAP.items():
        base_field = mapped_field.split(" ")[0].split(".")[0]
        if base_field == "deviceDetails":
            value = txn.get("deviceDetails")
        else:
            value = txn.get(base_field, txn.get(base_field.split("_")[0]))
        annotated[f"DE{de}"] = {"maps_to": mapped_field, "value": value}
    mti = "0200" if not txn.get("is_refund") else "0220"  # financial request vs reversal/refund-style
    return {"mti": mti, "data_elements": annotated}


if __name__ == "__main__":
    import json
    sample = {
        "txnId": "AXIe4f2a1b9c8d7e6f5",
        "payerVpa": "user@oksbi",
        "payeeVpa": "merchant@paytm",
        "amount": 15000,
        "currency": "INR",
        "timestamp": "2026-08-26T14:32:11+05:30",
        "deviceDetails": {"deviceId": "hash_abc123", "os": "Android 14",
                           "geocode": "18.5204,73.8567", "ipAddress": "hash_of_ip",
                           "appId": "com.phonepe.app"},
        "remarks": "Payment for order #4521",
        "mcc": "5411",
        "channel": "CNP",
        "initiationMode": "01",
        "purposeCode": "00",
        "is_refund": False,
    }
    print("UPI JSON:")
    print(json.dumps(to_upi_json(sample), indent=2))
    print("\nISO 8583 annotation:")
    print(json.dumps(to_iso8583_annotation(sample), indent=2))
