"""
Automated Report Generation — Module 4.

Builds the immutable, structured compliance record for a single detected
violation and persists it to both SQLite (queryable source of truth, used by
the dashboard) and an append-only CSV audit log (human-portable export
format), satisfying the "Output Formats" requirement with two of the three
allowed formats simultaneously.

Reports are never constructed by a human; build_report() is the only path
that creates one, and it is always called from the escalation pipeline
immediately after severity classification.
"""
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

CSV_LOG_PATH = Path(__file__).resolve().parents[2] / "outputs" / "compliance_log.csv"

CSV_FIELDS = [
    "event_id", "timestamp", "clip_id", "zone", "behavior_class",
    "policy_rule_ref", "event_description", "severity", "escalation_action",
]


def build_report(detection, severity, escalation_action, policy_rules_by_class):
    """
    detection: raw detection dict from Module 1 (see detection/detector.py),
               containing class_id, clip_id, clip_timestamp_sec, zone,
               description, context.
    severity: Severity enum/string from Module 2.
    escalation_action: str describing the routing action taken (Module 3
                        decides this string; passed in here rather than
                        re-derived, so the report always matches what
                        actually happened).
    """
    rule = policy_rules_by_class[detection["class_id"]]
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clip_id": detection["clip_id"],
        "clip_timestamp_sec": detection.get("clip_timestamp_sec"),
        "zone": detection.get("zone", "Unknown"),
        "class_id": detection["class_id"],
        "behavior_class": rule["unsafe_behavior"],
        "policy_rule_ref": rule["policy_rule_ref"],
        "event_description": detection["description"],
        "severity": severity.value if hasattr(severity, "value") else severity,
        "escalation_action": escalation_action,
        "context": detection.get("context", {}),
    }


def append_to_csv(report, csv_path=CSV_LOG_PATH):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: report[k] for k in CSV_FIELDS})
